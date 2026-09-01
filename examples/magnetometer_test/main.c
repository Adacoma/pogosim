// magnetometer_angle_display.c
// Pogobot/Pogosim controller:
//   1. robustly calibrate the magnetometer while rotating,
//   2. repeatedly stop/settle/read the calibrated heading,
//   3. display heading as an HSV LED color,
//   4. periodically rotate and repeat indefinitely.

#include "pogobase.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265f
#endif

#define PI_F 3.14159265f

/* ------------------------------------------------------------------------- */
/* Calibration configuration                                                  */
/* ------------------------------------------------------------------------- */

#define N_CAL 120
#define N_CAL_MIN 60
#define N_AVG_CAL 8
#define N_AVG_MES 12

#define N_BINS 36
#define N_BINS_MIN 12
#define N_STRAT_ITERS 2

#define STEP_MS_INIT 250
#define STEP_MS_MAX 1200
#define STEP_MS_INC 75
#define SETTLE_MS 250

#define MAG_TIMEOUT_MS 50
#define MAG_RETRY_MS 20
#define MIN_DELTA_MAG_SQ 100.0f
#define GLOBAL_MAX_ATTEMPTS (4 * N_CAL)

#define JACOBI_MAX_SWEEPS 15
#define JACOBI_EPSILON 1e-6f

/* ------------------------------------------------------------------------- */
/* Post-calibration behavior                                                  */
/* ------------------------------------------------------------------------- */

/* Each cycle is:
 *   stop -> settle -> robust heading read -> show angle -> rotate -> repeat.
 */
static uint32_t measurement_settle_ms = 350;
static uint32_t angle_display_ms = 700;
static uint32_t rotation_ms = 300;

static int calibration_turn_speed = motorHalf;
static int measurement_turn_speed = motorHalf;

/* main(10).c convention: calibrated magnetic angle is converted to the common
 * swarm/body convention using a sign and offset before being displayed. */
static float magnetometer_heading_offset_rad = 0.0f;
static float magnetometer_heading_sign = 1.0f;

#ifndef ENABLE_LIVE_DIAGNOSTICS
#define ENABLE_LIVE_DIAGNOSTICS 0
#endif

#define DIAG_PRINTF(...) do { \
    if (ENABLE_LIVE_DIAGNOSTICS) { \
        printf(__VA_ARGS__); \
    } \
} while (0)

#define FLOAT_TO_INT_FMT "%s%d.%02d"
#define FLOAT_TO_INT_ARGS(v) \
    (((v) < 0.0f && (v) > -1.0f) ? "-" : ""), \
    ((int)(v)), \
    ((int)(((v) < 0.0f ? -(v) : (v)) * 100.0f) % 100)

typedef enum {
    STATE_CAL_ROTATING = 0,
    STATE_CAL_SETTLING,
    STATE_CAL_READING,
    STATE_MEAS_SETTLING,
    STATE_MEAS_READING,
    STATE_MEAS_DISPLAY,
    STATE_MEAS_ROTATING,
    STATE_FATAL
} controller_state_t;

typedef enum {
    MAG_READ_PENDING = 0,
    MAG_READ_SUCCESS,
    MAG_READ_FAILURE
} mag_read_result_t;

typedef struct {
    uint8_t motor_dir_left_fwd;
    uint8_t motor_dir_right_fwd;

    controller_state_t state;
    uint32_t state_deadline_ms;

    int n_collected;
    int step_ms;
    int global_attempts;
    float last_cal_x;
    float last_cal_y;
    float last_cal_z;
    bool have_last_cal;

    int16_t cal_pts[N_CAL][3];
    float u_buf[N_CAL];
    float v_buf[N_CAL];
    float bin_su[N_BINS];
    float bin_sv[N_BINS];
    int bin_n[N_BINS];
    float bin_mu[N_BINS];
    float bin_mv[N_BINS];

    bool mag_read_active;
    int mag_n_target;
    int mag_n;
    int mag_attempts;
    uint32_t mag_next_ms;
    int16_t mag_bx[16];
    int16_t mag_by[16];
    int16_t mag_bz[16];

    float mean[3];
    float e1[3];
    float e2[3];
    float u0;
    float v0;
    float w2[2][2];
    float el1;
    float el2;
    float ev1x;
    float ev1y;
    float k_norm;
    float s_norm;
    bool fit_ok;

    float pending_x;
    float pending_y;
    float pending_z;
    float last_heading_rad;
    bool heading_valid;
} USERDATA;

DECLARE_USERDATA(USERDATA);
REGISTER_USERDATA(USERDATA);

/* ------------------------------------------------------------------------- */
/* Timing / motor helpers                                                     */
/* ------------------------------------------------------------------------- */

static uint32_t now_ms(void) {
    return (uint32_t)current_time_milliseconds();
}

static uint32_t deadline_after_ms(uint32_t delay_ms) {
    return now_ms() + delay_ms;
}

static bool deadline_reached(uint32_t deadline_ms) {
    return (int32_t)(now_ms() - deadline_ms) >= 0;
}

static float wrap_pi(float a) {
    while (a > PI_F) {
        a -= 2.0f * PI_F;
    }
    while (a < -PI_F) {
        a += 2.0f * PI_F;
    }
    return a;
}

static int16_t f2i16(float v) {
    return (int16_t)(v >= 0.0f ? v + 0.5f : v - 0.5f);
}

static void motor_stop(void) {
    pogobot_motor_set(motorL, motorStop);
    pogobot_motor_set(motorR, motorStop);
}

static void motor_set_signed(motor_id id, int speed_signed, uint8_t fwd_dir) {
    int magnitude = speed_signed >= 0 ? speed_signed : -speed_signed;
    if (magnitude > motorFull) {
        magnitude = motorFull;
    }

    uint8_t dir = speed_signed >= 0 ? fwd_dir : (fwd_dir == 0 ? 1 : 0);
    pogobot_motor_dir_set(id, dir);
    pogobot_motor_set(id, magnitude);
}

static void rotate_start(int speed) {
    motor_set_signed(motorL, speed, mydata->motor_dir_left_fwd);
    motor_set_signed(motorR, -speed, mydata->motor_dir_right_fwd);
}

static float median_i16(int16_t *a, int n) {
    for (int i = 1; i < n; i++) {
        int16_t key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = key;
    }

    if (n & 1) {
        return (float)a[n / 2];
    }
    return 0.5f * ((float)a[n / 2 - 1] + (float)a[n / 2]);
}

/* ------------------------------------------------------------------------- */
/* Non-blocking robust magnetometer reader                                    */
/* ------------------------------------------------------------------------- */

static void mag_read_robust_start(int n_target) {
    int16_t mx;
    int16_t my;
    int16_t mz;

    if (n_target > 16) {
        n_target = 16;
    }
    if (n_target < 1) {
        n_target = 1;
    }

    mydata->mag_n_target = n_target;
    mydata->mag_n = 0;
    mydata->mag_attempts = 0;
    mydata->mag_read_active = true;
    mydata->mag_next_ms = now_ms();

    /* Purge/starter read, as in the calibration controller. */
    magn_read_XYZ(&mx, &my, &mz, 0);
}

static mag_read_result_t mag_read_robust_step(float *x, float *y, float *z) {
    if (!mydata->mag_read_active) {
        return MAG_READ_FAILURE;
    }

    if (!deadline_reached(mydata->mag_next_ms)) {
        return MAG_READ_PENDING;
    }

    int16_t mx;
    int16_t my;
    int16_t mz;
    mydata->mag_attempts++;

    if (magn_read_XYZ(&mx, &my, &mz, MAG_TIMEOUT_MS) == 0) {
        int n = mydata->mag_n;
        mydata->mag_bx[n] = mx;
        mydata->mag_by[n] = my;
        mydata->mag_bz[n] = mz;
        mydata->mag_n++;
    }

    bool enough_samples = mydata->mag_n >= mydata->mag_n_target;
    bool attempts_exhausted = mydata->mag_attempts >= 4 * mydata->mag_n_target;

    if (enough_samples || attempts_exhausted) {
        mydata->mag_read_active = false;

        if (mydata->mag_n == 0) {
            return MAG_READ_FAILURE;
        }

        *x = median_i16(mydata->mag_bx, mydata->mag_n);
        *y = median_i16(mydata->mag_by, mydata->mag_n);
        *z = median_i16(mydata->mag_bz, mydata->mag_n);
        return MAG_READ_SUCCESS;
    }

    mydata->mag_next_ms = deadline_after_ms(MAG_RETRY_MS);
    return MAG_READ_PENDING;
}

/* ------------------------------------------------------------------------- */
/* Calibration linear algebra                                                 */
/* ------------------------------------------------------------------------- */

static void evd_jacobi_3x3(float c[3][3], float v[3][3], float l[3]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            v[i][j] = i == j ? 1.0f : 0.0f;
        }
    }

    for (int sweep = 0; sweep < JACOBI_MAX_SWEEPS; sweep++) {
        float max_off = 0.0f;
        int p_arr[] = {0, 0, 1};
        int q_arr[] = {1, 2, 2};

        for (int k = 0; k < 3; k++) {
            int p = p_arr[k];
            int q = q_arr[k];
            float abs_cpq = fabsf(c[p][q]);

            if (abs_cpq > max_off) {
                max_off = abs_cpq;
            }

            if (abs_cpq > JACOBI_EPSILON) {
                float theta = (c[q][q] - c[p][p]) / (2.0f * c[p][q]);
                float t = 1.0f / (fabsf(theta) + sqrtf(1.0f + theta * theta));
                if (theta < 0.0f) {
                    t = -t;
                }

                float co = 1.0f / sqrtf(1.0f + t * t);
                float si = t * co;
                float cpp = c[p][p];
                float cqq = c[q][q];
                float cpq = c[p][q];

                c[p][p] = co * co * cpp - 2.0f * si * co * cpq + si * si * cqq;
                c[q][q] = si * si * cpp + 2.0f * si * co * cpq + co * co * cqq;
                c[p][q] = 0.0f;
                c[q][p] = 0.0f;

                for (int i = 0; i < 3; i++) {
                    if (i != p && i != q) {
                        float cip = c[i][p];
                        float ciq = c[i][q];
                        c[i][p] = co * cip - si * ciq;
                        c[p][i] = c[i][p];
                        c[i][q] = si * cip + co * ciq;
                        c[q][i] = c[i][q];
                    }

                    float vip = v[i][p];
                    float viq = v[i][q];
                    v[i][p] = co * vip - si * viq;
                    v[i][q] = si * vip + co * viq;
                }
            }
        }

        if (max_off < JACOBI_EPSILON) {
            break;
        }
    }

    l[0] = c[0][0];
    l[1] = c[1][1];
    l[2] = c[2][2];
}

static bool solve5(float a[5][5], float b[5], float x[5]) {
    for (int col = 0; col < 5; col++) {
        int piv = col;
        float amax = fabsf(a[col][col]);

        for (int r = col + 1; r < 5; r++) {
            float ar = fabsf(a[r][col]);
            if (ar > amax) {
                amax = ar;
                piv = r;
            }
        }

        if (amax < 1e-9f) {
            return false;
        }

        if (piv != col) {
            for (int c = col; c < 5; c++) {
                float t = a[col][c];
                a[col][c] = a[piv][c];
                a[piv][c] = t;
            }
            float t = b[col];
            b[col] = b[piv];
            b[piv] = t;
        }

        for (int r = col + 1; r < 5; r++) {
            float f = a[r][col] / a[col][col];
            for (int c = col; c < 5; c++) {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }

    for (int r = 4; r >= 0; r--) {
        float s = b[r];
        for (int c = r + 1; c < 5; c++) {
            s -= a[r][c] * x[c];
        }
        x[r] = s / a[r][r];
    }

    return true;
}

static bool fit_conic(const float *u, const float *v, int n, float s, float coef[5]) {
    float normal[5][5] = {{0}};
    float rhs[5] = {0};

    for (int i = 0; i < n; i++) {
        float un = u[i] / s;
        float vn = v[i] / s;
        float phi[5] = {un * un, un * vn, vn * vn, un, vn};

        for (int r = 0; r < 5; r++) {
            rhs[r] += phi[r];
            for (int col = 0; col < 5; col++) {
                normal[r][col] += phi[r] * phi[col];
            }
        }
    }

    return solve5(normal, rhs, coef);
}

static bool conic_center(const float coef[5], float *u0n, float *v0n) {
    float det = 4.0f * coef[0] * coef[2] - coef[1] * coef[1];
    if (det <= 1e-9f || coef[0] <= 0.0f || coef[2] <= 0.0f) {
        return false;
    }

    *u0n = (-2.0f * coef[2] * coef[3] + coef[1] * coef[4]) / det;
    *v0n = (coef[1] * coef[3] - 2.0f * coef[0] * coef[4]) / det;
    return true;
}

static void eig_sym_2x2(float axx, float axy, float ayy,
                        float *l1, float *l2, float *v1x, float *v1y) {
    float trace = axx + ayy;
    float det = axx * ayy - axy * axy;
    float diff = axx - ayy;
    float sd = sqrtf(diff * diff + 4.0f * axy * axy);

    *l1 = 0.5f * (trace + sd);
    *l2 = *l1 > 1e-12f ? det / *l1 : 0.0f;

    float c1x = axy;
    float c1y = *l1 - axx;
    float c2x = *l1 - ayy;
    float c2y = axy;
    float n1 = c1x * c1x + c1y * c1y;
    float n2 = c2x * c2x + c2y * c2y;

    if (n1 > n2 && n1 > 0.0f) {
        float inv = 1.0f / sqrtf(n1);
        *v1x = c1x * inv;
        *v1y = c1y * inv;
    } else if (n2 > 0.0f) {
        float inv = 1.0f / sqrtf(n2);
        *v1x = c2x * inv;
        *v1y = c2y * inv;
    } else {
        *v1x = 1.0f;
        *v1y = 0.0f;
    }
}

static bool sqrtm_2x2(float axx, float axy, float ayy, float w[2][2]) {
    float l1;
    float l2;
    float vx;
    float vy;
    eig_sym_2x2(axx, axy, ayy, &l1, &l2, &vx, &vy);

    if (l1 <= 0.0f || l2 <= 0.0f) {
        return false;
    }

    float wx = -vy;
    float wy = vx;
    float s1 = sqrtf(l1);
    float s2 = sqrtf(l2);

    w[0][0] = s1 * vx * vx + s2 * wx * wx;
    w[0][1] = s1 * vx * vy + s2 * wx * wy;
    w[1][0] = w[0][1];
    w[1][1] = s1 * vy * vy + s2 * wy * wy;
    return true;
}

/* main(10).c improvement over the standalone calibration source: anchor the
 * in-plane X axis to the robot/sensor body X direction rather than selecting
 * the PCA major axis, whose sign/phase is data-dependent. */
static bool build_body_anchored_plane_basis(float nx, float ny, float nz) {
    float dot_x = nx;
    float ex = 1.0f - dot_x * nx;
    float ey = -dot_x * ny;
    float ez = -dot_x * nz;
    float norm = sqrtf(ex * ex + ey * ey + ez * ez);

    if (norm < 1e-3f) {
        float dot_y = ny;
        ex = -dot_y * nx;
        ey = 1.0f - dot_y * ny;
        ez = -dot_y * nz;
        norm = sqrtf(ex * ex + ey * ey + ez * ez);
    }

    if (norm < 1e-3f) {
        return false;
    }

    mydata->e1[0] = ex / norm;
    mydata->e1[1] = ey / norm;
    mydata->e1[2] = ez / norm;

    mydata->e2[0] = ny * mydata->e1[2] - nz * mydata->e1[1];
    mydata->e2[1] = nz * mydata->e1[0] - nx * mydata->e1[2];
    mydata->e2[2] = nx * mydata->e1[1] - ny * mydata->e1[0];
    return true;
}

static float calibrated_heading_deg(float mx, float my, float mz) {
    float px = mx - mydata->mean[0];
    float py = my - mydata->mean[1];
    float pz = mz - mydata->mean[2];

    float u = mydata->e1[0] * px + mydata->e1[1] * py + mydata->e1[2] * pz - mydata->u0;
    float v = mydata->e2[0] * px + mydata->e2[1] * py + mydata->e2[2] * pz - mydata->v0;

    float xc = mydata->w2[0][0] * u + mydata->w2[0][1] * v;
    float yc = mydata->w2[1][0] * u + mydata->w2[1][1] * v;

    float raw_rad = atan2f(yc, xc);
    float heading_rad = wrap_pi(magnetometer_heading_sign * raw_rad +
                                magnetometer_heading_offset_rad);
    if (heading_rad < 0.0f) {
        heading_rad += 2.0f * PI_F;
    }

    return heading_rad * (180.0f / PI_F);
}

/* ------------------------------------------------------------------------- */
/* Angle LED                                                                  */
/* ------------------------------------------------------------------------- */

static void set_angle_led(float angle_deg) {
    while (angle_deg >= 360.0f) {
        angle_deg -= 360.0f;
    }
    while (angle_deg < 0.0f) {
        angle_deg += 360.0f;
    }

    uint8_t r8;
    uint8_t g8;
    uint8_t b8;
    hsv_to_rgb(angle_deg, 1.0f, 1.0f, &r8, &g8, &b8);

    r8 = SCALE_0_255_TO_0_25(r8);
    g8 = SCALE_0_255_TO_0_25(g8);
    b8 = SCALE_0_255_TO_0_25(b8);

    if (r8 == 0 && g8 == 0 && b8 == 0) {
        r8 = 1;
    }

    pogobot_led_setColor(r8, g8, b8);
}

/* ------------------------------------------------------------------------- */
/* Calibration workflow                                                       */
/* ------------------------------------------------------------------------- */

static void enter_fatal_state(const char *reason) {
    motor_stop();
    mydata->fit_ok = false;
    mydata->heading_valid = false;
    mydata->state = STATE_FATAL;
    pogobot_led_setColor(255, 0, 255);

#ifdef SIMULATOR
    printf("Robot %u magnetometer controller fatal: %s\n",
           (unsigned)pogobot_helper_getid(), reason);
#else
    DIAG_PRINTF("# FATAL: %s\n", reason);
#endif
}

static void calibration_begin_rotation(void);

static void measurement_begin_settle(void) {
    motor_stop();
    mydata->state_deadline_ms = deadline_after_ms(measurement_settle_ms);
    mydata->state = STATE_MEAS_SETTLING;
}

static void calibration_fit_and_start_measurements(void) {
    const int n_used = mydata->n_collected;

    for (int k = 0; k < 3; k++) {
        mydata->mean[k] = 0.0f;
    }

    for (int i = 0; i < n_used; i++) {
        mydata->mean[0] += (float)mydata->cal_pts[i][0];
        mydata->mean[1] += (float)mydata->cal_pts[i][1];
        mydata->mean[2] += (float)mydata->cal_pts[i][2];
    }

    mydata->mean[0] /= (float)n_used;
    mydata->mean[1] /= (float)n_used;
    mydata->mean[2] /= (float)n_used;

    float cmat[3][3] = {{0}};
    for (int i = 0; i < n_used; i++) {
        float dx = (float)mydata->cal_pts[i][0] - mydata->mean[0];
        float dy = (float)mydata->cal_pts[i][1] - mydata->mean[1];
        float dz = (float)mydata->cal_pts[i][2] - mydata->mean[2];

        cmat[0][0] += dx * dx;
        cmat[1][1] += dy * dy;
        cmat[2][2] += dz * dz;
        cmat[0][1] += dx * dy;
        cmat[0][2] += dx * dz;
        cmat[1][2] += dy * dz;
    }

    cmat[0][0] /= (float)n_used;
    cmat[1][1] /= (float)n_used;
    cmat[2][2] /= (float)n_used;
    cmat[0][1] /= (float)n_used;
    cmat[0][2] /= (float)n_used;
    cmat[1][2] /= (float)n_used;
    cmat[1][0] = cmat[0][1];
    cmat[2][0] = cmat[0][2];
    cmat[2][1] = cmat[1][2];

    float eig_vec[3][3];
    float eig_val[3];
    evd_jacobi_3x3(cmat, eig_vec, eig_val);

    int i_min = 0;
    if (eig_val[1] < eig_val[i_min]) {
        i_min = 1;
    }
    if (eig_val[2] < eig_val[i_min]) {
        i_min = 2;
    }

    float nx = eig_vec[0][i_min];
    float ny = eig_vec[1][i_min];
    float nz = eig_vec[2][i_min];

    if (nz < 0.0f) {
        nx = -nx;
        ny = -ny;
        nz = -nz;
    }

    if (!build_body_anchored_plane_basis(nx, ny, nz)) {
        enter_fatal_state("could not construct magnetometer plane basis");
        return;
    }

    mydata->s_norm = 0.0f;
    for (int i = 0; i < n_used; i++) {
        float px = (float)mydata->cal_pts[i][0] - mydata->mean[0];
        float py = (float)mydata->cal_pts[i][1] - mydata->mean[1];
        float pz = (float)mydata->cal_pts[i][2] - mydata->mean[2];

        float u = mydata->e1[0] * px + mydata->e1[1] * py + mydata->e1[2] * pz;
        float v = mydata->e2[0] * px + mydata->e2[1] * py + mydata->e2[2] * pz;

        mydata->u_buf[i] = u;
        mydata->v_buf[i] = v;
        mydata->s_norm += sqrtf(u * u + v * v);
    }

    mydata->s_norm /= (float)n_used;
    if (mydata->s_norm < 1.0f) {
        mydata->s_norm = 1.0f;
    }

    float coef[5] = {0};
    mydata->u0 = 0.0f;
    mydata->v0 = 0.0f;
    mydata->fit_ok = false;

    if (fit_conic(mydata->u_buf, mydata->v_buf, n_used, mydata->s_norm, coef)) {
        float u0n;
        float v0n;
        if (conic_center(coef, &u0n, &v0n)) {
            mydata->u0 = mydata->s_norm * u0n;
            mydata->v0 = mydata->s_norm * v0n;
            mydata->fit_ok = true;
        }
    }

    int n_bins_used = 0;
    for (int it = 0; it < N_STRAT_ITERS && mydata->fit_ok; it++) {
        for (int b = 0; b < N_BINS; b++) {
            mydata->bin_su[b] = 0.0f;
            mydata->bin_sv[b] = 0.0f;
            mydata->bin_n[b] = 0;
        }

        for (int i = 0; i < n_used; i++) {
            float phi = atan2f(mydata->v_buf[i] - mydata->v0,
                               mydata->u_buf[i] - mydata->u0);
            int b = (int)((phi + PI_F) * ((float)N_BINS / (2.0f * PI_F)));
            if (b < 0) {
                b = 0;
            }
            if (b >= N_BINS) {
                b = N_BINS - 1;
            }

            mydata->bin_su[b] += mydata->u_buf[i];
            mydata->bin_sv[b] += mydata->v_buf[i];
            mydata->bin_n[b]++;
        }

        int m = 0;
        for (int b = 0; b < N_BINS; b++) {
            if (mydata->bin_n[b] > 0) {
                mydata->bin_mu[m] = mydata->bin_su[b] / (float)mydata->bin_n[b];
                mydata->bin_mv[m] = mydata->bin_sv[b] / (float)mydata->bin_n[b];
                m++;
            }
        }

        n_bins_used = m;
        if (m < N_BINS_MIN) {
            break;
        }

        float coef2[5];
        if (fit_conic(mydata->bin_mu, mydata->bin_mv, m, mydata->s_norm, coef2)) {
            float u0n;
            float v0n;
            if (conic_center(coef2, &u0n, &v0n)) {
                for (int k = 0; k < 5; k++) {
                    coef[k] = coef2[k];
                }
                mydata->u0 = mydata->s_norm * u0n;
                mydata->v0 = mydata->s_norm * v0n;
            }
        }
    }

    mydata->el1 = 1.0f;
    mydata->el2 = 1.0f;
    mydata->ev1x = 1.0f;
    mydata->ev1y = 0.0f;
    mydata->k_norm = 1.0f;
    mydata->w2[0][0] = 1.0f;
    mydata->w2[0][1] = 0.0f;
    mydata->w2[1][0] = 0.0f;
    mydata->w2[1][1] = 1.0f;

    if (mydata->fit_ok) {
        float c_a = coef[0];
        float c_b = coef[1];
        float c_c = coef[2];
        float u0n = mydata->u0 / mydata->s_norm;
        float v0n = mydata->v0 / mydata->s_norm;

        eig_sym_2x2(c_a, 0.5f * c_b, c_c,
                    &mydata->el1, &mydata->el2,
                    &mydata->ev1x, &mydata->ev1y);

        mydata->k_norm = 1.0f +
            c_a * u0n * u0n + c_b * u0n * v0n + c_c * v0n * v0n;

        if (!sqrtm_2x2(c_a, 0.5f * c_b, c_c, mydata->w2)) {
            mydata->fit_ok = false;
        }
    }

    if (!mydata->fit_ok) {
        enter_fatal_state("magnetometer conic fit failed");
        return;
    }

    DIAG_PRINTF("# Calibration OK: %d points, %d attempts, %d occupied bins\n",
                n_used, mydata->global_attempts, n_bins_used);

    motor_stop();
    mydata->heading_valid = false;
    measurement_begin_settle();
}

static void calibration_finish_collection(void) {
    motor_stop();

    if (mydata->n_collected < N_CAL_MIN) {
        enter_fatal_state("too few valid magnetometer calibration points");
        return;
    }

    calibration_fit_and_start_measurements();
}

static void calibration_begin_rotation(void) {
    if (mydata->n_collected >= N_CAL ||
        mydata->global_attempts >= GLOBAL_MAX_ATTEMPTS) {
        calibration_finish_collection();
        return;
    }

    mydata->global_attempts++;
    rotate_start(calibration_turn_speed);
    mydata->state_deadline_ms = deadline_after_ms((uint32_t)mydata->step_ms);
    mydata->state = STATE_CAL_ROTATING;
}

static void calibration_start(void) {
    memset(mydata, 0, sizeof(*mydata));

    uint8_t dir_mem[3] = {0, 0, 0};
    pogobot_motor_dir_mem_get(dir_mem);
    mydata->motor_dir_right_fwd = dir_mem[0];
    mydata->motor_dir_left_fwd = dir_mem[1];

    mydata->step_ms = STEP_MS_INIT;
    mydata->fit_ok = false;
    mydata->heading_valid = false;

    motor_stop();
    pogobot_led_setColor(255, 0, 255); /* magenta while calibrating */
    calibration_begin_rotation();
}

/* ------------------------------------------------------------------------- */
/* Post-calibration periodic measurement                                      */
/* ------------------------------------------------------------------------- */

static void print_and_display_heading(float x, float y, float z) {
    float angle_deg = calibrated_heading_deg(x, y, z);
    mydata->last_heading_rad = angle_deg * PI_F / 180.0f;
    mydata->heading_valid = true;

#ifdef SIMULATOR
    printf("ANGLE,%u," FLOAT_TO_INT_FMT "\n",
           (unsigned)pogobot_helper_getid(),
           FLOAT_TO_INT_ARGS(angle_deg));
#else
    printf("ANGLE," FLOAT_TO_INT_FMT "\n", FLOAT_TO_INT_ARGS(angle_deg));
#endif

    set_angle_led(angle_deg);
}

static void controller_step(void) {
    switch (mydata->state) {
    case STATE_CAL_ROTATING:
        if (!deadline_reached(mydata->state_deadline_ms)) {
            return;
        }
        motor_stop();
        mydata->state_deadline_ms = deadline_after_ms(SETTLE_MS);
        mydata->state = STATE_CAL_SETTLING;
        return;

    case STATE_CAL_SETTLING:
        if (!deadline_reached(mydata->state_deadline_ms)) {
            return;
        }
        mag_read_robust_start(N_AVG_CAL);
        mydata->state = STATE_CAL_READING;
        return;

    case STATE_CAL_READING: {
        float x;
        float y;
        float z;
        mag_read_result_t result = mag_read_robust_step(&x, &y, &z);

        if (result == MAG_READ_PENDING) {
            return;
        }
        if (result == MAG_READ_FAILURE) {
            calibration_begin_rotation();
            return;
        }

        if (mydata->have_last_cal) {
            float dx = x - mydata->last_cal_x;
            float dy = y - mydata->last_cal_y;
            float dz = z - mydata->last_cal_z;

            if (dx * dx + dy * dy + dz * dz < MIN_DELTA_MAG_SQ) {
                if (mydata->step_ms < STEP_MS_MAX) {
                    mydata->step_ms += STEP_MS_INC;
                    if (mydata->step_ms > STEP_MS_MAX) {
                        mydata->step_ms = STEP_MS_MAX;
                    }
                }
                calibration_begin_rotation();
                return;
            }
        }

        mydata->last_cal_x = x;
        mydata->last_cal_y = y;
        mydata->last_cal_z = z;
        mydata->have_last_cal = true;

        int i = mydata->n_collected;
        mydata->cal_pts[i][0] = f2i16(x);
        mydata->cal_pts[i][1] = f2i16(y);
        mydata->cal_pts[i][2] = f2i16(z);

        DIAG_PRINTF("CALPT,%d,%d,%d\n",
                    (int)mydata->cal_pts[i][0],
                    (int)mydata->cal_pts[i][1],
                    (int)mydata->cal_pts[i][2]);

        mydata->n_collected++;
        calibration_begin_rotation();
        return;
    }

    case STATE_MEAS_SETTLING:
        if (!deadline_reached(mydata->state_deadline_ms)) {
            return;
        }
        mag_read_robust_start(N_AVG_MES);
        mydata->state = STATE_MEAS_READING;
        return;

    case STATE_MEAS_READING: {
        float x;
        float y;
        float z;
        mag_read_result_t result = mag_read_robust_step(&x, &y, &z);

        if (result == MAG_READ_PENDING) {
            return;
        }

        if (result == MAG_READ_FAILURE) {
            DIAG_PRINTF("# [WARN] post-calibration magnetometer read failed\n");
            mydata->state_deadline_ms = deadline_after_ms(100);
            mydata->state = STATE_MEAS_SETTLING;
            return;
        }

        mydata->pending_x = x;
        mydata->pending_y = y;
        mydata->pending_z = z;
        print_and_display_heading(x, y, z);

        mydata->state_deadline_ms = deadline_after_ms(angle_display_ms);
        mydata->state = STATE_MEAS_DISPLAY;
        return;
    }

    case STATE_MEAS_DISPLAY:
        if (!deadline_reached(mydata->state_deadline_ms)) {
            return;
        }

        rotate_start(measurement_turn_speed);
        mydata->state_deadline_ms = deadline_after_ms(rotation_ms);
        mydata->state = STATE_MEAS_ROTATING;
        return;

    case STATE_MEAS_ROTATING:
        if (!deadline_reached(mydata->state_deadline_ms)) {
            return;
        }

        measurement_begin_settle();
        return;

    case STATE_FATAL:
    default:
        motor_stop();
        return;
    }
}

/* ------------------------------------------------------------------------- */
/* Pogobot callbacks                                                          */
/* ------------------------------------------------------------------------- */

void user_init(void) {
    srand(pogobot_helper_getRandSeed());

    main_loop_hz = 100;
    max_nb_processed_msg_per_tick = 0;
    percent_msgs_sent_per_ticks = 0;
    msg_rx_fn = NULL;
    msg_tx_fn = NULL;
    error_codes_led_idx = 3;

    calibration_start();
}

void user_step(void) {
    controller_step();
}

#ifdef SIMULATOR
static void create_data_schema(void) {
    data_add_column_int8("controller_state");
    data_add_column_int8("calibration_fit_ok");
    data_add_column_int8("heading_valid");
    data_add_column_double("heading_rad");
}

static void export_data(void) {
    data_set_value_int8("controller_state", (int8_t)mydata->state);
    data_set_value_int8("calibration_fit_ok", (int8_t)(mydata->fit_ok ? 1 : 0));
    data_set_value_int8("heading_valid", (int8_t)(mydata->heading_valid ? 1 : 0));
    data_set_value_double("heading_rad", mydata->last_heading_rad);
}

static void global_setup(void) {
    init_from_configuration(calibration_turn_speed);
    init_from_configuration(measurement_turn_speed);
    init_from_configuration(measurement_settle_ms);
    init_from_configuration(angle_display_ms);
    init_from_configuration(rotation_ms);
    init_from_configuration(magnetometer_heading_offset_rad);
}
#endif

int main(void) {
    pogobot_init();
    pogobot_start(user_init, user_step);
#ifdef SIMULATOR
    SET_CALLBACK(callback_global_setup, global_setup);
    SET_CALLBACK(callback_create_data_schema, create_data_schema);
    SET_CALLBACK(callback_export_data, export_data);
#endif
    return 0;
}
