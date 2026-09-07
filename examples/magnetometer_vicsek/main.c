#include "pogobase.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "pogo-utils/version.h"
#include "pogo-utils/wall_avoidance.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define PI_F 3.14159265f

/* ------------------------------------------------------------------------- */
/* Magnetometer calibration                                                   */
/* ------------------------------------------------------------------------- */

#define N_CAL 120
#define N_CAL_MIN 60
#define N_AVG_CAL 8
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

/* Recent raw magnetometer samples used during Vicsek locomotion. */
#define MAG_HEADING_WINDOW 5
#define MAG_HEADING_TIMEOUT_MS 20

#ifndef ENABLE_CALIBRATION_UART
#define ENABLE_CALIBRATION_UART 0
#endif

#define CAL_PRINTF(...) do { \
    if (ENABLE_CALIBRATION_UART) { \
        printf(__VA_ARGS__); \
    } \
} while (0)

/* ------------------------------------------------------------------------- */
/* Vicsek configuration                                                       */
/* ------------------------------------------------------------------------- */

#define MAX_NEIGHBORS 20u
#define BEACON_HZ 10u
#define BEACON_PERIOD_MS (1000u / BEACON_HZ)

static int forward_speed = 200; // motorHalf;
static int calibration_turn_speed = motorHalf;

uint32_t max_age = 600;
uint32_t vicsek_period_ms = 100;

float noise_eta_rad = 0.0f;
float align_gain = 1.0f;
bool include_self_in_avg = true;
bool broadcast_angle_when_avoiding_walls = true;
float vicsek_turn_gain = 0.8f;

bool vicsek_time_continuous = false;
float vicsek_beta_rad_per_s = 0.0f;
float cont_noise_sigma_rad = 0.0f;
float cont_max_dt_s = 0.05f;

/* Mandatory post-calibration hold requested for the experiment. */
uint32_t post_calibration_wait_ms = 5000;

/* The ellipse fit removes hard-/soft-iron distortion. The following parameters
 * express the resulting magnetic angle in the convention shared by the swarm.
 * "cw" means the calibrated angle already increases clockwise; "ccw" negates
 * it. The offset is applied after that sign conversion. */
float magnetometer_heading_offset_rad = 0.0f;
float magnetometer_heading_filter_gain = 1.0f;
uint32_t magnetometer_heading_max_age_ms = 500;
static float magnetometer_heading_sign = 1.0f;

uint32_t cluster_u_turn_duration_ms = 1500;
float phi_rad_min = 0.2f;
float phi_rad_max = 0.2f;

uint32_t wall_avoidance_memory_ms = 300;
uint32_t wall_avoidance_turn_duration_ms = 300;
uint32_t wall_avoidance_forward_commit_ms = 300;
float wall_avoidance_forward_speed_ratio = 0.5f;
wall_chirality_t wall_avoidance_chiralty_policy = WALL_MIN_TURN;

typedef enum {
    SHOW_STATE,
    SHOW_ANGLE
} main_led_display_type_t;

//main_led_display_type_t main_led_display_enum = SHOW_STATE;
main_led_display_type_t main_led_display_enum = SHOW_ANGLE;

#define VMSGF_CLUSTER_UTURN 0x01

typedef struct __attribute__((__packed__)) {
    uint16_t sender_id;
    int16_t theta_mrad;
    uint8_t flags;
    int16_t cluster_target_mrad;
    uint32_t cluster_wall_t0_ms;
    uint16_t cluster_msg_uid;
} vicsek_msg_t;

#define MSG_SIZE ((uint16_t)sizeof(vicsek_msg_t))

typedef struct {
    uint16_t id;
    uint32_t last_seen_ms;
    int16_t theta_mrad;
} neighbor_t;

typedef enum {
    CONTROLLER_CALIBRATING = 0,
    CONTROLLER_WAITING = 1,
    CONTROLLER_VICSEK = 2,
    CONTROLLER_FATAL = 3
} controller_state_t;

typedef enum {
    CAL_STATE_IDLE = 0,
    CAL_STATE_ROTATING,
    CAL_STATE_SETTLING,
    CAL_STATE_READING,
    CAL_STATE_DONE,
    CAL_STATE_FATAL
} calibration_state_t;

typedef enum {
    MAG_READ_PENDING = 0,
    MAG_READ_SUCCESS,
    MAG_READ_FAILURE
} mag_read_result_t;

typedef struct {
    uint8_t motor_dir_left_fwd;
    uint8_t motor_dir_right_fwd;
    uint16_t motor_power_left;
    uint16_t motor_power_right;

    controller_state_t controller_state;
    uint32_t controller_deadline_ms;

    calibration_state_t cal_state;
    uint32_t cal_deadline_ms;
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

    int16_t heading_mag_x[MAG_HEADING_WINDOW];
    int16_t heading_mag_y[MAG_HEADING_WINDOW];
    int16_t heading_mag_z[MAG_HEADING_WINDOW];
    uint8_t heading_mag_count;
    uint8_t heading_mag_pos;
    int16_t last_mag_x;
    int16_t last_mag_y;
    int16_t last_mag_z;
    float magnetometer_heading_rad;
    bool magnetometer_heading_valid;
    uint32_t last_magnetometer_heading_ms;

    neighbor_t neighbors[MAX_NEIGHBORS];
    uint8_t nb_neighbors;
    uint32_t last_beacon_ms;
    float theta_cmd_rad;
    uint32_t last_vicsek_update_ms;
    int diff_cmd;

    wall_avoidance_state_t wall_avoidance;
    bool doing_wall_avoidance;
    bool prev_doing_wall_avoidance;

    bool cluster_turn_active;
    float cluster_target_rad;
    uint32_t cluster_wall_t0_ms;
    uint32_t cluster_active_until_ms;
    uint16_t cluster_msg_uid;
    uint16_t last_seen_cluster_uid;
    bool have_seen_cluster_uid;
} USERDATA;

DECLARE_USERDATA(USERDATA);
REGISTER_USERDATA(USERDATA);

/* ------------------------------------------------------------------------- */
/* Generic helpers                                                            */
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
    while (a > M_PI) {
        a -= 2.0f * M_PI;
    }
    while (a < -M_PI) {
        a += 2.0f * M_PI;
    }
    return a;
}

static int round_float_to_int(float v) {
    return (int)(v >= 0.0f ? v + 0.5f : v - 0.5f);
}

static int16_t rad_to_mrad(float a) {
    a = wrap_pi(a);

    int v = round_float_to_int(a * 1000.0f);
    if (v > 32767) {
        v = 32767;
    }
    if (v < -32768) {
        v = -32768;
    }
    return (int16_t)v;
}

static float mrad_to_rad(int16_t m) {
    return ((float)m) / 1000.0f;
}

static float noise_uniform(float eta) {
    float u = (float)rand() / (float)RAND_MAX;
    return (u - 0.5f) * eta;
}

static float rand_uniform(float a, float b) {
    float u = (float)rand() / (float)RAND_MAX;
    return a + (b - a) * u;
}

static int16_t f2i16(float v) {
    return (int16_t)(v >= 0.0f ? v + 0.5f : v - 0.5f);
}

static void motor_stop(void) {
    pogobot_motor_set(motorL, motorStop);
    pogobot_motor_set(motorR, motorStop);
}

static int calibrated_motor_power(int nominal_power,
                                  uint16_t calibrated_full_power) {
    if (nominal_power <= 0) {
        return motorStop;
    }

    if (nominal_power > motorFull) {
        nominal_power = motorFull;
    }

    uint32_t scaled =
        (uint32_t)calibrated_full_power *
        (uint32_t)nominal_power;

    scaled += (uint32_t)motorFull / 2u;
    scaled /= (uint32_t)motorFull;

    return (int)scaled;
}

static void motor_set_signed(motor_id id,
                             int speed_signed,
                             uint8_t fwd_dir_mem) {
    int magnitude =
        speed_signed >= 0 ? speed_signed : -speed_signed;

    if (magnitude > motorFull) {
        magnitude = motorFull;
    }

    uint8_t dir =
        speed_signed >= 0
            ? fwd_dir_mem
            : (fwd_dir_mem == 0 ? 1 : 0);

    uint16_t calibrated_full_power;

    if (id == motorL) {
        calibrated_full_power = mydata->motor_power_left;
    } else {
        calibrated_full_power = mydata->motor_power_right;
    }

    int calibrated_power =
        calibrated_motor_power(magnitude,
                               calibrated_full_power);

    pogobot_motor_dir_set(id, dir);
    pogobot_motor_set(id, calibrated_power);
}


static void calibration_motor_step_start(void) {
    motor_set_signed(motorL, calibration_turn_speed, mydata->motor_dir_left_fwd);
    motor_set_signed(motorR, -calibration_turn_speed, mydata->motor_dir_right_fwd);
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
/* Non-blocking robust reader used during calibration                         */
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
            v[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    for (int sweep = 0; sweep < JACOBI_MAX_SWEEPS; sweep++) {
        float max_off = 0.0f;
        int p_arr[] = {0, 0, 1};
        int q_arr[] = {1, 2, 2};

        for (int k = 0; k < 3; k++) {
            int p = p_arr[k];
            int q = q_arr[k];

            if (fabsf(c[p][q]) > max_off) {
                max_off = fabsf(c[p][q]);
            }

            if (fabsf(c[p][q]) > JACOBI_EPSILON) {
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
            if (fabsf(a[r][col]) > amax) {
                amax = fabsf(a[r][col]);
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
    *l2 = (*l1 > 1e-12f) ? (det / *l1) : 0.0f;

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

static float calibrated_heading_deg(float mx, float my, float mz) {
    float px = mx - mydata->mean[0];
    float py = my - mydata->mean[1];
    float pz = mz - mydata->mean[2];

    float u = mydata->e1[0] * px + mydata->e1[1] * py + mydata->e1[2] * pz - mydata->u0;
    float v = mydata->e2[0] * px + mydata->e2[1] * py + mydata->e2[2] * pz - mydata->v0;

    float xc = mydata->w2[0][0] * u + mydata->w2[0][1] * v;
    float yc = mydata->w2[1][0] * u + mydata->w2[1][1] * v;

    float angle = atan2f(yc, xc) * (180.0f / PI_F);
    if (angle < 0.0f) {
        angle += 360.0f;
    }
    return angle;
}

static float calibrated_heading_rad(float mx, float my, float mz) {
    float raw_rad = calibrated_heading_deg(mx, my, mz) * PI_F / 180.0f;
    return wrap_pi(magnetometer_heading_sign * raw_rad +
                   magnetometer_heading_offset_rad);
}

/* ------------------------------------------------------------------------- */
/* Calibration state machine                                                  */
/* ------------------------------------------------------------------------- */

static void calibration_begin_rotation(void);

static void enter_fatal_state(const char *reason) {
    motor_stop();
    mydata->controller_state = CONTROLLER_FATAL;
    mydata->cal_state = CAL_STATE_FATAL;
    mydata->magnetometer_heading_valid = false;
    pogobot_led_setColor(255, 0, 255);

#ifdef SIMULATOR
    printf("Robot %u magnetometer controller fatal: %s\n",
           (unsigned)pogobot_helper_getid(), reason);
#else
    CAL_PRINTF("# FATAL: %s\n", reason);
#endif
}

static bool build_body_anchored_plane_basis(float nx, float ny, float nz) {
    /* Anchor heading phase to the physical sensor/body X axis rather than to
     * the PCA major axis. This avoids a data-dependent per-robot zero angle. */
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

static void calibration_fit_and_wait(void) {
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

    /* Deterministic normal sign. On a normally mounted Pogobot magnetometer,
     * the calibration-plane normal is close to the sensor Z axis. */
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
            (c_a * u0n * u0n + c_b * u0n * v0n + c_c * v0n * v0n);

        if (!sqrtm_2x2(c_a, 0.5f * c_b, c_c, mydata->w2)) {
            mydata->fit_ok = false;
        }
    }

    if (!mydata->fit_ok) {
        enter_fatal_state("magnetometer conic fit failed");
        return;
    }

    CAL_PRINTF("# Calibration OK: %d points, %d attempts, %d occupied bins\n",
               n_used, mydata->global_attempts, n_bins_used);

    motor_stop();
    mydata->cal_state = CAL_STATE_DONE;
    mydata->controller_state = CONTROLLER_WAITING;
    mydata->controller_deadline_ms = deadline_after_ms(post_calibration_wait_ms);
    pogobot_led_setColor(255, 80, 0); /* amber */
}

static void calibration_finish_collection(void) {
    motor_stop();

    if (mydata->n_collected < N_CAL_MIN) {
        enter_fatal_state("too few valid magnetometer calibration points");
        return;
    }

    calibration_fit_and_wait();
}

static void calibration_begin_rotation(void) {
    if (mydata->n_collected >= N_CAL ||
        mydata->global_attempts >= GLOBAL_MAX_ATTEMPTS) {
        calibration_finish_collection();
        return;
    }

    mydata->global_attempts++;
    calibration_motor_step_start();
    mydata->cal_deadline_ms = deadline_after_ms((uint32_t)mydata->step_ms);
    mydata->cal_state = CAL_STATE_ROTATING;
}

static void magnetometer_calibration_start(void) {
    motor_stop();
    pogobot_led_setColor(255, 0, 255);

    mydata->controller_state = CONTROLLER_CALIBRATING;
    mydata->cal_state = CAL_STATE_IDLE;
    mydata->n_collected = 0;
    mydata->step_ms = STEP_MS_INIT;
    mydata->global_attempts = 0;
    mydata->last_cal_x = 0.0f;
    mydata->last_cal_y = 0.0f;
    mydata->last_cal_z = 0.0f;
    mydata->have_last_cal = false;
    mydata->mag_read_active = false;
    mydata->fit_ok = false;

    calibration_begin_rotation();
}

static void magnetometer_calibration_step(void) {
    switch (mydata->cal_state) {
    case CAL_STATE_ROTATING:
        if (!deadline_reached(mydata->cal_deadline_ms)) {
            return;
        }

        motor_stop();
        mydata->cal_deadline_ms = deadline_after_ms(SETTLE_MS);
        mydata->cal_state = CAL_STATE_SETTLING;
        return;

    case CAL_STATE_SETTLING:
        if (!deadline_reached(mydata->cal_deadline_ms)) {
            return;
        }

        mag_read_robust_start(N_AVG_CAL);
        mydata->cal_state = CAL_STATE_READING;
        return;

    case CAL_STATE_READING: {
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

        CAL_PRINTF("CALPT,%d,%d,%d\n",
                   (int)mydata->cal_pts[i][0],
                   (int)mydata->cal_pts[i][1],
                   (int)mydata->cal_pts[i][2]);

        mydata->n_collected++;
        calibration_begin_rotation();
        return;
    }

    case CAL_STATE_IDLE:
    case CAL_STATE_DONE:
    case CAL_STATE_FATAL:
    default:
        return;
    }
}

/* ------------------------------------------------------------------------- */
/* Magnetometer heading during Vicsek motion                                  */
/* ------------------------------------------------------------------------- */

static bool magnetometer_heading_update(void) {
    int16_t mx;
    int16_t my;
    int16_t mz;

    if (!mydata->fit_ok) {
        return false;
    }

    if (magn_read_XYZ(&mx, &my, &mz, MAG_HEADING_TIMEOUT_MS) != 0) {
        return false;
    }

    uint8_t pos = mydata->heading_mag_pos;
    mydata->heading_mag_x[pos] = mx;
    mydata->heading_mag_y[pos] = my;
    mydata->heading_mag_z[pos] = mz;
    mydata->heading_mag_pos = (uint8_t)((pos + 1u) % MAG_HEADING_WINDOW);

    if (mydata->heading_mag_count < MAG_HEADING_WINDOW) {
        mydata->heading_mag_count++;
    }

    int16_t bx[MAG_HEADING_WINDOW];
    int16_t by[MAG_HEADING_WINDOW];
    int16_t bz[MAG_HEADING_WINDOW];
    int n = (int)mydata->heading_mag_count;

    for (int i = 0; i < n; i++) {
        bx[i] = mydata->heading_mag_x[i];
        by[i] = mydata->heading_mag_y[i];
        bz[i] = mydata->heading_mag_z[i];
    }

    float fx = median_i16(bx, n);
    float fy = median_i16(by, n);
    float fz = median_i16(bz, n);
    float new_heading = calibrated_heading_rad(fx, fy, fz);

    if (mydata->magnetometer_heading_valid) {
        float gain = magnetometer_heading_filter_gain;
        if (gain < 0.0f) {
            gain = 0.0f;
        }
        if (gain > 1.0f) {
            gain = 1.0f;
        }

        mydata->magnetometer_heading_rad = wrap_pi(
            mydata->magnetometer_heading_rad +
            gain * wrap_pi(new_heading - mydata->magnetometer_heading_rad));
    } else {
        mydata->magnetometer_heading_rad = new_heading;
    }

    mydata->last_mag_x = f2i16(fx);
    mydata->last_mag_y = f2i16(fy);
    mydata->last_mag_z = f2i16(fz);
    mydata->magnetometer_heading_valid = true;
    mydata->last_magnetometer_heading_ms = now_ms();
    return true;
}

static bool magnetometer_heading_is_fresh(uint32_t now) {
    if (!mydata->magnetometer_heading_valid) {
        return false;
    }

    return (uint32_t)(now - mydata->last_magnetometer_heading_ms) <=
           magnetometer_heading_max_age_ms;
}

/* ------------------------------------------------------------------------- */
/* Neighbor / cluster messaging                                               */
/* ------------------------------------------------------------------------- */

static void purge_old_neighbors(void) {
    uint32_t now = now_ms();
    for (int i = (int)mydata->nb_neighbors - 1; i >= 0; --i) {
        if ((uint32_t)(now - mydata->neighbors[i].last_seen_ms) > max_age) {
            mydata->neighbors[i] = mydata->neighbors[mydata->nb_neighbors - 1];
            mydata->nb_neighbors--;
        }
    }
}

static neighbor_t *upsert_neighbor(uint16_t id) {
    for (uint8_t i = 0; i < mydata->nb_neighbors; ++i) {
        if (mydata->neighbors[i].id == id) {
            return &mydata->neighbors[i];
        }
    }

    if (mydata->nb_neighbors >= MAX_NEIGHBORS) {
        return NULL;
    }

    neighbor_t *neighbor = &mydata->neighbors[mydata->nb_neighbors++];
    neighbor->id = id;
    neighbor->theta_mrad = 0;
    neighbor->last_seen_ms = 0;
    return neighbor;
}

static bool cluster_window_active(uint32_t now) {
    return mydata->cluster_turn_active &&
           (int32_t)(now - mydata->cluster_active_until_ms) < 0;
}

static void cluster_adopt_and_activate(int16_t target_mrad, uint32_t t0_ms, uint16_t uid) {
    bool newer = !mydata->cluster_turn_active ||
                 t0_ms > mydata->cluster_wall_t0_ms ||
                 !mydata->have_seen_cluster_uid ||
                 uid != mydata->last_seen_cluster_uid;

    if (newer) {
        mydata->cluster_target_rad = mrad_to_rad(target_mrad);
        mydata->cluster_wall_t0_ms = t0_ms;
        mydata->cluster_active_until_ms = t0_ms + cluster_u_turn_duration_ms;
        mydata->cluster_turn_active = true;
        mydata->cluster_msg_uid = uid;
        mydata->last_seen_cluster_uid = uid;
        mydata->have_seen_cluster_uid = true;
    }
}

bool send_message(void) {
    if (mydata->controller_state != CONTROLLER_VICSEK) {
        return false;
    }

    uint32_t now = now_ms();
    if (!magnetometer_heading_is_fresh(now)) {
        return false;
    }
    if ((uint32_t)(now - mydata->last_beacon_ms) < BEACON_PERIOD_MS) {
        return false;
    }

    bool advertise_cluster = cluster_window_active(now);
    if (!advertise_cluster &&
        !broadcast_angle_when_avoiding_walls &&
        mydata->doing_wall_avoidance) {
        return false;
    }

    vicsek_msg_t msg = {
        .sender_id = pogobot_helper_getid(),
        .theta_mrad = rad_to_mrad(mydata->theta_cmd_rad),
        .flags = 0u,
        .cluster_target_mrad = 0,
        .cluster_wall_t0_ms = 0u,
        .cluster_msg_uid = 0u
    };

    if (advertise_cluster) {
        msg.flags |= VMSGF_CLUSTER_UTURN;
        msg.cluster_target_mrad = rad_to_mrad(mydata->cluster_target_rad);
        msg.cluster_wall_t0_ms = mydata->cluster_wall_t0_ms;
        msg.cluster_msg_uid = mydata->cluster_msg_uid;
    }

    mydata->last_beacon_ms = now;
    return pogobot_infrared_sendShortMessage_omni((uint8_t *)&msg, MSG_SIZE);
}

void process_message(message_t *mr) {
    if (mydata->controller_state != CONTROLLER_VICSEK) {
        return;
    }

    if (wall_avoidance_process_message(&mydata->wall_avoidance, mr)) {
        return;
    }

    if (mr->header.payload_length < MSG_SIZE) {
        return;
    }

    const vicsek_msg_t *msg = (const vicsek_msg_t *)mr->payload;
    if (msg->sender_id == pogobot_helper_getid()) {
        return;
    }

    neighbor_t *neighbor = upsert_neighbor(msg->sender_id);
    if (neighbor == NULL) {
        return;
    }

    neighbor->theta_mrad = msg->theta_mrad;
    neighbor->last_seen_ms = now_ms();

    if (msg->flags & VMSGF_CLUSTER_UTURN) {
        cluster_adopt_and_activate(
            msg->cluster_target_mrad,
            msg->cluster_wall_t0_ms,
            msg->cluster_msg_uid);
    }
}

/* ------------------------------------------------------------------------- */
/* Vicsek controller                                                          */
/* ------------------------------------------------------------------------- */

static void vicsek_update_and_build_diff(void) {
    purge_old_neighbors();

    uint32_t now = now_ms();
    /* Keep using the latest valid heading during a transient read miss. */
    if (!mydata->magnetometer_heading_valid) {
        mydata->diff_cmd = 0;
        return;
    }

    float heading = mydata->magnetometer_heading_rad;
    float theta_cmd;

    float dt_s =
        (float)(uint32_t)(now - mydata->last_vicsek_update_ms) * 1e-3f;

    if (dt_s > cont_max_dt_s) {
        dt_s = cont_max_dt_s;
    }

    if (cluster_window_active(now)) {
        theta_cmd = mydata->cluster_target_rad;
    } else {
        float sx = 0.0f;
        float sy = 0.0f;

        if (include_self_in_avg) {
            sx += cosf(heading);
            sy += sinf(heading);
        }

        for (uint8_t i = 0; i < mydata->nb_neighbors; ++i) {
            float theta = mrad_to_rad(mydata->neighbors[i].theta_mrad);
            sx += cosf(theta);
            sy += sinf(theta);
        }

        float theta_mean = heading;
        if (!(sx == 0.0f && sy == 0.0f)) {
            theta_mean = atan2f(sy, sx);
        }

        if (vicsek_time_continuous) {
            float dtheta =
                vicsek_beta_rad_per_s *
                sinf(wrap_pi(theta_mean - heading)) *
                dt_s;

            if (cont_noise_sigma_rad > 0.0f) {
                float u1 =
                    ((float)rand() + 1.0f) /
                    ((float)RAND_MAX + 2.0f);

                float u2 =
                    ((float)rand() + 1.0f) /
                    ((float)RAND_MAX + 2.0f);

                float z =
                    sqrtf(-2.0f * logf(u1)) *
                    cosf(2.0f * PI_F * u2);

                dtheta +=
                    cont_noise_sigma_rad * sqrtf(dt_s) * z;
            }

            theta_cmd = wrap_pi(heading + dtheta);
        } else {
            float theta_blend = wrap_pi(
                heading +
                align_gain * wrap_pi(theta_mean - heading));

            theta_cmd =
                wrap_pi(theta_blend +
                        noise_uniform(noise_eta_rad));
        }
    }

    float err = wrap_pi(theta_cmd - heading);
    const float err_norm =
        err / (30.0f * PI_F / 180.0f);

    int diff = round_float_to_int(
        vicsek_turn_gain *
        err_norm *
        (float)forward_speed);

    if (diff > forward_speed) {
        diff = forward_speed;
    }
    if (diff < -forward_speed) {
        diff = -forward_speed;
    }

    mydata->theta_cmd_rad = theta_cmd;
    mydata->diff_cmd = diff;
}


static void vicsek_enter(void) {
    mydata->controller_state = CONTROLLER_VICSEK;
    mydata->heading_mag_count = 0;
    mydata->heading_mag_pos = 0;
    mydata->magnetometer_heading_valid = false;
    mydata->nb_neighbors = 0;
    mydata->last_beacon_ms = 0;
    mydata->diff_cmd = 0;
    mydata->cluster_turn_active = false;
    mydata->doing_wall_avoidance = false;
    mydata->prev_doing_wall_avoidance = false;

    magnetometer_heading_update();
    if (mydata->magnetometer_heading_valid) {
        mydata->theta_cmd_rad = wrap_pi(
            mydata->magnetometer_heading_rad + noise_uniform(noise_eta_rad));
    }

    mydata->last_vicsek_update_ms = now_ms();
    pogobot_led_setColor(0, 0, 255);
}

static void update_main_led(void) {
    if (mydata->controller_state == CONTROLLER_CALIBRATING) {
        pogobot_led_setColor(255, 0, 255);
        return;
    }

    if (mydata->controller_state == CONTROLLER_WAITING) {
        pogobot_led_setColor(255, 80, 0);
        return;
    }

    if (mydata->controller_state == CONTROLLER_FATAL) {
        pogobot_led_setColor(255, 0, 255);
        return;
    }

    uint32_t now = now_ms();
    if (!magnetometer_heading_is_fresh(now)) {
        pogobot_led_setColor(255, 0, 255);
        return;
    }

    if (main_led_display_enum == SHOW_STATE) {
        if (mydata->doing_wall_avoidance) {
            pogobot_led_setColor(255, 0, 0);
        } else if (cluster_window_active(now)) {
            pogobot_led_setColor(0, 255, 255);
        } else if (mydata->nb_neighbors == 0) {
            pogobot_led_setColor(0, 0, 255);
        } else {
            pogobot_led_setColor(0, 255, 0);
        }
        return;
    }

    float angle = (float)mydata->magnetometer_heading_rad;
    if (angle < 0.0f) {
        angle += 2.0f * M_PI;
    }

    float hue_deg = angle * 180.0f / M_PI;
    uint8_t r8;
    uint8_t g8;
    uint8_t b8;
    hsv_to_rgb(hue_deg, 1.0f, 1.0f, &r8, &g8, &b8);
    r8 = SCALE_0_255_TO_0_25(r8);
    g8 = SCALE_0_255_TO_0_25(g8);
    b8 = SCALE_0_255_TO_0_25(b8);
    if (r8 == 0 && g8 == 0 && b8 == 0) {
        r8 = 1;
    }
    pogobot_led_setColor(r8, g8, b8);
}

/* ------------------------------------------------------------------------- */
/* Main Pogobot callbacks                                                     */
/* ------------------------------------------------------------------------- */

void user_init(void) {
    srand(pogobot_helper_getRandSeed());
    memset(mydata, 0, sizeof(*mydata));

    /* The calibration state machine benefits from the same 100 Hz loop as the
     * standalone calibration controller. Vicsek itself remains rate-limited by
     * vicsek_period_ms. */
    main_loop_hz = 20;
    max_nb_processed_msg_per_tick = 3;
    percent_msgs_sent_per_ticks = 50;
    msg_rx_fn = process_message;
    msg_tx_fn = send_message;
    error_codes_led_idx = 3;

    uint8_t dir_mem[3] = {0, 0, 0};
    pogobot_motor_dir_mem_get(dir_mem);
    mydata->motor_dir_right_fwd = dir_mem[0];
    mydata->motor_dir_left_fwd = dir_mem[1];

    uint16_t power_mem[3] = {0, 0, 0};
    pogobot_motor_power_mem_get(power_mem);
    mydata->motor_power_right = power_mem[0];
    mydata->motor_power_left = power_mem[1];

    motor_calibration_t motors = {
        .motor_left = mydata->motor_power_left,
        .dir_left = mydata->motor_dir_left_fwd,
        .motor_right = mydata->motor_power_right,
        .dir_right = mydata->motor_dir_right_fwd
    };

    wall_avoidance_config_t default_config = {
        .wall_memory_ms = wall_avoidance_memory_ms,
        .turn_duration_ms = wall_avoidance_turn_duration_ms,
        .forward_commit_ms = wall_avoidance_forward_commit_ms,
        .forward_speed_ratio = wall_avoidance_forward_speed_ratio
    };

    wall_avoidance_init(&mydata->wall_avoidance, &default_config, &motors);
    wall_avoidance_set_policy(&mydata->wall_avoidance,
                              wall_avoidance_chiralty_policy, 0);

    mydata->cluster_turn_active = false;
    mydata->cluster_target_rad = 0.0f;
    mydata->cluster_wall_t0_ms = 0u;
    mydata->cluster_active_until_ms = 0u;
    mydata->cluster_msg_uid = 0u;
    mydata->have_seen_cluster_uid = false;

    magnetometer_calibration_start();
}

void user_step(void) {
    if (mydata->controller_state == CONTROLLER_CALIBRATING) {
        magnetometer_calibration_step();
        update_main_led();
        return;
    }

    if (mydata->controller_state == CONTROLLER_WAITING) {
        motor_stop();
        update_main_led();
        if (deadline_reached(mydata->controller_deadline_ms)) {
            vicsek_enter();
        }
        return;
    }

    if (mydata->controller_state == CONTROLLER_FATAL) {
        motor_stop();
        update_main_led();
        return;
    }

    magnetometer_heading_update();
    uint32_t now = now_ms();

    /* Do not stop for transient magnetometer read misses. The latest valid
     * heading remains available to the controller, and the motors keep their
     * continuous Vicsek/wall-avoidance motion. */

    bool wall_avoidance = wall_avoidance_step(&mydata->wall_avoidance, true);
    mydata->prev_doing_wall_avoidance = mydata->doing_wall_avoidance;
    mydata->doing_wall_avoidance = wall_avoidance;

    if (!mydata->prev_doing_wall_avoidance && mydata->doing_wall_avoidance) {
        float phi_sample = rand_uniform(phi_rad_min, phi_rad_max);
        float target = wrap_pi(mydata->magnetometer_heading_rad + phi_sample);

        mydata->cluster_target_rad = target;
        mydata->cluster_wall_t0_ms = now;
        mydata->cluster_active_until_ms = now + cluster_u_turn_duration_ms;
        mydata->cluster_turn_active = true;
        mydata->cluster_msg_uid = (uint16_t)(rand() & 0xFFFF);
        mydata->last_seen_cluster_uid = mydata->cluster_msg_uid;
        mydata->have_seen_cluster_uid = true;
    }

    if (vicsek_time_continuous ||
        (uint32_t)(now - mydata->last_vicsek_update_ms) >= vicsek_period_ms) {
        vicsek_update_and_build_diff();
        mydata->last_vicsek_update_ms = now;
    }

    if (!mydata->doing_wall_avoidance) {
        motor_set_signed(motorL,
                         forward_speed - mydata->diff_cmd,
                         mydata->motor_dir_left_fwd);
        motor_set_signed(motorR,
                         forward_speed + mydata->diff_cmd,
                         mydata->motor_dir_right_fwd);
    }

    update_main_led();
}

#ifdef SIMULATOR
static void create_data_schema(void) {
    data_add_column_int8("controller_state");
    data_add_column_int8("calibration_fit_ok");
    data_add_column_int8("nb_neighbors");
    data_add_column_int8("mag_heading_valid");
    data_add_column_int16("mag_x");
    data_add_column_int16("mag_y");
    data_add_column_int16("mag_z");
    data_add_column_double("theta_mag_rad");
    data_add_column_double("theta_cmd_rad");
    data_add_column_int16("diff_cmd");
    data_add_column_int8("cluster_active");
    data_add_column_double("cluster_target_rad");
    data_add_column_int32("cluster_t0_ms");
    data_add_column_int32("cluster_until_ms");
}

static void export_data(void) {
    uint32_t now = now_ms();
    data_set_value_int8("controller_state", (int8_t)mydata->controller_state);
    data_set_value_int8("calibration_fit_ok", (int8_t)(mydata->fit_ok ? 1 : 0));
    data_set_value_int8("nb_neighbors", (int8_t)mydata->nb_neighbors);
    data_set_value_int8("mag_heading_valid",
                        (int8_t)(magnetometer_heading_is_fresh(now) ? 1 : 0));
    data_set_value_int16("mag_x", mydata->last_mag_x);
    data_set_value_int16("mag_y", mydata->last_mag_y);
    data_set_value_int16("mag_z", mydata->last_mag_z);
    data_set_value_double("theta_mag_rad", mydata->magnetometer_heading_rad);
    data_set_value_double("theta_cmd_rad", mydata->theta_cmd_rad);
    data_set_value_int16("diff_cmd", (int16_t)mydata->diff_cmd);
    data_set_value_int8("cluster_active",
                        (int8_t)(cluster_window_active(now) ? 1 : 0));
    data_set_value_double("cluster_target_rad", mydata->cluster_target_rad);
    data_set_value_int32("cluster_t0_ms", (int32_t)mydata->cluster_wall_t0_ms);
    data_set_value_int32("cluster_until_ms", (int32_t)mydata->cluster_active_until_ms);
}

static void global_setup(void) {
    init_from_configuration(forward_speed);
    init_from_configuration(calibration_turn_speed);

    init_from_configuration(max_age);
    init_from_configuration(vicsek_period_ms);
    init_from_configuration(noise_eta_rad);
    init_from_configuration(include_self_in_avg);
    init_from_configuration(broadcast_angle_when_avoiding_walls);
    init_from_configuration(align_gain);
    init_from_configuration(vicsek_turn_gain);
    init_from_configuration(vicsek_time_continuous);
    init_from_configuration(vicsek_beta_rad_per_s);
    init_from_configuration(cont_noise_sigma_rad);
    init_from_configuration(cont_max_dt_s);

    init_from_configuration(post_calibration_wait_ms);
    init_from_configuration(magnetometer_heading_offset_rad);
    init_from_configuration(magnetometer_heading_filter_gain);
    init_from_configuration(magnetometer_heading_max_age_ms);

    char magnetometer_heading_chirality[128] = "cw";
    init_array_from_configuration(magnetometer_heading_chirality);
    if (strcasecmp(magnetometer_heading_chirality, "cw") == 0) {
        magnetometer_heading_sign = 1.0f;
    } else if (strcasecmp(magnetometer_heading_chirality, "ccw") == 0) {
        magnetometer_heading_sign = -1.0f;
    } else {
        printf("ERROR: unknown magnetometer_heading_chirality '%s' (use 'cw' or 'ccw').\n",
               magnetometer_heading_chirality);
        exit(1);
    }

    init_from_configuration(cluster_u_turn_duration_ms);
    init_from_configuration(phi_rad_min);
    init_from_configuration(phi_rad_max);
    if (phi_rad_min > phi_rad_max) {
        float tmp = phi_rad_min;
        phi_rad_min = phi_rad_max;
        phi_rad_max = tmp;
    }

    init_from_configuration(wall_avoidance_memory_ms);
    init_from_configuration(wall_avoidance_turn_duration_ms);
    init_from_configuration(wall_avoidance_forward_commit_ms);
    init_from_configuration(wall_avoidance_forward_speed_ratio);

    char wall_avoidance_policy[128] = "min_turn";
    init_array_from_configuration(wall_avoidance_policy);
    if (strcasecmp(wall_avoidance_policy, "cw") == 0) {
        wall_avoidance_chiralty_policy = WALL_CW;
    } else if (strcasecmp(wall_avoidance_policy, "ccw") == 0) {
        wall_avoidance_chiralty_policy = WALL_CCW;
    } else if (strcasecmp(wall_avoidance_policy, "random") == 0) {
        wall_avoidance_chiralty_policy = WALL_RANDOM;
    } else if (strcasecmp(wall_avoidance_policy, "min_turn") == 0) {
        wall_avoidance_chiralty_policy = WALL_MIN_TURN;
    } else {
        printf("ERROR: unknown wall_avoidance_policy '%s'.\n", wall_avoidance_policy);
        exit(1);
    }

    char main_led_display[128] = "state";
    init_array_from_configuration(main_led_display);
    if (strcasecmp(main_led_display, "state") == 0) {
        main_led_display_enum = SHOW_STATE;
    } else if (strcasecmp(main_led_display, "angle") == 0) {
        main_led_display_enum = SHOW_ANGLE;
    } else {
        printf("ERROR: unknown main_led_display '%s' (use 'state' or 'angle').\n",
               main_led_display);
        exit(1);
    }
}
#endif

int main(void) {
    pogobot_init();
    pogobot_start(user_init, user_step);
    pogobot_start(default_walls_user_init, default_walls_user_step, "walls");
#ifdef SIMULATOR
    SET_CALLBACK(callback_global_setup, global_setup);
    SET_CALLBACK(callback_create_data_schema, create_data_schema);
    SET_CALLBACK(callback_export_data, export_data);
#endif
    return 0;
}
