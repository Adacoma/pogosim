// main.c — magnetometer calibration/test adapted for Pogobot and Pogosim

#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>

// Main include for pogobots, both for real robots and for simulations
#include "pogobase.h"

//////  CALIBRATION CONFIGURATION  ////// {{{1

/* =========================================================================
 * CONFIGURATION  (Pogobot : VexRiscv lite, 128 kB RAM, 256 pages flash)
 * ========================================================================= */
#define N_CAL            120
#define N_CAL_MIN        60
#define N_AVG_CAL        8     /* samples per stop (MEDIAN per axis)             */
#define N_AVG_MES        12    /* same during mission (robot stopped)            */
#define MAX_MEASUREMENTS 100

#define STEP_MS_INIT     250
#define STEP_MS_MAX      1200
#define STEP_MS_INC      75
#define STEP_MS_MES      300
#define SETTLE_MS        250

#define MAG_TIMEOUT_MS   50
#define MAG_RETRY_MS     20
#define MIN_DELTA_MAG_SQ 100.0f

#define GLOBAL_MAX_ATTEMPTS (4 * N_CAL)

/* Select the complete calibration + mission workflow.
 * Set to 0 to use the lightweight continuous magnetometer test instead. */
#define RUN_FULL_CALIBRATION 1

/* Live diagnostic UART output.
 *
 * Structured experiment records are always printed, independently of this
 * option: CALPT, BINPT, CALIB_*, ANGLE, and MAG.
 *
 * 0: print structured records only.
 * 1: additionally print human-readable diagnostics such as progress,
 *    warnings, fit information, setup details, and memory-dump markers. */
#define ENABLE_LIVE_DIAGNOSTICS 0

#define DIAG_PRINTF(...) do { \
    if (ENABLE_LIVE_DIAGNOSTICS) { \
        printf(__VA_ARGS__); \
    } \
} while (0)

/* The full calibration uses a faster loop so 20 ms deadlines can be serviced
 * with useful temporal resolution. Timing itself is based on
 * current_time_milliseconds(), not on the loop frequency. The simple live
 * magnetometer test keeps the original 30 Hz. */
#if RUN_FULL_CALIBRATION
#define USER_MAIN_LOOP_HZ 100
#else
#define USER_MAIN_LOOP_HZ 30
#endif

#define FLASH_DUMP_PERIOD_MS 2000

/* Angular stratification to reduce fit density bias (cf. Kanatani) */
#define N_BINS           36    /* 10-degree sectors                              */
#define N_BINS_MIN       12    /* minimum occupied sectors required for refit    */
#define N_STRAT_ITERS    2     /* center -> binning -> refit iterations          */

#define PI_F 3.14159265f

typedef enum {
    CAL_STATE_IDLE = 0,
    CAL_STATE_ROTATING,
    CAL_STATE_SETTLING,
    CAL_STATE_READING,
    CAL_STATE_MISSION_INITIAL_WAIT,
    CAL_STATE_MISSION_SETTLING,
    CAL_STATE_MISSION_READING,
    CAL_STATE_MISSION_LED,
    CAL_STATE_MISSION_ROTATING,
    CAL_STATE_DONE,
    CAL_STATE_FATAL
} calibration_state_t;

typedef enum {
    MAG_READ_PENDING = 0,
    MAG_READ_SUCCESS,
    MAG_READ_FAILURE
} mag_read_result_t;

// "Global" variables should be inserted within the USERDATA struct.
// /!\ In simulation, don't declare non-const global variables outside this
// struct, otherwise they will be shared among all agents.
typedef struct {
    // Motor calibration retrieved from robot memory
    uint8_t motor_dir_left;
    uint8_t motor_dir_right;
    uint16_t motor_power_left;
    uint16_t motor_power_right;

    // Cooperative calibration state machine
    calibration_state_t cal_state;
    uint32_t state_deadline_ms;
    int n_collected;
    int step_ms;
    int global_attempts;
    float last_x;
    float last_y;
    float last_z;
    bool have_last;

    // Per-robot calibration buffers (must not be static globals in Pogosim)
    int16_t cal_pts[N_CAL][3];
    float u_buf[N_CAL];
    float v_buf[N_CAL];
    float bin_su[N_BINS];
    float bin_sv[N_BINS];
    int bin_n[N_BINS];
    float bin_mu[N_BINS];
    float bin_mv[N_BINS];

    // Non-blocking robust magnetometer reader
    bool mag_read_active;
    int mag_n_target;
    int mag_n;
    int mag_attempts;
    uint32_t mag_next_ms;
    int16_t mag_bx[16];
    int16_t mag_by[16];
    int16_t mag_bz[16];

    // Calibration result, retained across user_step() calls
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

    // Mission state
    int mission_point;
    float pending_x;
    float pending_y;
    float pending_z;

    // Flash buffers retained across mission steps
    uint16_t buffer_angles[128];
    uint16_t buffer_magn[128];
    int page_idx;
    int buffer_angles_idx;
    int buffer_magn_idx;

    // Cooperative flash dump state
    bool dump_active;
    bool dump_completed;
    int dump_page_idx;
    int dump_empty_pages_count;
    uint32_t dump_next_ms;
    uint32_t next_live_dump_ms;
} USERDATA;

// Call this macro in the same file (.h or .c) as the declaration of USERDATA
DECLARE_USERDATA(USERDATA);

// Don't forget to call this macro in the main .c file of your project (only once!)
REGISTER_USERDATA(USERDATA);

union FloatToUint16 { float f; uint16_t u16[2]; };

#define FLOAT_TO_INT_FMT "%s%d.%02d"
#define FLOAT_TO_INT_ARGS(v) \
    (((v) < 0.0f && (v) > -1.0f) ? "-" : ""), \
    ((int)(v)), \
    ((int)(((v) < 0.0f ? -(v) : (v)) * 100.0f) % 100)

/* =========================================================================
 * COOPERATIVE TIMING
 *
 * Timed operations use current_time_milliseconds() rather than msleep().
 * Each operation stores a millisecond deadline, returns from user_step(), and
 * resumes on the first later user_step() call at or after that deadline. This
 * lets Pogosim continue advancing simulation time and updating robot physics.
 * ========================================================================= */

static uint32_t now_ms(void) {
    return (uint32_t)current_time_milliseconds();
}

static uint32_t deadline_after_ms(uint32_t delay_ms) {
    return now_ms() + delay_ms;
}

static bool deadline_reached(uint32_t deadline_ms) {
    // Signed subtraction keeps comparisons valid across uint32_t wraparound,
    // provided no individual delay is >= 2^31 milliseconds (~24.9 days).
    return (int32_t)(now_ms() - deadline_ms) >= 0;
}

static void motor_stop(void) {
    pogobot_motor_set(motorL, motorStop);
    pogobot_motor_set(motorR, motorStop);
}

static void motor_step_start(void) {
    pogobot_motor_dir_set(motorR, 0);
    pogobot_motor_dir_set(motorL, 0);
    pogobot_motor_set(motorL, motorHalf);
    pogobot_motor_set(motorR, motorHalf);
}

static int16_t f2i16(float v) {
    return (int16_t)(v >= 0.0f ? v + 0.5f : v - 0.5f);
}

/* Median of a small int16 array (insertion sort, n <= 16).
 * Its 50% breakdown point makes it robust to magnetic transients, unlike
 * the arithmetic mean, whose breakdown point is zero. */
static float median_i16(int16_t *a, int n) {
    for (int i = 1; i < n; i++) {
        int16_t key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) { a[j + 1] = a[j]; j--; }
        a[j + 1] = key;
    }
    if (n & 1) return (float)a[n / 2];
    return 0.5f * ((float)a[n/2 - 1] + (float)a[n/2]);
}


/* =========================================================================
 * NON-BLOCKING ROBUST MAGNETOMETER READER
 * ========================================================================= */

static void mag_read_robust_start(int n_target) {
    int16_t mx, my, mz;

    if (n_target > 16) n_target = 16;
    if (n_target < 1) n_target = 1;

    mydata->mag_n_target = n_target;
    mydata->mag_n = 0;
    mydata->mag_attempts = 0;
    mydata->mag_read_active = true;
    mydata->mag_next_ms = now_ms();

    // Preserve the original purge read, but do not sleep afterwards.
    magn_read_XYZ(&mx, &my, &mz, 0);
}

static mag_read_result_t mag_read_robust_step(float *x, float *y, float *z) {
    if (!mydata->mag_read_active) {
        return MAG_READ_FAILURE;
    }

    if (!deadline_reached(mydata->mag_next_ms)) {
        return MAG_READ_PENDING;
    }

    int16_t mx, my, mz;
    mydata->mag_attempts++;

    if (magn_read_XYZ(&mx, &my, &mz, MAG_TIMEOUT_MS) == 0) {
        int n = mydata->mag_n;
        mydata->mag_bx[n] = mx;
        mydata->mag_by[n] = my;
        mydata->mag_bz[n] = mz;
        mydata->mag_n++;
    }

    bool enough_samples = mydata->mag_n >= mydata->mag_n_target;
    bool attempts_exhausted =
        mydata->mag_attempts >= 4 * mydata->mag_n_target;

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

static void uart_dump_floats(const char *tag, const float *vals, int n) {
    union FloatToUint16 u;
    printf("%s", tag);
    for (int i = 0; i < n; i++) {
        u.f = vals[i];
        printf(",%u,%u", (unsigned)u.u16[0], (unsigned)u.u16[1]);
    }
    printf("\n");
}

static int flash_put_float(uint16_t *buf, int idx, float v) {
    union FloatToUint16 u;
    u.f = v;
    buf[idx]     = u.u16[0];
    buf[idx + 1] = u.u16[1];
    return idx + 2;
}

/* =========================================================================
 * 3x3 JACOBI EVD (spectral theorem) - rotation plane
 * ========================================================================= */
#define JACOBI_MAX_SWEEPS 15
#define JACOBI_EPSILON    1e-6f

static void evd_jacobi_3x3(float C[3][3], float V[3][3], float L[3]) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            V[i][j] = (i == j) ? 1.0f : 0.0f;

    for (int sweep = 0; sweep < JACOBI_MAX_SWEEPS; sweep++) {
        float max_off = 0.0f;
        int p_arr[] = {0, 0, 1}, q_arr[] = {1, 2, 2};
        for (int k = 0; k < 3; k++) {
            int p = p_arr[k], q = q_arr[k];
            if (fabsf(C[p][q]) > max_off) max_off = fabsf(C[p][q]);
            if (fabsf(C[p][q]) > JACOBI_EPSILON) {
                float theta = (C[q][q] - C[p][p]) / (2.0f * C[p][q]);
                float t = 1.0f / (fabsf(theta) + sqrtf(1.0f + theta * theta));
                if (theta < 0.0f) t = -t;
                float c = 1.0f / sqrtf(1.0f + t * t);
                float s = t * c;
                float cpp = C[p][p], cqq = C[q][q], cpq = C[p][q];
                C[p][p] = c*c*cpp - 2.0f*s*c*cpq + s*s*cqq;
                C[q][q] = s*s*cpp + 2.0f*s*c*cpq + c*c*cqq;
                C[p][q] = 0.0f; C[q][p] = 0.0f;
                for (int i = 0; i < 3; i++) {
                    if (i != p && i != q) {
                        float cip = C[i][p], ciq = C[i][q];
                        C[i][p] = c*cip - s*ciq; C[p][i] = C[i][p];
                        C[i][q] = s*cip + c*ciq; C[q][i] = C[i][q];
                    }
                    float vip = V[i][p], viq = V[i][q];
                    V[i][p] = c*vip - s*viq;
                    V[i][q] = s*vip + c*viq;
                }
            }
        }
        if (max_off < JACOBI_EPSILON) break;
    }
    L[0] = C[0][0]; L[1] = C[1][1]; L[2] = C[2][2];
}

/* =========================================================================
 * 5x5 GAUSSIAN ELIMINATION WITH PARTIAL PIVOTING
 * ========================================================================= */
static bool solve5(float A[5][5], float b[5], float x[5]) {
    for (int col = 0; col < 5; col++) {
        int piv = col;
        float amax = fabsf(A[col][col]);
        for (int r = col + 1; r < 5; r++) {
            if (fabsf(A[r][col]) > amax) { amax = fabsf(A[r][col]); piv = r; }
        }
        if (amax < 1e-9f) return false;
        if (piv != col) {
            for (int c = col; c < 5; c++) {
                float t = A[col][c]; A[col][c] = A[piv][c]; A[piv][c] = t;
            }
            float t = b[col]; b[col] = b[piv]; b[piv] = t;
        }
        for (int r = col + 1; r < 5; r++) {
            float f = A[r][col] / A[col][col];
            for (int c = col; c < 5; c++) A[r][c] -= f * A[col][c];
            b[r] -= f * b[col];
        }
    }
    for (int r = 4; r >= 0; r--) {
        float s = b[r];
        for (int c = r + 1; c < 5; c++) s -= A[r][c] * x[c];
        x[r] = s / A[r][r];
    }
    return true;
}

/* =========================================================================
 * CONIC FIT: A.u^2 + B.uv + C.v^2 + D.u + E.v = 1
 * (normal equations, coordinates normalized by s - Hartley)
 * ========================================================================= */
static bool fit_conic(const float *u, const float *v, int n, float s,
                      float coef[5]) {
    float S[5][5] = {{0}}, bb[5] = {0};
    for (int i = 0; i < n; i++) {
        float un = u[i] / s, vn = v[i] / s;
        float phi[5] = { un*un, un*vn, vn*vn, un, vn };
        for (int r = 0; r < 5; r++) {
            bb[r] += phi[r];
            for (int c = 0; c < 5; c++) S[r][c] += phi[r] * phi[c];
        }
    }
    return solve5(S, bb, coef);
}

/* Conic center in normalized coordinates. Returns false if non-elliptic. */
static bool conic_center(const float coef[5], float *u0n, float *v0n) {
    float det = 4.0f * coef[0] * coef[2] - coef[1] * coef[1];
    if (det <= 1e-9f || coef[0] <= 0.0f || coef[2] <= 0.0f) return false;
    *u0n = (-2.0f * coef[2] * coef[3] + coef[1] * coef[4]) / det;
    *v0n = ( coef[1] * coef[3] - 2.0f * coef[0] * coef[4]) / det;
    return true;
}

/* =========================================================================
 * ANALYTIC 2x2 EVD (Vieta + largest-norm vector), l1 >= l2
 * ========================================================================= */
static void eig_sym_2x2(float axx, float axy, float ayy,
                        float *l1, float *l2, float *v1x, float *v1y) {
    float T = axx + ayy;
    float D = axx * ayy - axy * axy;
    float diff = axx - ayy;
    float sd = sqrtf(diff * diff + 4.0f * axy * axy);
    *l1 = 0.5f * (T + sd);
    *l2 = (*l1 > 1e-12f) ? (D / *l1) : 0.0f;

    float c1x = axy,       c1y = *l1 - axx;
    float c2x = *l1 - ayy, c2y = axy;
    float n1 = c1x*c1x + c1y*c1y;
    float n2 = c2x*c2x + c2y*c2y;
    if (n1 > n2 && n1 > 0.0f) {
        float inv = 1.0f / sqrtf(n1); *v1x = c1x * inv; *v1y = c1y * inv;
    } else if (n2 > 0.0f) {
        float inv = 1.0f / sqrtf(n2); *v1x = c2x * inv; *v1y = c2y * inv;
    } else {
        *v1x = 1.0f; *v1y = 0.0f;
    }
}

/* W = V diag(sqrt(lambda)) V^T (square root of a PSD matrix) */
static bool sqrtm_2x2(float axx, float axy, float ayy, float W[2][2]) {
    float l1, l2, vx, vy;
    eig_sym_2x2(axx, axy, ayy, &l1, &l2, &vx, &vy);
    if (l1 <= 0.0f || l2 <= 0.0f) return false;
    float wx = -vy, wy = vx;
    float s1 = sqrtf(l1), s2 = sqrtf(l2);
    W[0][0] = s1*vx*vx + s2*wx*wx;
    W[0][1] = s1*vx*vy + s2*wx*wy;
    W[1][0] = W[0][1];
    W[1][1] = s1*vy*vy + s2*wy*wy;
    return true;
}

static float heading_deg(float mx, float my, float mz,
                         float mean[3], float e1[3], float e2[3],
                         float u0, float v0, float W2[2][2]) {
    float px = mx - mean[0], py = my - mean[1], pz = mz - mean[2];
    float u = e1[0]*px + e1[1]*py + e1[2]*pz - u0;
    float v = e2[0]*px + e2[1]*py + e2[2]*pz - v0;
    float xc = W2[0][0]*u + W2[0][1]*v;
    float yc = W2[1][0]*u + W2[1][1]*v;
    float ang = atan2f(yc, xc) * (180.0f / PI_F);
    if (ang < 0.0f) ang += 360.0f;
    return ang;
}


/* =========================================================================
 * COOPERATIVE CALIBRATION / MISSION STATE MACHINE
 * ========================================================================= */

static void calibration_begin_rotation(void);

static void calibration_fit_and_prepare_mission(void) {
    const int n_used = mydata->n_collected;

    DIAG_PRINTF("# Collection OK: %d points, %d attempts.\n",
           n_used, mydata->global_attempts);

    /* --- PHASE 2a: MEAN + COVARIANCE -> PLANE --- */
    for (int k = 0; k < 3; k++) mydata->mean[k] = 0.0f;

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

    cmat[0][0] /= n_used;
    cmat[1][1] /= n_used;
    cmat[2][2] /= n_used;
    cmat[0][1] /= n_used;
    cmat[0][2] /= n_used;
    cmat[1][2] /= n_used;
    cmat[1][0] = cmat[0][1];
    cmat[2][0] = cmat[0][2];
    cmat[2][1] = cmat[1][2];

    float eig_vec[3][3], eig_val[3];
    evd_jacobi_3x3(cmat, eig_vec, eig_val);

    int i_min = 0;
    int i_max = 0;
    if (eig_val[1] < eig_val[i_min]) i_min = 1;
    if (eig_val[2] < eig_val[i_min]) i_min = 2;
    if (eig_val[1] > eig_val[i_max]) i_max = 1;
    if (eig_val[2] > eig_val[i_max]) i_max = 2;

    float nx = eig_vec[0][i_min];
    float ny = eig_vec[1][i_min];
    float nz = eig_vec[2][i_min];
    if (nz < 0.0f) {
        nx = -nx;
        ny = -ny;
        nz = -nz;
    }

    mydata->e1[0] = eig_vec[0][i_max];
    mydata->e1[1] = eig_vec[1][i_max];
    mydata->e1[2] = eig_vec[2][i_max];

    mydata->e2[0] = ny * mydata->e1[2] - nz * mydata->e1[1];
    mydata->e2[1] = nz * mydata->e1[0] - nx * mydata->e1[2];
    mydata->e2[2] = nx * mydata->e1[1] - ny * mydata->e1[0];

    /* --- PHASE 2b: PROJECTION --- */
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
    if (mydata->s_norm < 1.0f) mydata->s_norm = 1.0f;

    {
        int bins[12] = {0};
        for (int i = 0; i < n_used; i++) {
            float a = atan2f(mydata->v_buf[i], mydata->u_buf[i]);
            int b = (int)((a + PI_F) * (12.0f / (2.0f * PI_F)));
            if (b < 0) b = 0;
            if (b > 11) b = 11;
            bins[b]++;
        }

        int occupied = 0;
        for (int b = 0; b < 12; b++) {
            if (bins[b] > 0) occupied++;
        }

        DIAG_PRINTF("# Raw angular coverage: %d/12 sectors\n", occupied);
        if (occupied < 9) DIAG_PRINTF("# [WARN] insufficient arc coverage for the fit\n");
    }

    /* --- PHASE 2c: INITIAL FIT + ANGULAR STRATIFICATION --- */
    float coef[5] = {0};
    mydata->u0 = 0.0f;
    mydata->v0 = 0.0f;
    mydata->fit_ok = false;

    if (fit_conic(mydata->u_buf, mydata->v_buf, n_used,
                  mydata->s_norm, coef)) {
        float u0n, v0n;
        if (conic_center(coef, &u0n, &v0n)) {
            mydata->u0 = mydata->s_norm * u0n;
            mydata->v0 = mydata->s_norm * v0n;
            mydata->fit_ok = true;
        }
    }

    if (!mydata->fit_ok) DIAG_PRINTF("# [WARN] initial fit failed\n");

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
            int b = (int)((phi + PI_F) *
                          ((float)N_BINS / (2.0f * PI_F)));

            if (b < 0) b = 0;
            if (b >= N_BINS) b = N_BINS - 1;

            mydata->bin_su[b] += mydata->u_buf[i];
            mydata->bin_sv[b] += mydata->v_buf[i];
            mydata->bin_n[b]++;
        }

        int m = 0;
        int cnt_min = 1 << 30;
        int cnt_max = 0;

        for (int b = 0; b < N_BINS; b++) {
            if (mydata->bin_n[b] > 0) {
                mydata->bin_mu[m] =
                    mydata->bin_su[b] / (float)mydata->bin_n[b];
                mydata->bin_mv[m] =
                    mydata->bin_sv[b] / (float)mydata->bin_n[b];
                m++;

                if (mydata->bin_n[b] < cnt_min) cnt_min = mydata->bin_n[b];
                if (mydata->bin_n[b] > cnt_max) cnt_max = mydata->bin_n[b];
            }
        }

        n_bins_used = m;
        DIAG_PRINTF("# [Strat %d] %d/%d occupied sectors (min %d / max %d pts)\n",
               it + 1, m, N_BINS, cnt_min, cnt_max);

        if (m < N_BINS_MIN) {
            DIAG_PRINTF("# [WARN] too few sectors, refit skipped\n");
            break;
        }

        float coef2[5];
        if (fit_conic(mydata->bin_mu, mydata->bin_mv, m,
                      mydata->s_norm, coef2)) {
            float u0n, v0n;
            if (conic_center(coef2, &u0n, &v0n)) {
                for (int k = 0; k < 5; k++) coef[k] = coef2[k];
                mydata->u0 = mydata->s_norm * u0n;
                mydata->v0 = mydata->s_norm * v0n;
            } else {
                DIAG_PRINTF("# [WARN] non-elliptic refit, iteration skipped\n");
            }
        } else {
            DIAG_PRINTF("# [WARN] singular refit, iteration skipped\n");
        }
    }

    for (int b = 0; b < n_bins_used; b++) {
        printf("BINPT,%d,%d\n",
               (int)f2i16(mydata->bin_mu[b]),
               (int)f2i16(mydata->bin_mv[b]));
    }

    /* --- PHASE 2d: FINAL EXTRACTION (axes, k, W2) --- */
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

        mydata->k_norm =
            1.0f + (c_a * u0n * u0n +
                    c_b * u0n * v0n +
                    c_c * v0n * v0n);

        if (mydata->el1 > 0.0f &&
            mydata->el2 > 0.0f &&
            mydata->k_norm > 0.0f) {
            float ax_short =
                mydata->s_norm * sqrtf(mydata->k_norm / mydata->el1);
            float ax_long =
                mydata->s_norm * sqrtf(mydata->k_norm / mydata->el2);

            DIAG_PRINTF("# Ellipse: semi-axes " FLOAT_TO_INT_FMT " / "
                   FLOAT_TO_INT_FMT " (ratio " FLOAT_TO_INT_FMT ")\n",
                   FLOAT_TO_INT_ARGS(ax_long),
                   FLOAT_TO_INT_ARGS(ax_short),
                   FLOAT_TO_INT_ARGS(ax_long / ax_short));
        }

        if (!sqrtm_2x2(c_a, 0.5f * c_b, c_c, mydata->w2)) {
            DIAG_PRINTF("# [WARN] degenerate sqrtm, W2 = identity\n");
            mydata->w2[0][0] = 1.0f;
            mydata->w2[0][1] = 0.0f;
            mydata->w2[1][0] = 0.0f;
            mydata->w2[1][1] = 1.0f;
            mydata->fit_ok = false;
        }
    }

    DIAG_PRINTF("# Stratified conic fit: %s | u0=" FLOAT_TO_INT_FMT
           " v0=" FLOAT_TO_INT_FMT "\n",
           mydata->fit_ok ? "OK" : "FALLBACK",
           FLOAT_TO_INT_ARGS(mydata->u0),
           FLOAT_TO_INT_ARGS(mydata->v0));

    {
        float c2d[2] = {mydata->u0, mydata->v0};
        float w2f[4] = {
            mydata->w2[0][0], mydata->w2[0][1],
            mydata->w2[1][0], mydata->w2[1][1]
        };
        float axes[6] = {
            mydata->el1, mydata->el2,
            mydata->ev1x, mydata->ev1y,
            mydata->k_norm, mydata->s_norm
        };

        uart_dump_floats("CALIB_MEAN", mydata->mean, 3);
        uart_dump_floats("CALIB_E1", mydata->e1, 3);
        uart_dump_floats("CALIB_E2", mydata->e2, 3);
        uart_dump_floats("CALIB_C2D", c2d, 2);
        uart_dump_floats("CALIB_W2", w2f, 4);
        uart_dump_floats("CALIB_AXES", axes, 6);
    }

    /* --- FLASH calibration page; layout unchanged --- */
    erase_write_section_flash();

    mydata->page_idx = 0;
    mydata->buffer_angles_idx = 1;
    mydata->buffer_magn_idx = 2;

    mydata->buffer_angles[0] = 0x4444;
    mydata->buffer_magn[0] = 0x6666;
    mydata->buffer_magn[1] = 0x6666;

    uint16_t buffer_calibr[128];
    buffer_calibr[0] = 0x5555;

    int idx = 1;
    for (int k = 0; k < 3; k++) {
        idx = flash_put_float(buffer_calibr, idx, mydata->mean[k]);
    }
    for (int k = 0; k < 3; k++) {
        idx = flash_put_float(buffer_calibr, idx, mydata->e1[k]);
    }
    for (int k = 0; k < 3; k++) {
        idx = flash_put_float(buffer_calibr, idx, mydata->e2[k]);
    }

    idx = flash_put_float(buffer_calibr, idx, mydata->u0);
    idx = flash_put_float(buffer_calibr, idx, mydata->v0);
    idx = flash_put_float(buffer_calibr, idx, mydata->w2[0][0]);
    idx = flash_put_float(buffer_calibr, idx, mydata->w2[0][1]);
    idx = flash_put_float(buffer_calibr, idx, mydata->w2[1][0]);
    idx = flash_put_float(buffer_calibr, idx, mydata->w2[1][1]);
    idx = flash_put_float(buffer_calibr, idx, mydata->el1);
    idx = flash_put_float(buffer_calibr, idx, mydata->el2);
    idx = flash_put_float(buffer_calibr, idx, mydata->ev1x);
    idx = flash_put_float(buffer_calibr, idx, mydata->ev1y);
    idx = flash_put_float(buffer_calibr, idx, mydata->k_norm);
    idx = flash_put_float(buffer_calibr, idx, mydata->s_norm);

    for (int i = idx; i < 128; i++) buffer_calibr[i] = 0xFFFF;

    write_page_flash(mydata->page_idx++, (char *)buffer_calibr);

    DIAG_PRINTF("# Calibration complete. Stop-and-Go mission (%d measurements).\n",
           MAX_MEASUREMENTS);

    mydata->mission_point = 0;
    mydata->state_deadline_ms = deadline_after_ms(2000);
    mydata->cal_state = CAL_STATE_MISSION_INITIAL_WAIT;
}

static void calibration_finish_collection(void) {
    motor_stop();

    if (mydata->n_collected < N_CAL_MIN) {
        DIAG_PRINTF("# [FATAL] %d points (< %d). Check magnetometer / motors.\n",
               mydata->n_collected, N_CAL_MIN);
        pogobot_led_setColor(255, 0, 255);
        mydata->cal_state = CAL_STATE_FATAL;
        return;
    }

    calibration_fit_and_prepare_mission();
}

static void calibration_begin_rotation(void) {
    if (mydata->n_collected >= N_CAL ||
        mydata->global_attempts >= GLOBAL_MAX_ATTEMPTS) {
        calibration_finish_collection();
        return;
    }

    mydata->global_attempts++;
    motor_step_start();
    mydata->state_deadline_ms =
        deadline_after_ms((uint32_t)mydata->step_ms);
    mydata->cal_state = CAL_STATE_ROTATING;
}

static void mission_begin_rotation(void) {
    motor_step_start();
    mydata->state_deadline_ms = deadline_after_ms(STEP_MS_MES);
    mydata->cal_state = CAL_STATE_MISSION_ROTATING;
}

static void mission_commit_measurement(void) {
    float x = mydata->pending_x;
    float y = mydata->pending_y;
    float z = mydata->pending_z;

    int16_t ix = f2i16(x);
    int16_t iy = f2i16(y);
    int16_t iz = f2i16(z);

    mydata->buffer_magn[mydata->buffer_magn_idx++] = (uint16_t)ix;
    mydata->buffer_magn[mydata->buffer_magn_idx++] = (uint16_t)iy;
    mydata->buffer_magn[mydata->buffer_magn_idx++] = (uint16_t)iz;

    if (mydata->buffer_magn_idx >= 128) {
        write_page_flash(mydata->page_idx++, (char *)mydata->buffer_magn);
        mydata->buffer_magn_idx = 2;
    }

    float ang = heading_deg(
        x, y, z,
        mydata->mean, mydata->e1, mydata->e2,
        mydata->u0, mydata->v0, mydata->w2);

    printf("ANGLE," FLOAT_TO_INT_FMT "\n", FLOAT_TO_INT_ARGS(ang));
    printf("MAG,%d,%d,%d\n", (int)ix, (int)iy, (int)iz);
    DIAG_PRINTF(">>>> [POINT %d] Angle: " FLOAT_TO_INT_FMT " deg <<<<\n",
           mydata->mission_point, FLOAT_TO_INT_ARGS(ang));

    mydata->buffer_angles[mydata->buffer_angles_idx++] = (uint16_t)ang;
    if (mydata->buffer_angles_idx >= 128) {
        write_page_flash(mydata->page_idx++, (char *)mydata->buffer_angles);
        mydata->buffer_angles_idx = 1;
    }
}

static void calibration_flush_and_finish(void) {
    if (mydata->buffer_angles_idx > 1) {
        for (int i = mydata->buffer_angles_idx; i < 128; i++) {
            mydata->buffer_angles[i] = 0xFFFF;
        }
        write_page_flash(mydata->page_idx++, (char *)mydata->buffer_angles);
    }

    if (mydata->buffer_magn_idx > 2) {
        for (int i = mydata->buffer_magn_idx; i < 128; i++) {
            mydata->buffer_magn[i] = 0xFFFF;
        }
        write_page_flash(mydata->page_idx++, (char *)mydata->buffer_magn);
    }

    motor_stop();
    pogobot_led_setColor(255, 0, 0);
    DIAG_PRINTF("\n[SYSTEM] Capture complete. Pages written: %d\n",
           mydata->page_idx);

    mydata->cal_state = CAL_STATE_DONE;
}

/* Starts the cooperative calibration.
 * Unlike the original Pogobot-only function, this function returns
 * immediately. magn_calibration_step() must then be called from user_step(). */
void magn_calibration(void) {
    DIAG_PRINTF("# Init Pogobot - Conic fit + angular stratification\n");

    motor_stop();
    pogobot_led_setColor(0, 0, 0);

    mydata->n_collected = 0;
    mydata->step_ms = STEP_MS_INIT;
    mydata->global_attempts = 0;
    mydata->last_x = 0.0f;
    mydata->last_y = 0.0f;
    mydata->last_z = 0.0f;
    mydata->have_last = false;
    mydata->mag_read_active = false;
    mydata->dump_active = false;
    mydata->dump_completed = false;

    DIAG_PRINTF("# Phase 1: collecting %d points...\n", N_CAL);
    calibration_begin_rotation();
}

static void magn_calibration_step(void) {
    switch (mydata->cal_state) {
    case CAL_STATE_ROTATING:
        if (!deadline_reached(mydata->state_deadline_ms)) return;

        motor_stop();
        mydata->state_deadline_ms = deadline_after_ms(SETTLE_MS);
        mydata->cal_state = CAL_STATE_SETTLING;
        return;

    case CAL_STATE_SETTLING:
        if (!deadline_reached(mydata->state_deadline_ms)) return;

        mag_read_robust_start(N_AVG_CAL);
        mydata->cal_state = CAL_STATE_READING;
        return;

    case CAL_STATE_READING: {
        float x, y, z;
        mag_read_result_t result = mag_read_robust_step(&x, &y, &z);

        if (result == MAG_READ_PENDING) return;

        if (result == MAG_READ_FAILURE) {
            DIAG_PRINTF("# [WARN] magnetometer read failed (attempt %d)\n",
                   mydata->global_attempts);
            calibration_begin_rotation();
            return;
        }

        if (mydata->have_last) {
            float dx = x - mydata->last_x;
            float dy = y - mydata->last_y;
            float dz = z - mydata->last_z;

            if (dx * dx + dy * dy + dz * dz < MIN_DELTA_MAG_SQ) {
                if (mydata->step_ms < STEP_MS_MAX) {
                    mydata->step_ms += STEP_MS_INC;
                    if (mydata->step_ms > STEP_MS_MAX) {
                        mydata->step_ms = STEP_MS_MAX;
                    }
                }

                DIAG_PRINTF("# [ADAPT] insufficient rotation, step=%d ms\n",
                       mydata->step_ms);
                calibration_begin_rotation();
                return;
            }
        }

        mydata->last_x = x;
        mydata->last_y = y;
        mydata->last_z = z;
        mydata->have_last = true;

        int i = mydata->n_collected;
        mydata->cal_pts[i][0] = f2i16(x);
        mydata->cal_pts[i][1] = f2i16(y);
        mydata->cal_pts[i][2] = f2i16(z);

        printf("CALPT,%d,%d,%d\n",
               (int)mydata->cal_pts[i][0],
               (int)mydata->cal_pts[i][1],
               (int)mydata->cal_pts[i][2]);

        mydata->n_collected++;

        if ((mydata->n_collected % 20) == 0) {
            DIAG_PRINTF("# [Calibration] %d / %d (step=%d ms)\n",
                   mydata->n_collected, N_CAL, mydata->step_ms);
        }

        calibration_begin_rotation();
        return;
    }

    case CAL_STATE_MISSION_INITIAL_WAIT:
        if (!deadline_reached(mydata->state_deadline_ms)) return;

        motor_stop();
        mydata->state_deadline_ms =
            deadline_after_ms(SETTLE_MS + 100);
        mydata->cal_state = CAL_STATE_MISSION_SETTLING;
        return;

    case CAL_STATE_MISSION_SETTLING:
        if (!deadline_reached(mydata->state_deadline_ms)) return;

        mag_read_robust_start(N_AVG_MES);
        mydata->cal_state = CAL_STATE_MISSION_READING;
        return;

    case CAL_STATE_MISSION_READING: {
        float x, y, z;
        mag_read_result_t result = mag_read_robust_step(&x, &y, &z);

        if (result == MAG_READ_PENDING) return;

        if (result == MAG_READ_FAILURE) {
            DIAG_PRINTF("# [WARN] measurement %d lost (no flash emitted)\n",
                   mydata->mission_point);
            mission_begin_rotation();
            return;
        }

        mydata->pending_x = x;
        mydata->pending_y = y;
        mydata->pending_z = z;

        // Signal camera: keep the robot stopped and the LED green for 500 ms.
        pogobot_led_setColor(0, 255, 0);
        mydata->state_deadline_ms = deadline_after_ms(500);
        mydata->cal_state = CAL_STATE_MISSION_LED;
        return;
    }

    case CAL_STATE_MISSION_LED:
        if (!deadline_reached(mydata->state_deadline_ms)) return;

        pogobot_led_setColor(0, 0, 0);
        mission_commit_measurement();
        mission_begin_rotation();
        return;

    case CAL_STATE_MISSION_ROTATING:
        if (!deadline_reached(mydata->state_deadline_ms)) return;

        motor_stop();
        mydata->mission_point++;

        if (mydata->mission_point >= MAX_MEASUREMENTS) {
            calibration_flush_and_finish();
            return;
        }

        mydata->state_deadline_ms =
            deadline_after_ms(SETTLE_MS + 100);
        mydata->cal_state = CAL_STATE_MISSION_SETTLING;
        return;

    case CAL_STATE_IDLE:
    case CAL_STATE_DONE:
    case CAL_STATE_FATAL:
    default:
        return;
    }
}


//////  FLASH DUMP  ////// {{{1

/* =========================================================================
 * DUMP FLASH -> UART
 *
 * This is also cooperative: one page is read per eligible user_step().
 * The original 5 ms pause is represented by a 5 ms time deadline; the
 * operation resumes on the first user_step() call after that deadline.
 * ========================================================================= */

static void dump_u16_range(const char *tag, uint16_t *buf, int start, int n) {
    printf("%s", tag);
    for (int i = 0; i < n; i++) printf(",%u", buf[start + i]);
    printf("\n");
}

void dump_flash_uart(void) {
    if (mydata->dump_active) return;

    DIAG_PRINTF("\n# --- BEGIN MEMORY DUMP ---\n");
    mydata->dump_page_idx = 0;
    mydata->dump_empty_pages_count = 0;
    mydata->dump_next_ms = now_ms();
    mydata->dump_active = true;
    mydata->dump_completed = false;
}

static void dump_flash_uart_step(void) {
    if (!mydata->dump_active) return;
    if (!deadline_reached(mydata->dump_next_ms)) return;

    uint16_t buffer[128];
    read_page_flash(mydata->dump_page_idx, (char *)buffer);

    if (buffer[0] == 0x5555) {
        dump_u16_range("CALIB_MEAN", buffer, 1, 6);
        dump_u16_range("CALIB_E1", buffer, 7, 6);
        dump_u16_range("CALIB_E2", buffer, 13, 6);
        dump_u16_range("CALIB_C2D", buffer, 19, 4);
        dump_u16_range("CALIB_W2", buffer, 23, 8);
        dump_u16_range("CALIB_AXES", buffer, 31, 12);
        mydata->dump_empty_pages_count = 0;
    } else if (buffer[0] == 0x4444) {
        for (int i = 1; i < 128; i++) {
            if (buffer[i] == 0xFFFF) break;
            printf("ANGLE,%d\n", buffer[i]);
        }
        mydata->dump_empty_pages_count = 0;
    } else if (buffer[0] == 0x6666 && buffer[1] == 0x6666) {
        int16_t *sb = (int16_t *)buffer;
        for (int i = 2; i < 126; i += 3) {
            if (sb[i] == -1 && sb[i + 1] == -1 && sb[i + 2] == -1) break;
            printf("MAG,%d,%d,%d\n", sb[i], sb[i + 1], sb[i + 2]);
        }
        mydata->dump_empty_pages_count = 0;
    } else {
        mydata->dump_empty_pages_count++;
    }

    mydata->dump_page_idx++;

    if (mydata->dump_empty_pages_count >= 2) {
        DIAG_PRINTF("# --- END MEMORY DUMP ---\n");
        mydata->dump_active = false;
        mydata->dump_completed = true;
        return;
    }

    mydata->dump_next_ms = deadline_after_ms(5);
}


//////  MAIN FUNCTIONS  ////// {{{1

// Init function. Called once at the beginning of the program.
void user_init(void) {
#ifndef SIMULATOR
    DIAG_PRINTF("setup ok\n");
#endif

    srand(pogobot_helper_getRandSeed());

    main_loop_hz = USER_MAIN_LOOP_HZ;
    max_nb_processed_msg_per_tick = 0;
    percent_msgs_sent_per_ticks = 0;
    msg_rx_fn = NULL;
    msg_tx_fn = NULL;

    error_codes_led_idx = 3;

    uint8_t dir_mem[3];
    int8_t res_dir_mem_get = pogobot_motor_dir_mem_get(dir_mem);
    mydata->motor_dir_right = dir_mem[0];
    mydata->motor_dir_left = dir_mem[1];

    uint16_t power_mem[3];
    int8_t res_power_mem_get = pogobot_motor_power_mem_get(power_mem);
    mydata->motor_power_left = power_mem[1];
    mydata->motor_power_right = power_mem[0];

    DIAG_PRINTF("calibrated dir_mem:   (R:%u L:%u res=%d)\n",
           mydata->motor_dir_left, mydata->motor_dir_right,
           res_dir_mem_get);
    DIAG_PRINTF("calibrated power_mem: (R:%u L:%u res=%d)\n",
           mydata->motor_power_left, mydata->motor_power_right,
           res_power_mem_get);

    mydata->cal_state = CAL_STATE_IDLE;
    mydata->dump_active = false;
    mydata->dump_completed = false;
    mydata->next_live_dump_ms = deadline_after_ms(FLASH_DUMP_PERIOD_MS);

#if RUN_FULL_CALIBRATION
    // Start the previously blocking calibration workflow.
    magn_calibration();
#endif
}

static void live_magnetometer_test_step(void) {
    int16_t mag_x = 0;
    int16_t mag_y = 0;
    int16_t mag_z = 0;

    pogobot_led_setColor(0, 0, 255);

    if (magn_read_XYZ(&mag_x, &mag_y, &mag_z, 10) == 0) {
#ifdef SIMULATOR
        if (pogobot_helper_getid() == 0)
#endif
        {
            DIAG_PRINTF("%d %d %d\n", mag_x, mag_y, mag_z);
        }
        pogobot_led_setColor(0, 255, 0);
    }

    // Preserve the original continuous spin used by the simple test.
    pogobot_motor_dir_set(motorL, mydata->motor_dir_left);
    pogobot_motor_dir_set(motorR, mydata->motor_dir_right == 0 ? 1 : 0);
    pogobot_motor_set(motorL, (int)(mydata->motor_power_left * 0.3f));
    pogobot_motor_set(motorR, (int)(mydata->motor_power_right * 0.3f));

    // The original dump_flash_uart() blocked and used msleep(5).
    // Here it is started periodically and progresses one page at a time.
    if (!mydata->dump_active &&
        deadline_reached(mydata->next_live_dump_ms)) {
        dump_flash_uart();
        mydata->next_live_dump_ms = deadline_after_ms(FLASH_DUMP_PERIOD_MS);
    }

    dump_flash_uart_step();
}

// Step function. Called continuously at each step of the Pogobot main loop.
void user_step(void) {
#if RUN_FULL_CALIBRATION
    magn_calibration_step();

    // Dump flash once after the calibration/mission workflow finishes.
    if (mydata->cal_state == CAL_STATE_DONE &&
        !mydata->dump_active &&
        !mydata->dump_completed) {
#ifdef SIMULATOR
        if (pogobot_helper_getid() == 0) {
            dump_flash_uart();
        } else {
            // Avoid repeated dump attempts from robots whose UART output is
            // intentionally suppressed in simulation.
            mydata->dump_completed = true;
        }
#else
        dump_flash_uart();
#endif
    }

    dump_flash_uart_step();
#else
    live_magnetometer_test_step();
#endif
}

// Entrypoint of the program
int main(void) {
    pogobot_init();
#ifndef SIMULATOR
    DIAG_PRINTF("init ok\n");
#endif

    pogobot_start(user_init, user_step);
    return 0;
}

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
