
#include "pogobase.h"
#include <math.h>        // sinf, cosf, atan2f, hypotf, remainderf, isfinite
#include <float.h>       // FLT_EPSILON

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884197169399375105820974944
#endif

// -----------------------------------------------------------------------------
//  Global tunables (over-ridden by the simulator at launch)
// -----------------------------------------------------------------------------
float lighthouse_omega     = 6.0f;   // rad · s⁻¹
//float lighthouse_omega     = 3.141593f;   // rad · s⁻¹
//float lighthouse_omega     = 6.2831f;   // rad · s⁻¹
float edge_delta           = 50.f;   // ADC counts above dark level ⇒ "hit"
float lm_lambda0           = 1e-3f;  // initial LM damping (m²)
float robot_radius         = 0.026f; // 26 mm
float sensor_angle_deg[3]  = {  90.f, 210.f, 330.f };   // CCW, sensor 0 = +Y

uint16_t white_level_min =  1500;


// -----------------------------------------------------------------------------
//  Types
// -----------------------------------------------------------------------------
typedef struct { float x, y;                     } vec2;
typedef struct { vec2  pos; float yaw;           } pose_t;
typedef struct { uint32_t t;  int16_t v;         } sample_t;
typedef struct {
    bool valid;
    float time_s;
    bool is_sweep_start;  // Distinguish sweep start from ray hits
} hit_t;

// -----------------------------------------------------------------------------
//  Per-robot user data
// -----------------------------------------------------------------------------
typedef struct {
    sample_t last[3];
    hit_t    hit[3];
    bool     got[3];
    uint8_t  state;             /* 0 WAIT_WHITE  1 WAIT_FALL  2 COLLECT    */

    uint32_t white_start_ms;
    float    t0_s;
    float    sweep_deadline_s;

    float sweep_start_time_s;       // t_0: sweep start time in seconds
    float sweep_timeout_s;          // timeout for current sweep
    float alpha[3];                 // Bearings [rad]
    pose_t pose;                    // Estimated pose
    vec2 sensor_pos[3];             // In robot frame

    // Sweep start detection parameters
    float sweep_start_threshold;   // Brightness threshold for full-white detection
    float ray_threshold;           // Brightness threshold for ray detection
    float max_sweep_duration_s;    // Maximum expected time for full sweep
} USERDATA;
DECLARE_USERDATA(USERDATA);
REGISTER_USERDATA(USERDATA);

// -----------------------------------------------------------------------------
//  Small helpers
// -----------------------------------------------------------------------------
static inline vec2 rot(vec2 v, float yaw)
{
    float c = cosf(yaw), s = sinf(yaw);
    return (vec2){ c*v.x - s*v.y, s*v.x + c*v.y };
}

static void compute_sensor_positions(void)
{
    for (int k = 0; k < 3; ++k) {
        float a = sensor_angle_deg[k]*(float)M_PI/180.f;
        mydata->sensor_pos[k] = (vec2){ robot_radius*cosf(a),
                                        robot_radius*sinf(a) };
    }
}


// Enhanced interpolation function that handles both sweep start and ray hits
static float interpolate_hit_time(uint32_t t_prev, int16_t v_prev,
                                 uint32_t t_now, int16_t v_now,
                                 float threshold)
{
    if (v_now == v_prev) return (float)t_now;

    float t_hit = (float)t_prev + (float)(t_now - t_prev) *
                  (threshold - (float)v_prev) / ((float)v_now - (float)v_prev);
    return t_hit;
}


// -----------------------------------------------------------------------------
//  3×3 linear solver  (Gauss–Jordan, row-pivoting, *float*)
// -----------------------------------------------------------------------------
static bool solve_3x3(float A[3][3], const float b[3], float x[3])
{
    float M[3][4];   // augmented
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) M[i][j] = A[i][j];
        M[i][3] = b[i];
    }

    for (int k = 0; k < 3; ++k) {
        /* pivot */
        int piv = k;
        for (int i = k+1; i < 3; ++i)
            if (fabsf(M[i][k]) > fabsf(M[piv][k])) piv = i;
        if (fabsf(M[piv][k]) < 1e-9f) return false;   // singular

        /* swap rows */
        if (piv != k)
            for (int j = k; j < 4; ++j) {
                float tmp = M[k][j]; M[k][j] = M[piv][j]; M[piv][j] = tmp;
            }

        /* normalise & eliminate */
        float inv = 1.f / M[k][k];
        for (int j = k; j < 4; ++j) M[k][j] *= inv;        // diag = 1

        for (int i = 0; i < 3; ++i)
            if (i != k) {
                float f = M[i][k];
                for (int j = k; j < 4; ++j) M[i][j] -= f*M[k][j];
            }
    }
    for (int i = 0; i < 3; ++i) x[i] = M[i][3];
    return true;
}

// -----------------------------------------------------------------------------
//  Pose solver: 5 × Levenberg–Marquardt per sweep
// -----------------------------------------------------------------------------
static float bearing_residual(const float alpha[3], const pose_t *p)
{
    float sum = 0.f;
    for (int k = 0; k < 3; ++k) {
        vec2 sk  = rot(mydata->sensor_pos[k], p->yaw);
        vec2 g   = (vec2){ p->pos.x + sk.x, p->pos.y + sk.y };
        float mod = atan2f(g.y, g.x);
        float err = remainderf(alpha[k] - mod, 2.f*(float)M_PI);
        sum += err*err;
    }
    return sum;
}

static bool lm_solve_pose(const float alpha[3], pose_t *pose)
{
    float lambda = lm_lambda0;
    const int  max_iter = 10;

    for (int it = 0; it < max_iter; ++it) {

        /* build J and residual */
        float J[3][3] = {{0}},  r[3] = {0};
        for (int k = 0; k < 3; ++k) {
            vec2 sk = rot(mydata->sensor_pos[k], pose->yaw);
            vec2 g  = (vec2){ pose->pos.x + sk.x,
                              pose->pos.y + sk.y };
            float rho2 = g.x*g.x + g.y*g.y;
            float mod  = atan2f(g.y, g.x);
            float err  = remainderf(alpha[k] - mod, 2.f*(float)M_PI);
            r[k] = err;

            /* Row of J */
            J[k][0] = -g.y / rho2;    // ∂(atan2(g.y,g.x))/∂x = -g.y/rho2
            J[k][1] =  g.x / rho2;    // ∂(atan2(g.y,g.x))/∂y =  g.x/rho2
            J[k][2] = -(sk.x*g.y - sk.y*g.x) / rho2;  // ∂(atan2(g.y,g.x))/∂yaw
        }

        /* JTJ and JTr */
        float A[3][3] = {{0}},  b[3] = {0};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                    A[i][j] += J[k][i]*J[k][j];

        for (int i = 0; i < 3; ++i)
            for (int k = 0; k < 3; ++k)
                b[i] += J[k][i]*r[k];

        /* LM damping */
        for (int i = 0; i < 3; ++i) A[i][i] += lambda;

        /* solve */
        float delta[3];
        if (!solve_3x3(A, b, delta)) {
            lambda *= 10.f;
            continue;    // retry with larger damping
        }

        pose_t trial = { { pose->pos.x + delta[0],
                           pose->pos.y + delta[1]},
                         remainderf(pose->yaw + delta[2],
                                    2.f*(float)M_PI) };

        /* accept / reject step */
        float err_old = bearing_residual(alpha, pose);
        float err_new = bearing_residual(alpha, &trial);

        if (err_new < err_old) {           // success → move & shrink λ
            *pose  = trial;
            lambda = fmaxf(lambda * 0.3f, 1e-6f);
        } else                             // fail → enlarge λ
            lambda *= 10.f;
    }
    return isfinite(pose->pos.x) && isfinite(pose->pos.y) &&
           isfinite(pose->yaw);
}

static void lighthouse_loop(void)
{
    uint32_t now_ms = current_time_milliseconds();
    int16_t  v[3]; for (int i = 0; i < 3; ++i) v[i] = pogobot_photosensors_read(i);

    /* ---------- 0. WAIT FOR WHITE --------------------------------------- */
    if (mydata->state == 0) {                        /* WAIT_WHITE */
        bool all_white = true;
        for (int i = 0; i < 3; ++i)
            if (v[i] < white_level_min) { all_white = false; break; }

        if (all_white) {                            /* rising edge */
            mydata->white_start_ms = now_ms;
            mydata->state = 1;                      /* → WAIT_FALL */
            printf("# ALL_WHITE detected at t=%.3fs\n", now_ms);
        }
        goto update_last;
    }

    /* ---------- 1. STAY IN WHITE UNTIL IT ENDS -------------------------- */
    if (mydata->state == 1) {                        /* WAIT_FALL */
        bool all_white = true;
        for (int i = 0; i < 3; ++i)
            if (v[i] < white_level_min) { all_white = false; break; }

        if (!all_white) {                           /* falling edge */
            /* use the mid-point of the plateau as t0 (±5 ms at 100 Hz) */
            mydata->t0_s            = 0.001f * (mydata->white_start_ms + now_ms) * 0.5f;
            mydata->sweep_deadline_s = mydata->t0_s + (M_PI * 2.f / lighthouse_omega);

            memset(mydata->got, 0, sizeof mydata->got);
            mydata->state = 2;                      /* → COLLECT */
            printf("# SWEEP_START detected at t=%.3fs\n", now_ms);
        }
        goto update_last;
    }

    /* ---------- 2. NORMAL RAY HITS -------------------------------------- */
    if (mydata->state == 2) {                        /* COLLECT */
        float now_s = 0.001f * now_ms;

        /* sweep missed? */
        if (now_s > mydata->sweep_deadline_s) {
            mydata->state = 0;                      /* restart    */
            goto update_last;
        }

        for (int k = 0; k < 3; ++k) {
            int16_t dv = v[k] - mydata->last[k].v;
            if (!mydata->got[k] && dv > (edge_delta * 0.6f) && v[k] > mydata->last[k].v + 15) {
                float thr  = mydata->last[k].v + edge_delta;
                float hit_ms = interpolate_hit_time(mydata->last[k].t,
                                                    mydata->last[k].v,
                                                    now_ms, v[k], thr);
                mydata->hit[k] = (hit_t){ true, 0.001f*hit_ms - mydata->t0_s };
                mydata->got[k] = true;
                    printf("# HIT sensor %d at Δt=%.3fs (dv=%d, val=%d)\n",
                           k, mydata->hit[k].time_s, dv, v[k]);

                // XXX
                //float relative_time_s = (hit_ms * 1e-3f) - mydata->sweep_start_time_s;
                //if (relative_time_s > 0.f && relative_time_s < 1.0f) {
                //    mydata->hit[k] = (hit_t){ true, relative_time_s };
                //    mydata->got[k] = true;
                //    printf("# HIT sensor %d at Δt=%.3fs (dv=%d, val=%d)\n",
                //           k, relative_time_s, dv, v[k]);
                //}

            }
        }

        // Allow 2-sensor solutions after 1 second
        int sensor_count = mydata->got[0] + mydata->got[1] + mydata->got[2];
        //bool timeout_partial = (now_ms / 1000.f > mydata->sweep_start_time_s + 1.0f) && (sensor_count >= 2);

        // Check if all rays detected
        //if (sensor_count == 3 || timeout_partial) {
        if (sensor_count == 3) {
            // Convert to absolute bearings using sweep start reference
            for (int k = 0; k < 3; ++k) {
                float delta_time_s = mydata->hit[k].time_s - mydata->sweep_start_time_s;
                mydata->alpha[k] = lighthouse_omega * delta_time_s;

                // Normalize to [0, 2π)
                mydata->alpha[k] = fmodf(mydata->alpha[k], 2.f*(float)M_PI);
                if (mydata->alpha[k] < 0.f) mydata->alpha[k] += 2.f*(float)M_PI;
            }

            // Solve pose with absolute bearings
            bool solve_ok = lm_solve_pose(mydata->alpha, &mydata->pose);

            printf("# POSE: sensors:%d | Δt = %.3f %.3f %.3f | α = %.3f %.3f %.3f | pose = (%.3f,%.3f,%.2f°) %s\n",
                   sensor_count,
                   mydata->hit[0].time_s - mydata->sweep_start_time_s,
                   mydata->hit[1].time_s - mydata->sweep_start_time_s,
                   mydata->hit[2].time_s - mydata->sweep_start_time_s,
                   mydata->alpha[0], mydata->alpha[1], mydata->alpha[2],
                   mydata->pose.pos.x, mydata->pose.pos.y,
                   mydata->pose.yaw*180.f/(float)M_PI,
                   solve_ok ? "OK" : "FAIL");

            mydata->state = 0;   // Wait for next sweep start
        }
    }

update_last:
    for (int i = 0; i < 3; ++i) mydata->last[i] = (sample_t){ now_ms, v[i] };
}



// -----------------------------------------------------------------------------
//  Simulator integration
// -----------------------------------------------------------------------------
#ifdef SIMULATOR
void global_setup(void)
{
    init_from_configuration(lighthouse_omega);
    init_from_configuration(edge_delta);
    init_from_configuration(lm_lambda0);
}
void create_data_schema(void)
{
    data_add_column_double("pred_x");
    data_add_column_double("pred_y");
    data_add_column_double("pred_yaw");
}
void export_data(void)
{
    data_set_value_double("pred_x",   mydata->pose.pos.x);
    data_set_value_double("pred_y",   mydata->pose.pos.y);
    data_set_value_double("pred_yaw", mydata->pose.yaw);
}
#endif

// -----------------------------------------------------------------------------
//  Entry points
// -----------------------------------------------------------------------------

void user_init(void)
{
    srand(pogobot_helper_getRandSeed());
    main_loop_hz = 250;
    max_nb_processed_msg_per_tick = 0;
    msg_rx_fn = NULL; msg_tx_fn = NULL;
    error_codes_led_idx = 3;

    pogobot_led_setColor(0, 0, 255);     // blue = initialising

    uint32_t t0 = current_time_milliseconds();
    for (int i = 0; i < 3; ++i) {
        mydata->last[i] = (sample_t){ t0, pogobot_photosensors_read(i) };
        mydata->hit[i] = (hit_t){ false, 0.f, false };
        mydata->got[i] = false;
    }

    mydata->pose  = (pose_t){ { 0.05f, 0.f }, 0.2f };
    mydata->state = 0;  // Start in sweep start detection mode
    mydata->sweep_start_time_s = 0.f;
    mydata->sweep_timeout_s = 0.f;

    // Configure thresholds (these should be tuned experimentally)
    mydata->sweep_start_threshold = edge_delta * 3.0f;  // Sweep start is much brighter
    mydata->max_sweep_duration_s = 0.5f;  // Slightly longer than one rotation period

    compute_sensor_positions();
    pogobot_led_setColor(0, 255, 0);     // green = ready
}

void user_step(void)
{
    lighthouse_loop();
}

int main(void)
{
    pogobot_init();
    pogobot_start(user_init, user_step);
#ifdef SIMULATOR
    SET_CALLBACK(callback_global_setup,        global_setup);
    SET_CALLBACK(callback_create_data_schema,  create_data_schema);
    SET_CALLBACK(callback_export_data,         export_data);
#endif
    return 0;
}

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
