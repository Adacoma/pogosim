#include "pogobase.h"
#include <math.h>        // sinf, cosf, atan2f, hypotf, remainderf, isfinite
#include <float.h>       // FLT_EPSILON

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884197169399375105820974944
#endif

// -----------------------------------------------------------------------------
//  Global tunables (over-ridden by the simulator at launch)
// -----------------------------------------------------------------------------
float lighthouse_omega     = 1.0f;   // rad · s⁻¹ (sweep speed for π/2 quadrant)
float edge_delta           = 50.f;   // ADC counts above dark level ⇒ "hit"
float max_dist_from_center = 1.5f;

uint16_t white_level_min = 1000;

// Timing constants for dual lighthouse system
float long_white_duration_s  = 0.5f;   // Long white frame duration
float short_white_duration_s = 0.1f;   // Short white frame duration
float long_white_min_s       = 0.3f;   // Minimum to detect long white
float short_white_min_s      = 0.010f; // Minimum to detect short white (10ms)
float short_white_max_s      = 0.25f;  // Maximum for short white
float white_timeout_s        = 1.0f;   // Max white duration before reset

// Lighthouse positions (arena assumed centered at origin)
float lighthouse1_x = -0.75f;  // Top-left lighthouse
float lighthouse1_y =  0.75f;
float lighthouse2_x =  0.75f;  // Top-right lighthouse  
float lighthouse2_y =  0.75f;

// State definitions
#define STATE_WAIT_LONG_WHITE  0
#define STATE_WAIT_RAY1_FALL   1  
#define STATE_COLLECT_RAY1     2
#define STATE_WAIT_SHORT_WHITE 3
#define STATE_WAIT_RAY2_FALL   4
#define STATE_COLLECT_RAY2     5

// -----------------------------------------------------------------------------
//  Types
// -----------------------------------------------------------------------------
typedef struct { float x, y; } vec2;
typedef struct { vec2 pos; } pose_t;  // Only position, no orientation
typedef struct { uint32_t t; int16_t v; } sample_t;
typedef struct {
    bool valid;
    float time_s;
} hit_t;

// -----------------------------------------------------------------------------
//  Per-robot user data
// -----------------------------------------------------------------------------
typedef struct {
    sample_t last_sample;
    uint8_t state;
    
    uint32_t white_start_ms;
    float white_start_s;
    
    hit_t ray1_hit;
    hit_t ray2_hit;
    bool got_ray1;
    bool got_ray2;
    
    float ray1_start_s;
    float ray2_start_s;
    float sweep_duration_s;  // Duration of π/2 sweep
    
    pose_t pose;
    bool pose_valid;
    
} USERDATA;
DECLARE_USERDATA(USERDATA);
REGISTER_USERDATA(USERDATA);

// -----------------------------------------------------------------------------
//  Helper functions
// -----------------------------------------------------------------------------
static bool is_white_condition(void)
{
    int16_t v[3];
    for (int i = 0; i < 3; ++i) v[i] = pogobot_photosensors_read(i);
    
    // Count how many sensors are above white threshold
    int white_count = 0;
    for (int i = 0; i < 3; ++i) {
        if (v[i] >= white_level_min) white_count++;
    }
    
    // Require at least 2 out of 3 sensors to be bright for "white" condition
    // This prevents single ray hits from being detected as white
    bool is_white = (white_count >= 2);
    
    // Debug: Print when white detection changes
    static bool last_white = false;
    if (is_white != last_white) {
        printf("# WHITE_CHANGE: %s (sensors: %d/%d/%d, count: %d/3)\n", 
               is_white ? "YES" : "NO", v[0], v[1], v[2], white_count);
        last_white = is_white;
    }
    
    return is_white;
}

static int16_t get_max_sensor_value(void)
{
    int16_t v[3];
    for (int i = 0; i < 3; ++i) v[i] = pogobot_photosensors_read(i);
    
    int16_t max_v = v[0];
    if (v[1] > max_v) max_v = v[1];
    if (v[2] > max_v) max_v = v[2];
    
    // Debug: Print sensor values occasionally when they change significantly
    static int16_t last_debug_max = 0;
    static uint32_t last_debug_time = 0;
    uint32_t now_ms = current_time_milliseconds();
    
    if (abs(max_v - last_debug_max) > 1000 || (now_ms - last_debug_time) > 1000) {
        printf("# SENSORS: [%d, %d, %d] max=%d\n", v[0], v[1], v[2], max_v);
        last_debug_max = max_v;
        last_debug_time = now_ms;
    }
    
    return max_v;
}

static float interpolate_hit_time(uint32_t t_prev, int16_t v_prev,
                                 uint32_t t_now, int16_t v_now,
                                 float threshold)
{
    if (v_now == v_prev) return (float)t_now;

    float t_hit = (float)t_prev + (float)(t_now - t_prev) *
                  (threshold - (float)v_prev) / ((float)v_now - (float)v_prev);
    return t_hit;
}

// Triangulate position from two lighthouse angles
static bool triangulate_position(float angle1, float angle2, vec2 *pos)
{
    // Lighthouse 1 ray direction
    vec2 dir1 = { cosf(angle1), sinf(angle1) };
    // Lighthouse 2 ray direction  
    vec2 dir2 = { cosf(angle2), sinf(angle2) };
    
    // Lighthouse positions
    vec2 l1 = { lighthouse1_x, lighthouse1_y };
    vec2 l2 = { lighthouse2_x, lighthouse2_y };
    
    // Solve intersection of two rays:
    // l1 + t1 * dir1 = l2 + t2 * dir2
    // l1.x + t1 * dir1.x = l2.x + t2 * dir2.x
    // l1.y + t1 * dir1.y = l2.y + t2 * dir2.y
    //
    // Rearranging:
    // t1 * dir1.x - t2 * dir2.x = l2.x - l1.x
    // t1 * dir1.y - t2 * dir2.y = l2.y - l1.y
    
    float dx = l2.x - l1.x;
    float dy = l2.y - l1.y;
    
    float det = dir1.x * (-dir2.y) - dir1.y * (-dir2.x);
    det = dir1.x * dir2.y - dir1.y * dir2.x;
    
    if (fabsf(det) < 1e-6f) {
        return false; // Parallel rays
    }
    
    float t1 = (dx * dir2.y - dy * dir2.x) / det;
    
    // Calculate intersection point
    pos->x = l1.x + t1 * dir1.x;
    pos->y = l1.y + t1 * dir1.y;
    
    // Check if position is reasonable
    if (fabsf(pos->x) > max_dist_from_center || fabsf(pos->y) > max_dist_from_center) {
        return false;
    }
    
    return true;
}

// -----------------------------------------------------------------------------
//  Main lighthouse loop
// -----------------------------------------------------------------------------
static void lighthouse_loop(void)
{
    uint32_t now_ms = current_time_milliseconds();
    float now_s = now_ms * 0.001f;
    int16_t max_v = get_max_sensor_value();
    
    bool is_white = is_white_condition();
    
    // Debug state changes
    static uint8_t last_state = 255;
    if (mydata->state != last_state) {
        const char* state_names[] = {"WAIT_LONG_WHITE", "WAIT_RAY1_FALL", "COLLECT_RAY1", 
                                   "WAIT_SHORT_WHITE", "WAIT_RAY2_FALL", "COLLECT_RAY2"};
        printf("# STATE_CHANGE: %s -> %s at t=%.3fs\n", 
               (last_state < 6) ? state_names[last_state] : "UNKNOWN",
               (mydata->state < 6) ? state_names[mydata->state] : "UNKNOWN", now_s);
        last_state = mydata->state;
    }
    
    switch (mydata->state) {
        
        case STATE_WAIT_LONG_WHITE: {
            if (is_white) {
                if (mydata->white_start_s == 0.f) {
                    // Start of white period
                    mydata->white_start_s = now_s;
                    mydata->white_start_ms = now_ms;
                }
                
                float white_duration = now_s - mydata->white_start_s;
                if (white_duration >= long_white_min_s) {
                    // Long white detected, ready for ray 1
                    mydata->state = STATE_WAIT_RAY1_FALL;
                    printf("# LONG_WHITE detected, duration=%.3fs\n", white_duration);
                }
                
                // Reset if white period too long (missed cycle)
                if (white_duration > white_timeout_s) {
                    printf("# WHITE_TIMEOUT, resetting\n");
                    mydata->white_start_s = now_s;
                    mydata->white_start_ms = now_ms;
                }
                
            } else {
                // Not white, reset white start time
                mydata->white_start_s = 0.f;
            }
            break;
        }
        
        case STATE_WAIT_RAY1_FALL: {
            if (!is_white) {
                // End of white, start ray 1 collection
                mydata->ray1_start_s = now_s;
                mydata->sweep_duration_s = (M_PI * 0.5f) / lighthouse_omega; // π/2 sweep
                mydata->got_ray1 = false;
                mydata->state = STATE_COLLECT_RAY1;
                printf("# RAY1_START at t=%.3fs\n", now_s);
            }
            break;
        }
        
        case STATE_COLLECT_RAY1: {
            // Check for ray 1 hit (during dark period)
            int16_t dv = max_v - mydata->last_sample.v;
            if (!mydata->got_ray1 && !is_white && dv > edge_delta && max_v > mydata->last_sample.v + 15) {
                float threshold = mydata->last_sample.v + edge_delta;
                float hit_ms = interpolate_hit_time(mydata->last_sample.t,
                                                   mydata->last_sample.v,
                                                   now_ms, max_v, threshold);
                float hit_time_s = hit_ms * 0.001f;
                mydata->ray1_hit = (hit_t){ true, hit_time_s - mydata->ray1_start_s };
                mydata->got_ray1 = true;
                printf("# RAY1_HIT at Δt=%.3fs (dv=%d, val=%d) - during dark period\n", 
                       mydata->ray1_hit.time_s, dv, max_v);
            }
            
            // Check for timeout first
            if ((now_s - mydata->ray1_start_s) > mydata->sweep_duration_s * 1.5f) {
                printf("# RAY1_TIMEOUT after %.3fs, resetting\n", now_s - mydata->ray1_start_s);
                mydata->state = STATE_WAIT_LONG_WHITE;
                mydata->white_start_s = 0.f;
                break;
            }
            
            // Check for return to white (end of ray 1 sweep period)
            if (is_white) {
                printf("# RAY1_SWEEP_END detected, transitioning to short white at t=%.3fs (got_hit=%d)\n", 
                       now_s, mydata->got_ray1);
                mydata->white_start_s = now_s;
                mydata->white_start_ms = now_ms;
                mydata->state = STATE_WAIT_SHORT_WHITE;
            }
            break;
        }
        
        case STATE_WAIT_SHORT_WHITE: {
            if (is_white) {
                float white_duration = now_s - mydata->white_start_s;
                printf("# SHORT_WHITE continuing, duration=%.3fs\n", white_duration);
                
                // If white period too long, assume we missed ray and reset
                if (white_duration > long_white_min_s) {
                    printf("# LONG_WHITE detected instead of short (%.3fs), resetting\n", white_duration);
                    mydata->state = STATE_WAIT_LONG_WHITE;
                    // Keep current white_start_s as it's already a long white
                }
                
            } else {
                // White ended, check duration
                float white_duration = now_s - mydata->white_start_s;
                printf("# SHORT_WHITE ended after %.3fs\n", white_duration);
                
                if (white_duration >= short_white_min_s) {
                    // Valid short white detected, ready for ray 2
                    printf("# SHORT_WHITE valid, transitioning to ray 2\n");
                    mydata->state = STATE_WAIT_RAY2_FALL;
                } else {
                    // White ended too early, reset
                    printf("# SHORT_WHITE too short (%.3fs < %.3fs), resetting\n", 
                           white_duration, short_white_min_s);
                    mydata->state = STATE_WAIT_LONG_WHITE;
                    mydata->white_start_s = 0.f;
                }
            }
            break;
        }
        
        case STATE_WAIT_RAY2_FALL: {
            if (!is_white) {
                // End of white, start ray 2 collection
                mydata->ray2_start_s = now_s;
                mydata->got_ray2 = false;
                mydata->state = STATE_COLLECT_RAY2;
                printf("# RAY2_START at t=%.3fs\n", now_s);
            }
            break;
        }
        
        case STATE_COLLECT_RAY2: {
            // Check for ray 2 hit (during dark period)
            int16_t dv = max_v - mydata->last_sample.v;
            if (!mydata->got_ray2 && !is_white && dv > edge_delta && max_v > mydata->last_sample.v + 15) {
                float threshold = mydata->last_sample.v + edge_delta;
                float hit_ms = interpolate_hit_time(mydata->last_sample.t,
                                                   mydata->last_sample.v,
                                                   now_ms, max_v, threshold);
                float hit_time_s = hit_ms * 0.001f;
                mydata->ray2_hit = (hit_t){ true, hit_time_s - mydata->ray2_start_s };
                mydata->got_ray2 = true;
                printf("# RAY2_HIT at Δt=%.3fs (dv=%d, val=%d) - during dark period\n", 
                       mydata->ray2_hit.time_s, dv, max_v);
            }
            
            // Check for timeout first
            if ((now_s - mydata->ray2_start_s) > mydata->sweep_duration_s * 1.5f) {
                printf("# RAY2_TIMEOUT after %.3fs\n", now_s - mydata->ray2_start_s);
                
                // Compute pose if we have both hits even on timeout
                if (mydata->got_ray1 && mydata->got_ray2) {
                    // Calculate angles from lighthouse sweep times
                    float angle1 = -M_PI/2.f + (mydata->ray1_hit.time_s / mydata->sweep_duration_s) * (M_PI/2.f);
                    float angle2 = M_PI + (mydata->ray2_hit.time_s / mydata->sweep_duration_s) * (M_PI/2.f);
                    
                    vec2 pos;
                    bool solve_ok = triangulate_position(angle1, angle2, &pos);
                    
                    if (solve_ok) {
                        mydata->pose.pos = pos;
                        mydata->pose_valid = true;
                        printf("# POSE_COMPUTED: (%.3f, %.3f) from angles %.1f°, %.1f° (timeout)\n",
                               pos.x, pos.y, angle1 * 180.f / M_PI, angle2 * 180.f / M_PI);
                    } else {
                        mydata->pose_valid = false;
                        printf("# POSE_ERROR: Failed to triangulate position from angles %.1f°, %.1f° (timeout)\n",
                               angle1 * 180.f / M_PI, angle2 * 180.f / M_PI);
                    }
                } else {
                    mydata->pose_valid = false;
                    printf("# POSE_ERROR: Missing ray hits on timeout (ray1=%d, ray2=%d)\n", 
                           mydata->got_ray1, mydata->got_ray2);
                }
                
                // Reset for next cycle
                mydata->state = STATE_WAIT_LONG_WHITE;
                mydata->white_start_s = 0.f;
                break;
            }
            
            // Check for end of ray 2 (return to white)
            if (is_white) {
                printf("# RAY2_SWEEP_END detected at t=%.3fs (got_hit=%d)\n", now_s, mydata->got_ray2);
                
                // Compute pose if we have both hits
                if (mydata->got_ray1 && mydata->got_ray2) {
                    
                    // Calculate angles from lighthouse sweep times
                    // Lighthouse 1 sweeps from 270° to 360° (or -90° to 0°)
                    float angle1 = -M_PI/2.f + (mydata->ray1_hit.time_s / mydata->sweep_duration_s) * (M_PI/2.f);
                    
                    // Lighthouse 2 sweeps from 180° to 270° (or π to 3π/2)
                    float angle2 = M_PI + (mydata->ray2_hit.time_s / mydata->sweep_duration_s) * (M_PI/2.f);
                    
                    vec2 pos;
                    bool solve_ok = triangulate_position(angle1, angle2, &pos);
                    
                    if (solve_ok) {
                        mydata->pose.pos = pos;
                        mydata->pose_valid = true;
                        printf("# POSE_COMPUTED: (%.3f, %.3f) from angles %.1f°, %.1f°\n",
                               pos.x, pos.y, angle1 * 180.f / M_PI, angle2 * 180.f / M_PI);
                    } else {
                        mydata->pose_valid = false;
                        printf("# POSE_ERROR: Failed to triangulate position from angles %.1f°, %.1f°\n",
                               angle1 * 180.f / M_PI, angle2 * 180.f / M_PI);
                    }
                    
                } else {
                    mydata->pose_valid = false;
                    printf("# POSE_ERROR: Missing ray hits (ray1=%d, ray2=%d)\n", 
                           mydata->got_ray1, mydata->got_ray2);
                }
                
                // Reset for next cycle
                mydata->state = STATE_WAIT_LONG_WHITE;
                mydata->white_start_s = now_s;
                mydata->white_start_ms = now_ms;
            }
            break;
        }
    }
    
    // Update last sample
    mydata->last_sample = (sample_t){ now_ms, max_v };
}

// -----------------------------------------------------------------------------
//  Simulator integration
// -----------------------------------------------------------------------------
#ifdef SIMULATOR
void global_setup(void)
{
    init_from_configuration(lighthouse_omega);
    init_from_configuration(edge_delta);
    init_from_configuration(max_dist_from_center);
    init_from_configuration(long_white_duration_s);
    init_from_configuration(short_white_duration_s);
    init_from_configuration(lighthouse1_x);
    init_from_configuration(lighthouse1_y);
    init_from_configuration(lighthouse2_x);
    init_from_configuration(lighthouse2_y);
}

void create_data_schema(void)
{
    data_add_column_double("pred_x");
    data_add_column_double("pred_y");
    data_add_column_bool("pose_valid");
}

void export_data(void)
{
    data_set_value_double("pred_x", mydata->pose.pos.x);
    data_set_value_double("pred_y", mydata->pose.pos.y);
    data_set_value_bool("pose_valid", mydata->pose_valid);
}
#endif

// -----------------------------------------------------------------------------
//  Entry points
// -----------------------------------------------------------------------------

void user_init(void)
{
    srand(pogobot_helper_getRandSeed());
    main_loop_hz = 60;
    max_nb_processed_msg_per_tick = 0;
    msg_rx_fn = NULL; msg_tx_fn = NULL;
    error_codes_led_idx = 3;

    pogobot_led_setColor(0, 0, 255);     // blue = initialising

    uint32_t t0 = current_time_milliseconds();
    mydata->last_sample = (sample_t){ t0, get_max_sensor_value() };
    
    mydata->ray1_hit = (hit_t){ false, 0.f };
    mydata->ray2_hit = (hit_t){ false, 0.f };
    mydata->got_ray1 = false;
    mydata->got_ray2 = false;

    mydata->pose = (pose_t){ { 0.f, 0.f } };
    mydata->pose_valid = false;
    mydata->state = STATE_WAIT_LONG_WHITE;
    mydata->white_start_s = 0.f;
    mydata->sweep_duration_s = (M_PI * 0.5f) / lighthouse_omega;

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
    SET_CALLBACK(callback_global_setup, global_setup);
    SET_CALLBACK(callback_create_data_schema, create_data_schema);
    SET_CALLBACK(callback_export_data, export_data);
#endif
    return 0;
}

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
