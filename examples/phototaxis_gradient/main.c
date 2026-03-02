// Main include for pogobots, both for real robots and for simulations
#include "pogobase.h"

#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884197169399375105820974944
#endif


// -----------------------------------------------------------------------------
// Global tunables (can be overridden by the simulator YAML via global_setup())
// -----------------------------------------------------------------------------
float base_speed_ratio = 0.20f;     // 0..1 of motorFull
float turn_gain_ratio  = 0.25f;     // 0..1 of motorFull (multiplied by normalized left-right diff)
uint16_t dark_threshold = 210;       // below this, consider it "dark" and do a gentle search turn
float search_turn_ratio = 0.12f;    // 0..1 of motorFull when in the dark

float p_tumble = 0.02f;          // tumble probability per tick while searching (60Hz -> ~1.2 tumbles/s)
uint16_t tumble_min_ticks = 12;  // 0.2s
uint16_t tumble_max_ticks = 36;  // 0.6s
float tumble_turn_ratio = 0.18f; // turning speed during tumble


// -----------------------------------------------------------------------------
// Per-robot state
// -----------------------------------------------------------------------------
typedef struct {
    uint8_t motor_dir_left;
    uint8_t motor_dir_right;

    // Run&tumble state
    uint8_t searching;          // 1 if in "search" mode
    uint8_t tumbling;           // 1 if currently tumbling
    int8_t tumble_sign;         // -1 or +1
    uint16_t tumble_ticks_left; // countdown
    uint32_t last_light_sum;    // for biasing
} USERDATA;

DECLARE_USERDATA(USERDATA);
REGISTER_USERDATA(USERDATA);

static inline int16_t clamp_i16(int16_t v, int16_t lo, int16_t hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static inline float clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

// Uses sensor angles:
// 0: 180°, 1: -40° (320°), 2: +40°
static void phototaxis_step_from_3_sensors(int16_t s0_back, int16_t s1_fr, int16_t s2_fl,
                                          float base_speed_ratio, float turn_gain_ratio,
                                          int16_t *out_left_cmd, int16_t *out_right_cmd) {
    // Normalize to reduce saturation effects
    float w0 = (float)s0_back;
    float w1 = (float)s1_fr;
    float w2 = (float)s2_fl;
    float sum = w0 + w1 + w2;
    if (sum < 1.0f) sum = 1.0f;
    w0 /= sum; w1 /= sum; w2 /= sum;

    // Angles in radians: back=pi, fr=-40deg, fl=+40deg
    const float a0 = (float)M_PI;
    const float a1 = -40.0f * (float)M_PI / 180.0f;
    const float a2 =  40.0f * (float)M_PI / 180.0f;

    // Resultant direction in robot frame
    float gx = w0 * cosf(a0) + w1 * cosf(a1) + w2 * cosf(a2);
    float gy = w0 * sinf(a0) + w1 * sinf(a1) + w2 * sinf(a2);

    // Bearing error toward brighter direction
    float err = -atan2f(gy, gx);

    // Base speed
    int16_t base = (int16_t)((float)motorFull * base_speed_ratio);

    // Photokinesis (helps a lot to break the ring):
    // Slow down when very bright (near the peak)
    float bright = clampf(sum / 3.0f, 0.0f, 1.0f);   // crude, but works
    float speed_scale = clampf(1.0f - 0.7f * bright, 0.15f, 1.0f);
    base = (int16_t)((float)base * speed_scale);

    // Turn command proportional to angular error
    float turn_f = (float)motorFull * turn_gain_ratio * err;
    int16_t turn = (int16_t)clampf(turn_f, -(float)motorFull, (float)motorFull);

    // Differential drive:
    // err>0 => light on left => turn left => left motor slower, right motor faster
    int16_t left_cmd  = clamp_i16(base - turn, motorStop, motorFull);
    int16_t right_cmd = clamp_i16(base + turn, motorStop, motorFull);

    *out_left_cmd = left_cmd;
    *out_right_cmd = right_cmd;
}

void user_init(void) {
#ifndef SIMULATOR
    printf("setup ok\n");
#endif

    srand(pogobot_helper_getRandSeed());

    // 60 Hz control loop
    main_loop_hz = 60;

    // No messaging in this demo.
    max_nb_processed_msg_per_tick = 0;
    msg_rx_fn = NULL;
    msg_tx_fn = NULL;

    // Retrieve calibration data from robot memory (motor direction)
    uint8_t dir_mem[3];
    pogobot_motor_dir_mem_get(dir_mem);
    mydata->motor_dir_left = dir_mem[1];
    mydata->motor_dir_right = dir_mem[0];
    pogobot_motor_dir_set(motorL, mydata->motor_dir_left);
    pogobot_motor_dir_set(motorR, mydata->motor_dir_right);

    mydata->searching = 0;
    mydata->tumbling = 0;
    mydata->tumble_sign = 1;
    mydata->tumble_ticks_left = 0;
    mydata->last_light_sum = 0;

    // LED index for error codes (negative disables)
    error_codes_led_idx = 3;
}

void user_step(void) {
    int16_t s0 = pogobot_photosensors_read(0); // back
    int16_t s1 = pogobot_photosensors_read(1); // front-right (-40°)
    int16_t s2 = pogobot_photosensors_read(2); // front-left  (+40°)

    uint32_t sum = (uint32_t)(s0 + s1 + s2);

    // Decide if we're in "searching" (low-signal) regime
    mydata->searching = (sum < (uint32_t)dark_threshold * 3u);

    int16_t base = (int16_t)((float)motorFull * base_speed_ratio);

    if (mydata->searching) {
        // --- RUN & TUMBLE SEARCH MODE ---------------------------------------

        // Bias: if light is improving, tumble less; if worsening, tumble more
        float p = p_tumble;
        if (mydata->last_light_sum > 0) {
            int32_t d = (int32_t)sum - (int32_t)mydata->last_light_sum;
            if (d > 0) p *= 0.5f;
            else if (d < 0) p *= 1.5f;
        }

        // If currently tumbling, keep tumbling
        if (mydata->tumbling) {
            int16_t turn = (int16_t)((float)motorFull * tumble_turn_ratio);
            int16_t l = clamp_i16(base - mydata->tumble_sign * turn, motorStop, motorFull);
            int16_t r = clamp_i16(base + mydata->tumble_sign * turn, motorStop, motorFull);

            pogobot_motor_set(motorL, l);
            pogobot_motor_set(motorR, r);

            if (mydata->tumble_ticks_left > 0) mydata->tumble_ticks_left--;
            if (mydata->tumble_ticks_left == 0) mydata->tumbling = 0;

            pogobot_led_setColor(0, 0, 255); // blue = searching/tumbling
        } else {
            // Not tumbling: RUN forward, maybe start a tumble with probability p
            // (Use libc rand() seeded in user_init via pogobot_helper_getRandSeed())
            float u = (float)(rand() & 0xFFFF) / 65535.0f;
            if (u < p) {
                mydata->tumbling = 1;
                mydata->tumble_sign = (rand() & 1) ? 1 : -1;
                uint16_t span = (tumble_max_ticks > tumble_min_ticks)
                                    ? (tumble_max_ticks - tumble_min_ticks)
                                    : 0;
                mydata->tumble_ticks_left = tumble_min_ticks + (span ? (rand() % (span + 1)) : 0);
            }

            // RUN: straight
            pogobot_motor_set(motorL, base);
            pogobot_motor_set(motorR, base);
            pogobot_led_setColor(0, 255, 255); // cyan = searching/run
        }

        mydata->last_light_sum = sum;
        return;
    }

    // --- GRADIENT FOLLOWING MODE --------------------------------------------
    // Reset tumble state once we have enough light
    mydata->tumbling = 0;
    mydata->tumble_ticks_left = 0;

    int16_t l = 0, r = 0;
    phototaxis_step_from_3_sensors(s0, s1, s2, base_speed_ratio, turn_gain_ratio, &l, &r);
    pogobot_motor_set(motorL, l);
    pogobot_motor_set(motorR, r);

    pogobot_led_setColor(0, 255, 0); // green = tracking
    mydata->last_light_sum = sum;
}


#ifdef SIMULATOR
void global_setup(void) {
    init_from_configuration(base_speed_ratio);
    init_from_configuration(turn_gain_ratio);
    init_from_configuration(dark_threshold);
    init_from_configuration(search_turn_ratio);
}
#endif

int main(void) {
    pogobot_init();
#ifndef SIMULATOR
    printf("init ok\n");
#endif

    pogobot_start(user_init, user_step);

    // Simulator-only callback(s)
    SET_CALLBACK(callback_global_setup, global_setup);

    return 0;
}
