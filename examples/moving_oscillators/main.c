
#include "pogobase.h"
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

/* ================================================================================================
 * CONSTANTS & TYPEDEFS
 * ============================================================================================== */

/* ‑‑‑  Motion ‑‑‑ */
static float max_speed_frac   = 0.4f;       /* 0–1: fraction of motorFull while RUNNING           */
static uint32_t run_duration_min  =  800;   /* [ms] */
static uint32_t run_duration_max  = 4000;   /* [ms] */
static uint32_t tumble_duration_min = 200;  /* [ms] */
static uint32_t tumble_duration_max = 600;  /* [ms] */

#define MS_TO_S(ms)  ((float)(ms) * 0.001f)

/* ‑‑‑  Kuramoto oscillator ‑‑‑ */
static float natural_freq_mean = 1.0f;      /* [rad s⁻¹]  – average ω₀                                   */
static float natural_freq_std  = 0.1f;      /* [rad s⁻¹]  – per‑robot Gaussian jitter                   */
static float coupling_k        = 1.0f;      /* [rad s⁻¹]  – coupling gain                               */

// θ - locomotion weak-coupling
static bool enable_locomotion_coupling = true;     // Whether to couple the oscillator theta with the run-and-tumble locomotion.
                                                   //  If false, disable this coupling: locomotion if fully independent from the oscillator.
                                                   //  If True, create a weak coupling between the two (cf gains below).
static float osc_to_move_gain = 0.25f;   /* θ → run/tumble timing (dimensionless)      */
static float move_to_osc_gain = 0.25f;   /* run/tumble → θ  (rad s⁻¹)                  */

typedef enum {
    LED_MODE_SIGN,           // Red / blue, sign of phase
    LED_MODE_RAINBOW_PHASE,  // Rainbow θ (phase)
    LED_MODE_PERIOD_COLOR    // Rainbow period
} led_mode_t;
static led_mode_t led_mode_enum = LED_MODE_SIGN;

/* ‑‑‑  Messaging ‑‑‑ */
#define BROADCAST_HZ        30U
#define broadcast_period_ms (1000U / BROADCAST_HZ)         /* 33 ms */
#define MAX_NEIGHBORS       20U
#define IR_RANGE_MAX_AGE    200U    /* [ms] keep neighbour for coupling this long               */

/* ================================================================================================
 * ENUM & STRUCTS
 * ============================================================================================== */

typedef enum { STATE_RUN = 0, STATE_TUMBLE = 1 } robot_state_t;

typedef struct {
    uint16_t id;                /* neighbour ID           */
    float    phase;             /* neighbour θ [rad]      */
    uint32_t last_seen_ms;      /* timestamp              */
} neighbor_t;

/* ==== USERDATA ================================================================================= */

typedef struct {
    /* neighbour list */
    neighbor_t neighbors[MAX_NEIGHBORS];
    uint8_t    nb_neighbors;

    /* oscillator */
    float theta;                /* own phase θ [rad]      */
    float natural_freq;         /* ω₀  [rad s⁻¹]          */

    /* mobility FSM */
    robot_state_t fsm;
    uint32_t   timer_ms;
    bool       tumble_left;

    /* misc */
    uint32_t last_step_ms;
    uint32_t last_broadcast_ms;

    /* cached motor polarity (EEPROM) */
    uint8_t motor_dir_left;
    uint8_t motor_dir_right;

    // Period computation
    uint32_t  last_cross_time;   // [s]
    float     period_est;        // [s]
} USERDATA;

DECLARE_USERDATA(USERDATA);
REGISTER_USERDATA(USERDATA);

/* ================================================================================================
 * UTILS
 * ============================================================================================== */

static inline float rand_unitf(void) {
    return (float)rand() / (float)(RAND_MAX + 1UL);
}

static inline float random_uniform(float a, float b) {
    return a + (b - a) * rand_unitf();
}

static inline float clampf(float v, float lo, float hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

static inline void wrap_phase(float *theta) {
    /* wrap to (‑π, π] for numerical stability */
    if (*theta > (float)M_PI)  { *theta -= 2.0f * (float)M_PI; }
    if (*theta <= (-(float)M_PI)) { *theta += 2.0f * (float)M_PI; }
}

static inline uint8_t period_to_color(float T) {
    //const float T_min = 4.0f, T_max = 12.0f;
    const float T_min = 0.0f, T_max = 8000.0f;
    float clamped = fmaxf(T_min, fminf(T, T_max));
    return (uint8_t)(255.0f * (clamped - T_min) / (T_max - T_min));
}

/* ================================================================================================
 * MOTION PRIMITIVES
 * ============================================================================================== */

static inline void motors_forward(float vfrac) {
    pogobot_motor_set(motorL, motorFull * vfrac);
    pogobot_motor_set(motorR, motorFull * vfrac);
}

static inline void motors_stop(void) {
    pogobot_motor_set(motorL, motorStop);
    pogobot_motor_set(motorR, motorStop);
}

static void choose_new_heading(void) {
    /* simple 50‑50 chance to flip both polarities → random walk */
    bool flip = (rand() & 1U);
    pogobot_motor_dir_set(motorL, flip ? !mydata->motor_dir_left  : mydata->motor_dir_left);
    pogobot_motor_dir_set(motorR, flip ? !mydata->motor_dir_right : mydata->motor_dir_right);
}

static void tumble_pivot(bool left) {
    if (left) {
        pogobot_motor_set(motorL, motorStop);
        pogobot_motor_set(motorR, motorFull * max_speed_frac);
    } else {
        pogobot_motor_set(motorL, motorFull * max_speed_frac);
        pogobot_motor_set(motorR, motorStop);
    }
}

/* ================================================================================================
 * IR MESSAGING  (phase sharing)
 * ============================================================================================== */

typedef struct __attribute__((__packed__)) {
    uint16_t sender_id;
    int16_t  phase_q15;        /* q15 encoding of θ  ∈ (‑π, π]  → (‑32768, 32767] */
} phase_msg_t;

#define PHASE_MSG_SIZE  ((uint16_t)sizeof(phase_msg_t))

static int16_t float_to_q15(float theta) {
    /* map (‑π, π] → (‑32768, 32767] */
    wrap_phase(&theta);
    return (int16_t)lroundf(theta / (float)M_PI * 32767.0f);
}

static float q15_to_float(int16_t q) {
    return ((float)q) * (float)M_PI / 32767.0f;
}

static bool send_phase_msg(void) {
    uint32_t now = current_time_milliseconds();
    if (now - mydata->last_broadcast_ms < broadcast_period_ms) { return false; }

    phase_msg_t m = { .sender_id = pogobot_helper_getid(), .phase_q15 = float_to_q15(mydata->theta) };
    pogobot_infrared_sendShortMessage_omni((uint8_t *)&m, PHASE_MSG_SIZE);
    mydata->last_broadcast_ms = now;
    return true;
}

static void process_message(message_t *msg) {
    if (msg->header.payload_length < PHASE_MSG_SIZE) { return; }

    phase_msg_t const *pm = (phase_msg_t const *)msg->payload;
    uint16_t sender = pm->sender_id;
    if (sender == pogobot_helper_getid()) { return; }

    /* search / insert neighbour */
    uint8_t idx;
    for (idx = 0; idx < mydata->nb_neighbors; ++idx) {
        if (mydata->neighbors[idx].id == sender) { break; }
    }
    if (idx == mydata->nb_neighbors) {
        if (mydata->nb_neighbors >= MAX_NEIGHBORS) { return; }
        mydata->nb_neighbors++;
    }

    mydata->neighbors[idx].id           = sender;
    mydata->neighbors[idx].phase        = q15_to_float(pm->phase_q15);
    mydata->neighbors[idx].last_seen_ms = current_time_milliseconds();
}

static void purge_old_neighbors(uint32_t now) {
    for (int8_t i = (int8_t)mydata->nb_neighbors - 1; i >= 0; --i) {
        if (now - mydata->neighbors[i].last_seen_ms > IR_RANGE_MAX_AGE) {
            mydata->neighbors[i] = mydata->neighbors[mydata->nb_neighbors - 1];
            mydata->nb_neighbors--;
        }
    }
}

/* ================================================================================================
 * LED RENDERING  (θ visualisation)
 * ============================================================================================== */

static void update_leds(void) {
    uint8_t r=0,g=0,b=0;
    if (led_mode_enum == LED_MODE_SIGN) {
        /* sign( sin θ ) → red vs blue */
        bool positive = (sinf(mydata->theta) >= 0.0f);
        if (positive) { r = 255; g = 0;   b = 0; } else { r = 0; g = 0; b = 255; }
    } else if (led_mode_enum == LED_MODE_RAINBOW_PHASE) {
        float norm = fmodf(mydata->theta + 2.0f * (float)M_PI, 2.0f * (float)M_PI) / (2.0f * (float)M_PI);
        uint8_t val = (uint8_t)lroundf(norm * 255.0f);
        rainbow_colormap(val, &r, &g, &b);
    } else {        // LED_MODE_PERIOD_COLOR
        uint8_t idx = period_to_color(mydata->period_est);
        rainbow_colormap(idx, &r, &g, &b);
        //printf("period: %f\n", mydata->period_est);
    }
    pogobot_led_setColors(r, g, b, 0);      /* centre LED */

    /* Ring LED 1 shows motion state */
    switch (mydata->fsm) {
        case STATE_RUN:    r = 0; g = 255; b = 0; break;   /* green */
        case STATE_TUMBLE: r = 255; g = 160; b = 0; break; /* orange */
    }
    pogobot_led_setColors(r, g, b, 1);
}

/* ================================================================================================
 * OSCILLATOR UPDATE
 * ============================================================================================== */

static void update_phase(float dt) {
    float prev_phase = mydata->theta;

    // Update phase
    float coupling_sum = 0.0f;
    uint8_t n = mydata->nb_neighbors;
    for (uint8_t i = 0; i < n; ++i) {
        coupling_sum += sinf(mydata->neighbors[i].phase - mydata->theta);
    }
    float coupling_term = (n > 0) ? (coupling_k / (float)n) * coupling_sum : 0.0f;
    mydata->theta += (mydata->natural_freq + coupling_term) * dt;

    if (enable_locomotion_coupling) {
        /* locomotion → oscillator: RUN = +1, TUMBLE = −1 (square-wave drive) */
        float loco_signal = (mydata->fsm == STATE_RUN) ? 1.0f : -1.0f;
        mydata->theta += move_to_osc_gain * loco_signal * dt;
    }

    wrap_phase(&mydata->theta);

    // Compute the period
    bool zero_crossing = (prev_phase < 0.0f && mydata->theta >= 0.0f);
    if (zero_crossing) {
        uint32_t now = current_time_milliseconds();
        //data->period_est = now - data->last_cross_time;
        mydata->period_est = 0.8f * mydata->period_est + 0.2f * (now - mydata->last_cross_time);
        mydata->last_cross_time = now;
    }
}

/* ================================================================================================
 * RUN/TUMBLE SCHEDULING
 * ============================================================================================== */

static uint32_t draw_run_time_ms(void) {
    return (uint32_t)lroundf(random_uniform(run_duration_min, run_duration_max));
}

static uint32_t draw_tumble_time_ms(void) {
    return (uint32_t)lroundf(random_uniform(tumble_duration_min, tumble_duration_max));
}

/* ================================================================================================
 * INITIALISATION
 * ============================================================================================== */

static void user_init(void) {
    srand(pogobot_helper_getRandSeed());
    pogobot_infrared_set_power(2);

    memset(mydata, 0, sizeof(*mydata));

    /* motor polarity from EEPROM */
    uint8_t dir_mem[3];
    pogobot_motor_dir_mem_get(dir_mem);
    mydata->motor_dir_left  = dir_mem[1];
    mydata->motor_dir_right = dir_mem[0];

    /* oscillator params */
    float jitter = natural_freq_std * (rand_unitf() * 2.0f - 1.0f);
    mydata->natural_freq = natural_freq_mean + jitter;
    mydata->theta = random_uniform(-(float)M_PI, (float)M_PI);

    /* FSM init */
    mydata->fsm = STATE_TUMBLE;
    mydata->timer_ms = 100;             /* tiny initial tumble */
    mydata->tumble_left = true;

    /* simulator / messaging */
    main_loop_hz                  = BROADCAST_HZ;  /* 30 Hz loop */
    msg_rx_fn                     = process_message;
    msg_tx_fn                     = send_phase_msg;
    max_nb_processed_msg_per_tick = 4;
    percent_msgs_sent_per_ticks   = 50;
    error_codes_led_idx           = 3;

    mydata->last_step_ms      = current_time_milliseconds();
    mydata->last_broadcast_ms = current_time_milliseconds();
}

/* ================================================================================================
 * MAIN CONTROL LOOP
 * ============================================================================================== */

static void user_step(void) {
    uint32_t now = current_time_milliseconds();
    uint32_t elapsed_ms = now - mydata->last_step_ms;
    mydata->last_step_ms = now;

    /* oscillator update */
    update_phase(MS_TO_S(elapsed_ms));

    /* neighbour maintenance */
    purge_old_neighbors(now);

    /* run/tumble FSM */
    if (mydata->timer_ms > elapsed_ms) { mydata->timer_ms -= elapsed_ms; } else { mydata->timer_ms = 0; }

    switch (mydata->fsm) {
        case STATE_RUN:
            motors_forward(max_speed_frac);
            if (mydata->timer_ms == 0) {
                mydata->tumble_left = (rand() & 1U);
                if (enable_locomotion_coupling) {
                    /* θ → locomotion: spend a bit longer tumbling around θ ≈ +π/2,
                       shorter around θ ≈ −π/2 (weak sine modulation, clamped) */
                    float factor = 1.0f + osc_to_move_gain * sinf(mydata->theta);
                    factor = clampf(factor, 0.5f, 1.5f);
                    mydata->timer_ms = (uint32_t)lroundf(draw_tumble_time_ms() * factor);
                } else {
                    mydata->timer_ms = draw_tumble_time_ms();
                }
                mydata->fsm = STATE_TUMBLE;
            }
            break;

        case STATE_TUMBLE:
        default:
            tumble_pivot(mydata->tumble_left);
            if (mydata->timer_ms == 0) {
                choose_new_heading();
                if (enable_locomotion_coupling) {
                    /* θ → locomotion: run longer when sin θ is negative,
                       shorter when positive (opposite sign to tumble) */
                    float factor = 1.0f - osc_to_move_gain * sinf(mydata->theta);
                    factor = clampf(factor, 0.5f, 1.5f);
                    mydata->timer_ms = (uint32_t)lroundf(draw_run_time_ms() * factor);
                } else {
                    mydata->timer_ms = draw_run_time_ms();
                }
                mydata->fsm = STATE_RUN;
            }
            break;
    }

    /* LED feedback & beacon */
    update_leds();
    send_phase_msg();
}

/* ================================================================================================
 * SIMULATOR CALLBACKS  (YAML config & data export)
 * ============================================================================================== */
#ifdef SIMULATOR
#include <strings.h>

static void global_setup(void) {
    init_from_configuration(max_speed_frac);
    init_from_configuration(run_duration_min);
    init_from_configuration(run_duration_max);
    init_from_configuration(tumble_duration_min);
    init_from_configuration(tumble_duration_max);

    init_from_configuration(natural_freq_mean);
    init_from_configuration(natural_freq_std);
    init_from_configuration(coupling_k);

    init_from_configuration(max_speed_frac);

    char led_mode[128] = "sign"; // Initialized with default value
    init_array_from_configuration(led_mode);
    if (strcasecmp(led_mode, "sign") == 0) {
        led_mode_enum = LED_MODE_SIGN;
    } else if (strcasecmp(led_mode, "phase") == 0) {
        led_mode_enum = LED_MODE_RAINBOW_PHASE;
    } else if (strcasecmp(led_mode, "period") == 0) {
        led_mode_enum = LED_MODE_PERIOD_COLOR;
    } else {
        printf("ERROR: unknown led_mode parameter value: '%s', use either 'sign', 'phase' or 'period'.\n", led_mode);
        exit(1);
    }

    init_from_configuration(enable_locomotion_coupling);
    init_from_configuration(osc_to_move_gain);
    init_from_configuration(move_to_osc_gain);
}

static void create_data_schema(void) {
    data_add_column_double("theta");
    data_add_column_int8("nb_neighbors");
}

static void export_data(void) {
    data_set_value_double("theta", mydata->theta);
    data_set_value_int8("nb_neighbors", mydata->nb_neighbors);
}
#endif /* SIMULATOR */

/* ================================================================================================
 * ENTRY POINT
 * ============================================================================================== */

int main(void) {
    pogobot_init();

    pogobot_start(user_init, user_step);
#ifdef SIMULATOR
    SET_CALLBACK(callback_global_setup,       global_setup);
    SET_CALLBACK(callback_create_data_schema, create_data_schema);
    SET_CALLBACK(callback_export_data,        export_data);
#endif
    return 0;
}

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
