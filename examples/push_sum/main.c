/**
 * @file push_sum.c
 * @brief Baseline push-sum gossip controller for PogoBot (in C11).
 *
 * Every robot starts with an initial scalar value @c s (here: its UID cast to
 * float) and weight @c w ≔ 1. At a fixed rate it broadcasts **half** of both
 * (s/2,w/2) omnidirectionally and keeps the other half. Upon reception the
 * pairs are accumulated locally. The running estimate of the global average is
 * @c estimate = s/w and converges exponentially fast to the exact mean for a
 * connected swarm.
 *
 * Telemetry exported by the simulator:
 *   • current s, w and estimate for every robot
 *   • current push-sum round counter (monotonically increasing)
 *
 * LEDs: colour-codes the estimate (mod 10) with the qualitative palette so the
 * convergence is observable in real time.
 */

#include "pogobase.h"
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

/* -------------------------------------------------------------------------- */
/* Global constants (configurable from YAML via global_setup)                 */
/* -------------------------------------------------------------------------- */

/// Broadcast frequency (Hz).
uint32_t gossip_hz = 10;

/// Set true to enable the run-and-tumble demo motion.
bool moving_robots = false;

/* -------------------------------------------------------------------------- */
/* Compile-time parameters                                                    */
/* -------------------------------------------------------------------------- */
#define GOSSIP_PERIOD_MS (1000U / gossip_hz)

/* -------------------------------------------------------------------------- */
/* Message definition                                                         */
/* -------------------------------------------------------------------------- */
typedef struct __attribute__((__packed__)) {
    float    s;          ///< Half of the sender's current s value.
    float    w;          ///< Half of the sender's current weight.
    uint16_t sender_id;  ///< UID of the sender (for stats / filtering).
} push_sum_msg_t;

#define MSG_SIZE ((uint16_t)sizeof(push_sum_msg_t))

/* -------------------------------------------------------------------------- */
/* USERDATA                                                                   */
/* -------------------------------------------------------------------------- */
typedef struct {
    float    s;              ///< Local scalar value.
    float    w;              ///< Local weight.
    float    estimate;       ///< Current estimate s / w.
    uint32_t last_send_ms;   ///< Timestamp of last broadcast.
    uint32_t round;          ///< Push-sum round counter.
} USERDATA;

DECLARE_USERDATA(USERDATA);
REGISTER_USERDATA(USERDATA);

/* -------------------------------------------------------------------------- */
/* Utility helpers                                                            */
/* -------------------------------------------------------------------------- */

/**
 * @brief Map estimate (mod 10) → LED colour.
 */
static void set_led_from_estimate(float estimate) {
    uint8_t r, g, b;
    uint8_t idx = (uint8_t)fmodf(fabsf(estimate), 10.0f);
    qualitative_colormap(idx, &r, &g, &b);
    pogobot_led_setColors(r, g, b, 0); // LED 0 shows global estimate
}

/**
 * @brief Simple periodic tumbling motion (demo only).
 */
static void tumbling_motion(void) {
    if ((current_time_milliseconds() / 10000U) % 2U == 0U) {
        pogobot_motor_set(motorL, motorFull);
        pogobot_motor_set(motorR, motorStop);
    } else {
        pogobot_motor_set(motorL, motorStop);
        pogobot_motor_set(motorR, motorFull);
    }
}

/* -------------------------------------------------------------------------- */
/* Messaging callbacks                                                        */
/* -------------------------------------------------------------------------- */

/**
 * @brief Send half of the local (s,w) pair.
 */
bool send_message(void) {
    uint32_t now = current_time_milliseconds();
    if (now - mydata->last_send_ms < GOSSIP_PERIOD_MS) {
        return false;
    }

    // Split mass in half for push-sum
    float send_s = mydata->s * 0.5f;
    float send_w = mydata->w * 0.5f;
    mydata->s   -= send_s;
    mydata->w   -= send_w;

    push_sum_msg_t msg = {
        .s = send_s,
        .w = send_w,
        .sender_id = pogobot_helper_getid()
    };
    pogobot_infrared_sendShortMessage_omni((uint8_t *)&msg, MSG_SIZE);

    mydata->last_send_ms = now;
    mydata->round++;
    return true;
}

/**
 * @brief Add received (s,w) to local state.
 */
void process_message(message_t *mr) {
    if (mr->header.payload_length < MSG_SIZE) {
        return;
    }
    push_sum_msg_t const *msg = (push_sum_msg_t const *)mr->payload;
    if (msg->sender_id == pogobot_helper_getid()) {
        return; // ignore echo
    }

    mydata->s += msg->s;
    mydata->w += msg->w;
}

/* -------------------------------------------------------------------------- */
/* Controller logic                                                           */
/* -------------------------------------------------------------------------- */

/**
 * @brief Robot initialisation.
 */
void user_init(void) {
    srand(pogobot_helper_getRandSeed());

    // Initial values
    mydata->s = (float)pogobot_helper_getid(); // could be any sensor reading
    mydata->w = 1.0f;
    mydata->estimate = mydata->s; // w==1
    mydata->last_send_ms = 0;
    mydata->round = 0;

    // Radio power & scheduler
    pogobot_infrared_set_power(2);
    main_loop_hz                  = 60;
    max_nb_processed_msg_per_tick = 4;
    percent_msgs_sent_per_ticks   = 50;
    msg_rx_fn = process_message;
    msg_tx_fn = send_message;

    // LED to show estimate immediately
    set_led_from_estimate(mydata->estimate);
}

/**
 * @brief Main control loop.
 */
void user_step(void) {
    // Update running estimate
    if (mydata->w > 1e-6f) { // prevent division by zero
        mydata->estimate = mydata->s / mydata->w;
    }

    set_led_from_estimate(mydata->estimate);

    if (moving_robots) {
        tumbling_motion();
    }

    // Debug print every 1000 ticks for robot 0
    if (pogobot_ticks % 1000U == 0U && pogobot_helper_getid() == 0U) {
        printf("[Round %lu] est = %.4f (s=%.4f, w=%.4f)\n",
               (unsigned long)mydata->round,
               mydata->estimate, mydata->s, mydata->w);
    }
}

/* -------------------------------------------------------------------------- */
/* Simulator hooks                                                            */
/* -------------------------------------------------------------------------- */
#ifdef SIMULATOR

void global_setup(void) {
    init_from_configuration(gossip_hz);
    init_from_configuration(moving_robots);
}

static void create_data_schema(void) {
    data_add_column_double("s_value");
    data_add_column_double("w_value");
    data_add_column_double("estimate");
    data_add_column_int32("round");
}

static void export_data(void) {
    data_set_value_double("s_value",     mydata->s);
    data_set_value_double("w_value",     mydata->w);
    data_set_value_double("estimate",    mydata->estimate);
    data_set_value_int32("round",        mydata->round);
}

#endif /* SIMULATOR */

/* -------------------------------------------------------------------------- */
/* Entry point                                                                */
/* -------------------------------------------------------------------------- */
int main(void) {
    pogobot_init();
    pogobot_start(user_init, user_step);

    SET_CALLBACK(callback_global_setup,       global_setup);
    SET_CALLBACK(callback_create_data_schema, create_data_schema);
    SET_CALLBACK(callback_export_data,        export_data);

    return 0;
}

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
