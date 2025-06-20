/**
 * @file rtc_neighbor_control.c
 * @brief 3-metric (ρ, χ, ε) swarm controller for PogoBot — C99.
 *
 *          · neighbour count inside sensing disk  (D)
 *          · adaptive run–tumble motion rate       γ
 *          · probabilistic WAIT pauses             p_wait
 *
 *  Tunable set-points:
 *      – rho_star   : mean fill of sensing disk
 *      – chi_star   : index of dispersion   (variance control)
 *      – eps_star   : non-ergodicity target (mobility / freezing)
 *
 *  Copyright 2025  (feel free to reuse under MIT licence)
 */

#include "pogobase.h"
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

// "Global" variables set by the YAML configuration file (in simulation) by the function global_setup, or with a fixed values (in experiments). These values should be seen as constants shared by all robots.

// ρ  ≡  μ  / n_max     (mean fill fraction in [0…1])
float rho_star          = 0.30f;        // 0 … 1  : wall/lattice ⇢ gas
// χ  ≡  σ²/μ – 1       (index of dispersion)
float chi_star          = 3.00f;        // >0   : lattice,  0 : Poisson, <0 : cluster
// ε  ≡  1 – K_act      (fraction of recent time _not_ moving)
float eps_star          = 0.00f;        // 0…1  : ergodic ⇢ frozen

// Gains
float k_rho             = 0.10f;        // s⁻¹  – density  loop gain (p_wait)
float k_chi             = 0.02f;        // s⁻¹  – variance loop gain (γ)
float k_eps             = 0.10f;        // s⁻¹  – ergodicity loop gain (p_wait)

// EWMA windows
float tau_stats	        = 10.0f;        // s  – EWMA window for μ & σ²
float tau_act           = 5.0f;         // s  – EWMA window for activity (K_act)

// Gamma bounds and constants
float gamma_min         = 0.05;         // s⁻¹
float gamma_max         = 1.00f;        // s⁻¹
float gamma_0           = 0.15f;        // Initial value of gamma

// Neighbors constants
#define MAX_NEIGHBORS   20U             // Max number of robots that can be stored in the neighbors array. Must be above n_max
float n_max             = 15.0f;        // The largest number of robots that could ever fit inside a single robot’s sensing. Must be below MAX_NEIGHBORS

// Timing constants
uint32_t t_wait_ms      = 4000U;        // ms  - length of a WAIT pause
uint8_t heartbeat_hz    = 10U;
#define heartbeat_period_ms     (1000U / heartbeat_hz)
uint8_t main_loop_freq_hz    = 60U;
uint32_t bootstrap_time_ms = 4000U;


#define DBG_PRINTF(...)  printf(__VA_ARGS__)

/* -------------------------------------------------------------------------- */
/* ROBOT STATES                                                               */
/* -------------------------------------------------------------------------- */
typedef enum {
    STATE_RUN,
    STATE_TUMBLE,
    STATE_WAIT
} robot_state_t;

/* -------------------------------------------------------------------------- */
/* MESSAGES                                                                   */
/* -------------------------------------------------------------------------- */
typedef struct __attribute__((__packed__)) {
    uint16_t sender_id;
} heartbeat_t;

#define MSG_SIZE ((uint16_t)sizeof(heartbeat_t))

/* -------------------------------------------------------------------------- */
/* NEIGHBORS                                                                  */
/* -------------------------------------------------------------------------- */
typedef struct {
    uint16_t id;
    uint32_t last_seen_ms;
    uint8_t  direction;
} neighbor_t;

/* -------------------------------------------------------------------------- */
/* USER-DEFINED MUTABLE VARIABLES                                             */
/* -------------------------------------------------------------------------- */

// Normal "Global" variables should be inserted within the USERDATA struct.
// /!\  In simulation, don't declare non-const global variables outside this struct, elsewise they will be shared among all agents (and this is not realistic).

typedef struct {
    /* ─ Neighbour tracking ─ */
    neighbor_t neighbors[MAX_NEIGHBORS];
    uint8_t    nb_neighbors;
    uint8_t    dir_counts[IR_RX_COUNT];
    uint8_t    total_neighbors;
    uint32_t   last_heartbeat_ms;

    /* ─ Motion state machine ─ */
    robot_state_t state;
    uint32_t   wait_timer_ms;
    uint32_t   tumble_timer_ms;
    uint32_t   tumble_duration_ms;   /* duration  | MSB = left/right bit */
    uint32_t   run_timer_ms;

    /* ─ Statistical metrics ─ */
    float      mu_ewma;      /* ⟨D⟩ */
    float      m2_ewma;      /* ⟨D²⟩ */
    float      k_act;        /* mobility (1=moving) */
    float      rho_curr;
    float      chi_curr;
    float      eps_curr;

    /* ─ Control variables ─ */
    float      gamma_i;     /* current run rate (s⁻¹) */
    float      p_wait;       /* probability to enter WAIT next tick */

    /* ─ Timers ─ */
    uint32_t   last_step_ms;
    uint32_t   last_tumble_decision_ms;

    /* ─ Misc (motor dir persisted in EEPROM) ─ */
    uint8_t    motor_dir_left;
    uint8_t    motor_dir_right;
} USERDATA;

// Call this macro in the same file (.h or .c) as the declaration of USERDATA
DECLARE_USERDATA(USERDATA);
// Don't forget to call this macro in the main .c file of your project (only once!)
REGISTER_USERDATA(USERDATA);
// Now, members of the USERDATA struct can be accessed through mydata->MEMBER. E.g. mydata->data_foo
//  On real robots, the compiler will automatically optimize the code to access member variables as if they were true globals.

/* -------------------------------------------------------------------------- */
/* UTILITIES                                                                  */
/* -------------------------------------------------------------------------- */
static float clip(float min_val, float val, float max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

static float rand_unitf(void) {          /* uniform in [0,1) */
    return (float)rand() / (float)(RAND_MAX + 1UL);
}

/** Pick an RGB colour for the current motion state */
static void colour_for_state(robot_state_t st, uint8_t *r,
                             uint8_t *g, uint8_t *b) {
    switch (st) {
    default:                   /* fallthrough – treat unknown as RUN */
    case STATE_RUN:    *r = 0;   *g = 255; *b = 0;   break; /* green   */
    case STATE_TUMBLE: *r = 255; *g = 160; *b = 0;   break; /* yellow  */
    case STATE_WAIT:   *r = 255; *g = 0;   *b = 0;   break; /* red     */
    }
}

/* -------------------------------------------------------------------------- */
/* NEIGHBOR BOOK-KEEPING                                                      */
/* -------------------------------------------------------------------------- */
static void purge_old_neighbors(uint32_t now, uint32_t max_age_ms) {
    for (int8_t i = (int8_t)mydata->nb_neighbors - 1; i >= 0; --i) {
        if (now - mydata->neighbors[i].last_seen_ms > max_age_ms) {
            mydata->neighbors[i] = mydata->neighbors[mydata->nb_neighbors - 1];
            mydata->nb_neighbors--;
        }
    }
}

static void recalc_dir_counts(void) {
    memset(mydata->dir_counts, 0, sizeof(mydata->dir_counts));

    for (uint8_t i = 0; i < mydata->nb_neighbors; ++i) {
        uint8_t d = mydata->neighbors[i].direction;
        if (d < IR_RX_COUNT) { mydata->dir_counts[d]++; }
    }

    /* total neighbours */
    mydata->total_neighbors = 0;
    for (uint8_t d = 0; d < IR_RX_COUNT; ++d) {
        mydata->total_neighbors += mydata->dir_counts[d];
    }
}

/* -------------------------------------------------------------------------- */
/* MOTION PRIMITIVES  (RUN–TUMBLE)                                            */
/* -------------------------------------------------------------------------- */
static uint32_t sample_tumble_duration(void) { /* uniform 200–800 ms */
    return 200U + (uint32_t)(rand() % 600U);
}

/* mean run duration = 1/γ  (capped 0.5–20 s) */
static uint32_t sample_run_duration(void) {
    float mean_ms = 1000.0f / mydata->gamma_i;        /* exponential μ */
    if (mean_ms < 500.0f)   mean_ms = 500.0f;
    if (mean_ms > 20000.0f) mean_ms = 20000.0f;

    uint32_t base  = (uint32_t)(0.5f * mean_ms);
    uint32_t range = (uint32_t)mean_ms;
    return base + (uint32_t)(rand() % (range + 1U));
}

/* helper: change forward/backward motor polarity randomly                */
static void set_robot_direction(bool backward) {
    if (backward) {
        pogobot_motor_dir_set(motorL, (mydata->motor_dir_left  == 0 ? 1 : 0));
        pogobot_motor_dir_set(motorR, (mydata->motor_dir_right == 0 ? 1 : 0));
    } else {
        pogobot_motor_dir_set(motorL, mydata->motor_dir_left);
        pogobot_motor_dir_set(motorR, mydata->motor_dir_right);
    }
}

/* predicates */
static bool tumble_due(uint32_t now) {
    return (now - mydata->last_tumble_decision_ms) >= mydata->run_timer_ms;
}
static bool tumble_done(uint32_t now) {
    uint32_t real_dur = mydata->tumble_duration_ms & 0x7FFFFFFFU;
    return (now - mydata->tumble_timer_ms) >= real_dur;
}

/* RUN & TUMBLE state-machine step ----------------------------------------- */
static void run_and_tumble_step(void) {
    uint32_t now = current_time_milliseconds();

    if (mydata->state == STATE_RUN) {
        /* full speed ahead */
        pogobot_motor_set(motorL, motorFull);
        pogobot_motor_set(motorR, motorFull);

        if (tumble_due(now)) {
            mydata->state               = STATE_TUMBLE;
            mydata->tumble_timer_ms     = now;
            mydata->tumble_duration_ms  = sample_tumble_duration();
            set_robot_direction(false);                 /* ensure forward */

            /* random left/right turn stored in MSB */
            if (rand() & 1U) { mydata->tumble_duration_ms |= 0x80000000U; }

            DBG_PRINTF("[R%u] RUN → TUMBLE  (dur=%ums)\n",
                       pogobot_helper_getid(),
                       mydata->tumble_duration_ms & 0x7FFFFFFFU);
        }
    }
    else if (mydata->state == STATE_TUMBLE) {
        /* pivot */
        if (mydata->tumble_duration_ms & 0x80000000U) { /* left */
            pogobot_motor_set(motorL, motorStop);
            pogobot_motor_set(motorR, motorFull);
        } else {                                       /* right */
            pogobot_motor_set(motorL, motorFull);
            pogobot_motor_set(motorR, motorStop);
        }

        if (tumble_done(now)) {
            mydata->state                  = STATE_RUN;
            mydata->last_tumble_decision_ms= now;
            mydata->run_timer_ms           = sample_run_duration();
            set_robot_direction(rand() & 1U);

            DBG_PRINTF("[R%u] TUMBLE → RUN   (γ=%.3f, run=%ums)\n",
                       pogobot_helper_getid(),
                       mydata->gamma_i,
                       mydata->run_timer_ms);
        }
    }
}

/* -------------------------------------------------------------------------- */
/* MESSAGING                                                                  */
/* -------------------------------------------------------------------------- */
bool send_message(void) {
    uint32_t now = current_time_milliseconds();
    if (now - mydata->last_heartbeat_ms < heartbeat_period_ms) { return false; }

    heartbeat_t hb = { .sender_id = pogobot_helper_getid() };
    pogobot_infrared_sendShortMessage_omni((uint8_t *)&hb, MSG_SIZE);
    mydata->last_heartbeat_ms = now;
    return true;
}

void process_message(message_t *mr) {
    if (mr->header.payload_length < MSG_SIZE) { return; }

    uint8_t dir = mr->header._receiver_ir_index;
    if (dir >= IR_RX_COUNT) { return; }

    heartbeat_t const *hb = (heartbeat_t const *)mr->payload;
    uint16_t sender       = hb->sender_id;
    if (sender == pogobot_helper_getid()) { return; }

    /* search for existing neighbour                                     */
    uint8_t idx;
    for (idx = 0; idx < mydata->nb_neighbors; ++idx) {
        if (mydata->neighbors[idx].id == sender) { break; }
    }

    if (idx == mydata->nb_neighbors) {                   /* new neighbour */
        if (mydata->nb_neighbors >= MAX_NEIGHBORS) { return; }
        mydata->nb_neighbors++;
    }

    mydata->neighbors[idx].id           = sender;
    mydata->neighbors[idx].last_seen_ms = current_time_milliseconds();
    mydata->neighbors[idx].direction    = dir;
}

/* -------------------------------------------------------------------------- */
/* INITIALISATION                                                             */
/* -------------------------------------------------------------------------- */
void user_init(void) {
    srand(pogobot_helper_getRandSeed()); // initialize the random number generator
    pogobot_infrared_set_power(2); // set the power level used to send all the next messages

    memset(mydata, 0, sizeof(*mydata));

    /* infra-red setup */
    pogobot_infrared_set_power(2);
    main_loop_hz                  = main_loop_freq_hz;
    max_nb_processed_msg_per_tick = 3;
    percent_msgs_sent_per_ticks   = 50;
    msg_rx_fn                     = process_message;
    msg_tx_fn                     = send_message;
    error_codes_led_idx           = 3;

    /* read motor polarity from EEPROM */
    uint8_t dir_mem[3];
    pogobot_motor_dir_mem_get(dir_mem);
    mydata->motor_dir_left  = dir_mem[1];
    mydata->motor_dir_right = dir_mem[0];

    /* initial state */
    mydata->state     = STATE_RUN;
    mydata->gamma_i  = gamma_0;
    mydata->p_wait    = 0.0f;

    mydata->mu_ewma   = 0.0f;
    mydata->m2_ewma   = 0.0f;
    mydata->k_act     = 1.0f;      /* moving */
    mydata->rho_curr  = 0.0f;
    mydata->chi_curr  = 0.0f;
    mydata->eps_curr  = 0.0f;

    uint32_t now = current_time_milliseconds();
    mydata->last_step_ms             = now;
    mydata->last_tumble_decision_ms  = now;
    mydata->run_timer_ms             = sample_run_duration();
    mydata->wait_timer_ms            = 0;

    set_robot_direction(rand() & 1U);
}

/* -------------------------------------------------------------------------- */
/* MAIN CONTROL LOOP                                                          */
/* -------------------------------------------------------------------------- */
void user_step(void) {
    /* ─–––– timing ─–––– */
    uint32_t now        = current_time_milliseconds();
    uint32_t elapsed_ms = now - mydata->last_step_ms;
    mydata->last_step_ms = now;

    const float dt_s       = elapsed_ms * 0.001f;
    const float alpha_stats= dt_s / tau_stats;
    const float alpha_act  = dt_s / tau_act;

    /* ─–––– neighbour statistics ─–––– */
    purge_old_neighbors(now, 1200U);          /* 1.2 s age cutoff */
    recalc_dir_counts();

    /* ─–––– MOTION STATE MACHINE ─–––– */
    if (mydata->state == STATE_WAIT) {
        /* decrement WAIT timer                                             */
        if (mydata->wait_timer_ms > elapsed_ms) {
            mydata->wait_timer_ms -= elapsed_ms;
        } else {
            mydata->wait_timer_ms = 0;
        }

        if (mydata->wait_timer_ms == 0) {      /* resume RUN */
            mydata->state                  = STATE_RUN;
            mydata->last_tumble_decision_ms= now;
            mydata->run_timer_ms           = sample_run_duration();
            set_robot_direction(rand() & 1U);
            DBG_PRINTF("[R%u] WAIT → RUN     (pause=%.1fs)\n",
                       pogobot_helper_getid(),
                       t_wait_ms / 1000.0f);
        }

        /* motors OFF while waiting                                         */
        pogobot_motor_set(motorL, motorStop);
        pogobot_motor_set(motorR, motorStop);
    }
    else {                                     /* RUN / TUMBLE */
        run_and_tumble_step();
    }

    /*  moved?  (RUN counts as 1, WAIT & TUMBLE count as 0)                 */
    float moved = (mydata->state == STATE_RUN) ? 1.0f : 0.0f;

    /* ─–––– UPDATE METRICS (μ, σ², K_act) ─–––– */
    float D          = (float)mydata->total_neighbors;
    mydata->mu_ewma += alpha_stats * (D     - mydata->mu_ewma);
    mydata->m2_ewma += alpha_stats * (D*D   - mydata->m2_ewma);
    mydata->k_act   += alpha_act   * (moved - mydata->k_act);

    /* compute derived scalars                                              */
    float sigma2 = mydata->m2_ewma - mydata->mu_ewma * mydata->mu_ewma;
    if (sigma2 < 0.0f) { sigma2 = 0.0f; }                       /* numeric */

    mydata->rho_curr = mydata->mu_ewma / n_max;                 /* ρ */
    if (mydata->mu_ewma > 1e-4f) {
        mydata->chi_curr = sigma2 / mydata->mu_ewma - 1.0f;     /* χ */
    } else {
        mydata->chi_curr = 0.0f;
    }
    mydata->eps_curr = 1.0f - mydata->k_act;                    /* ε */

    /* ─–––– VARIANCE LOOP  (γ) ─–––– */
    float err_chi  = chi_star - mydata->chi_curr;     /* >0 ⇒ need less var */
    mydata->gamma_i += k_chi * err_chi;
    mydata->gamma_i  = clip(gamma_min, mydata->gamma_i, gamma_max);

    /* ─–––– DENSITY & ERGODICITY LOOP  (p_wait) ─–––– */
    float err_rho  = mydata->rho_curr - rho_star;     /* >0 ⇒ too dense */
    float err_eps  = eps_star - mydata->eps_curr;     /* >0 ⇒ too frozen */

    mydata->p_wait += k_rho * err_rho + k_eps * err_eps;
    mydata->p_wait  = clip(0.0f, mydata->p_wait, 1.0f);
    //mydata->p_wait  = clip(0.0f, mydata->p_wait, 0.9f);

    /* Disable the WAIT state at the beginning of the simulation, to ensure the robots disperse */
    if (now < bootstrap_time_ms) {          // first K s after power-on
        mydata->p_wait = 0.0f;                     // forbid WAIT
    }

    /* ─–––– DECIDE ON ENTERING WAIT ─–––– */
    if (mydata->state == STATE_RUN && rand_unitf() < mydata->p_wait) {
        float factor = 0.5f + rand_unitf();            // 0.5 ... 1.5
        mydata->wait_timer_ms = (uint32_t)(t_wait_ms * factor);
        mydata->state         = STATE_WAIT;

        /* stop immediately */
        pogobot_motor_set(motorL, motorStop);
        pogobot_motor_set(motorR, motorStop);

        DBG_PRINTF("[R%u] RUN → WAIT    (ρ=%.2f χ=%.2f ε=%.2f, p=%.2f)\n",
                   pogobot_helper_getid(),
                   mydata->rho_curr, mydata->chi_curr, mydata->eps_curr,
                   mydata->p_wait);
    }

    /* ------------------------------------------------------------------ */
    /* LED-0 : neighbour count (as before)                                */
    /* LED-1 : current behaviour  (RUN/TUMBLE/WAIT)                       */
    /* ------------------------------------------------------------------ */
    uint8_t r, g, b;

    /* central LED 0 = neighbour count, dimmed when waiting */
    uint8_t idx = (mydata->total_neighbors > 9U) ? 9U : mydata->total_neighbors;
    qualitative_colormap(idx, &r, &g, &b);
    if (mydata->state == STATE_WAIT) { r/=3U; g/=3U; b/=3U; }
    pogobot_led_setColors(r, g, b, 0);

    /* LED 1 = behaviour */
    colour_for_state(mydata->state, &r, &g, &b);
    pogobot_led_setColors(r, g, b, 1);

//    DBG_PRINTF("[R%u] ρ=%.2f χ=%.2f ε=%.2f  p_wait=%.2f  γ=%.2f\n",
//               pogobot_helper_getid(),
//               mydata->rho_curr, mydata->chi_curr, mydata->eps_curr,
//               mydata->p_wait,   mydata->gamma_i);
}

/* -------------------------------------------------------------------------- */
/* MAIN FUNCTIONS                                                             */
/* -------------------------------------------------------------------------- */

#ifdef SIMULATOR

/**
 * @brief Function called once to initialize global values (e.g. configuration-specified constants)
 */
void global_setup() {
    init_from_configuration(rho_star);
    init_from_configuration(chi_star);
    init_from_configuration(eps_star);
    init_from_configuration(k_rho);
    init_from_configuration(k_chi);
    init_from_configuration(k_eps);
    init_from_configuration(tau_stats);
    init_from_configuration(tau_act);
    init_from_configuration(gamma_min);
    init_from_configuration(gamma_max);
    init_from_configuration(gamma_0);
    init_from_configuration(n_max);
    if (n_max > MAX_NEIGHBORS) {
        printf("ERROR: 'n_max' must be less or equal to MAX_NEIGHBORS.\n");
        assert(n_max > MAX_NEIGHBORS);
    }
    init_from_configuration(heartbeat_hz);
    init_from_configuration(main_loop_freq_hz);
    init_from_configuration(bootstrap_time_ms);
}

// Function called once by the simulator to specify user-defined data fields to add to the exported data files
static void create_data_schema(void) {
    data_add_column_int8 ("total_neighbors");
    data_add_column_double("rho");
    data_add_column_double("chi");
    data_add_column_double("eps");
    data_add_column_double("lambda");
    data_add_column_double("p_wait");
    data_add_column_int8 ("state");
}

// Function called periodically by the simulator each time data is saved (cf config parameter "save_data_period" in seconds)
static void export_data(void) {
    data_set_value_int8 ("total_neighbors", mydata->total_neighbors);
    data_set_value_double("rho",            mydata->rho_curr);
    data_set_value_double("chi",            mydata->chi_curr);
    data_set_value_double("eps",            mydata->eps_curr);
    data_set_value_double("lambda",         mydata->gamma_i);
    data_set_value_double("p_wait",         mydata->p_wait);
    data_set_value_int8 ("state",          (int8_t)mydata->state);
}
#endif /* SIMULATOR */

/**
 * @brief Program entry point.
 *
 * This function initializes the robot system and starts the main execution loop by
 * passing the user initialization and control functions to the platform's startup routine.
 *
 * @return int Returns 0 upon successful completion.
 */
int main(void) {
    // Initialization routine for the robots
    pogobot_init();

    // Start the robot's main loop with the defined user_init and user_step functions.
    pogobot_start(user_init, user_step);

    // Specify the callback functions. Only called by the simulator.
    //  In particular, they serve to add data fields to the exported data files
    SET_CALLBACK(callback_global_setup, global_setup);              // Called once to initialize global values (e.g. configuration-specified constants)
    SET_CALLBACK(callback_create_data_schema, create_data_schema);  // Called once to specify the data format
    SET_CALLBACK(callback_export_data, export_data);                // Called at each configuration-specified period (e.g. every second) on each robot to register exported data
    return 0;
}

