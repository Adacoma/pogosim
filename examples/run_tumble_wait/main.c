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

#define PBI_BASED_CONTROL

// "Global" variables set by the YAML configuration file (in simulation) by the function global_setup, or with a fixed values (in experiments). These values should be seen as constants shared by all robots.

#ifdef PBI_BASED_CONTROL
// pbi = (µ / σ²) / (µ² / n_max)
float pbi_star          = 0.0f;         // Spatial structure control. From clustered (<0), Poisson/random (0), Binomial (1) to sub-binomial/lattice (>1)
#else
// ρ  ≡  μ  / n_max     (mean fill fraction in [0…1])
float rho_star          = 0.30f;        // 0 … 1  : wall/lattice ⇢ gas
// χ  ≡  σ²/μ – 1       (index of dispersion)
float chi_star          = 0.00f;        // >0   : lattice,  0 : Poisson, <0 : cluster
#endif

// ε  ≡  1 – K_act      (fraction of recent time _not_ moving)
float eps_star          = 0.00f;        // 0…1  : ergodic ⇢ frozen

// Movement
bool enable_backward_dir = true;

// Gains
#ifdef PBI_BASED_CONTROL
float k_pbi             = 0.02;        // s⁻¹  – variance loop gain (γ) - PBI
#else
float k_rho             = 0.10f;        // s⁻¹  – density  loop gain (p_wait)
float k_chi             = 0.02f;        // s⁻¹  – variance loop gain (γ)
#endif
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
uint32_t max_age_ms     = 1500U;        // ms  - max age of an heartbeat message
uint8_t heartbeat_hz    = 10U;
#define heartbeat_period_ms     (1000U / heartbeat_hz)
uint8_t main_loop_freq_hz    = 60U;
uint32_t bootstrap_time_ms = 4000U;


#define DBG_PRINTF(...)  printf(__VA_ARGS__)

#define TURN_LEFT  1
#define TURN_RIGHT 0

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
#ifdef PBI_BASED_CONTROL
    float      pbi_curr;
#endif
    float      rho_curr;
    float      chi_curr;
    float      eps_curr;

    /* ─ Control variables ─ */
    float      gamma_i;     /* current run rate (s⁻¹) */
    float      p_wait;       /* probability to enter WAIT next tick */

    /* ─ Timers ─ */
    uint32_t   last_step_ms;
    uint32_t   last_tumble_decision_ms;

    /* ─ Motor dir persisted in EEPROM ─ */
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

#ifdef PBI_BASED_CONTROL
float calculate_pbi(void) {
    float mu = mydata->mu_ewma;
    float var = mydata->m2_ewma - mu * mu;
    if (var < 0.0f) var = 0.0f;
    
//    float denom = (mu * mu) / n_max;
////    if (denom < 1e-6f) return 0.0f;  // avoid division by zero
////    
////    return (mu - var) / denom;
//    if (denom < 0.1f) return 0.0f;  // Increased threshold
//    
//    // Add stability margin
//    return clip(-2.0f, (mu - var) / denom, 2.0f);

    // XXX
    return var / mu;
}
#endif

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
    //return 100U + (uint32_t)(rand() % 300U);
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
    if (enable_backward_dir && backward) {
        pogobot_motor_dir_set(motorL, (mydata->motor_dir_left  == 0 ? 1 : 0));
        pogobot_motor_dir_set(motorR, (mydata->motor_dir_right == 0 ? 1 : 0));
    } else {
        pogobot_motor_dir_set(motorL, mydata->motor_dir_left);
        pogobot_motor_dir_set(motorR, mydata->motor_dir_right);
    }
}


/**
 * Choose tumble direction based on chi, or on spatial PBI control
 */
uint8_t choose_tumble_direction(void) {
    // For clustering (χ > 0): turn toward sparse areas
    // For dispersal (χ < 0): turn toward dense areas
    
    float left_density = mydata->dir_counts[ir_left];
    float right_density = mydata->dir_counts[ir_right];
    
    // Add some randomness to prevent deterministic patterns
    float random_bias = (rand_unitf() - 0.5f) * 0.2f;  // ±10% random component

#ifdef PBI_BASED_CONTROL
    float pbi_curr = calculate_pbi();
    float pbi_error = pbi_star - pbi_curr;  // Assuming pbi_star is defined globally

    if (pbi_error > 0.1f) {  
        // Need HIGHER PBI (more regular/uniform spacing)
        // Turn AWAY FROM denser areas to promote uniformity
        float bias = (right_density - left_density) + random_bias;
        return (bias < 0) ? TURN_LEFT : TURN_RIGHT;
        
    } else if (pbi_error < -0.1f) {  
        // Need LOWER PBI (more clustering/heterogeneity)
        // Turn TOWARD denser areas to promote clustering
        float bias = (left_density - right_density) + random_bias;
        return (bias < 0) ? TURN_LEFT : TURN_RIGHT;
        
    } else {
        // Near target - use random tumbling
        return (rand() & 1U) ? TURN_LEFT : TURN_RIGHT;
    }
#else
    float chi_error = chi_star - mydata->chi_curr;

    if (chi_error > 0.1f) {  
        // Need MORE clustering (higher χ)
        // Turn toward SPARSER areas to allow cluster formation
        float bias = (left_density - right_density) + random_bias;
        return (bias < 0) ? TURN_LEFT : TURN_RIGHT;
        
    } else if (chi_error < -0.1f) {  
        // Need LESS clustering (lower χ) 
        // Turn toward DENSER areas to break up clusters
        float bias = (right_density - left_density) + random_bias;
        return (bias < 0) ? TURN_LEFT : TURN_RIGHT;
        
    } else {
        // Near target - use random tumbling
        return (rand() & 1U) ? TURN_LEFT : TURN_RIGHT;
    }
#endif
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
            //if (rand() & 1U) { mydata->tumble_duration_ms |= 0x80000000U; }

            // Density-based tumbling
            uint8_t turn_direction = choose_tumble_direction();  // based on neighbor density
            if (turn_direction == TURN_LEFT) { 
                mydata->tumble_duration_ms |= 0x80000000U; 
            }

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
#ifdef PBI_BASED_CONTROL
    mydata->pbi_curr  = 0.0f;
#endif
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
    purge_old_neighbors(now, max_age_ms);
    recalc_dir_counts();

    /* ─–––– MOTION STATE MACHINE ─–––– */
    if (mydata->state == STATE_WAIT) {
        /* decrement WAIT timer */
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

        /* motors OFF while waiting */
        pogobot_motor_set(motorL, motorStop);
        pogobot_motor_set(motorR, motorStop);
    } else {                                     /* RUN / TUMBLE */
        run_and_tumble_step();
    }

    /*  moved?  (RUN counts as 1, WAIT & TUMBLE count as 0) */
    float moved = (mydata->state == STATE_RUN) ? 1.0f : 0.0f;

    /* ─–––– UPDATE METRICS (μ, σ², K_act) ─–––– */
    float D          = (float)mydata->total_neighbors;
    mydata->mu_ewma += alpha_stats * (D     - mydata->mu_ewma);
    mydata->m2_ewma += alpha_stats * (D*D   - mydata->m2_ewma);
    mydata->k_act   += alpha_act   * (moved - mydata->k_act);

    /* compute derived scalars */
    float sigma2 = mydata->m2_ewma - mydata->mu_ewma * mydata->mu_ewma;
    if (sigma2 < 0.0f) { sigma2 = 0.0f; }                       /* numeric */

    mydata->rho_curr = mydata->mu_ewma / n_max;                 /* ρ */
    if (mydata->mu_ewma > 1e-4f) {
        mydata->chi_curr = sigma2 / mydata->mu_ewma - 1.0f;     /* χ */
    } else {
        mydata->chi_curr = 0.0f;
    }
    mydata->eps_curr = 1.0f - mydata->k_act;                    /* ε */

#ifdef PBI_BASED_CONTROL
    /* ─–––– VARIANCE LOOP  (γ) ─–––– */
    mydata->pbi_curr = calculate_pbi();
    float err_pbi = pbi_star - mydata->pbi_curr;
    mydata->gamma_i += k_pbi * err_pbi;
    mydata->gamma_i  = clip(gamma_min, mydata->gamma_i, gamma_max);

    /* ─–––– DENSITY & ERGODICITY LOOP  (p_wait) ─–––– */
    float err_eps  = eps_star - mydata->eps_curr;     /* >0 ⇒ too frozen */
    mydata->p_wait += k_eps * err_eps;
    mydata->p_wait  = clip(0.0f, mydata->p_wait, 1.0f);
    //mydata->p_wait  = clip(0.0f, mydata->p_wait, 0.9f);

#else
    /* ─–––– VARIANCE LOOP  (γ) ─–––– */
    //float err_chi  = chi_star - mydata->chi_curr;     /* >0 ⇒ need less var */
    float err_chi  = mydata->chi_curr - chi_star;     /* <0 ⇒ need less var */
    mydata->gamma_i += k_chi * err_chi;
    mydata->gamma_i  = clip(gamma_min, mydata->gamma_i, gamma_max);

    /* ─–––– DENSITY & ERGODICITY LOOP  (p_wait) ─–––– */
    float err_rho  = mydata->rho_curr - rho_star;     /* >0 ⇒ too dense */
    float err_eps  = eps_star - mydata->eps_curr;     /* >0 ⇒ too frozen */
    mydata->p_wait += k_rho * err_rho + k_eps * err_eps;
    mydata->p_wait  = clip(0.0f, mydata->p_wait, 1.0f);
    //mydata->p_wait  = clip(0.0f, mydata->p_wait, 0.9f);
#endif

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

#ifdef PBI_BASED_CONTROL
        DBG_PRINTF("[R%u] RUN → WAIT    (pbi=%.2f ρ=%.2f χ=%.2f ε=%.2f, p=%.2f)\n",
                pogobot_helper_getid(),
                mydata->pbi_curr,
                mydata->rho_curr, mydata->chi_curr, mydata->eps_curr,
                mydata->p_wait);
#else
        DBG_PRINTF("[R%u] RUN → WAIT    (ρ=%.2f χ=%.2f ε=%.2f, p=%.2f)\n",
                pogobot_helper_getid(),
                mydata->rho_curr, mydata->chi_curr, mydata->eps_curr,
                mydata->p_wait);
#endif
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

    if (pogobot_helper_getid() == 0) {     // Only print messages for robot 0
#ifdef PBI_BASED_CONTROL
        DBG_PRINTF("[R%u] pbi=%.2f   ρ=%.2f χ=%.2f ε=%.2f  p_wait=%.2f  γ=%.5f  mu_ewma=%.5f m2_ewma=%.5f  tot_neighbors=%u\n",
                   pogobot_helper_getid(),
                   mydata->pbi_curr,
                   mydata->rho_curr, mydata->chi_curr, mydata->eps_curr,
                   mydata->p_wait,   mydata->gamma_i,
                   mydata->mu_ewma, mydata->m2_ewma,
                   mydata->total_neighbors);
#else
        DBG_PRINTF("[R%u] ρ=%.2f χ=%.2f ε=%.2f   err_chi=%.2f   p_wait=%.2f  γ=%.5f  mu_ewma=%.5f m2_ewma=%.5f  tot_neighbors=%u\n",
                   pogobot_helper_getid(),
                   mydata->rho_curr, mydata->chi_curr, mydata->eps_curr,
                   err_chi,
                   mydata->p_wait,   mydata->gamma_i,
                   mydata->mu_ewma, mydata->m2_ewma,
                   mydata->total_neighbors);
#endif
    }
}

/* -------------------------------------------------------------------------- */
/* MAIN FUNCTIONS                                                             */
/* -------------------------------------------------------------------------- */

#ifdef SIMULATOR

/**
 * @brief Function called once to initialize global values (e.g. configuration-specified constants)
 */
void global_setup() {
#ifdef PBI_BASED_CONTROL
    init_from_configuration(pbi_star);
    init_from_configuration(k_pbi);
#else
    init_from_configuration(rho_star);
    init_from_configuration(chi_star);
    init_from_configuration(k_rho);
    init_from_configuration(k_chi);
#endif
    init_from_configuration(eps_star);
    init_from_configuration(enable_backward_dir);
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
    init_from_configuration(t_wait_ms);
    init_from_configuration(max_age_ms);
    init_from_configuration(heartbeat_hz);
    init_from_configuration(main_loop_freq_hz);
    init_from_configuration(bootstrap_time_ms);
}

// Function called once by the simulator to specify user-defined data fields to add to the exported data files
static void create_data_schema(void) {
    data_add_column_int8 ("total_neighbors");
#ifdef PBI_BASED_CONTROL
    data_add_column_double("pbi");
#endif
    data_add_column_double("rho");
    data_add_column_double("chi");
    data_add_column_double("eps");
    data_add_column_double("gamma");
    data_add_column_double("p_wait");
    data_add_column_int8 ("state");
}

// Function called periodically by the simulator each time data is saved (cf config parameter "save_data_period" in seconds)
static void export_data(void) {
    data_set_value_int8 ("total_neighbors", mydata->total_neighbors);
#ifdef PBI_BASED_CONTROL
    data_set_value_double("pbi",            mydata->pbi_curr);
#endif
    data_set_value_double("rho",            mydata->rho_curr);
    data_set_value_double("chi",            mydata->chi_curr);
    data_set_value_double("eps",            mydata->eps_curr);
    data_set_value_double("gamma",         mydata->gamma_i);
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

