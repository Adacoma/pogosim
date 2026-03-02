// main.c — "Blooming flowers" hop-distance diffusion to a spontaneously appearing seed.

// Main include for pogobots, both for real robots and for simulations
#include "pogobase.h"

#include <string.h>
#include <stdbool.h>
#include <stdint.h>

// -----------------------------------------------------------------------------
// Configuration (can be overridden by simulator via global_setup())
// -----------------------------------------------------------------------------
// "Global" variables set by the YAML configuration file (in simulation) by the function global_setup, or with a fixed values (in experiments). These values should be seen as constants shared by all robots.
bool moving_robots         = true;
bool enable_backward_dir   = false;

// Run&tumble timing (ms)
uint32_t run_duration_min_ms    = 250;
uint32_t run_duration_max_ms    = 1200;
uint32_t tumble_duration_min_ms = 120;
uint32_t tumble_duration_max_ms = 900;
float    max_speed_frac         = 0.20f;

// "Seed creation" probability per tick: p = 1 / den_p_become_seed
bool enable_random_seed    = true;  // If false, the seed is the robot with the lowest ID.
                                    // If true, robots can self-promote themselves as the seed, with probability p
uint32_t den_p_become_seed = 25000; // 105000;   // lower => more frequent blooms

#define BOOT_TIME 1000             // Waiting time before the start of the experience in ms
#define ENABLE_PHOTO_START      // Whether to enable photo start, i.e. wait at the beginning of a experiment for a large instantaneous difference in light level
                                //  --> allow robots to start the experiment all at the same time by just adjusting quickly the light level in the experimental setup.
                                //  Comment this macro to disable photo start
#define LIGHT_THRESHOLD 40      // You can tweak this parameter to change the sensitivity to the light changes

// Hop diffusion / communication
#define INFRARED_POWER 2
#define FQCY 60
#define MAX_NB_OF_MSG 6
#define SEND_OMNI true
#define MSG_FULL_HEADER true
#define IGNORE_SYSTEM_MESSAGES false
uint8_t percent_msg_sent       = 10;    // slow-ish broadcast, helps "wavefront" look

// Visuals
#define MAX_HOPS 250            // clamp for safety
bool enable_blinking_when_updated = true;      // LEDs blink when state updates
uint32_t blink_total_ms           = 200; // total blink duration when state updates
uint32_t blink_half_period_ms     = 50;  // total blink duration when state updates


// -----------------------------------------------------------------------------

/**
 * @brief Enumeration for the robot's behavioral phases.
 *
 * The robot operates in two distinct modes:
 * - PHASE_RUN: The robot moves straight ahead.
 * - PHASE_TUMBLE: The robot rotates in place to change its heading.
 */
typedef enum {
    PHASE_RUN = 0,
    PHASE_TUMBLE = 1
} PhaseState;

typedef struct {
    uint8_t r, g, b;
} rgb_t;

static rgb_t const k_white = {25, 25, 25};
static rgb_t const k_off   = {0, 0, 0};

// A small rainbow palette (repeat with modulo).
static rgb_t const k_rainbow[] = {
    {25,  0,  0}, // red
    {25, 10,  0}, // orange
    {25, 25,  0}, // yellow
    { 0, 25,  0}, // green
    { 0, 25, 25}, // cyan
    { 0,  0, 25}, // blue
    {10,  0, 25}, // violet
    {25,  0, 25}, // magenta
};
static uint8_t const k_rainbow_n = (uint8_t)(sizeof(k_rainbow) / sizeof(k_rainbow[0]));

/* -------------------------------------------------------------------------- */
/* Colour names (for the start‑up table)                                      */
/* -------------------------------------------------------------------------- */
static char const *const k_rainbow_color_names[8] = {
    "Red", "Orange", "Yellow", "Green", "Cyan",
    "Blue", "Violet", "Magenta",
};

// Seed/hop diffusion message (small, user payload).
typedef struct __attribute__((packed)) {
    uint16_t seed_id;
    uint32_t seed_time_ms; // "newest" seed is the one with the largest time
    uint16_t dist_hops;    // sender distance to that seed
} BloomMsg;

#define MSG_SIZE ((uint16_t)sizeof(BloomMsg))

typedef union {
    uint8_t  bytes[MSG_SIZE];
    BloomMsg v;
} bloom_msg_u;


// Normal "Global" variables, not related to configuration parameters, should be inserted within the USERDATA struct.
// /!\  In simulation, don't declare non-const global variables outside this struct, elsewise they will be shared among all agents (and this is not realistic).

// "Global" variables should be inserted within the USERDATA struct.
// Non-const global variables used by each robot. They will be accessible through the mydata pointer, declared by the macro "REGISTER_USERDATA"
typedef struct {
    bool started;
    uint32_t start_of_experiment_ms;

    // Photo start values;
#ifdef ENABLE_PHOTO_START
    int16_t last_data_b;
    int16_t last_data_fl;
    int16_t last_data_fr;
#endif

    // ID + motor calibration
    uint16_t my_id;
    uint8_t motor_dir_left;
    uint8_t motor_dir_right;

    // Run&tumble
    PhaseState phase;
    uint32_t phase_start_ms;
    uint32_t phase_duration_ms;
    uint8_t tumble_dir; // 0 left, 1 right
    bool run_backward;

    // Diffusion state
    uint16_t seed_id;
    uint32_t seed_time_ms;
    uint16_t dist_hops; // 0 => seed, 1 => neighbor, ...
    bool has_seed;
    uint32_t last_seed_heard_ms; // last time (ms) we heard info about current seed
    uint32_t last_route_refresh_ms; // last time we heard a neighbor supporting our current route

    // Blink state
    uint32_t blink_end_ms;
    uint32_t blink_next_toggle_ms;
    bool blink_on;          // if true show color, else show off
    rgb_t target_color;     // steady color to show after blink
} USERDATA;

// Call this macro in the same file (.h or .c) as the declaration of USERDATA
DECLARE_USERDATA(USERDATA);

// Don't forget to call this macro in the main .c file of your project (only once!)
REGISTER_USERDATA(USERDATA);
// Now, members of the USERDATA struct can be accessed through mydata->MEMBER. E.g. mydata->age
//  On real robots, the compiler will automatically optimize the code to access member variables as if they were true globals.


// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

static uint32_t rand_between_u32(uint32_t lo, uint32_t hi) {
    if (hi <= lo) return lo;
    return lo + (uint32_t)(rand() % (int)(hi - lo + 1));
}

static uint16_t clamp_u16(uint32_t x, uint16_t maxv) {
    if (x > maxv) return maxv;
    return (uint16_t)x;
}

static void set_robot_direction(bool backward) {
    if (!enable_backward_dir) return;

    if (backward) {
        pogobot_motor_dir_set(motorL, (mydata->motor_dir_left  == 0 ? 1 : 0));
        pogobot_motor_dir_set(motorR, (mydata->motor_dir_right == 0 ? 1 : 0));
    } else {
        pogobot_motor_dir_set(motorL, mydata->motor_dir_left);
        pogobot_motor_dir_set(motorR, mydata->motor_dir_right);
    }
}

static rgb_t color_for_dist(uint16_t dist_hops) {
    if (dist_hops == 0) return k_white;
    uint16_t d = dist_hops - 1;
    return k_rainbow[d % k_rainbow_n];
}

static void apply_led(rgb_t c) {
    pogobot_led_setColor(c.r, c.g, c.b);
}

static void start_blink(rgb_t target, uint32_t now_ms) {
    mydata->target_color = target;
    mydata->blink_end_ms = now_ms + blink_total_ms;
    mydata->blink_next_toggle_ms = now_ms + blink_half_period_ms;
    mydata->blink_on = true;
    apply_led(target);
}

static void update_blink(uint32_t now_ms) {
    if (mydata->blink_end_ms == 0) {
        apply_led(mydata->target_color);
        return;
    }
    if (now_ms >= mydata->blink_end_ms) {
        mydata->blink_end_ms = 0;
        apply_led(mydata->target_color);
        return;
    }
    if (now_ms >= mydata->blink_next_toggle_ms) {
        mydata->blink_next_toggle_ms += blink_half_period_ms;
        mydata->blink_on = !mydata->blink_on;
        apply_led(mydata->blink_on ? mydata->target_color : k_off);
    }
}

// A seed is "newer" if it has larger seed_time_ms; tie-break by seed_id (smaller wins).
static bool seed_is_newer(uint32_t t_a, uint16_t id_a, uint32_t t_b, uint16_t id_b) {
    if (t_a != t_b) return t_a > t_b;
    return id_a < id_b;
}

/** Print the hop‑count→colour lookup table. */
static void print_colormap_table(void) {
    printf("\nDistance to seed -> LED colour\n");
    printf("  Distance | Colour  | RGB\n");
    printf(" -------+---------+-------------\n");
    for(uint8_t i = 0; i < k_rainbow_n; ++i) {
        rgb_t c = k_rainbow[i];
        printf("  %5u | %-7s| (%3u,%3u,%3u)\n", i, k_rainbow_color_names[i], c.r, c.g, c.b);
    }
    printf("\nCounts ≥10 repeat modulo %d.\n\n", k_rainbow_n);
}


// -----------------------------------------------------------------------------
// Messaging callbacks
// -----------------------------------------------------------------------------

// Called by the pogobot main loop before 'user_step', if there are messages to be processed
static void process_message(message_t *mr) {
#if IGNORE_SYSTEM_MESSAGES
    // Discard system messages (i.e. from other devices than robots). This check only works with long headers
    if (MSG_FULL_HEADER && mr->header._packet_type != ir_t_user) {
        return;
    }
#endif

    // Discard messages that are too small
    if (MSG_SIZE > mr->header.payload_length) {
        return;
    }

    // Copy payload
    bloom_msg_u u;
    for (uint16_t i = 0; i < MSG_SIZE; i++) {
        u.bytes[i] = mr->payload[i];
    }

    // Candidate info from neighbor
    uint16_t cand_seed_id = u.v.seed_id;
    uint32_t cand_seed_t  = u.v.seed_time_ms;
    uint16_t cand_dist    = clamp_u16((uint32_t)u.v.dist_hops + 1u, MAX_HOPS);

    // Ignore empty/uninitialized broadcasts
    if (cand_seed_t == 0) return;

    uint32_t now = current_time_milliseconds();
    bool updated = false;

    if (!mydata->has_seed) {
        // First seed info ever
        mydata->has_seed = true;
        mydata->seed_id = cand_seed_id;
        mydata->seed_time_ms = cand_seed_t;
        mydata->dist_hops = cand_dist;
        updated = true;
    } else {
        if (enable_random_seed) {
            // Prefer newest seed
            if (seed_is_newer(cand_seed_t, cand_seed_id, mydata->seed_time_ms, mydata->seed_id)) {
                mydata->seed_id = cand_seed_id;
                mydata->seed_time_ms = cand_seed_t;
                mydata->dist_hops = cand_dist;
                updated = true;
            } else if (cand_seed_t == mydata->seed_time_ms && cand_seed_id == mydata->seed_id) {
                // Same seed: prefer shorter hop distance (or track changes when robots move)
                if (cand_dist < mydata->dist_hops) {
                    mydata->dist_hops = cand_dist;
                    updated = true;
                } else if (cand_dist == mydata->dist_hops) {
                    // Route confirmation: refresh without changing distance
                    mydata->last_route_refresh_ms = now;
                }
            }

        } else {
            // enable_random_seed == false: global seed is the lowest ID
            if (cand_seed_id < mydata->seed_id) {
                mydata->seed_id = cand_seed_id;
                mydata->seed_time_ms = cand_seed_t;
                mydata->dist_hops = cand_dist;
                updated = true;
            } else if (cand_seed_id == mydata->seed_id) {
                // Same seed: prefer shorter distance (or track changes when robots move)
                if (cand_dist < mydata->dist_hops) {
                    mydata->dist_hops = cand_dist;
                    updated = true;
                } else if (cand_dist == mydata->dist_hops) {
                    // Route confirmation: refresh without changing distance
                    mydata->last_route_refresh_ms = now;
                }
            }

        }
    }

    if (updated) {
        mydata->last_seed_heard_ms = now;
        mydata->last_route_refresh_ms = now;
        rgb_t c = color_for_dist(mydata->dist_hops);
        if (enable_blinking_when_updated)
            start_blink(c, now);
    }
}

// Called by the pogobot main loop before 'user_step'. Used to send IR messages to the neighborhood.
static bool send_message(void) {
    if (!mydata->has_seed) return false;

    bloom_msg_u u;
    u.v.seed_id = mydata->seed_id;
    u.v.seed_time_ms = mydata->seed_time_ms;
    u.v.dist_hops = mydata->dist_hops;

    if (SEND_OMNI) {
        if (MSG_FULL_HEADER) {
            pogobot_infrared_sendLongMessage_omniGen((uint8_t *)u.bytes, MSG_SIZE);
        } else {
            pogobot_infrared_sendShortMessage_omni((uint8_t *)u.bytes, MSG_SIZE);
        }
    } else {
        for (uint16_t dir = 0; dir < 4; dir++) {
            if (MSG_FULL_HEADER) {
                pogobot_infrared_sendLongMessage_uniSpe(dir, (uint8_t *)u.bytes, MSG_SIZE);
            } else {
                pogobot_infrared_sendShortMessage_uni(dir, (uint8_t *)u.bytes, MSG_SIZE);
            }
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
// Controller
// -----------------------------------------------------------------------------

// Init function. Called once at the beginning of the program (cf 'pogobot_start' call in main())
void user_init(void) {
    srand(pogobot_helper_getRandSeed());
    pogobot_infrared_set_power(INFRARED_POWER);

    // Set mydata variables to 0
    memset(mydata, 0, sizeof(*mydata));
    mydata->my_id = pogobot_helper_getid();

    // Motor direction calibration (needed for backward mode)
    uint8_t dir_mem[3];
    pogobot_motor_dir_mem_get(dir_mem);
    mydata->motor_dir_left  = dir_mem[1];
    mydata->motor_dir_right = dir_mem[0];
    set_robot_direction(false);

    // Run&tumble init
    mydata->phase = (rand() % 2) ? PHASE_RUN : PHASE_TUMBLE;
    mydata->phase_start_ms = current_time_milliseconds();
    mydata->phase_duration_ms = rand_between_u32(run_duration_min_ms, run_duration_max_ms);
    mydata->tumble_dir = (uint8_t)(rand() % 2);
    mydata->run_backward = false;

    // Diffusion init
    if (!enable_random_seed) {
        mydata->has_seed = true;         // we have a candidate (ourselves)
        mydata->seed_id = mydata->my_id; // propose self
        mydata->seed_time_ms = 1;        // can be constant in this mode
        mydata->dist_hops = 0;           // distance to self
        mydata->last_seed_heard_ms = current_time_milliseconds();
        mydata->last_route_refresh_ms = mydata->last_seed_heard_ms;
    } else {
        // No seed known yet
        mydata->has_seed = false;
        mydata->seed_time_ms = 0;
        mydata->seed_id = 0;
        mydata->dist_hops = MAX_HOPS;
        mydata->last_seed_heard_ms = 0;
        mydata->last_route_refresh_ms = 0;
    }
    mydata->last_seed_heard_ms = current_time_milliseconds();

    // LED init
    mydata->target_color = (rgb_t){0, 0, 0};
    apply_led(k_off);

    // Main loop + messaging knobs
    main_loop_hz = FQCY;
    max_nb_processed_msg_per_tick = MAX_NB_OF_MSG;
    percent_msgs_sent_per_ticks = percent_msg_sent;
    msg_rx_fn = process_message;
    msg_tx_fn = send_message;

    error_codes_led_idx = 3;

#ifdef SIMULATOR
    if (pogobot_helper_getid() == 0) {
        print_colormap_table();
    }
#else
    print_colormap_table();
#endif

#ifdef ENABLE_PHOTO_START
    mydata->started = false;
    mydata->last_data_b  = pogobot_photosensors_read(0);
    mydata->last_data_fl = pogobot_photosensors_read(1);
    mydata->last_data_fr = pogobot_photosensors_read(2);
#else
    mydata->started = true;
#endif
}

static void maybe_become_seed(uint32_t now_ms) {
    if (!enable_random_seed) return;  // If enable_random_seed==false, the seed is always the robot with the lowest ID
    if (den_p_become_seed == 0) return;
    if ((uint32_t)(rand() % (int)den_p_become_seed) != 0) return;

    // This robot becomes the newest seed
    mydata->has_seed = true;
    mydata->seed_id = mydata->my_id;
    mydata->seed_time_ms = now_ms;
    mydata->dist_hops = 0;
    mydata->last_seed_heard_ms = now_ms;

    // White seed, blink on update (looks like a "spark" starting the bloom)
    if (enable_blinking_when_updated)
        start_blink(k_white, now_ms);
}

static void step_motility(uint32_t now_ms) {
    if (!moving_robots) {
        pogobot_motor_set(motorL, motorStop);
        pogobot_motor_set(motorR, motorStop);
        return;
    }

    uint16_t const motor_low = (uint16_t)(motorFull * 1.00);

    // Phase switch?
    if (now_ms - mydata->phase_start_ms >= mydata->phase_duration_ms) {
        if (mydata->phase == PHASE_RUN) {
            mydata->phase = PHASE_TUMBLE;
            mydata->phase_duration_ms = rand_between_u32(tumble_duration_min_ms, tumble_duration_max_ms);
            mydata->tumble_dir = (uint8_t)(rand() % 2);
            set_robot_direction(false);
        } else {
            mydata->phase = PHASE_RUN;
            mydata->phase_duration_ms = rand_between_u32(run_duration_min_ms, run_duration_max_ms);
            mydata->run_backward = enable_backward_dir ? ((rand() % 10) == 0) : false;
            set_robot_direction(mydata->run_backward);
        }
        mydata->phase_start_ms = now_ms;
    }

    // Execute
    if (mydata->phase == PHASE_RUN) {
        pogobot_motor_set(motorL, motor_low);
        pogobot_motor_set(motorR, motor_low);
    } else {
        // gentle tumble at low speed
        if (mydata->tumble_dir == 0) {
            pogobot_motor_set(motorL, motorStop);
            pogobot_motor_set(motorR, motor_low);
        } else {
            pogobot_motor_set(motorL, motor_low);
            pogobot_motor_set(motorR, motorStop);
        }
    }
}


#ifdef ENABLE_PHOTO_START
// ********************************************************************************
// * Start-up phase for simultaneous start of the robots when the lights turn off
// ********************************************************************************
bool photo_start(void) {
    if (mydata->started) {
        return true;
    }
    // Set initial led color to white
    pogobot_led_setColor(k_off.r, k_off.g, k_off.b);

    // Stopping if the difference between the last value and the current value is more than the threshold
    int16_t const data_b  = pogobot_photosensors_read(0);
    int16_t const data_fl = pogobot_photosensors_read(1);
    int16_t const data_fr = pogobot_photosensors_read(2);
    int16_t const diff_b  = data_b  - mydata->last_data_b;  // Positive if data > last_data, i.e more light than before
    int16_t const diff_fl = data_fl - mydata->last_data_fl;
    int16_t const diff_fr = data_fr - mydata->last_data_fr;
    mydata->last_data_b  = data_b;
    mydata->last_data_fl = data_fl;
    mydata->last_data_fr = data_fr;

    if(diff_b >= LIGHT_THRESHOLD || diff_fl >= LIGHT_THRESHOLD || diff_fr >= LIGHT_THRESHOLD) {
        mydata->started = true;
        mydata->start_of_experiment_ms = current_time_milliseconds();
        return true;
    } else {
        return false; // Quit function if experiment has not started
    }
}
#endif



void user_step(void) {
#ifdef ENABLE_PHOTO_START
    if (!photo_start())
        return;
#endif
    uint32_t now_ms = current_time_milliseconds();

    // Experiment has started. Wait for some time
    if (now_ms - mydata->start_of_experiment_ms < BOOT_TIME) {
        pogobot_led_setColor(k_white.r, k_white.g, k_white.b); // set boot led color to white
        return; // Wait
    }

    // Occasionally spark a new seed (starting a new "flower")
    maybe_become_seed(now_ms);

    // If robots are moving, connectivity changes can break previously valid shortest routes.
    // We use a simple "route aging" rule: distances can always decrease immediately, but
    // they only increase slowly if we don't receive confirmations for a while.
    if (moving_robots && mydata->has_seed) {
        // Seed (dist==0) never ages.
        if (mydata->dist_hops > 0 && mydata->dist_hops < MAX_HOPS) {
            uint32_t const route_age_ms = 800u;
            if (now_ms - mydata->last_route_refresh_ms > route_age_ms) {
                mydata->dist_hops = clamp_u16((uint32_t)mydata->dist_hops + 1u, MAX_HOPS);
                rgb_t c = color_for_dist(mydata->dist_hops);
                if (enable_blinking_when_updated) start_blink(c, now_ms);
                mydata->last_route_refresh_ms = now_ms;
            }
        }
    }

    // Maintain target color (blink if we just updated)
    if (mydata->has_seed) {
        mydata->target_color = color_for_dist(mydata->dist_hops);
    } else {
        mydata->target_color = k_off;
    }
    update_blink(now_ms);

    // Optional motility
    step_motility(now_ms);
}

#ifdef SIMULATOR
void global_setup(void) {
    init_from_configuration(moving_robots);
    init_from_configuration(enable_backward_dir);

    init_from_configuration(run_duration_min_ms);
    init_from_configuration(run_duration_max_ms);
    init_from_configuration(tumble_duration_min_ms);
    init_from_configuration(tumble_duration_max_ms);
    init_from_configuration(max_speed_frac);

    init_from_configuration(enable_random_seed);
    init_from_configuration(den_p_become_seed);

    init_from_configuration(percent_msg_sent);
    init_from_configuration(enable_blinking_when_updated);
    init_from_configuration(blink_total_ms);
    init_from_configuration(blink_half_period_ms);
}

// Function called once by the simulator to specify user-defined data fields to add to the exported data files
void create_data_schema() {
    data_add_column_int16("seed_id");
    data_add_column_int32("seed_time_ms");
    data_add_column_int16("dist_hops");
    data_add_column_int32("last_seed_heard_ms");
    data_add_column_int32("last_route_refresh_ms");
    data_add_column_bool("blink_on");
}

// Function called periodically by the simulator each time data is saved (cf config parameter "save_data_period" in seconds)
void export_data() {
    if (mydata->started) { // Only store data after the photostart period
        enable_data_export(); // Enable data export this time
        data_set_value_int16("seed_id", mydata->seed_id);
        data_set_value_int32("seed_time_ms", mydata->seed_time_ms);
        data_set_value_int16("dist_hops", mydata->dist_hops);
        data_set_value_int32("last_seed_heard_ms", mydata->last_seed_heard_ms);
        data_set_value_int32("last_route_refresh_ms", mydata->last_route_refresh_ms);
        data_set_value_bool("blink_on", mydata->blink_on);
    } else { // Disable data export this time
        disable_data_export();
    }
}
#endif


// Entrypoint of the program
int main(void) {
    pogobot_init();     // Initialization routine for the robots
    // Specify the user_init and user_step functions
    pogobot_start(user_init, user_step);

    // Specify the callback functions. Only called by the simulator.
    //  In particular, they serve to add data fields to the exported data files
    SET_CALLBACK(callback_global_setup, global_setup);              // Called once at the start of a simulation. Useful to set configuration parameters from the config file
    SET_CALLBACK(callback_create_data_schema, create_data_schema);  // Called once to specify the data format
    SET_CALLBACK(callback_export_data, export_data);                // Called at each configuration-specified period (e.g. every second) on each robot to register exported data
    return 0;
}
