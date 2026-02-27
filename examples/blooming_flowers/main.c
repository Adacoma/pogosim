// main.c — "Blooming flowers" hop-distance diffusion to a spontaneously appearing seed.
//

#include "pogobase.h"
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

// -----------------------------------------------------------------------------
// Configuration (can be overridden by simulator via global_setup())
// -----------------------------------------------------------------------------
bool moving_robots   = true;
bool enable_backward_dir   = false;

// Run&tumble timing (ms)
uint32_t run_duration_min_ms    = 250;
uint32_t run_duration_max_ms    = 1200;
uint32_t tumble_duration_min_ms = 120;
uint32_t tumble_duration_max_ms = 900;

// "Seed creation" probability per tick: p = 1 / den_p_become_seed
uint32_t den_p_become_seed = 25000; // 105000;   // lower => more frequent blooms

// Hop diffusion / communication
#define INFRARED_POWER 2
#define FQCY 60
#define MAX_NB_OF_MSG 6
#define PERCENT_MSG_SENT 10     // slow-ish broadcast, helps "wavefront" look
#define SEND_OMNI true
#define MSG_FULL_HEADER true

// Visuals
#define MAX_HOPS 250            // clamp for safety
#define BLINK_TOTAL_MS 6 // 420      // total blink duration when state updates
#define BLINK_HALF_PERIOD_MS 1 // 70 // blink toggles every 70ms

// -----------------------------------------------------------------------------

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

// Per-robot state
typedef struct {
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

    // Blink state
    uint32_t blink_end_ms;
    uint32_t blink_next_toggle_ms;
    bool blink_on;          // if true show color, else show off
    rgb_t target_color;     // steady color to show after blink
} USERDATA;

DECLARE_USERDATA(USERDATA);
REGISTER_USERDATA(USERDATA);

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
    mydata->blink_end_ms = now_ms + BLINK_TOTAL_MS;
    mydata->blink_next_toggle_ms = now_ms + BLINK_HALF_PERIOD_MS;
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
        mydata->blink_next_toggle_ms += BLINK_HALF_PERIOD_MS;
        mydata->blink_on = !mydata->blink_on;
        apply_led(mydata->blink_on ? mydata->target_color : k_off);
    }
}

// A seed is "newer" if it has larger seed_time_ms; tie-break by seed_id (smaller wins).
static bool seed_is_newer(uint32_t t_a, uint16_t id_a, uint32_t t_b, uint16_t id_b) {
    if (t_a != t_b) return t_a > t_b;
    return id_a < id_b;
}

// -----------------------------------------------------------------------------
// Messaging callbacks
// -----------------------------------------------------------------------------

static void process_message(message_t *mr) {
    if (MSG_FULL_HEADER && mr->header._packet_type != ir_t_user) {
        return;
    }

    if (MSG_SIZE > mr->header.payload_length) {
        return;
    }

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

    bool updated = false;

    if (!mydata->has_seed) {
        // First seed info ever
        mydata->has_seed = true;
        mydata->seed_id = cand_seed_id;
        mydata->seed_time_ms = cand_seed_t;
        mydata->dist_hops = cand_dist;
        updated = true;
    } else {
        // 1) Prefer newest seed
        if (seed_is_newer(cand_seed_t, cand_seed_id, mydata->seed_time_ms, mydata->seed_id)) {
            mydata->seed_id = cand_seed_id;
            mydata->seed_time_ms = cand_seed_t;
            mydata->dist_hops = cand_dist;
            updated = true;
        } else if (cand_seed_t == mydata->seed_time_ms && cand_seed_id == mydata->seed_id) {
            // 2) Same seed: prefer shorter hop distance
            if (cand_dist < mydata->dist_hops) {
                mydata->dist_hops = cand_dist;
                updated = true;
            }
        }
    }

    if (updated) {
        uint32_t now = current_time_milliseconds();
        rgb_t c = color_for_dist(mydata->dist_hops);
        start_blink(c, now);
    }
}

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

void user_init(void) {
    srand(pogobot_helper_getRandSeed());
    pogobot_infrared_set_power(INFRARED_POWER);

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

    // Diffusion init: no seed known yet
    mydata->has_seed = false;
    mydata->seed_time_ms = 0;
    mydata->seed_id = 0;
    mydata->dist_hops = MAX_HOPS;

    // LED init
    mydata->target_color = (rgb_t){0, 0, 0};
    apply_led(k_off);

    // Main loop + messaging knobs
    main_loop_hz = FQCY;
    max_nb_processed_msg_per_tick = MAX_NB_OF_MSG;
    percent_msgs_sent_per_ticks = PERCENT_MSG_SENT;
    msg_rx_fn = process_message;
    msg_tx_fn = send_message;

    error_codes_led_idx = 3;
}

static void maybe_become_seed(uint32_t now_ms) {
    if (den_p_become_seed == 0) return;
    if ((uint32_t)(rand() % (int)den_p_become_seed) != 0) return;

    // This robot becomes the newest seed
    mydata->has_seed = true;
    mydata->seed_id = mydata->my_id;
    mydata->seed_time_ms = now_ms;
    mydata->dist_hops = 0;

    // White seed, blink on update (looks like a "spark" starting the bloom)
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

void user_step(void) {
    uint32_t now_ms = current_time_milliseconds();

    // 1) Occasionally spark a new seed (starting a new "flower")
    maybe_become_seed(now_ms);

    // 2) Maintain target color (blink if we just updated)
    if (mydata->has_seed) {
        mydata->target_color = color_for_dist(mydata->dist_hops);
    } else {
        mydata->target_color = k_off;
    }
    update_blink(now_ms);

    // 3) Optional motility
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

    init_from_configuration(den_p_become_seed);
}
#endif

int main(void) {
    pogobot_init();
    pogobot_start(user_init, user_step);
#ifdef SIMULATOR
    SET_CALLBACK(callback_global_setup, global_setup);
#endif
    return 0;
}
