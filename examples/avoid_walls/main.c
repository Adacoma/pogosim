// Simple reactive wall avoidance for wheeled robots
// Robots detect "wall" messages and turn away from detected walls
// Red LEDs indicate which faces recently detected walls

// Main include for pogobots, both for real robots and for simulations
#include "pogobase.h"

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* ------------------------- Parameters ------------------------------- */
// "Global" variables set by the YAML configuration file (in simulation) by the function global_setup, or with a fixed values (in experiments). These values should be seen as constants shared by all robots.

float forward_speed_ratio = 1.0f;
uint16_t forward_speed = 0;

static inline void recompute_speeds(void) {
    forward_speed = (uint16_t)((float)motorFull * forward_speed_ratio);
}

uint32_t wall_memory_ms = 300;        // How long wall detection persists (short to avoid spinning)
uint32_t turn_duration_ms = 300;      // How long to execute a turn (long enough to clear wall angle)
uint32_t forward_commit_ms = 1000;    // Move forward after turn for this long


/* ------------------------- Types ------------------------------- */
typedef enum {
    ACTION_FORWARD,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
    ACTION_FORWARD_COMMIT  // Forward movement after a turn, but can still react to front walls
} action_t;

/**
 * @brief Extended USERDATA structure for the run-and-tumble behavior.
 *
 * This structure holds all the global variables that are unique to each robot.
 */
typedef struct {
    // Put all global variables you want here.

    // Calibration (wheels only)
    uint16_t motorLeft;
    uint8_t dirLeft;
    uint16_t motorRight;
    uint8_t dirRight;

    // Face detection times - 0:Front, 1:Right, 2:Back, 3:Left
    uint32_t last_wall_seen_ms[4];

    // Current action state
    action_t current_action;
    uint32_t action_until_ms;

    time_reference_t timer_it;
} USERDATA;

// Call this macro in the same file (.h or .c) as the declaration of USERDATA
DECLARE_USERDATA(USERDATA);

// Don't forget to call this macro in the main .c file of your project (only once!)
REGISTER_USERDATA(USERDATA);
// Now, members of the USERDATA struct can be accessed through mydata->MEMBER. E.g. mydata->data_foo
//  On real robots, the compiler will automatically optimize the code to access member variables as if they were true globals.

/* ------------------------- Utilities ------------------------------- */
static inline uint32_t now_ms(void) {
    return current_time_milliseconds();
}

static inline bool face_active(uint8_t face, uint32_t tnow) {
    uint32_t t = mydata->last_wall_seen_ms[face];
    return (t != 0) && (tnow - t <= wall_memory_ms);
}

// Show red LED on faces that recently detected walls
static inline void update_lateral_leds(uint32_t tnow) {
    for(uint8_t face = 0; face < 4; face++) {
        bool active = face_active(face, tnow);
        uint8_t r = active ? 255 : 0;
        pogobot_led_setColors(r, 0, 0, face + 1);
    }
}

static inline bool is_wall_payload(const message_t *msg) {
    const uint8_t *p = msg->payload;
    const uint16_t n = msg->header.payload_length;
    return (n >= 4) && (p[0]=='w' && p[1]=='a' && p[2]=='l' && p[3]=='l');
}

/* ------------------------- Motion ------------------------------- */
static inline void move_forward(void) {
    pogobot_motor_set(motorL, mydata->motorLeft);
    pogobot_motor_set(motorR, mydata->motorRight);
    pogobot_motor_dir_set(motorL, mydata->dirLeft);
    pogobot_motor_dir_set(motorR, mydata->dirRight);
    pogobot_led_setColor(0, 100, 0); // Dim green: moving forward
}

static inline void spin_left(void) {
    pogobot_motor_set(motorL, motorHalf);
    pogobot_motor_set(motorR, motorHalf);
    pogobot_motor_dir_set(motorL, (mydata->dirLeft + 1) % 2);
    pogobot_motor_dir_set(motorR, mydata->dirRight);
    pogobot_led_setColor(100, 100, 255); // Blue: turning left
}

static inline void spin_right(void) {
    pogobot_motor_set(motorL, motorHalf);
    pogobot_motor_set(motorR, motorHalf);
    pogobot_motor_dir_set(motorL, mydata->dirLeft);
    pogobot_motor_dir_set(motorR, (mydata->dirRight + 1) % 2);
    pogobot_led_setColor(255, 100, 100); // Red: turning right
}

/* ------------------------- IR messages ------------------------------- */

static void process_message(message_t *mr) {
    if (!is_wall_payload(mr))
        return;
    uint32_t t = now_ms();

    int face = mr->header._receiver_ir_index; // 0..3 for directional faces
    if (face >= 0 && face < 4) {
        mydata->last_wall_seen_ms[face] = t;
    }
    update_lateral_leds(t);
}


/* ------------------------- Simple reactive behavior ------------------------------- */
static void decide_action(void) {
    uint32_t tnow = now_ms();

    // If currently executing a turn, keep turning until duration expires
    if ((mydata->current_action == ACTION_TURN_LEFT || mydata->current_action == ACTION_TURN_RIGHT)
        && tnow < mydata->action_until_ms) {
        return; // Continue current turn
    }

    // Check which faces recently detected walls
    bool front = face_active(0, tnow);  // Face 0: Front
    bool right = face_active(1, tnow);  // Face 1: Right
    bool back  = face_active(2, tnow);  // Face 2: Back
    bool left  = face_active(3, tnow);  // Face 3: Left

    // Count active walls
    int wall_count = (front ? 1 : 0) + (right ? 1 : 0) + (back ? 1 : 0) + (left ? 1 : 0);

    // Special case: surrounded by walls on 3+ sides
    if (wall_count >= 3) {
        // If not already turning or in forward commit, start a short turn
        if (mydata->current_action != ACTION_TURN_LEFT &&
            mydata->current_action != ACTION_TURN_RIGHT &&
            mydata->current_action != ACTION_FORWARD_COMMIT) {
            // Pick random direction to turn
            if (rand() & 1) {
                mydata->current_action = ACTION_TURN_LEFT;
            } else {
                mydata->current_action = ACTION_TURN_RIGHT;
            }
            mydata->action_until_ms = tnow + turn_duration_ms;
            return;
        }
        // If we just finished turning while surrounded, commit to forward
        if ((mydata->current_action == ACTION_TURN_LEFT || mydata->current_action == ACTION_TURN_RIGHT)
            && tnow >= mydata->action_until_ms) {
            mydata->current_action = ACTION_FORWARD_COMMIT;
            mydata->action_until_ms = tnow + forward_commit_ms * 2;
            // Clear wall memory to break out of the surrounded state
            for(uint8_t i = 0; i < 4; i++) {
                mydata->last_wall_seen_ms[i] = 0;
            }
            return;
        }
        // If in forward commit while surrounded, just keep going (don't react)
        if (mydata->current_action == ACTION_FORWARD_COMMIT && tnow < mydata->action_until_ms) {
            return; // Ignore all walls and push through
        }
    }

    // If in committed forward (after a turn), keep going forward but react to front walls
    if (mydata->current_action == ACTION_FORWARD_COMMIT && tnow < mydata->action_until_ms) {
        if (front) {
            // Emergency: still heading into wall, turn more aggressively
            if (rand() & 1) {
                mydata->current_action = ACTION_TURN_LEFT;
            } else {
                mydata->current_action = ACTION_TURN_RIGHT;
            }
            mydata->action_until_ms = tnow + turn_duration_ms;
        }
        return; // Otherwise continue committed forward
    }

    // If just finished a turn, enter committed forward
    if ((mydata->current_action == ACTION_TURN_LEFT || mydata->current_action == ACTION_TURN_RIGHT)
        && tnow >= mydata->action_until_ms) {
        mydata->current_action = ACTION_FORWARD_COMMIT;
        mydata->action_until_ms = tnow + forward_commit_ms;
        return;
    }

    // Normal reactive decision logic
    if (front) {
        // Wall ahead: turn away from it
        if (left && !right) {
            // Left wall too, turn right
            mydata->current_action = ACTION_TURN_RIGHT;
            mydata->action_until_ms = tnow + turn_duration_ms;
        } else if (right && !left) {
            // Right wall too, turn left
            mydata->current_action = ACTION_TURN_LEFT;
            mydata->action_until_ms = tnow + turn_duration_ms;
        } else {
            // Pick random direction to turn
            if (rand() & 1) {
                mydata->current_action = ACTION_TURN_LEFT;
            } else {
                mydata->current_action = ACTION_TURN_RIGHT;
            }
            mydata->action_until_ms = tnow + turn_duration_ms;
        }
    }
    else if (left && !right) {
        // Wall only on left: turn right
        mydata->current_action = ACTION_TURN_RIGHT;
        mydata->action_until_ms = tnow + turn_duration_ms;
    }
    else if (right && !left) {
        // Wall only on right: turn left
        mydata->current_action = ACTION_TURN_LEFT;
        mydata->action_until_ms = tnow + turn_duration_ms;
    }
    else {
        // No problematic walls (corridor, back only, or clear): go forward
        mydata->current_action = ACTION_FORWARD;
    }
}

static void execute_action(void) {
    switch (mydata->current_action) {
        case ACTION_TURN_LEFT:
            spin_left();
            break;
        case ACTION_TURN_RIGHT:
            spin_right();
            break;
        case ACTION_FORWARD_COMMIT:
        case ACTION_FORWARD:
        default:
            move_forward();
            break;
    }
}

/* ------------------------- Lifecycle ------------------------------- */

/**
 * @brief Initialization function for the robot.
 *
 * This function is executed once at startup (cf 'pogobot_start' call in main()).
 * It seeds the random number generator, initializes timers and system parameters,
 * sets up the main loop frequency, and configures the initial state for the
 * run-and-tumble behavior.
 */
static void user_init(void) {
#ifndef SIMULATOR
    printf("setup ok\n");
#endif

    // Empty out user data
    memset(mydata, 0, sizeof(*mydata));

    // Initialize the random number generator
    srand(pogobot_helper_getRandSeed());

    // Reset the internal stopwatch timer for measuring phase durations.
    pogobot_stopwatch_reset(&mydata->timer_it);

    // Set the main loop frequency to 30 Hz (i.e., user_step() is called 30 times per second).
    main_loop_hz = 30;
    // Only enable message processing (not sending), to receive messages from the pogowalls.
    max_nb_processed_msg_per_tick = 100;
    percent_msgs_sent_per_ticks = 0;
    msg_rx_fn = process_message;
    msg_tx_fn = NULL;
    // Specify LED index for error codes (negative values disable this feature).
    error_codes_led_idx = -1;

    pogobot_infrared_set_power(2);

    // Load wheel calibration from memory
    uint16_t p[3];
    uint8_t d[3];
    pogobot_motor_power_mem_get(p);
    pogobot_motor_dir_mem_get(d);
    mydata->motorLeft  = p[1];
    mydata->motorRight = p[0];
    mydata->dirLeft    = d[1];
    mydata->dirRight   = d[0];

    mydata->current_action = ACTION_FORWARD;
    mydata->action_until_ms = 0;
}

/**
 * @brief Main control loop for executing behavior.
 *
 * This function is called continuously at the frequency defined in user_init().
 */
static void user_step(void) {
    uint32_t tnow = now_ms();
    update_lateral_leds(tnow);
    decide_action();
    execute_action();
}

/* ------------------------- Wall functions ------------------------------- */

/**
 * @brief Function called each time the walls send a message
 *
 * This function is called continuously at the frequency defined in walls_user_init().
 */
static bool ping_walls(void) {
    uint8_t msg[5] = {"wall"};
    return pogobot_infrared_sendLongMessage_omniSpe(msg, sizeof(msg));
}


/**
 * @brief Initialization function for the pogowalls.
 *
 * This function is executed once at startup (cf 'pogobot_start' call in main()).
 */
static void walls_user_init(void) {
#ifndef SIMULATOR
    printf("Wall initialized\n");
#endif
    // Initialize the random number generator
    srand(pogobot_helper_getRandSeed());
    pogobot_infrared_set_power(2); // Set the power level used to send all the next messages.

    // Set the main loop frequency to 10 Hz (i.e., walls_user_step() is called 10 times per second).
    main_loop_hz = 10;

    // Disable message reception, but enable message send by the pogowalls.
    max_nb_processed_msg_per_tick = 0;
    percent_msgs_sent_per_ticks = 75; // 75% of chance each step to send a message.
    msg_rx_fn = NULL;
    msg_tx_fn = ping_walls;
    // Specify LED index for error codes (negative values disable this feature).
    error_codes_led_idx = -1;
}

/**
 * @brief Main control loop for the Pogowalls
 *
 * This function is called continuously at the frequency defined in walls_user_init().
 */
static void walls_user_step(void) {
    // ...
}

/* ------------------------- Main ------------------------------- */

#ifdef SIMULATOR
/**
 * @brief Function called once to initialize global values (e.g. configuration-specified constants)
 */
void global_setup(void) {
    init_from_configuration(wall_memory_ms);
    init_from_configuration(turn_duration_ms);
    init_from_configuration(forward_commit_ms);
    init_from_configuration(forward_speed_ratio);
    recompute_speeds();
}
#endif


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
#ifndef SIMULATOR
    printf("init ok\n");
#endif

    // Start the robot's main loop with the defined user_init and user_step functions. Ignored by the pogowalls.
    pogobot_start(user_init, user_step);
    // Use robots category "robots" by default. Same behavior as: pogobot_start(user_init, user_step, "robots");
    //   --> Make sure that the category "robots" is used to declare the pogobots in your configuration file. Cf conf/test.yaml

    // Init and main loop functions for the walls (pogowalls). Ignored by the robots.
    pogobot_start(walls_user_init, walls_user_step, "walls");

    // Specify the callback functions. Only called by the simulator.
    SET_CALLBACK(callback_global_setup, global_setup);
    return 0;
}

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
