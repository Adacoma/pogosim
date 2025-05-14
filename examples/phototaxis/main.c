
// Main include for pogobots, both for real robots and for simulations
#include "pogobase.h"

/**
 * @brief Enumeration for the robot's behavioral phases.
 *
 * The robot operates in three distinct modes:
 * - PHASE_RUN: The robot moves straight ahead.
 * - PHASE_TUMBLE: The robot rotates in place to change its heading.
 * - PHASE_GOAL: The robot stops and stay immobile, except if it is pushed away from the goal.
 */
typedef enum {
    PHASE_RUN = 0,   ///< Moving forward, sampling light
    PHASE_TUMBLE,    ///< Rotating in place to reorient
    PHASE_GOAL       ///< Goal reached, stopped at light spot
} state_t;

// "Global" variables set by the YAML configuration file (in simulation) by the function global_setup, or with a fixed values (in experiments). These values should be seen as constants shared by all robots.
uint32_t run_duration_min     = 200;
uint32_t run_duration_max     = 1200;
uint32_t tumble_duration_min  = 100;
uint32_t tumble_duration_max  = 1100;
uint16_t light_goal_threshold = 800;


// Normal "Global" variables should be inserted within the USERDATA struct.
// /!\  In simulation, don't declare non-const global variables outside this struct, elsewise they will be shared among all agents (and this is not realistic).

/**
 * @brief Extended USERDATA structure for the run-and-tumble behavior.
 *
 * This structure holds all the global variables that are unique to each robot.
 * It includes:
 * - data_foo: A placeholder array for miscellaneous data.
 * - timer_it: A timer used for measuring durations.
 * - phase: The current phase of the robot (either PHASE_RUN or PHASE_TUMBLE).
 * - phase_start_time: The timestamp (in milliseconds) when the current phase began.
 * - phase_duration: How long (in milliseconds) the current phase should last.
 * - tumble_direction: The direction to turn during the tumble phase (0 for left, 1 for right).
 */
typedef struct {
    // Put all global variables you want here.
    time_reference_t timer_it;    // Timer for internal timing operations.
    state_t phase;                // Current behavioral phase.
    uint32_t phase_start_time;    // Time when the current phase started (ms).
    uint32_t phase_duration;      // Duration of the current phase (ms).
    uint8_t tumble_direction;     // Tumble direction: 0 = left, 1 = right.
} USERDATA;

// Call this macro in the same file (.h or .c) as the declaration of USERDATA
DECLARE_USERDATA(USERDATA);

// Don't forget to call this macro in the main .c file of your project (only once!)
REGISTER_USERDATA(USERDATA);
// Now, members of the USERDATA struct can be accessed through mydata->MEMBER. E.g. mydata->data_foo
//  On real robots, the compiler will automatically optimize the code to access member variables as if they were true globals.

/**
 * @brief Generates a random duration for the run phase.
 *
 * The run phase duration is randomly chosen
 *
 * @return uint32_t Random run duration in milliseconds.
 */
static uint32_t get_run_duration(void) {
    return run_duration_min + (rand() % (run_duration_max - run_duration_min + 1));
}

/**
 * @brief Generates a random duration for the tumble phase.
 *
 * The tumble phase duration is randomly chosen
 *
 * @return uint32_t Random tumble duration in milliseconds.
 */
static uint32_t get_tumble_duration(void) {
    return tumble_duration_min + (rand() % (tumble_duration_max - tumble_duration_min + 1));
}

/**
 * @brief Initialization function for the robot.
 *
 * This function is executed once at startup (cf 'pogobot_start' call in main()).
 * It seeds the random number generator, initializes timers and system parameters,
 * sets up the main loop frequency, and configures the initial state for the
 * run-and-tumble behavior.
 */
void user_init(void) {
#ifndef SIMULATOR
    printf("setup ok\n");
#endif

    // Initialize the random number generator
    srand(pogobot_helper_getRandSeed());

    // Reset the internal stopwatch timer for measuring phase durations.
    pogobot_stopwatch_reset(&mydata->timer_it);

    // Set the main loop frequency to 60 Hz (i.e., user_step() is called 60 times per second).
    main_loop_hz = 60;
    // Disable message processing (as messaging is not used in this example).
    max_nb_processed_msg_per_tick = 0;
    msg_rx_fn = NULL;
    msg_tx_fn = NULL;
    // Specify LED index for error codes (negative values disable this feature).
    error_codes_led_idx = 3;

    // Initialize the run-and-tumble behavior.
    mydata->phase = PHASE_TUMBLE;                      // Start with the tumble phase.
    mydata->phase_start_time = current_time_milliseconds();  // Record the start time.
    mydata->phase_duration = get_run_duration();    // Set a random duration for running.
    mydata->tumble_direction = rand() % 2;            // Choose a random tumble direction.
}

/**
 * @brief Main control loop for executing behavior.
 *
 * This function is called continuously at the frequency defined in user_init().
 * It checks if the current phase duration has elapsed and, if so, transitions to
 * the next phase. Depending on the current phase, it sets the robot's motors to
 * either move straight (run phase) or rotate (tumble phase). It also provides periodic
 * debugging output.
 */
void user_step(void) {
    uint32_t now = current_time_milliseconds();  // Get the current time in milliseconds.

    // Read all three photodiodes and average
    int16_t p0 = pogobot_photosensors_read(0);
    int16_t p1 = pogobot_photosensors_read(1);
    int16_t p2 = pogobot_photosensors_read(2);
    int32_t curr_light = (p0 + p1 + p2) / 3;

    if (pogobot_ticks % 1000 == 0 && pogobot_helper_getid() == 0) {     // Only print messages for robot 0
        printf("# Robot ID: %d   pogobot_ticks: %lu  curr_light: %ld  p0: %d  phase: %d\n",
                pogobot_helper_getid(),
                pogobot_ticks,       // Increased by one at each execution of user_step
                curr_light, p0, mydata->phase);
    }

    // If light intensity crosses goal threshold -> success
    if (curr_light >= light_goal_threshold) {
        mydata->phase = PHASE_GOAL;
        // Indicate success: stop and blue LED
        pogobot_led_setColor(0, 0, 255);
        pogobot_motor_set(motorL, motorStop);
        pogobot_motor_set(motorR, motorStop);
        return;
    }

    // Check if the current phase duration has elapsed.
    if (now - mydata->phase_start_time >= mydata->phase_duration) {
        // Transition between phases.
        if (mydata->phase == PHASE_RUN) {
            // Switch from run to tumble.
            mydata->phase = PHASE_TUMBLE;
            mydata->phase_duration = get_tumble_duration();
            // Randomly choose a tumble direction: 0 for left, 1 for right.
            mydata->tumble_direction = rand() % 2;
        } else {
            // Switch from tumble back to run.
            mydata->phase = PHASE_RUN;
            mydata->phase_duration = get_run_duration();
        }
        // Reset the phase start time.
        mydata->phase_start_time = now;
    }

    // Execute behavior based on the current phase.
    if (mydata->phase == PHASE_RUN) {
        // Run phase: the robot moves forward.
        // Set LED color to green to indicate running.
        pogobot_led_setColor(0, 255, 0);
        // Set both motors to full speed for straight-line motion.
        pogobot_motor_set(motorL, motorFull);
        pogobot_motor_set(motorR, motorFull);
    } else {
        // Tumble phase: the robot rotates in place.
        // Set LED color to red to indicate tumbling.
        pogobot_led_setColor(255, 0, 0);
        // Rotate the robot by driving one motor while stopping the other.
        if (mydata->tumble_direction == 0) {
            // Tumble left: stop the left motor.
            pogobot_motor_set(motorL, motorStop);
            pogobot_motor_set(motorR, motorFull);
        } else {
            // Tumble right: stop the right motor.
            pogobot_motor_set(motorL, motorFull);
            pogobot_motor_set(motorR, motorStop);
        }
    }

    // Debug: Print phase information periodically for robot with ID 0.
    if (pogobot_ticks % 1000 == 0 && pogobot_helper_getid() == 0) {
        printf("Phase: %s, Phase Duration: %lums, Elapsed: %lums\n",
               (mydata->phase == PHASE_RUN) ? "RUN" : "TUMBLE",
               mydata->phase_duration,
               now - mydata->phase_start_time);
    }
}

#ifdef SIMULATOR
/**
 * @brief Function called once to initialize global values (e.g. configuration-specified constants)
 */
void global_setup() {
    init_from_configuration(run_duration_min);
    init_from_configuration(run_duration_max);
    init_from_configuration(tumble_duration_min);
    init_from_configuration(tumble_duration_max);
    init_from_configuration(light_goal_threshold);
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

    // Start the robot's main loop with the defined user_init and user_step functions.
    pogobot_start(user_init, user_step);

    // Specify the callback functions. Only called by the simulator.
    SET_CALLBACK(callback_global_setup, global_setup);              // Called once to initialize global values (e.g. configuration-specified constants)
    return 0;
}

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
