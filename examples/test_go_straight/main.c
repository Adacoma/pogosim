
// Main include for pogobots, both for real robots and for simulations
#include "pogobase.h"

// "Global" variables set by the YAML configuration file (in simulation) by the function global_setup, or with a fixed values (in experiments). These values should be seen as constants shared by all robots.
float test_vect[4]           = {1.f, 2.f, 3.f, 4.f};


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
    uint8_t data_foo[8];          // Example data storage.
    uint8_t motor_dir_left;       // Calibrated value for left motor direction from robot memory
    uint8_t motor_dir_right;      // Calibrated value for right motor direction from robot memory
    uint16_t motor_power_left;    // Calibrated value for left motor power from robot memory
    uint16_t motor_power_right;   // Calibrated value for right motor power from robot memory
} USERDATA;

// Call this macro in the same file (.h or .c) as the declaration of USERDATA
DECLARE_USERDATA(USERDATA);

// Don't forget to call this macro in the main .c file of your project (only once!)
REGISTER_USERDATA(USERDATA);
// Now, members of the USERDATA struct can be accessed through mydata->MEMBER. E.g. mydata->data_foo
//  On real robots, the compiler will automatically optimize the code to access member variables as if they were true globals.


/**
 * @brief Set the direction of the robot to go forward or backward
 *
 * @param dir false for forward, true for backward
 */
static void set_robot_direction(bool dir) {
    if (dir) {
        pogobot_motor_dir_set(motorL, (mydata->motor_dir_left  == 0 ? 1 : 0));
        pogobot_motor_dir_set(motorR, (mydata->motor_dir_right == 0 ? 1 : 0));
    } else {
        pogobot_motor_dir_set(motorL, mydata->motor_dir_left);
        pogobot_motor_dir_set(motorR, mydata->motor_dir_right);
    }
}


void user_init(void) {
#ifndef SIMULATOR
    printf("setup ok\n");
#endif

    // Initialize the random number generator
    srand(pogobot_helper_getRandSeed());

    // Set the main loop frequency to 60 Hz (i.e., user_step() is called 60 times per second).
    main_loop_hz = 60;
    // Disable message processing (as messaging is not used in this example).
    max_nb_processed_msg_per_tick = 0;
    msg_rx_fn = NULL;
    msg_tx_fn = NULL;
    // Specify LED index for error codes (negative values disable this feature).
    error_codes_led_idx = 3;

    // Retrieve calibration data from the robots
    uint8_t dir_mem[3];
    int8_t res_dir_mem_get = pogobot_motor_dir_mem_get(dir_mem);
    mydata->motor_dir_right = dir_mem[0];
    mydata->motor_dir_left = dir_mem[1];

    uint16_t power_mem[3];
    int8_t res_power_mem_get = pogobot_motor_power_mem_get(power_mem);
    mydata->motor_power_left = power_mem[1];
    mydata->motor_power_right = power_mem[0];
    printf("calibrated dir_mem:   (R:%u L:%u res=%d)\n", mydata->motor_dir_left, mydata->motor_dir_right, res_dir_mem_get);
    printf("calibrated power_mem: (R:%u L:%u res=%d)\n", mydata->motor_power_left, mydata->motor_power_right, res_power_mem_get);

    // In simulation, test_vect values are directly set in configuration files.
    // In experiments with real robots, test_vect has the default values set at initialization.
    if (pogobot_helper_getid() == 0) {     // Only print messages for robot 0
        printf("Test global values: (%d,%d,%d,%d)\n", (int)test_vect[0], (int)test_vect[1], (int)test_vect[2], (int)test_vect[3]);
    }
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
    set_robot_direction(0);
    pogobot_motor_set(motorL, mydata->motor_power_left);
    pogobot_motor_set(motorR, mydata->motor_power_right);

    // Example use of the USERDATA data array.
    mydata->data_foo[0] = 42;
}

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
    return 0;
}

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
