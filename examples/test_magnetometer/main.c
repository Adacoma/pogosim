// main.c — test several miscellaneous functions of the Pogobot API

// Main include for pogobots, both for real robots and for simulations
#include "pogobase.h"

// "Global" variables should be inserted within the USERDATA struct.
// /!\  In simulation, don't declare non-const global variables outside this struct, elsewise they will be shared among all agents (and this is not realistic).
typedef struct {
    // Put all global variables you want here.
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


// Init function. Called once at the beginning of the program (cf 'pogobot_start' call in main())
void user_init(void) {
#ifndef SIMULATOR
    printf("setup ok\n");
#endif

    // Initialize the random number generator
    srand(pogobot_helper_getRandSeed());

    // Set main loop frequency, message sending frequency, message processing frequency
    main_loop_hz = 30;      // Call the 'user_step' function only 1 time per second
    max_nb_processed_msg_per_tick = 0;
    // Specify functions to send/transmit messages. See the "blooming" example to see message sending/processing in action!
    percent_msgs_sent_per_ticks = 0;
    msg_rx_fn = NULL;
    msg_tx_fn = NULL;

    // Set led index to show error codes (e.g. time overflows)
    error_codes_led_idx = 3; // Default value, negative values to disable

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
}


// Step function. Called continuously at each step of the pogobot main loop
void user_step(void) {
    int16_t mag_x = 0;
    int16_t mag_y = 0;
    int16_t mag_z = 0;  

    pogobot_led_setColor(0, 0, 255);

    // Read LIS2MDL magnetometer X,Y,Z axis
    if (magn_read_XYZ(&mag_x, &mag_y, &mag_z, 10) == 0) {
#ifdef SIMULATOR
        if (pogobot_helper_getid() == 0)     // Only print messages for robot 0
#endif
        printf("%d %d %d\n", mag_x, mag_y, mag_z);
        mag_x = 0; 
        mag_y = 0; 
        mag_z = 0;
        pogobot_led_setColor(0, 255, 0);
    }

    // Spin the robots
    pogobot_motor_dir_set(motorL, mydata->motor_dir_left);
    pogobot_motor_dir_set(motorR, mydata->motor_dir_right == 0 ? 1 : 0);
    pogobot_motor_set(motorL, mydata->motor_power_left * 0.3);
    pogobot_motor_set(motorR, mydata->motor_power_right * 0.3);
}


// Entrypoint of the program
int main(void) {
    pogobot_init();     // Initialization routine for the robots
#ifndef SIMULATOR
    printf("init ok\n");
#endif

    // Specify the user_init and user_step functions
    pogobot_start(user_init, user_step);
    return 0;
}

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
