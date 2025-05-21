/**
 * POGOBOT
 *
 * Copyright © 2023 Sorbonne Université ISIR
 * This file is licensed under the Expat License, sometimes known as the MIT License.
 * Please refer to file LICENCE for details.
 *
 * Alessia Loi 2023-12 hanabi code.
 *  - Start-up phase for simultaneous start of the robots when the lights turn off: at start, the upper led is red
 *    and stays on until the environmental light is lower than LIGHT_THRESHOLD; then, the upper led switches to white
 *    for a period of BOOT_TIME seconds; after this BOOT_TIME period, hanabi starts
 *  - A robot has a state, consisting of an age and a color. The age is the number of color changes since initialization.
 *    Age value goes from 0 to max 65535 (16 bits unsigned). The robot broadcasts continuously its state to the neighbors.
 *  - A robot receives messages from neighbors, and elaborates the first MAX_NB_OF_MSG received. One message contains the sender state.
 *    If the receiver detects that the (neighbor) sender age is higher, then it copies its age and color.
 *  - A robot changes autonomously its state with a probability of 1/den_p_change_led_color.
 *  - A stagnation prevention mechanism tests if the age stays the same in a period of timeout_age_s seconds. This could happen
 *    if the maximum age value is exceeded, or if a robot doesn't receive messages from aged neighbors and doesn't change its state autonomously.
 *
 * Leo Cazenille 2025-01.
 *  - Adapted the hanabi code to be used in the pogosim simulator.
 *  - Updated photo start to use difference in light levels rather than direct threshold

 * This algorithm shows the rapidity of the diffusion of information among the swarm and by which path it propagates.
 * One robot changes autonomously its color and increments its age, this causes a cascade of changes affecting the robots with lower age.
 * If the probability to change the color autonomously (den_p_change_led_color) is too high, it is possible to get 2 distant robots autonomous changes
 * and observe partitions with groups of same age.
**/

// Main include for pogobots, both for real robots and for simulations
#include "pogobase.h"

#include "time.h"

#define CODENAME "HANABI"

#define INFRARED_POWER 2    // 1,2,3

#define FQCY 60             // control update frequency. 30Hz | 60 Hz | 90 Hz | etc.
#define MAX_NB_OF_MSG 3     // max. number of messages per step which this robot can process
#define PERCENT_MSG_SENT 50 // Percent of messages sent per tick

#define SEND_MODE_ALLDIRECTION true // true: all direction at once; false: 4x one-direction
#define MSG_MODE_FULL_HEADER true   // true: full header; false: short header

#define DEBUG_LEVEL 0           // 0: nothing; 1: debug; 2: synchronzation; 3: communication

#define BOOT_TIME 5             // Waiting time before the start of the experience in seconds
#define ENABLE_PHOTO_START      // Whether to enable photo start, i.e. wait at the beginning of a experiment for a large instantaneous difference in light level
                                //  --> allow robots to start the experiment all at the same time by just adjusting quickly the light level in the experimental setup.
                                //  Comment this macro to disable photo start
#define LIGHT_THRESHOLD 40      // You can tweak this parameter to change the sensitivity to the light changes

// Uncomment to have moving robots
//#define MOVING_ROBOTS


// ********************************************************************************
// * Initialization: Pogobot message structure
// ********************************************************************************

typedef struct RawMessage {
    uint16_t sender_id;
    uint16_t msg_id;
    uint16_t age;
    uint8_t rgb_colors_index;
} RawMessage;

#define MSG_SIZE sizeof(RawMessage) // Number of bytes

typedef union message_template {
    uint8_t msg_array[MSG_SIZE];
    RawMessage msg_values;
} message;

void process_message(message_t* mr);
bool send_message(void);
bool photo_start(void);


// ********************************************************************************
// * Initialization: Pogobot led colors structure
// ********************************************************************************

typedef struct {
    char name[16];
    uint8_t r;
    uint8_t g;
    uint8_t b;
} rgb_color;

#define red        ((rgb_color){ .name = "red", .r = 25, .g = 0, .b = 0 })
#define green      ((rgb_color){ .name = "green", .r = 0, .g = 25, .b = 0 })
#define blue       ((rgb_color){ .name = "blue", .r = 0, .g = 0, .b = 25 })
#define magenta    ((rgb_color){ .name = "magenta", .r = 25, .g = 0, .b = 25 })
#define yellow     ((rgb_color){ .name = "yellow", .r = 12, .g = 6, .b = 0 })
#define cyan       ((rgb_color){ .name = "cyan", .r = 0, .g = 25, .b = 25 })
#define orange     ((rgb_color){ .name = "orange", .r = 25, .g = 6, .b = 0 })
#define purple     ((rgb_color){ .name = "purple", .r = 6, .g = 0, .b = 25 })
#define light_pink ((rgb_color){ .name = "light_pink", .r = 12, .g = 3, .b = 12 })
#define mint_green ((rgb_color){ .name = "mint_green", .r = 6, .g = 25, .b = 6 })
#define white      ((rgb_color){ .name = "white", .r = 25, .g = 25, .b = 25 })


// ********************************************************************************
// * main
// ********************************************************************************

// Const global variables (same values for all robots)
static rgb_color const rgb_colors[] = { red, green, blue, magenta, yellow, cyan, orange, purple, light_pink, mint_green }; // 10 hanabi colors
uint8_t const nb_rgb_colors = sizeof(rgb_colors) / sizeof(rgb_colors[0]);
uint16_t const den_p_change_led_color = 20000; // probability to change led color independently (1/den_p_change_led_color) (p=3000 for 15 robots; p=20000 for 72 robots)


// "Global" variables should be inserted within the USERDATA struct.
// Non-const global variables used by each robot. They will be accessible through the mydata pointer, declared by the macro "REGISTER_USERDATA"
typedef struct {
    uint32_t start_of_experiment_ms;

    uint16_t my_pogobot_id;
    uint16_t age; // number of times this robot has changed color from the start of the algorithm, from 0 to max 65535 (16 bits unsigned)

    uint8_t rgb_colors_index; // index of the rgb_colors array

    // Photo start values;
#ifdef ENABLE_PHOTO_START
    bool started;
    int16_t last_data_b;
    int16_t last_data_fl;
    int16_t last_data_fr;
#endif
} USERDATA;

// Call this macro in the same file (.h or .c) as the declaration of USERDATA
DECLARE_USERDATA(USERDATA);

// Don't forget to call this macro in the main .c file of your project (only once!)
REGISTER_USERDATA(USERDATA);
// Now, members of the USERDATA struct can be accessed through mydata->MEMBER. E.g. mydata->age
//  On real robots, the compiler will automatically optimize the code to access member variables as if they were true globals.


// Called by the pogobot main loop before 'user_step', if there are messages to be processed
void process_message(message_t* mr) {
    // Elaborate robot messages only (avoid controllers messages). NB: this condition works only with long headers
    if (MSG_MODE_FULL_HEADER && mr->header._packet_type != ir_t_user) {
        printf("[I'm Pogobot %d] [RECV] This message is discarded because it didn't come from a Pogobot\n", mydata->my_pogobot_id);
        return;
    }

    // Read the received message and stock it in the msg_from_neighbor structure
    message msg_from_neighbor;
    for (uint16_t i = 0; i != MSG_SIZE; i++)
        msg_from_neighbor.msg_array[i] = mr->payload[i];

    if (msg_from_neighbor.msg_values.age > mydata->age && mydata->rgb_colors_index != msg_from_neighbor.msg_values.rgb_colors_index &&
            mydata->rgb_colors_index < nb_rgb_colors && (msg_from_neighbor.msg_values.age - mydata->age) < 10000) {
        mydata->age = msg_from_neighbor.msg_values.age; // update to the greatest age in the neighborhood
        // pogobot_stopwatch_reset(&timeout_age_watch); // reset of the timer, for age timeout
        if (DEBUG_LEVEL == 1)
            printf("[I'm Pogobot %d] [HANABI] Copying state from neighbor: passing from old color %s to new color %s, copying age to %d  *____*\n",
                    mydata->my_pogobot_id, rgb_colors[mydata->rgb_colors_index].name, rgb_colors[msg_from_neighbor.msg_values.rgb_colors_index].name, mydata->age);
        mydata->rgb_colors_index = msg_from_neighbor.msg_values.rgb_colors_index;
        pogobot_led_setColor(rgb_colors[mydata->rgb_colors_index].r, rgb_colors[mydata->rgb_colors_index].g, rgb_colors[mydata->rgb_colors_index].b);

    } else {
        if (DEBUG_LEVEL == 1)
            printf("[I'm Pogobot %d] [HANABI] NOT Changing color, not incrementing age %d  ç____ç\n", mydata->my_pogobot_id, mydata->age);
    }
}


// ********************************************************************************
// * Send message
// * Each robot transmits 2 information to the neighbors: a color and an iterator.
// * The iterator (age) corresponds to how many times the sender has changed color.
// ********************************************************************************
bool send_message(void) {        // Called by the pogobot main loop before 'user_step'
    uint16_t msg_id;
    uint8_t data[MSG_SIZE]; // message to send, containing uint8_t data
    message msg_from_neighbor;

    msg_id = nb_msgs_sent;

    // Composing a new message to send
    msg_from_neighbor.msg_values.sender_id = mydata->my_pogobot_id;
    msg_from_neighbor.msg_values.msg_id = msg_id;
    msg_from_neighbor.msg_values.age = mydata->age;
    msg_from_neighbor.msg_values.rgb_colors_index = mydata->rgb_colors_index; // value from 0 to nb_rgb_colors-1

    // Convert the message into an uint8_t pointer
    for ( uint16_t i = 0; i != MSG_SIZE; i++ )
        data[i] = msg_from_neighbor.msg_array[i];

    if (SEND_MODE_ALLDIRECTION == true) {
        if (MSG_MODE_FULL_HEADER == true)
            pogobot_infrared_sendLongMessage_omniGen((uint8_t *)(data), MSG_SIZE);
        else
            pogobot_infrared_sendShortMessage_omni((uint8_t *)(data), MSG_SIZE);
    } else {
        for (uint16_t i  = 0; i < 4 ; i++) { // i is one of the 4 possible directions
            if (MSG_MODE_FULL_HEADER == true)
                pogobot_infrared_sendLongMessage_uniSpe(i, (uint8_t *)(data), MSG_SIZE);
            else
                pogobot_infrared_sendShortMessage_uni(i, (uint8_t *)(data), MSG_SIZE);
        }
    }

    if (DEBUG_LEVEL == 3)
        printf("[I'm Pogobot %d] [SEND] Sent message msg_id=%d.\n", mydata->my_pogobot_id, msg_id);

    return true;
}


// Init function. Called once at the beginning of the program (cf 'pogobot_start' call in main())
void user_init(void) {
    srand(pogobot_helper_getRandSeed()); // initialize the random number generator
    pogobot_infrared_set_power(INFRARED_POWER); // set the power level used to send all the next messages

    // Set mydata variables to 0
    memset(mydata, 0, sizeof(*mydata));

    // Set main loop frequency, message sending frequency, message processing frequency
    main_loop_hz = FQCY;
    max_nb_processed_msg_per_tick = MAX_NB_OF_MSG;
    percent_msgs_sent_per_ticks = PERCENT_MSG_SENT;
    // Specify functions to send/transmit messages
    msg_rx_fn = process_message;
    msg_tx_fn = send_message;

    // Set led index to show error codes
    error_codes_led_idx = 3; // Default value, negative values to disable

    mydata->my_pogobot_id = pogobot_helper_getid();
    mydata->start_of_experiment_ms = current_time_milliseconds();

#ifdef ENABLE_PHOTO_START
    mydata->started = false;
    mydata->last_data_b  = pogobot_photosensors_read(0);
    mydata->last_data_fl = pogobot_photosensors_read(1);
    mydata->last_data_fr = pogobot_photosensors_read(2);
#endif

    if (DEBUG_LEVEL) {
        printf("\n");
        printf("[INFO] =-=-=-=-=-=-=-=-=-=-=-=-=-=\n");
        printf("[INFO] =-=- POGOBOT::METADATA -=-=\n");
        printf("[INFO] =-=-=-=-=-=-=-=-=-=-=-=-=-=\n");
        printf("[INFO] CODENAME : %s\n", CODENAME);
        printf("[INFO] INFRARED_POWER         %d\n", INFRARED_POWER);
        printf("[INFO] FQCY                   %d\n", FQCY);
        printf("[INFO] MAX_NB_OF_MSG          %d\n", MAX_NB_OF_MSG);
        printf("[INFO] PERCENT_MSG_SENT       %d\n", PERCENT_MSG_SENT);
        printf("[INFO] SEND_MODE_ALLDIRECTION %d\n", SEND_MODE_ALLDIRECTION);
        printf("[INFO] MSG_MODE_FULL_HEADER   %d\n", MSG_MODE_FULL_HEADER);
        printf("[INFO] DEBUG_LEVEL            %d\n", DEBUG_LEVEL);
        printf("[INFO] POGOBOT_ID             %d\n", mydata->my_pogobot_id);
        printf("[INFO] =-=-=-=-=-=-=-=-=-=-=-=-=-=\n");
    }

    if (DEBUG_LEVEL == 1 && den_p_change_led_color > RAND_MAX)
        printf("Warning: the maximal value allowed for den_p_change_led_color is %d\n", RAND_MAX);
}


#ifdef ENABLE_PHOTO_START
// ********************************************************************************
// * Start-up phase for simultaneous start of the robots when the lights turn off
// ********************************************************************************
bool photo_start(void) {
    if (mydata->started) {
        return true;
    }
    // Set initial led color to red
    pogobot_led_setColor(red.r, red.g, red.b);

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

// Step function. Called continuously at each step of the pogobot main loop
void user_step(void) {
#ifdef ENABLE_PHOTO_START
    if (!photo_start())
        return;
#endif

    // Experiment has started. Wait for some time
    if (current_time_milliseconds() - mydata->start_of_experiment_ms < BOOT_TIME * 1000) {
        pogobot_led_setColor(white.r, white.g, white.b); // set boot led color to white
        return; // Wait
    }

    // ********************************************************************************
    // * Main loop
    // ********************************************************************************
    // A robot changes autonomously its state with a probability of 1/den_p_change_led_color
    if (rand() % den_p_change_led_color <= 1) {
        uint8_t rgb_colors_choice = rand()%nb_rgb_colors;  // random index to choose the next led color
        if (rgb_colors_choice != mydata->rgb_colors_index) {
            mydata->age++;
            // pogobot_stopwatch_reset(&timeout_age_watch); // reset of the timer, for age timeout

            if (DEBUG_LEVEL == 1)
                printf("[I'm Pogobot %d] [HANABI] Changing color from old color %s to new color %s because I'm lucky, age %d  *____*\n",
                        mydata->my_pogobot_id, rgb_colors[mydata->rgb_colors_index].name, rgb_colors[rgb_colors_choice].name, mydata->age);

            mydata->rgb_colors_index = rgb_colors_choice;
            pogobot_led_setColor(rgb_colors[mydata->rgb_colors_index].r, rgb_colors[mydata->rgb_colors_index].g, rgb_colors[mydata->rgb_colors_index].b);
        } else {
            if (DEBUG_LEVEL == 1)
                printf("[I'm Pogobot %d] [HANABI] NOT Changing color because old color %s = new color %s, age %d  ç____ç\n",
                        mydata->my_pogobot_id, rgb_colors[mydata->rgb_colors_index].name, rgb_colors[rgb_colors_choice].name, mydata->age);
        }
    }

    // ********************************************************************************
    // * Motility
    // ********************************************************************************
#ifdef MOVING_ROBOTS
    if ((uint32_t)(current_time_milliseconds() / 10000) % 2 == 0) {
        pogobot_led_setColors(blue.r, blue.g, blue.b, 1);
        pogobot_motor_set(motorL, motorFull);
        pogobot_motor_set(motorR, motorStop);
    } else {
        pogobot_led_setColors(red.r, red.g, red.b, 1);
        pogobot_motor_set(motorL, motorStop);
        pogobot_motor_set(motorR, motorFull);
    }
#else
    // * Robots do not move in this experience.
    // ... NOTHING TO DO ...
#endif
}


#ifdef SIMULATOR
// Function called once by the simulator to specify user-defined data fields to add to the exported data files
void create_data_schema() {
    data_add_column_int32("age");
    data_add_column_int16("rgb_colors_index");
}

// Function called periodically by the simulator each time data is saved (cf config parameter "save_data_period" in seconds)
void export_data() {
    if (mydata->started) { // Only store data after the photostart period
        enable_data_export(); // Enable data export this time
        data_set_value_int32("age", mydata->age);
        data_set_value_int16("rgb_colors_index", mydata->rgb_colors_index);
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
    SET_CALLBACK(callback_create_data_schema, create_data_schema);  // Called once to specify the data format
    SET_CALLBACK(callback_export_data, export_data);                // Called at each configuration-specified period (e.g. every second) on each robot to register exported data
    return 0;
}

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
