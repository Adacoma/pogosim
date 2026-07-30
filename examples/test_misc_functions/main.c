// main.c — test several miscellaneous functions of the Pogobot API

// Main include for pogobots, both for real robots and for simulations
#include "pogobase.h"

// "Global" variables should be inserted within the USERDATA struct.
// /!\  In simulation, don't declare non-const global variables outside this struct, elsewise they will be shared among all agents (and this is not realistic).
typedef struct {
    // Put all global variables you want here.
    uint8_t data_foo[8];
    time_reference_t timer_it;
} USERDATA;

// Call this macro in the same file (.h or .c) as the declaration of USERDATA
DECLARE_USERDATA(USERDATA);

// Don't forget to call this macro in the main .c file of your project (only once!)
REGISTER_USERDATA(USERDATA);
// Now, members of the USERDATA struct can be accessed through mydata->MEMBER. E.g. mydata->data_foo
//  On real robots, the compiler will automatically optimize the code to access member variables as if they were true globals.


uint8_t data[] = "hello";

// Called by the pogobot main loop before 'user_step', if there are messages to be processed
static void process_message(message_t *mr) {
    // ...
}

// Called by the pogobot main loop before 'user_step'. Used to send IR messages to the neighborhood.
static bool send_message(void) {
    pogobot_infrared_sendLongMessage_omniGen(data, 6);
    return true;
}


/* Erase 64kB in flash in RW authorized space */
/* Write and Read in each page (256 bytes) */
static void test_flash_write_read(void) {
    int cmp = 0;

    /* One page */
    char vals_in[256];
    for (int i = 0; i<256; i++){
        vals_in[i] = i;
    }

    char vals_out[256];
    erase_write_section_flash();

    for (int k = 0; k < 4; k++) {
        pogobot_led_setColor(255,0,0);
        write_page_flash(k, vals_in);

        pogobot_led_setColor(0,255,0);
        read_page_flash(k, vals_out);

        pogobot_led_setColor(0,0,255);
        printf("Read page %d : ", k);
        cmp = 0;
        for (int i = 0; i<256;i++) {
            cmp += !(vals_out[i]==vals_in[i]);
        }
        if (cmp==0) {
            printf("OK \n");
        } else {
            printf("NOK \n");  
        }

        erase_write_section_flash();
        read_page_flash(k, vals_out);
        printf("Read page %d after erasing : ", k);
        cmp = 0;
        for (int i = 0; i<256;i++) {
            cmp += !((uint8_t)vals_out[i]==255);
        }
        if (cmp==0) {
            printf("OK \n");
        } else {
            printf("NOK \n");  
        }
    }
}


// Init function. Called once at the beginning of the program (cf 'pogobot_start' call in main())
void user_init(void) {
#ifndef SIMULATOR
    printf("setup ok\n");
#endif

    // Init timer
    pogobot_stopwatch_reset(&mydata->timer_it);

    // Set main loop frequency, message sending frequency, message processing frequency
    main_loop_hz = 1;      // Call the 'user_step' function only 1 time per second
    max_nb_processed_msg_per_tick = 10;
    // Specify functions to send/transmit messages. See the "blooming" example to see message sending/processing in action!
    percent_msgs_sent_per_ticks = 30;
    msg_rx_fn = process_message;
    msg_tx_fn = send_message;

    // Set led index to show error codes (e.g. time overflows)
    error_codes_led_idx = 3; // Default value, negative values to disable

    // Internal function to reset IR flags. May be useful if IR sensors are stuck.
    IR_reset_interrupt_flags();

    // Test writing and reading parts of the flash
    test_flash_write_read();
}


// Step function. Called continuously at each step of the pogobot main loop
void user_step(void) {
    mydata->data_foo[0] = 42;

    bool verbose = (pogobot_ticks % 1000 == 0);
    if (is_muted()) {
        pogobot_led_setColor(255,0,0);
        if (verbose) printf("Mute !\n");
    } else {
        pogobot_led_setColor(0,255,0);
        if (verbose) printf("Unmute !\n");
    }


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
