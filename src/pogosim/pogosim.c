
#include <math.h>
#include "pogosim.h"
#include "colormaps.h"

#ifdef SIMULATOR
void (*callback_create_data_schema)(void) = NULL;
void (*callback_export_data)(void) = NULL;
#endif

uint8_t main_loop_hz = 60;
uint8_t max_nb_processed_msg_per_tick = 3;
void (*msg_rx_fn)(message_t *) = NULL;
bool (*msg_tx_fn)(void) = NULL;
int8_t error_codes_led_idx = 3; // Default value, negative values to disable
uint32_t pogobot_ticks = 0;
uint32_t _current_time_milliseconds = 0LL;
uint32_t _error_code_initial_time = 0LL;
time_reference_t _global_timer;
time_reference_t timer_main_loop;

uint8_t percent_msgs_sent_per_ticks = 20;
uint32_t nb_msgs_sent = 0;
uint32_t nb_msgs_recv = 0;


#ifndef SIMULATOR // Compiling for real robots
void pogobot_start(void (*user_init)(void), void (*user_step)(void)) {
    pogobot_ticks = 0;
    pogobot_stopwatch_reset(&_global_timer);
    pogobot_stopwatch_reset(&timer_main_loop);

    // Launch user-specified init function
    user_init();
    // Main loop
    for (;;) {
        pogo_main_loop_step(user_step);
    }
}
#endif

uint32_t current_time_milliseconds(void) {
    _current_time_milliseconds += pogobot_stopwatch_get_elapsed_microseconds(&_global_timer) / 1000;
    pogobot_stopwatch_reset(&_global_timer);
    return _current_time_milliseconds;
}

void display_led_error_code(error_code_t const c) {
    // Check if led error codes are enabled
    if (error_codes_led_idx < 0)
        return;

    // Find color from error code using the qualitative color map
    uint8_t r, g, b;
    qualitative_colormap(c, &r, &g, &b);
    pogobot_led_setColors(r, g, b, error_codes_led_idx);
    _error_code_initial_time = current_time_milliseconds();
}

void pogo_main_loop_step(void (*user_step)(void)) {
    // Update millisecond clock
    pogobot_stopwatch_reset(&timer_main_loop);

    // Check if we received a system message (e.g. rc_stop)
    // TODO

    // Messages I/O (send & receive)
    if (msg_tx_fn) {
        if (rand() % 100 < percent_msgs_sent_per_ticks) {
            bool const success_tx = msg_tx_fn();
            if (success_tx)
                ++nb_msgs_sent;
        }
    }
    if (msg_rx_fn) {
        pogobot_infrared_update(); // infrared checks for received data. Then, messages are decoded and insered in a FIFO.
        // Identify number of messages to handle, depending on the message processing frequency
        for (uint8_t i = 0; i < max_nb_processed_msg_per_tick; i++) {
            if (pogobot_infrared_message_available()) { // read FIFO buffer - any message(s)?
                // Recover the next message inside the message queue and stock it in the "mr" message_t structure, then in the "msg_from_neighbor" structure.  
                message_t mr;
                pogobot_infrared_recover_next_message(&mr);
                msg_rx_fn(&mr);
                ++nb_msgs_recv;
            }
        }
    }

    // Call user-specified step function
    user_step();

    if (main_loop_hz > 0) {
        // Detect and handle time overflows
        uint64_t const elapsed_µs = pogobot_stopwatch_get_elapsed_microseconds(&timer_main_loop);
        uint64_t const step_max_duration = 1000000LL/main_loop_hz;
        if (elapsed_µs > step_max_duration) {
            printf("[TIME] Error code: %u: TIME OVERFLOW ERROR! Step took %llu µs, should be less than %llu µs.", ERROR_TIME_OVERFLOW, elapsed_µs, step_max_duration);
            display_led_error_code(ERROR_TIME_OVERFLOW);
        }

        // Sleep to align temporally to the next main loop step
        if (elapsed_µs < step_max_duration) {
            uint32_t const ms_to_sleep = (step_max_duration - elapsed_µs) / 1000;
            msleep(ms_to_sleep); // Wait for next step
        }
    }

    // Check if the error code has expired. When 'display_led_error_code' is called, the error code led is lighted only for 1 seccond.
    if (error_codes_led_idx >= 0 && _error_code_initial_time > 0 && current_time_milliseconds() >= _error_code_initial_time + 1000) {
        pogobot_led_setColors(0, 0, 0, error_codes_led_idx);
    }

    // Update pogobot_ticks
    pogobot_ticks++;
}

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
