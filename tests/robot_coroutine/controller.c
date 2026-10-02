#include "pogobase.h"
#include <string.h>

/* Shared counters are intentional test instrumentation, not controller state. */
static uint32_t test_mode = 0;
static unsigned global_steps = 0;
static unsigned transmit_calls = 0;
static unsigned receive_calls = 0;

typedef struct {
    uint16_t id;
    int initialized;
    unsigned steps;
    int nested_result;
    int timers_checked;
    time_reference_t lifetime;
} USERDATA;
DECLARE_USERDATA(USERDATA);
REGISTER_USERDATA(USERDATA);

/* The C++ bridge checks exceptions, RAII and physics across nested C frames. */
void fixture_require(int condition, const char *message);
uint64_t fixture_now(void);
void fixture_motion_and_c_locals(void);
void fixture_suspend_with_cleanup(void);
void fixture_init_with_cleanup(void);
void fixture_throw(void);
void fixture_queue_message(void);
void fixture_timer_wrap(void);
unsigned fixture_cleanup_count(uint16_t id);

static void global_setup(void) {
    init_from_configuration(test_mode);
    if (test_mode == 2) msleep(1); /* Must reject sleep outside a controller. */
}

static void global_step(void) {
    if (test_mode == 10) fixture_throw();
    main_loop_hz = 0;
    ++global_steps;
    msleep(3);
}

void user_init(void) {
    memset(mydata, 0, sizeof(*mydata));
    mydata->id = pogobot_helper_getid();
    pogobot_stopwatch_reset(&mydata->lifetime);
    main_loop_hz = mydata->id == 0 ? 0 : 100;
    if (test_mode == 5) fixture_throw();
    if (mydata->id == 0) {
        USERDATA *saved = mydata;
        fixture_init_with_cleanup();
        fixture_require(mydata == saved && main_loop_hz == 0,
                        "Initialization did not restore robot C globals");
        fixture_require(fixture_now() >= 5000, "Initialization sleep returned early");
    } else {
        fixture_require(fixture_now() == 0, "Sleeping initialization blocked another robot");
        fixture_queue_message();
    }
    mydata->initialized = 1;
}

void fixture_init_c_sleep(void) { msleep(5); }

/* Ordinary C locals and stack-local stopwatches must survive a nested sleep. */
static int inner_sleep(int value) {
    time_reference_t local;
    pogobot_stopwatch_reset(&local);
    int saved = value + 7;
    msleep(10);
    int32_t elapsed = pogobot_stopwatch_get_elapsed_microseconds(&local);
    fixture_require(elapsed >= 10000 && elapsed < 11000, "Nested sleep duration mismatch");
    fixture_require(pogobot_stopwatch_get_elapsed_microseconds(&local) == elapsed,
                    "Reading a stopwatch accumulated time twice");
    return saved + value;
}

int fixture_nested_c_sleep(void) {
    int saved = 35;
    return saved + inner_sleep(saved);
}

void fixture_long_c_sleep(void) {
    if (test_mode == 8) stop_simulation();
    msleep(1000);
    fixture_require(0, "Cancelled controller continued after its sleep");
}

static bool transmit(void) {
    ++transmit_calls;
    if (transmit_calls == 1) msleep(1);
    fixture_require(mydata->id == 1 && main_loop_hz == 100,
                    "Transmit callback lost robot C globals");
    return true;
}

static void receive(message_t *message) {
    (void)message;
    ++receive_calls;
    msleep(2);
    fixture_require(mydata->id == 1 && main_loop_hz == 100,
                    "Receive callback lost robot C globals");
}

void user_step(void) {
    fixture_require(mydata->initialized, "user_step ran before initialization finished");
    fixture_require(mydata->id == pogobot_helper_getid(), "USERDATA changed robots");
    ++mydata->steps;
    if (test_mode == 3) fixture_throw();
    if (test_mode == 4) msleep(-1);
    if (mydata->id != 0) return;
    if (mydata->steps > 1) {
        fixture_suspend_with_cleanup();
        return;
    }
    uint64_t now = fixture_now();
    msleep(0);
    fixture_require(fixture_now() == now, "Zero sleep advanced simulation time");
    fixture_motion_and_c_locals();
    mydata->nested_result = 112;

    time_reference_t timer;
    pogobot_timer_init(&timer, 2000);
    fixture_require(pogobot_timer_get_remaining_microseconds(&timer) == 2000,
                    "Timer remaining time has the wrong sign");
    msleep(2);
    fixture_require(pogobot_timer_get_remaining_microseconds(&timer) == 0 &&
                    !pogobot_timer_has_expired(&timer), "Timer expired at its origin");
    pogobot_timer_wait_for_expiry(&timer);
    fixture_require(pogobot_timer_has_expired(&timer), "Timer wait returned before expiry");
    pogobot_stopwatch_reset(&timer);
    pogobot_stopwatch_offset_origin_microseconds(&timer, -123);
    fixture_require(pogobot_stopwatch_get_elapsed_microseconds(&timer) == 123,
                    "Negative stopwatch offset was not preserved");
    fixture_timer_wrap();
    mydata->timers_checked = 1;
}

static void export_data(void) {
    if (test_mode == 6) msleep(1);
}

static void robot_end(void) {
    if (test_mode == 1) msleep(1);
    if (test_mode == 9) {
        // Zero-duration simulations must also cancel a sleeping initializer.
        fixture_require(mydata->steps == 0 && fixture_now() == 0,
                        "Zero-duration simulation ran a step or advanced time");
        fixture_require(mydata->id == 0 ? !mydata->initialized : mydata->initialized,
                        "Initializer continued after simulation termination");
        if (mydata->id == 0) fixture_require(fixture_cleanup_count(0) == 1,
                                           "Initializer's C++ local was not unwound");
        return;
    }
    if (test_mode == 8) {
        fixture_require(current_time_milliseconds() < 90, "stop_simulation was ignored");
        if (mydata->id == 0) fixture_require(fixture_cleanup_count(0) == 2,
                                           "Early stop did not unwind its controller");
        return;
    }
    fixture_require(test_mode == 0, "Unexpected error-mode end callback");
    fixture_require(current_time_milliseconds() >= 90, "Final clock was not synchronized");
    fixture_require(pogobot_stopwatch_get_elapsed_microseconds(&mydata->lifetime) >= 90000,
                    "A sleeping robot's stopwatch stopped advancing");
    fixture_require(global_steps >= 29 && global_steps <= 31, "Global callback pacing mismatch");
    if (mydata->id == 0) {
        fixture_require(mydata->steps == 2 && mydata->nested_result == 112 && mydata->timers_checked,
                        "Nested controller did not finish its checks");
        fixture_require(fixture_cleanup_count(mydata->id) == 2,
                        "Suspended C++ local was not destroyed before robot_end");
    } else {
        fixture_require(mydata->steps == 9, "Main-loop pacing acquired an extra physics tick");
        fixture_require(transmit_calls == 9 && receive_calls == 1,
                        "Message callbacks were not resumed correctly");
    }
}

int main(void) {
    pogobot_init();
    pogobot_start(user_init, user_step);
    /* Configuration initialization belongs in global_setup, never user_init. */
    SET_CALLBACK(callback_global_setup, global_setup);
    SET_CALLBACK(callback_global_step, global_step);
    SET_CALLBACK(callback_robot_end, robot_end);
    SET_CALLBACK(callback_export_data, export_data);
    if (pogobot_helper_getid() == 1) {
        msg_tx_fn = transmit;
        msg_rx_fn = receive;
        percent_msgs_sent_per_ticks = 100;
    }
    return 0;
}
