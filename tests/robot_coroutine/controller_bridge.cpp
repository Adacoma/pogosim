#include <stdexcept>
#include <cstdint>
#include <cmath>
#include "pogosim/simulator.h"
#include "pogosim/robot.h"

extern "C" int fixture_nested_c_sleep(void);
extern "C" void fixture_long_c_sleep(void);
extern "C" void fixture_init_c_sleep(void);

namespace {
unsigned cleanups[2] = {};

struct Cleanup {
    PogobotObject* robot = current_robot;
    void* userdata = robot->data;
    ~Cleanup() {
        // Cancellation must run on the correct robot, before global simulation,
        // physics bodies or USERDATA are destroyed. Violations fail the process.
        if (!simulation || current_robot != robot || mydata != userdata ||
            !std::isfinite(robot->get_position().x)) std::terminate();
        ++cleanups[robot->id];
    }
};
}

extern "C" void fixture_require(int condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

extern "C" std::uint64_t fixture_now(void) {
    return current_robot->current_time_microseconds;
}

extern "C" void fixture_motion_and_c_locals(void) {
    auto* robot = current_robot;
    auto* userdata = mydata;
    const auto start = robot->get_position();
    pogobot_motor_set(motorL, motorFull);
    pogobot_motor_set(motorR, motorFull);
    fixture_require(fixture_nested_c_sleep() == 112, "Nested C locals changed during sleep");
    fixture_require(current_robot == robot && mydata == userdata,
                    "Resuming a nested C call selected the wrong robot");
    fixture_require(b2Distance(start, robot->get_position()) > 0.000001f,
                    "Physics did not advance while the controller slept");
    pogobot_motor_set(motorL, motorStop);
    pogobot_motor_set(motorR, motorStop);
}

extern "C" void fixture_suspend_with_cleanup(void) {
    Cleanup cleanup;
    fixture_long_c_sleep();
}

extern "C" void fixture_init_with_cleanup(void) {
    Cleanup cleanup;
    fixture_init_c_sleep();
}

extern "C" void fixture_throw(void) {
    Cleanup cleanup;
    throw std::runtime_error("fixture controller exception");
}

extern "C" unsigned fixture_cleanup_count(std::uint16_t id) { return cleanups[id]; }

extern "C" void fixture_queue_message(void) {
    current_robot->messages.push(message_t{});
}

extern "C" void fixture_timer_wrap(void) {
    // Exercise wrapping clock arithmetic without running a 71-minute simulation.
    auto& clock = current_robot->current_time_microseconds;
    const auto saved = clock;
    clock = UINT32_MAX - 5ull;
    time_reference_t timer;
    pogobot_timer_init(&timer, 10);
    clock += 11;
    fixture_require(pogobot_timer_get_remaining_microseconds(&timer) == -1,
                    "Timer failed across the 32-bit clock wrap");
    pogobot_stopwatch_reset(&timer);
    clock = 1000;
    clock += 700;
    const auto first = current_time_milliseconds();
    clock += 700;
    fixture_require(current_time_milliseconds() > first,
                    "Frequent millisecond reads lost fractional time");
    clock = saved;
}
