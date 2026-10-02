#include "pogosim/robot_coroutine.h"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" int coroutine_nested_c(void (*)(void *, uint64_t), void *);

namespace {
using pogosim::RobotCoroutine;

void require(bool condition, const char* message) {
    // assert() would disappear in the Release configurations used by CI.
    if (!condition) throw std::runtime_error(message);
}

void sleep_from_c(void* context, uint64_t deadline) {
    static_cast<RobotCoroutine*>(context)->sleep_until(deadline);
}

struct Cleanup {
    int& count;
    ~Cleanup() { ++count; }
};

void run_tests() {
    int initialized = 0;
    int completed = 0;
    int c_result = 0;
    RobotCoroutine* controller = nullptr;
    RobotCoroutine nested([&] { ++initialized; }, [&] {
        c_result = coroutine_nested_c(sleep_from_c, controller);
        ++completed;
    });
    controller = &nested;
    nested.resume(0, true);
    require(initialized == 1 && completed == 0, "Startup ran user_step");
    nested.resume(0);
    require(!nested.ready(4999) && nested.ready(5000), "Wakeup readiness did not match the deadline");
    nested.resume(4999);
    require(completed == 0, "C sleep returned before its deadline");
    nested.resume(5000);
    require(c_result == 112 && completed >= 1, "Nested C locals were not preserved");
    nested.stop();
    require(!nested.ready(5000), "Stopped controller remained ready");

    std::vector<int> events;
    RobotCoroutine* first_ptr = nullptr;
    RobotCoroutine first([&] {
        events.push_back(1);
        first_ptr->sleep_until(3000);
        events.push_back(3);
    }, [&] { events.push_back(4); });
    first_ptr = &first;
    RobotCoroutine second({}, [&] { events.push_back(2); });
    first.resume(0, true);
    second.resume(0, true);
    second.resume(0);
    first.resume(2999);
    require(events == std::vector<int>({1, 2}), "Sleeping initialization blocked another robot");
    first.resume(3000);
    require(first.initialized() && events == std::vector<int>({1, 2, 3, 4}),
            "Initialization did not complete on its wake tick");

    int iterations = 0;
    RobotCoroutine no_sleep({}, [&] { ++iterations; });
    no_sleep.resume(0, true);
    for (uint64_t t = 0; t < 10000; t += 1000) no_sleep.resume(t);
    require(iterations == 10, "No-sleep controller did not yield once per tick");

    int starts = 0;
    RobotCoroutine* paced_ptr = nullptr;
    RobotCoroutine paced({}, [&] {
        ++starts;
        paced_ptr->sleep_until(paced_ptr->now() + 1000);
    });
    paced_ptr = &paced;
    paced.resume(0, true);
    for (uint64_t t = 0; t <= 5000; t += 1000) paced.resume(t);
    require(starts == 6, "Rate-limiter sleep acquired an extra tick of delay");

    int cleanups = 0;
    RobotCoroutine* suspended_ptr = nullptr;
    RobotCoroutine suspended({}, [&] {
        Cleanup cleanup{cleanups};
        suspended_ptr->sleep_until(1000000);
        throw std::runtime_error("Cancelled code unexpectedly continued");
    });
    suspended_ptr = &suspended;
    suspended.resume(0);
    require(cleanups == 0, "Yield destroyed a live C++ local");
    suspended.stop();
    suspended.stop();
    require(cleanups == 1, "Stopping did not unwind exactly once");

    RobotCoroutine failing({}, [] { throw std::runtime_error("controller failure"); });
    bool caught = false;
    try { failing.resume(0); }
    catch (const std::runtime_error& error) { caught = std::string(error.what()) == "controller failure"; }
    require(caught && !failing.active(), "Controller exception was not returned to scheduler");

    caught = false;
    try { no_sleep.sleep_until(20000); }
    catch (const std::logic_error&) { caught = true; }
    require(caught, "Sleep outside a coroutine was accepted");
    caught = false;
    try { RobotCoroutine invalid({}, {}, 0); }
    catch (const std::invalid_argument&) { caught = true; }
    require(caught, "Invalid stack size was accepted");
    caught = false;
    try { RobotCoroutine invalid({}, {}, std::numeric_limits<std::size_t>::max()); }
    catch (const std::invalid_argument&) { caught = true; }
    require(caught, "Overflowing guarded stack size was accepted");
    caught = false;
    try { no_sleep.resume(0); }
    catch (const std::invalid_argument&) { caught = true; }
    require(caught, "Backwards simulation time was accepted");
}
} // namespace

int main() {
    try {
        run_tests();
        std::cout << "Robot coroutine runtime tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
