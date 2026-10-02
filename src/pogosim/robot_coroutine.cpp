#include "robot_coroutine.h"

#include <boost/context/fiber.hpp>
#include <boost/context/protected_fixedsize_stack.hpp>
#include <boost/context/detail/exception.hpp>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>

namespace pogosim {

struct RobotCoroutine::Impl {
    // Keep the scheduler handle alive until the worker has been unwound.
    boost::context::fiber scheduler;
    boost::context::fiber worker;
    std::exception_ptr failure;
    std::uint64_t now = 0;
    std::uint64_t wakeup = 0;
    bool active = false;
    bool initialized = false;
    bool initialization_only = false;

    Impl(std::function<void()> initialize, std::function<void()> step,
         std::size_t stack_size) {
        using traits = boost::context::stack_traits;
        // Protected stacks round up to pages and add a guard page. Reject
        // sizes that would overflow that allocation on unbounded platforms.
        // Parentheses prevent Windows headers' function-like max macro expanding.
        if (stack_size > (std::numeric_limits<std::size_t>::max)() - 2 * traits::page_size() ||
            stack_size < traits::minimum_size() ||
            (!traits::is_unbounded() && stack_size > traits::maximum_size())) {
            throw std::invalid_argument("Invalid coroutine_stack_size for this platform");
        }

        worker = boost::context::fiber(
            std::allocator_arg, boost::context::protected_fixedsize_stack(stack_size),
            [this, initialize = std::move(initialize), step = std::move(step)]
            (boost::context::fiber&& caller) {
                scheduler = std::move(caller);
                try {
                    if (initialize) initialize();
                    initialized = true;
                    if (initialization_only) yield();
                    for (;;) {
                        const auto started = now;
                        if (step) step();
                        // Always give no-sleep controllers a tick boundary.
                        // A sleeping iteration may finish and begin the next
                        // one on its wake tick, avoiding a second physics dt
                        // after pogo_main_loop_step's own rate-limiter sleep.
                        if (now == started) yield();
                    }
                } catch (const boost::context::detail::forced_unwind&) {
                    // Destroying a suspended fiber must unwind C++ locals.
                    throw;
                } catch (...) {
                    // Exceptions cannot escape a context entry function.
                    failure = std::current_exception();
                }
                return std::move(scheduler);
            });
    }

    void yield() {
        scheduler = std::move(scheduler).resume();
    }
};

RobotCoroutine::RobotCoroutine(std::function<void()> initialize,
                               std::function<void()> step, std::size_t stack_size)
    : impl_(std::make_unique<Impl>(std::move(initialize), std::move(step), stack_size)) {}

RobotCoroutine::~RobotCoroutine() = default;

void RobotCoroutine::resume(std::uint64_t now, bool initialization_only) {
    if (impl_->active) throw std::logic_error("Cannot resume an active robot coroutine");
    if (now < impl_->now) throw std::invalid_argument("Coroutine simulation time moved backwards");
    impl_->now = now;
    impl_->initialization_only = initialization_only;
    if (now >= impl_->wakeup && impl_->worker) {
        impl_->active = true;
        impl_->worker = std::move(impl_->worker).resume();
        impl_->active = false;
    }
    if (impl_->failure) std::rethrow_exception(impl_->failure);
}

void RobotCoroutine::sleep_until(std::uint64_t deadline) {
    if (!impl_->active) {
        throw std::logic_error("Sleep requires an active robot controller coroutine");
    }
    if (deadline <= impl_->now) return;
    impl_->wakeup = deadline;
    impl_->yield();
}

void RobotCoroutine::stop() {
    if (impl_->active) throw std::logic_error("Cannot stop an active robot coroutine");
    impl_->worker = {};
}

bool RobotCoroutine::active() const { return impl_->active; }
bool RobotCoroutine::ready(std::uint64_t now) const {
    return impl_->worker && now >= impl_->wakeup;
}
bool RobotCoroutine::initialized() const { return impl_->initialized; }
std::uint64_t RobotCoroutine::now() const { return impl_->now; }

} // namespace pogosim
