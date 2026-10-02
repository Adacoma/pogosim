#ifndef POGOSIM_ROBOT_COROUTINE_H
#define POGOSIM_ROBOT_COROUTINE_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

namespace pogosim {

// One stackful controller on the simulator thread. Boost types stay out of
// controller-facing headers, and deadlines are simulation time, not wall time.
class RobotCoroutine {
public:
    static constexpr std::size_t default_stack_size = 128 * 1024;

    RobotCoroutine(std::function<void()> initialize, std::function<void()> step,
                   std::size_t stack_size = default_stack_size);
    ~RobotCoroutine();
    RobotCoroutine(const RobotCoroutine&) = delete;
    RobotCoroutine& operator=(const RobotCoroutine&) = delete;

    // The startup pass may run initialization, but must not start user_step.
    // Initialization that sleeps can finish later during ordinary tick resumes.
    void resume(std::uint64_t now, bool initialization_only = false);
    void sleep_until(std::uint64_t deadline);
    void stop();
    bool active() const;
    bool initialized() const;
    std::uint64_t now() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace pogosim

#endif
