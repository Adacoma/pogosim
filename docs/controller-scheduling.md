# Cooperative controller scheduling

Pogosim uses one Boost.Context stackful coroutine per robot, plus one for
`callback_global_step`. All run on the simulator thread, in the established
global/object/robot order. This is cooperative scheduling, not parallel execution
or a real-time operating system.

## Sleeping in controller code

The Pogobot-facing C API is unchanged. A positive `msleep()` suspends execution
at the call site, including nested C or C++ functions and their local variables.
Physics and other robots continue; the sleeping robot's motors keep their last
settings. Execution resumes on the first simulation tick at or after the sleep
deadline, with that robot's `USERDATA` and API globals restored.

For example, a controller can now express a complete movement sequence directly:

```c
void user_step(void) {
    pogobot_motor_set(motorL, motorFull);
    pogobot_motor_set(motorR, motorFull);
    msleep(500); /* Other robots and physics advance while this robot moves. */
    pogobot_motor_set(motorL, motorStop);
    pogobot_motor_set(motorR, motorStop);
    msleep(100);
}
```

The same source uses hardware sleep when compiled for physical robots. No Boost
dependency is introduced into firmware builds.

- Sleep is supported in `user_init`, `user_step`, message receive/transmit
  callbacks, and `callback_global_step`, including nested helper calls.
- `robot_main` registration, global setup, schema creation, data export, click,
  and end callbacks run synchronously. A positive sleep from these callbacks
  reports an error; there is no suspended callback stack for the scheduler to own.
- `msleep(0)` is a no-op, not an explicit yield. Negative durations are rejected.
- Sleep uses simulation time, not wall time. Tick resolution still limits timing
  accuracy; choose `time_step` appropriately. GUI pauses do not consume sleep time.
- A returning controller with `main_loop_hz == 0` runs once per tick. Positive
  frequencies retain the shared C main loop's existing millisecond pacing. A
  pacing sleep does not introduce an additional idle tick after waking.
- Temporal noise remains a per-iteration microsecond overhead; it is not a
  separate scheduling thread or a model of CPU instruction execution time.

An infinite loop containing positive sleeps can cooperate with the simulator.
An infinite loop without sleeps still blocks the entire simulation: there is no
automatic preemption, timeout, or loop instrumentation. If `user_step` never
returns, the shared main loop's message processing and rate limiting after that
callback also cannot run; the controller must perform any needed processing itself.
Mutable globals remain shared across robots: keep per-robot persistent controller
state in `USERDATA`, even though stack-local variables are now independent.

## Initialization, clocks, and termination

All robots register their controllers before global setup. Each initializer then
starts at simulation time zero on its own stack. An initializer that sleeps
finishes during later ticks; its robot cannot start `user_step` before it returns.
Other robots can initialize and run in the meantime. Configuration loading belongs
in the normal configuration callbacks, not in `user_init`.

Stopwatches measure the simulated robot clock, including time spent asleep;
repeated reads at the same time return the same elapsed duration. Timers retain
the firmware's wrapping 32-bit microsecond arithmetic and positive-origin-offset
semantics. A timer expires strictly after its deadline, and
`pogobot_timer_wait_for_expiry()` cooperatively sleeps until then.
`current_time_milliseconds()` derives directly from the simulated clock, so
frequent reads do not discard fractional milliseconds.

At normal termination or `stop_simulation()`, suspended stacks are unwound before
per-robot end callbacks, with the correct robot selected and its data and physics
resources still alive. Code after a pending sleep is not executed merely to finish
a controller. End callbacks can therefore see incomplete initialization if the
simulation ended before an initializer's sleep expired. Flash export remains after
end callbacks and stores no coroutine, timer, or other transient execution state.
Controller exceptions are returned to the scheduler and reported as simulator
errors; other suspended controllers are unwound before teardown.

For C++ controllers, do not mark sleeping functions `noexcept`, disable exception
unwinding, or swallow Boost.Context's cancellation exception in a `catch (...)`:
rethrow it. Do not sleep inside a catch block or while holding a resource another
controller needs (for example, a shared mutex). Destructors used during cancellation
must not sleep or throw.
These constraints follow [Boost.Context's fiber cleanup rules](https://www.boost.org/doc/libs/latest/libs/context/doc/html/context/ff.html).

## Installation and stack budget

Boost.Context is a compiled dependency, not header-only. The installation commands,
example/template Makefiles, container recipes, and CI include it:

- Ubuntu/WSL: `libboost-context-dev`.
- macOS: the existing Homebrew `boost` package.
- MSYS2/UCRT64: the existing Boost package; Makefiles select `-lboost_context-mt`.
- Native MSVC/vcpkg: `boost-context:x64-windows`.

Rebuild Pogosim and controller executables after upgrading. Existing containers
also need rebuilding. Custom Unix Makefiles must link `-lboost_context` and compile
C controller frames with `-fexceptions`; C++ unwinding must remain enabled. MSVC
targets inherit `/EHs /EHc- /GL-` from Pogosim for safe mixed-language unwinding.

Each controller allocates a fixed 128 KiB stack by default, with a guard page and
context overhead. This is additional to `USERDATA` and simulated flash memory.
For unusually deep call stacks or large stack-local arrays, increase the top-level
YAML setting (bytes):

```yaml
coroutine_stack_size: 262144  # 256 KiB per robot and the global-step controller
```

Platform-invalid stack sizes fail during initialization. The guard page detects
stack overflow through an OS fault, not a recoverable simulation error. Reduce
stack size only after measuring controller stack usage. Large-population runtime
and memory scaling has not yet been benchmarked.

## Regression tests

The normal CMake build includes a standalone mixed-C/C++ runtime test and real
headless simulator fixtures. Run them from the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The tests cover nested sleeps, independent initialization, physics during sleep,
robot globals, message callbacks, stopwatch/timer arithmetic, pacing, invalid
sleep/stack configuration, exception propagation, and cancellation during
initialization, normal completion, and early stop. CI runs them on Linux, macOS,
WSL, MSYS2, and MSVC. To build just the library, configure with
`-DBUILD_TESTING=OFF`.
