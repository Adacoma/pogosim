# Current project state

Last updated: 2026-10-02

## Inspected

- Top-level documentation, build scripts, package metadata, CI, tracked repository layout, and working-tree status.
- Core configuration, simulator lifecycle, object hierarchy and factories, robot state switching, physics scheduling, communication, sensing, geometry, rendering, and Arrow data logging.
- Controller template and the available example/configuration families.
- Python batch, optimization, arena, locomotion, coverage, neighbor, network, and active-matter tools at an architectural level.

## Understood

- Pogosim is a two-dimensional behavioral and physical simulator for Pogobot swarm controllers. Its central goal is to let one C controller source target both simulation and physical robots through a compatibility API.
- A simulated controller's `main` is renamed to `robot_main`. The simulator invokes it for each robot, allocates separate `USERDATA`, and swaps per-robot API state through `set_current_robot` before controller calls.
- Robot initialization and main-loop calls now run on independent guarded Boost.Context stacks. Positive sleeps suspend at the call site until physics time reaches their deadline; synchronous lifecycle/export/UI callbacks cannot sleep.
- YAML defines the arena, boundary conditions, initial formation, objects, physics, sensing, communication, timing, rendering, and output.
- Each simulation tick runs global/object/robot controllers, advances Box2D, applies periodic wrapping when selected, recomputes directional neighbors, logs data, and optionally renders.
- The object model covers Pogobots, Pogobjects, Pogowalls, flexible membranes, active/passive objects, and static or time-varying lights. Communication supports directional range, optional occlusion, and static or density-dependent reception probability.
- Results are buffered into compressed Arrow/Feather files with configuration, arena, and version metadata. Pogobatch expands parameter choices and seeds, runs tasks locally or through Rundra, records manifests, and merges task outputs.
- Pogoptim now reuses Pogobatch's public local-campaign API for Random Search, CMA-ES, and MAP-Elites, with YAML/CLI precedence and recorded evaluation provenance.
- Optional flash-state archives now carry each robot's 1,472 KiB v3 user flash plus motor direction/power calibration memories between simulator invocations, without checkpointing transient or physical state.
- CI combines Linux, macOS, WSL2, MSYS2/MinGW-w64, and native MSVC builds with headless simulator smoke runs and focused Python regression tests for Pogobatch and Pogoptim.

## Remains unknown

- Quantitative agreement between each simulated physical/sensor/communication model and measurements from real Pogobots.
- Performance and memory scaling across robot counts, arena complexity, logging rates, and GUI/headless execution.
- Which untracked configurations and example directories are active research work intended for integration.
- Compatibility of optional analysis tools other than Pogoptim with current dependency releases.
- Whether every example remains suitable for physical firmware as well as simulation, especially examples using optional `pogo-utils` functionality.
- Earlier MSYS2/MinGW-w64 and PowerShell/MSVC library builds passed on hosted Windows. The new coroutine integration and CMake simulator fixtures still need hosted macOS/Windows verification; standard example Makefiles remain GNU-specific.

## Working analyses

- Boost.Context scheduling and simulated clocks are integrated. Fourteen runtime/simulator tests pass locally in Release and Debug with undefined-behavior checks, including mixed-language sleeps, physics, callbacks, timers, pacing, exceptions, cancellation, legacy linking, and periodic wall suppression. C and C++ example smoke runs, eight flash-state smoke configurations, and all 17 Python regression tests pass; hosted cross-platform and full container rebuilds remain unverified.
- CMake supports both yaml-cpp's legacy and namespaced imported targets; all 14 tests pass against installed yaml-cpp 0.7 and 0.9 packages. Static Pogosim archives bundle Context members to preserve existing external Makefile link flags; a copied pre-coroutine Makefile built and ran against a temporary installation without edits. GNU/LLVM archive merging also passed paths-with-spaces checks; native Apple/MSVC verification remains with CI.
- Periodic boundaries now return immediately when skipping Pogowalls, avoiding a pre-existing null dereference during object creation. The regression covers both implicit and explicit wall coordinates; short 50-robot Toner–Tu runs passed headless/UBSan and the GUI code path with SDL's dummy display, with walls still declared and outputs disabled in temporary configs.
- No analysis is currently running as part of this milestone.
- The Pogoptim/Pogobatch migration has passed focused unit tests, optional CMA-ES/QDpy smoke tests, a nested-worker fake-simulator campaign, and one-evaluation Random/MAP-Elites runs against the built `run_and_tumble` controller; longer scientific runs have not been benchmarked.
- Flash persistence has been implemented with pre-controller restore, post-callback atomic export, strict robot identity validation, and task-local Pogobatch outputs; round-trip and rejection checks cover the archive boundary.
- Flash import now accepts an archive with extra robot identities while requiring every simulated robot to match. Export from an imported archive retains unused records, including same-path replacement; a five-to-two-to-five simulator sequence passed locally and is covered by the native Windows smoke test.
- A dedicated `test_flash_state` example now provides a reproducible two-run simulator smoke test and a two-boot real-robot test while preserving hardware motor calibration values.
- The simulated flash API now matches `pogobot/Software`: 5,888 pages of 256 bytes, `uint16_t` page indices, and non-mutating rejection of out-of-range pages. The archive stores the full region; old 64 KiB archives are rejected. The simulator example checks the final page and bounds behavior.
- A missing configured flash-state input now creates a valid zero-record archive by default, leaving fresh flash indeterminate; `create_if_missing: false` retains strict missing-file failure, and same-path export replaces the empty marker with a full archive.
- The Windows jobs now distinguish MSYS2/UCRT64 with MinGW-w64 from a native PowerShell/MSVC build. Both use the same pinned Box2D 3.x revision; the MSYS2 job retains the simulator flash-state smoke test and native-CPython suite, while the MSVC job initializes the latest installed `cl.exe`, builds through Ninja, exports Box2D's library and public headers through one explicit install prefix, and validates the CMake library artifact.
- All six Apptainer/Singularity recipes now preinstall `packaging==24.2` without uninstalling Ubuntu's apt-owned copy before LiteX setup; a full image rebuild has not yet been rerun locally.
- CMake resolves GNU make explicitly for optional example targets, so generated Ninja builds no longer contain Make-only `$(MAKE)` syntax.
- MSVC builds define `_USE_MATH_DEFINES` at the target level so existing public headers and sources can use the standard math constants consistently.
- The README documents the validated PowerShell/MSVC, Ninja, vcpkg, and pinned Box2D library installation path without requiring MSYS2 or WSL.
- The README keeps WSL, MacOSX, native Windows, individual troubleshooting instructions, and the AI-agent guidance in collapsed `<details>` blocks while retaining section headings for navigation.
- A worked run-and-tumble/Pogobatch/Rundra tutorial now includes a 100-robot, two-condition, 128-seed batch configuration, a merged-Feather MSD analysis script, and a link to the existing demonstration video. A one-seed local simulation and analysis completed; no 256-Task cluster campaign was submitted for this documentation change.
- The worktree contains untracked quadrant-motility and magnetometer-related examples/configurations; their scientific status has not been assessed.

## Current scientific decisions

- Model motion and collisions in 2D with zero-gravity Box2D dynamics.
- Support both solid and periodic boundary conditions; displacement analyses must account for the selected topology.
- Treat robot communication as directional and probabilistic, with optional line-of-sight occlusion and a configurable reception-success model.
- Represent photosensor, timing, locomotion, and magnetometer variability explicitly through configurable bias/noise models.
- Use explicit seeds, effective YAML configurations, and source/build identity as the basis of reproducible stochastic experiments.
- Give each optimization candidate a deterministic, disjoint simulation-seed block; do not reuse common random numbers across candidates.
- Require explicit descriptor domains for custom MAP-Elites features and reject out-of-domain values rather than clipping them.
- Preserve simulation/hardware controller parity by storing mutable per-robot controller state in `USERDATA`.
- Use cooperative same-thread scheduling with simulation-time wakeups, not preemption. Cancel suspended stacks before end callbacks and resource teardown; keep hardware sleep and firmware build dependencies unchanged.
- Treat only the user flash section and motor calibration memories as persistent robot state; fresh user flash remains indeterminate unless an archive is loaded.
- Keep archive loading strict after the v3 flash expansion; do not fabricate newly exposed pages for obsolete 64 KiB archives.
- Permit smaller follow-up robot populations to restore their identity-matched flash records; preserve unused source records on export rather than silently shrinking a shared archive.
- Use Feather output with embedded provenance and restrict logged fields/categories for large experiments.
- For the run-and-tumble tutorial, compute origin-relative MSD per robot, average robots within each seed first, and describe uncertainty across independent seed-level curves.

## Known data limitations

- Logged timestamps lie on simulator ticks and may not exactly match the requested logging period or final simulation time.
- Long-time displacement in solid arenas is bounded; periodic trajectories require coordinate unwrapping before conventional unbounded-space MSD analysis.
- Changing robot count at fixed arena area also changes density, collision frequency, and communication connectivity.
- Custom controller fields have experiment-specific units and semantics unless documented alongside their configuration.
- Empirical calibration/validation datasets were not inspected, so model fidelity must not be inferred solely from API coverage.
- Generated results in local ignored directories are not canonical repository data and were not used to establish scientific conclusions.
- Pogoptim is local-only, has no resume/checkpoint workflow, and relies on the optional QDpy package for MAP-Elites.
- Flash archives validate simulator-level structure and robot identity but cannot determine whether controller-defined byte layouts are application-compatible.
- The expanded flash consumes 1,472 KiB of RAM and archive space per robot, roughly 23 times the prior flash allocation; its large-population cost has not been benchmarked.
- The bundled `pogobot-sdk` header still declares the older flash API; high-page coverage in `test_flash_state` currently runs only in simulation.
- The tutorial's solid disk arena bounds late-time MSD; its one-seed local smoke result is not evidence for a population-level condition effect.
- Corrected sleep/timer semantics can change trajectories and tick counts relative to older Pogosim results. The additional guarded stacks default to 128 KiB per robot plus the global controller; large-population scaling has not been benchmarked.

## Next concrete tasks

1. Verify the new coroutine runtime and simulator fixtures on hosted Linux/macOS/WSL/MinGW/MSVC CI; general native CMake example targets remain a separate task.
2. Decide whether the untracked quadrant-motility and magnetometer work should be documented, tested, and committed.
3. Add quantitative validation references or datasets for motion, sensing, timing, and infrared communication models.
4. Extend unit/regression coverage beyond the new batch/optimization tests to simulation scheduling, neighbor detection, and data logging.
5. Benchmark nested Pogoptim parallelism and optimizer convergence on representative built controllers.
6. Reconcile the CMake project version with the canonical C/Python version and document the release process.
7. Record benchmark envelopes for runtime and memory as robot count and logging volume increase.
8. Run the documented 256-Task experiment on a configured cluster and check its retrieved data, run-level MSD uncertainty, and resource envelope.
