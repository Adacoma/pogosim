# Current project state

Last updated: 2026-09-18

## Inspected

- Top-level documentation, build scripts, package metadata, CI, tracked repository layout, and working-tree status.
- Core configuration, simulator lifecycle, object hierarchy and factories, robot state switching, physics scheduling, communication, sensing, geometry, rendering, and Arrow data logging.
- Controller template and the available example/configuration families.
- Python batch, optimization, arena, locomotion, coverage, neighbor, network, and active-matter tools at an architectural level.

## Understood

- Pogosim is a two-dimensional behavioral and physical simulator for Pogobot swarm controllers. Its central goal is to let one C controller source target both simulation and physical robots through a compatibility API.
- A simulated controller's `main` is renamed to `robot_main`. The simulator invokes it for each robot, allocates separate `USERDATA`, and swaps per-robot API state through `set_current_robot` before controller calls.
- YAML defines the arena, boundary conditions, initial formation, objects, physics, sensing, communication, timing, rendering, and output.
- Each simulation tick runs global/object/robot controllers, advances Box2D, applies periodic wrapping when selected, recomputes directional neighbors, logs data, and optionally renders.
- The object model covers Pogobots, Pogobjects, Pogowalls, flexible membranes, active/passive objects, and static or time-varying lights. Communication supports directional range, optional occlusion, and static or density-dependent reception probability.
- Results are buffered into compressed Arrow/Feather files with configuration, arena, and version metadata. Pogobatch expands parameter choices and seeds, runs tasks locally or through Rundra, records manifests, and merges task outputs.
- Pogoptim now reuses Pogobatch's public local-campaign API for Random Search, CMA-ES, and MAP-Elites, with YAML/CLI precedence and recorded evaluation provenance.
- Optional flash-state archives now carry each robot's 64 KiB user flash plus motor direction/power calibration memories between simulator invocations, without checkpointing transient or physical state.
- CI combines Linux, macOS, WSL2, MSYS2/MinGW-w64, and native MSVC builds with headless simulator smoke runs and focused Python regression tests for Pogobatch and Pogoptim.

## Remains unknown

- Quantitative agreement between each simulated physical/sensor/communication model and measurements from real Pogobots.
- Performance and memory scaling across robot counts, arena complexity, logging rates, and GUI/headless execution.
- Which untracked configurations and example directories are active research work intended for integration.
- Compatibility of optional analysis tools other than Pogoptim with current dependency releases.
- Whether every example remains suitable for physical firmware as well as simulation, especially examples using optional `pogo-utils` functionality.
- The MSYS2/MinGW-w64 and PowerShell/MSVC library builds have both passed on hosted Windows. Native MSVC controller/example executable targets remain unavailable because the example Makefiles use GNU-specific compiler and linker options.

## Working analyses

- No analysis is currently running as part of this milestone.
- The Pogoptim/Pogobatch migration has passed focused unit tests, optional CMA-ES/QDpy smoke tests, a nested-worker fake-simulator campaign, and one-evaluation Random/MAP-Elites runs against the built `run_and_tumble` controller; longer scientific runs have not been benchmarked.
- Flash persistence has been implemented with pre-controller restore, post-callback atomic export, strict robot identity validation, and task-local Pogobatch outputs; round-trip and rejection checks cover the archive boundary.
- A dedicated `test_flash_state` example now provides a reproducible two-run simulator smoke test and a two-boot real-robot test while preserving hardware motor calibration values.
- The Windows jobs now distinguish MSYS2/UCRT64 with MinGW-w64 from a native PowerShell/MSVC build. Both use the same pinned Box2D 3.x revision; the MSYS2 job retains the simulator flash-state smoke test and native-CPython suite, while the MSVC job initializes the latest installed `cl.exe`, builds through Ninja, exports Box2D's library and public headers through one explicit install prefix, and validates the CMake library artifact.
- CMake resolves GNU make explicitly for optional example targets, so generated Ninja builds no longer contain Make-only `$(MAKE)` syntax.
- MSVC builds define `_USE_MATH_DEFINES` at the target level so existing public headers and sources can use the standard math constants consistently.
- The README documents the validated PowerShell/MSVC, Ninja, vcpkg, and pinned Box2D library installation path without requiring MSYS2 or WSL.
- The README keeps WSL, MacOSX, native Windows, individual troubleshooting instructions, and the AI-agent guidance in collapsed `<details>` blocks while retaining section headings for navigation.
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
- Treat only the user flash section and motor calibration memories as persistent robot state; fresh user flash remains indeterminate unless an archive is loaded.
- Use Feather output with embedded provenance and restrict logged fields/categories for large experiments.

## Known data limitations

- Logged timestamps lie on simulator ticks and may not exactly match the requested logging period or final simulation time.
- Long-time displacement in solid arenas is bounded; periodic trajectories require coordinate unwrapping before conventional unbounded-space MSD analysis.
- Changing robot count at fixed arena area also changes density, collision frequency, and communication connectivity.
- Custom controller fields have experiment-specific units and semantics unless documented alongside their configuration.
- Empirical calibration/validation datasets were not inspected, so model fidelity must not be inferred solely from API coverage.
- Generated results in local ignored directories are not canonical repository data and were not used to establish scientific conclusions.
- Pogoptim is local-only, has no resume/checkpoint workflow, and relies on the optional QDpy package for MAP-Elites.
- Flash archives validate simulator-level structure and robot identity but cannot determine whether controller-defined byte layouts are application-compatible.

## Next concrete tasks

1. Add native CMake controller/example executable targets and an MSVC simulator smoke test.
2. Decide whether the untracked quadrant-motility and magnetometer work should be documented, tested, and committed.
3. Add quantitative validation references or datasets for motion, sensing, timing, and infrared communication models.
4. Extend unit/regression coverage beyond the new batch/optimization tests to simulation scheduling, neighbor detection, and data logging.
5. Benchmark nested Pogoptim parallelism and optimizer convergence on representative built controllers.
6. Reconcile the CMake project version with the canonical C/Python version and document the release process.
7. Record benchmark envelopes for runtime and memory as robot count and logging volume increase.
