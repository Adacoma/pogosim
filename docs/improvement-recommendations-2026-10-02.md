# Pogosim improvement recommendations

Date: 2026-10-02. These are proposals based on inspection, not measured performance gains or evidence of hardware-model fidelity. Prefer incremental changes that preserve the C controller API and external Makefiles.

## Near-term priorities

1. **Broaden regression coverage** (small to medium effort). Extend the scheduling/flash/batch tests to object factories, solid and periodic boundaries, lighting, communication, logging, and failed initialization. Test installed-library consumers as well as in-tree builds, and add a Linux undefined-behavior sanitizer job. Trade-off: somewhat longer CI, but better protection against portability and lifecycle regressions.

2. **Lightweight configuration validation** (medium effort). Provide a check-only command and an optional strict mode, with YAML paths in errors for invalid types, counts, time steps, and probabilities. Keep missing-value defaults, unknown controller keys, and existing permissive runs compatible. Avoid a mandatory schema or duplicate registration for every new parameter. This proposal is implemented in the [configuration validation guide](configuration-validation.md).

3. **Predictable rebuilds and installation** (small to medium effort). Make example executables depend on the simulator library so reinstalling it causes relinking without `make -B`. Complete an installed `pogosim::pogosim` CMake package alongside the legacy archive-linking path. Trade-off: dependency tracking must distinguish local and installed builds and work across toolchains. See the [CMake importing/exporting guide](https://cmake.org/cmake/help/latest/guide/importing-exporting/index.html).

## Scientific correctness and robustness

4. **Clarify simulation clocks and logged timestamps** (medium effort). The main loop advances physics before exporting state but increments its time afterward; document which instant each row represents. Record the effective step when controller pacing reduces it. Consider fixed-step ticks, with a versioned change if this affects trajectories or logging. Trade-off: timing corrections can invalidate direct comparisons with earlier results. [Box2D's simulation guide](https://box2d.org/documentation/md_simulation.html) provides the fixed-step background.

5. **More useful scientific output** (medium effort). Optionally log periodic crossing counts or unwrapped positions for displacement/MSD analyses. Include controller/build/dependency identity and effective timing in output provenance. Trade-off: extra columns increase output size; unwrapping conventions and units must be explicit. Cross-platform bitwise determinism should not be promised without evidence.

6. **Reliable output completion** (medium effort). Explicitly finalize Feather output and report finalization failures through the exit status, rather than relying only on destructor diagnostics. Write to a partial file and publish the final name only after successful completion. Trade-off: replacement and recovery semantics must remain portable and compatible with Pogobatch retrieval/merging.

## Measured performance work

7. **Profile before optimizing neighbors** (medium effort). Measure time spent in controllers, physics, directional-neighbor detection, logging, and rendering. Investigate reusable buffers and once-per-tick body-pose caching rather than recomputing the same inputs for four IR directions. Trade-off: neighbor ordering, occlusion geometry, random-number consumption, and seeded behavior must be preserved; benchmark representative densities and arenas before claiming speedups.

8. **Reduce per-robot memory if benchmarks justify it** (medium to high effort). Each robot currently carries roughly 1.44 MiB of user flash plus a default 128 KiB guarded controller stack. Measure actual resident memory and scaling before considering lazy flash allocation or page-based storage. Trade-off: fresh flash must remain indeterminate, archives must retain their semantics, and page bookkeeping may slow reads/writes or complicate ownership.

## Longer-term scope

Calibrate locomotion, sensing, and infrared reception against physical Pogobot measurements, documenting uncertainty and validity ranges. API coverage alone does not establish simulation-to-hardware agreement.

Suggested order: configuration validation, broader regression/installed-consumer coverage, automatic relinking, then timing/provenance improvements and benchmark-driven optimization. Infinite-loop preemption is outside this roadmap; controller scheduling remains cooperative.
