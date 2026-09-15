# Pogosim documentation map

## Start here

- [Project README](../README.md) — purpose, installation, quickstart, simulator controls, configuration examples, data export, containers, and reproducible experiments.
- [Current project state](current_state.md) — inspected areas, current understanding, open questions, scientific decisions, data limitations, and next tasks.
- [Change log](../CHANGELOG.md) — release history and user-visible changes.

## Experiment execution

- [Pogobatch and Rundra guide](pogobatch-rundra-guide.md) — local parameter sweeps, cluster execution, task manifests, retrieval, merging, recovery, and provenance.
- [Pogoptim guide](pogoptim-guide.md) — parameter domains, optimization algorithms, objectives, parallel execution, seeds, failure handling, and outputs.
- [Persistent robot flash state](flash-state.md) — export/import lifecycle, stored fields, identity validation, and Pogobatch behavior.
- [Template controller](../template_prj/main.c) and [template Makefile](../template_prj/Makefile) — minimal controller that can target both the simulator and physical Pogobots.
- [Example controllers](../examples) — runnable demonstrations of locomotion, communication, sensing, localization, synchronization, and collective-behavior algorithms.
- [Flash-state import/export test](../examples/test_flash_state/README.md) — minimal two-run validation of persistent user flash and motor calibration memories.
- [Example configurations](../conf) — YAML configurations for individual runs, batches, and optimization.

## APIs and implementation

- [Public controller entry header](../src/pogobase.h) — common include used by simulated and physical controllers.
- [Pogosim controller API](../src/pogosim/pogosim.h) — controller lifecycle, per-robot state, callbacks, and shared helpers.
- [Simulated Pogobot API](../src/pogosim/spogobot.h) — simulation implementations of the Pogobot-facing C API.
- [Simulator architecture](../src/pogosim/simulator.h) — simulation ownership, initialization, main loop, objects, rendering, and logging.
- [Doxygen configuration](../Doxyfile) — generates the detailed source/API reference; generated output is not tracked.
- [Pogobot reception-success model](pogobot_P_reception_success.pdf) — supporting communication-model document.

## Data and analysis

- [Python tools](../scripts/pogosim) — Feather loading, Pogobatch, optimization, arena utilities, and analyses of locomotion, coverage, neighbors, networks, and active matter.
- [Python package metadata](../scripts/setup.py) — installable package, optimization extras, and `pogobatch`/`pogoptim` console entry points.

## Build and deployment

- [CMake build](../CMakeLists.txt) and [build script](../build.sh) — build and install the simulator library and example programs.
- [CI workflow](../.github/workflows/ci.yaml) — multi-platform builds and headless smoke tests.
- [Apptainer definitions](..) — root-level `pogosim-*.def` files for reproducible and cluster-oriented environments.
