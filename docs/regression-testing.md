# Simulator regression testing

Last updated: 2026-10-02

The CMake/CTest suite contains 60 tests: the existing 16 configuration and
controller-scheduling tests, plus 44 simulator regressions. These fixtures do
not modify runtime code, public APIs, configuration defaults, controller
Makefiles, or installed dependencies.

## Run locally

With the normal simulator build dependencies installed:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

CTest supplies SDL's dummy video driver; no interactive display is needed.
Tests use repository assets and keep generated configurations, Feather files,
flash archives, and a temporary installation under the build directory. They
do not need a system-wide Pogosim installation or write into example folders.
Expected failures check the exact exit status and a diagnostic, so a crash
cannot pass as a successful rejection test. Assertions remain active in
Release builds. Individual tests have timeouts, and independent cases can run
with `ctest --test-dir build --parallel 3 --output-on-failure`.

To run only the added simulator tests:

```sh
ctest --test-dir build -R '^simulator_' --output-on-failure
```

`-DBUILD_TESTING=OFF` retains the existing library-only build path and excludes
all test executables. Test binaries are never installed.

## Coverage

- Geometry: disk, rectangle, triangle, arena and global contours, bounds,
  distances, and occupancy grids.
- Lighting: spatial lookup, current outside-map clamping, saturation, callback
  reset/composition, uniform/radial/plane illumination, and startup pulses.
- Neighbors and communication: spatial cells, periodic ghosts, angular seams,
  field of view, occlusion, reception models, and real C-controller delivery
  and dropping of messages. Simulator runs also check periodic wrapping,
  preserved orientation, and isolated controller state across two categories.
- Object creation: all ten factory types with solid and periodic boundaries,
  concrete classes, initialization guards, tangibility, and light updates.
- Flash archives: deterministic identity ordering, full-region restoration,
  motor calibration, subset and same-path saves, retained unused records,
  empty markers, corrupt/truncated/duplicate records, identity mismatches,
  and preservation of the destination after failed export. Nine existing
  `test_flash_state` configurations run in sequence without modifying the
  hardware-compatible example.
- Feather logging: typed values, 64-bit integer precision, UTF-8, float16
  special values, null/reset semantics, filtering, metadata, multiple buffered
  batches and destructor flush. Real simulation outputs are reopened and
  checked for effective CLI seed, arena/version metadata, controller fields,
  per-robot time ordering, and field/category filtering.
- Error paths: malformed YAML, missing arena, unknown object/geometry/sensor
  source, invalid output destinations, and schema/export callback exceptions.
- Installed-library compatibility: a separate C consumer uses only installed
  simulator headers and the raw archive, with the pre-coroutine dependency
  list and no explicit Boost.Context link flag. Its installation prefix
  contains spaces. Dependency-discovery modules are reused, but the consumer
  does not link the source-tree Pogosim target.

Fixtures live in [tests/simulator](../tests/simulator); their registration is
kept separate from the runtime build in `register_tests.cmake`. Add a model
case to that list or a CLI case to `run_scenario.cmake`, using explicit model
settings and deterministic inputs. Do not adjust runtime behavior just to
make a regression fixture pass.

## CI and undefined-behavior checks

The existing Linux, macOS, WSL2, MinGW-w64 and MSVC CTest steps run the complete
suite. Ubuntu-latest also builds a separate Debug/UBSan configuration:

```sh
cmake -S . -B build-ubsan -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_FLAGS="-fsanitize=undefined -fno-omit-frame-pointer" \
  -DCMAKE_CXX_FLAGS="-fsanitize=undefined -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=undefined"
cmake --build build-ubsan --parallel 2
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  ctest --test-dir build-ubsan --output-on-failure
```

Instrumentation is confined to that build; normal builds and controller flags
are unchanged. The existing Python batch/optimization suite remains separate:

```sh
python3 -m unittest discover -s scripts/python_tests -v
```

All 60 CTest cases passed locally in Release, Debug/UBSan, and a build using
yaml-cpp 0.7; all 17 Python tests also passed. Hosted macOS/Windows execution
of the expanded suite still requires CI confirmation.

## Limits

These are behavioral regressions, not empirical validation of motion or sensor
accuracy, performance benchmarks, GUI interaction tests, or real-robot tests.
They avoid probabilistic pass/fail thresholds and cross-platform trajectory
golden files. UBSan does not establish leak freedom or instrument third-party
libraries. The installed-consumer test checks headers/linking/execution, not
complete runtime-asset relocatability. Fresh indeterminate flash is explicitly
initialized before byte-level comparisons; tests do not assign it a default
erased value.
