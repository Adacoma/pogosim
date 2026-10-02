# Configuration validation

Added: 2026-10-02.

Rebuild the simulator library and relink your controller executable to use the new options:

```sh
./examples/run_and_tumble/run_and_tumble -c conf/simple.yaml --check-config
./examples/run_and_tumble/run_and_tumble -c conf/simple.yaml --strict-config -g
```

`--check-config` reads YAML and checks core configuration constraints, then exits with status 0 on success or 1 on error. It does not initialize SDL, execute controllers, create missing flash archives, write outputs, or delete existing output files. Explicit CLI seed/GUI/progress settings retain their normal precedence.

`--strict-config` performs the same preflight, then runs normally with type checking at every consumed `Configuration::get<T>()` lookup, including the existing C `init_from_configuration` bridge. Later failures use the existing simulation-error exit status 2. Errors identify the YAML path and, when available, source line, for example:

```text
Invalid configuration 'objects.robots.nb': expected a nonnegative robot/object count (line 12)
```

Without either flag, existing permissive conversions and defaults remain unchanged. Unknown keys are always allowed. Missing/null entries retain their historical behavior. Batch `default_option` and `batch_hierarchical_options.default` are resolved as before; inactive alternatives are not validated as simulation inputs. Decimal/scientific representations of whole integers remain accepted in strict mode, but fractional integers, out-of-range integer casts, and numeric booleans other than 0/1 are rejected.

## Scope and limits

Preflight covers the root/objects/parameters/flash mappings, finite positive time steps and arena size, simulation duration/tick-count bounds, window/bin/buffer sizes, counts, basic physical dimensions/noise, static reception probabilities, and selected output options. Zero counts and disabled output-period sentinels remain valid; infinite maximum formation spacing remains supported.

This is not a complete schema or a guarantee that a simulation will start successfully. Check-only mode does not validate arbitrary controller parameters, all specialized model options, arena/CSV files, flash archive contents, callback behavior, or platform-specific stack limits. Strict runs check additional types when those settings are actually used. Unknown keys also mean that spelling mistakes in unused keys cannot be detected. For batch campaigns, check the concrete generated simulation YAML (or usable scalar defaults), not unexpanded batch-only choices.

## Adding parameters

A new parameter still needs only its usual lookup—no registration, schema update, or new dependency:

```cpp
const float new_rate = config["new_rate"].get(1.0f);
```

For a parameter-specific domain constraint, optionally add a local check:

```cpp
const auto rate_config = config["new_rate"];
const float new_rate = rate_config.get(1.0f);
// Domain checks are enabled only for strict runs, preserving legacy behavior.
rate_config.require(std::isfinite(new_rate) && new_rate >= 0,
                    "a finite nonnegative rate");
```

`Configuration::enable_validation()` enables type checks for a C++ configuration and its subsequently retrieved children. `validate_simulator()` performs core preflight without changing the original object's mode. Only add a core preflight constraint when it is needed before simulation initialization; ordinary new parameters do not require editing that list.

Validation introduces no dependency and no whole-document walk during normal runs. Diagnostic paths are assembled only when validation is enabled. This does not change the controller-facing C API or real-robot firmware behavior.
