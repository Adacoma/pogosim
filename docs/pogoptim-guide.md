# Pogoptim guide

Pogoptim searches parameters in a Pogosim YAML configuration. It uses the same
local task, retry, manifest, and merge implementation as `pogobatch run`.
Optimization through Rundra is not currently supported.

## Installation and first run

Random Search needs only the normal Pogosim Python dependencies. CMA-ES and
MAP-Elites use the optional optimization dependencies:

```console
cd scripts
python3 -m pip install -e '.[optim]'
cd ..
```

Run the checked-in example after building its controller:

```console
pogoptim -c conf/optim/simple.yaml \
  -S examples/run_and_tumble/run_and_tumble
```

Values explicitly supplied on the command line override `optimization:` YAML
values. Omitted values fall back first to YAML and then to Pogoptim defaults.
The configuration file and simulator binary are always required command-line
arguments. Relative paths are interpreted from the invocation directory.

## Optimization configuration

An optimizable scalar or option owner has an `optimization_domain`:

```yaml
parameters:
  gain:
    default_option: 0.5
    optimization_domain: {type: float, min: 0.01, max: 2.0, log: true}
  count:
    default_option: 5
    optimization_domain: {type: int, min: 1, max: 20}
```

Float, integer, and categorical domains are represented internally in
`[0,1]`. The complete runtime schema is illustrated by
[`conf/optim/simple.yaml`](../conf/optim/simple.yaml). Important settings are:

- `algorithm`: `random`, `cmaes`, or `mapelites`;
- `budget`: number of candidate evaluations;
- `runs`: fresh simulation seeds used for each batch combination and candidate;
- `parallelism.candidate_jobs`: concurrent candidate evaluations;
- `parallelism.batch_jobs`: concurrent Pogobatch tasks inside each candidate,
  where zero means automatic;
- `cmaes`: normalized step size and optional population size;
- `qd`: grid shape, batch size, descriptor domains, and QDpy algorithm options.

Pogoptim automatically caps the product of candidate and batch workers to the
detected CPU count. Candidate-level concurrency uses coordinator threads, while
Pogobatch owns simulator processes or Ray tasks. A custom objective used with
concurrent candidates must therefore be thread-safe.

## Objectives and MAP-Elites descriptors

`--objective FILE` loads `compute_objective(df)` by default. The function may
return a scalar fitness or `(fitness, features)`. Without a custom function,
fitness is mean per-agent MSD. MAP-Elites uses polar order and trajectory
straightness as its default descriptors; both lie in `[0,1]`.

Custom MAP-Elites descriptors require explicit `qd.features_domain` ranges.
Their count must match the grid dimensions, and out-of-range or non-finite
descriptors fail that candidate rather than being silently clipped. A custom
objective returning only fitness continues to use the built-in descriptors.

## Seeds, retries, and failures

The optimization seed controls both candidate generation and a separate,
deterministic simulation-seed allocation. Every candidate receives a disjoint
block containing its initial seeds and reserved retry seeds. Consequently,
parallel completion order does not change the experiment, and separate
candidates do not reuse stochastic simulations.

An exhausted simulator task or candidate-specific objective exception assigns
the candidate a worst fitness and records the error. Configuration errors,
invalid objective contracts, and merge errors stop the optimization. If no
candidate succeeds, Pogoptim writes its history and a failed summary, then exits
nonzero. Failed Pogobatch task directories are retained for diagnosis.

## Outputs

Every run writes `opt_history.csv` and `summary.json`. The history includes
decoded parameters, normalized genomes, requested/effective seeds, retries,
fitness, descriptors, status, and errors.

Random Search and CMA-ES also write:

- `best_config.yaml`;
- `best_results.feather` with configuration metadata;
- `fitness_vs_eval.png`.

MAP-Elites instead writes `qd_archive.csv`, `qd_final.p`, and QDpy grid plots.
The archive contains each successful elite's fitness, descriptors, normalized
genome, and decoded parameter values. Failed candidates are not exported as
elites.
