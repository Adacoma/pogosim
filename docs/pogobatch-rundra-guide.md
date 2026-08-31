# Running Pogobatch locally and with Rundra

Pogobatch expands parameter choices, runs repeated Pogosim simulations, and
merges their Feather data. Local batches use the `run` subcommand and do not
require Rundra. Rundra is required only for the remote `cluster` workflow and
the explicit cluster lifecycle later in this guide.

## Quickstart: run a local batch

Run these commands from the root of a Pogosim checkout. Install the Python
package from the current checkout so the command and configuration schema match
the source being used:

```bash
python3 -m pip install -e ./scripts
```

Build the simulator program before launching the batch. For example:

```bash
make -C examples/test_go_straight clean sim
```

### Add choices to a normal Pogosim configuration

A batch configuration is an ordinary Pogosim YAML file with one or more batch
markers. The following abbreviated example varies the arena and robot count:

```yaml
pogobatch:
  # A placeholder can select which combinations are merged together.
  result_filename_format: "result_{objects.robots.nb}.feather"

  # Copy selected configuration values into columns of the merged data.
  result_new_columns:
    - arena_file

arena_file:
  default_option: arenas/disk.csv
  batch_options:
    - arenas/disk.csv
    - arenas/annulus.csv

objects:
  robots:
    nb:
      default_option: 50
      batch_options: [50, 100]
```

`batch_options` lists the values Pogobatch should test. `default_option` is
required and is the value used when the same configuration is resolved for one
direct Task. Batch-only result settings now belong under the reserved
`pogobatch:` mapping; Pogobatch removes that mapping and all batch markers
before invoking the simulator.

Choices at different paths form a Cartesian product. The abbreviated example
above has four combinations. The current `conf/batch/simple.yaml` also varies a
hierarchical communication model, so it expands to eight combinations: two
arenas × two population sizes × two communication models.

Use `plan` to inspect the expansion without running the simulator:

```bash
./scripts/pogosim/pogobatch.py plan \
  -c conf/batch/simple.yaml \
  --seed 17
```

### Execute and merge locally

The `run` subcommand expands the configuration, executes all atomic simulations,
and merges their Feather files in one command:

```bash
./scripts/pogosim/pogobatch.py run \
  -c conf/batch/simple.yaml \
  -S ./examples/test_go_straight/test_go_straight \
  -j 4 \
  -o results/test_go_straight \
  --progress
```

A successful invocation looks like:

```text
Pogosim: 100%|...| 8/8
Completed 8 local task(s).
  results/test_go_straight/result_100.feather
  results/test_go_straight/result_50.feather
```

In this example:

- `-c` selects the batch-aware Pogosim configuration;
- `-S` selects an already compiled simulator executable;
- `-j 4` allows at most four local simulations to run concurrently using the
  default multiprocessing backend;
- `-o` selects the directory for merged results;
- `--progress` displays completed Tasks; and
- execution is headless unless `--gui` is supplied.

The eight Tasks are grouped into two outputs because
`result_filename_format` contains only the robot-count placeholder. Each file
therefore contains all arena and communication-model choices for that robot
count. Pogobatch adds `run`, `seed`, and `retry_attempt` columns, plus the values
listed by `pogobatch.result_new_columns`. It also writes
`pogobatch_run.json` beside the merged results. Files produced by a new `run`
replace same-named results rather than appending to them.

### Choose repeats and seeds explicitly

If no seed option is present and the configuration has no `_rundr.seeds`,
Pogobatch generates one seed and uses it for every combination in that
invocation. For reproducible work, select the seeds explicitly:

| Option | Meaning |
|---|---|
| `--seed 17` | one Task per combination, all using seed 17 |
| `-r 4` | four Tasks per combination, using seeds 0 through 3 |
| `--seeds 10:13` | four Tasks per combination; the range is inclusive |

The total Task count is the number of combinations multiplied by the number of
seeds. Thus eight combinations with `-r 4` produce 32 Tasks. Use the same seed
option with `plan` and `run` when comparing the preview with an execution.

Temporary Task shards are created below `tmp/` by default and removed after a
successful merge. Use `-t PATH` to select another temporary base and
`--keep-temp` to preserve successful shards; failed campaigns retain their
temporary directory for diagnosis. Local Tasks retry crashes up to five times
by default, using replacement seeds. Use `--retry-same-seed` when retries must
preserve the requested seed, or `-R 0` to disable retries.

For command-specific options, run:

```bash
./scripts/pogosim/pogobatch.py help run
```

This guide describes the Pogobatch subcommand interface in the Pogosim `dev`
branch and its integration with Rundra. It covers configuration, target
preflight, source/container preparation, cluster submission, retrieval, merge,
failure recovery, and cleanup.

Pogobatch and Rundra have separate responsibilities:

- **Pogosim** executes one simulation from one effective YAML configuration and
  one seed, then writes one or more Feather files.
- **Pogobatch** expands Pogosim batch choices, executes an atomic Task, writes a
  `pogobatch_task.json` sidecar, and merges self-describing Task outputs.
- **[Rundra](https://github.com/leo-cazenille/rundra)** seals or acquires source,
  prepares the application and container, expands seeds and parameters, submits
  scheduler work, tracks durable Run/Task state, and retrieves raw outputs.

The normal remote lifecycle is:

```text
Pogobatch plan
    -> Rundra doctor and plan
    -> Rundra submit
    -> Rundra await
    -> Rundra fetch
    -> Pogobatch merge
```

`pogobatch cluster` automates that lifecycle. The explicit lifecycle is useful
for unattended agents, long Runs, custom Rundra experiments, and workflows that
must retain raw retrievals separately from merged results.

## 1. Pogobatch on cluster -- first verify the installed interfaces

The recommended way to install Rundra is as an isolated Python 3.12 tool with
[`uv`](https://docs.astral.sh/uv/):

```bash
uv tool install --python 3.12 rundra
rundr --version
rundr help
```

Upgrade it later with `uv tool upgrade rundra`. When developing Rundra from a
clone of its [GitHub repository](https://github.com/leo-cazenille/rundra), run
`uv sync --locked` in that checkout and prefix commands with `uv run`, for
example `uv run rundr --version`.

The Pogosim `dev` branch contains a newer Pogobatch interface than some released
or globally installed `pogobatch` commands. Before preparing a remote campaign,
check both tools:

```bash
python3 -m pogosim.pogobatch --version
rundr version
rundr help
```

The dev interface has the following subcommands:

```text
help  plan  task  run  merge  cluster
```

From a Pogosim checkout, use the checked-out Python package explicitly if it is
not installed into the active environment:

```bash
export PYTHONPATH="$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
python3 -m pogosim.pogobatch --version
```

Alternatively, build and install the Python package from `scripts/` into an
isolated environment. Do not assume that a `pogobatch` executable found earlier
on `PATH` implements the dev interface.

Required Python packages include PyYAML and PyArrow. The target container must
also provide Python plus these dependencies when `pogobatch task` runs inside
the container.

## 2. Configure a batch

Batch-only settings belong under the reserved top-level `pogobatch` mapping.
They are removed before Pogosim receives the effective configuration.

```yaml
pogobatch:
  # Keep this constant to merge every parameter choice into one Feather file.
  # Use placeholders to produce one merged file per selected choice instead.
  result_filename_format: results.feather

  # Copy selected configuration values or hierarchical choice names into each
  # row of the merged result.
  result_new_columns:
    - condition

  # Paths excluded while staging working-tree source through Rundra.
  rundra:
    sync:
      exclude:
        - results/
        - retrieved/
        - frames/
        - build/
        - "*.sif"

enable_data_logging: true
data_filename: frames/data.feather
save_data_period: 1.0
save_video_period: -1.0
GUI: false

# Restrict large campaigns to the fields used by the analysis.
data_logger_fields:
  - time
  - robot_category
  - robot_id
  - x
  - y

parameters:
  batch_hierarchical_options:
    name: condition
    default:
      parameter_a: 1
      parameter_b: 2
    condition_a:
      parameter_a: 10
      parameter_b: 20
    condition_b:
      parameter_a: 30
      parameter_b: 40

_rundr:
  version: 1
  seeds: "0:127"       # inclusive: 128 seeds per parameter combination
```

Pogobatch supports three batch marker forms.

### Discrete values

```yaml
objects:
  robots:
    nb:
      default_option: 100
      batch_options: [50, 100, 200]
```

### Numeric ranges

```yaml
some_parameter:
  default_option: 0.1
  batch_options_range:
    start: 0.1
    stop: 0.5
    step: 0.1
    inclusive: true
    type: float
```

### Correlated/hierarchical choices

Use `batch_hierarchical_options` when several values must change together. The
optional `name` creates a choice alias suitable for `result_new_columns` and
result filename templates.

Every simple batch node must have `default_option` so that the same file can be
used by one direct `pogobatch task`. Every hierarchical node must have a
`default` mapping.

Seed ranges use inclusive `START:STOP` syntax. CLI `--seed`, `--seeds`, or
`--runs` values override `_rundr.seeds`. `--runs N` selects seeds `0..N-1`.

## 3. Inspect the Pogobatch expansion

Planning is local and does not execute simulations or contact a cluster:

```bash
python3 -m pogosim.pogobatch plan \
  --config conf/experiment.yaml \
  --seeds 0:127 \
  --json
```

Review at least:

- `combinations`;
- `seeds`;
- total `task_count`;
- computed output filenames;
- configuration hashes;
- any `--only-output` filtering.

The total Task count is the Cartesian product of parameter combinations and
seeds. For example, two conditions and seeds `0:127` produce 256 Tasks.

## 4. Define and audit a Rundra target

Target definitions are site-specific and should normally live in
`~/.config/rundra/targets.yaml`. Do not place credentials in target, experiment,
or project YAML files. SSH authentication and host verification remain in the
normal OpenSSH configuration.

A typical remote Slurm target has this structure:

```yaml
version: 11
targets:
  cluster:
    transport:
      type: ssh
      host: cluster-login
    scheduler:
      type: slurm
      partition_routes:
        - name: cpu-hour
          resource_class: cpu
          max_walltime: "01:00:00"
          partition: cpu-short
    staging:
      type: rsync
    container:
      type: apptainer
      executable: apptainer
    workspace: /shared/users/USERNAME/.rundra
    execution:
      hard_task_limit: 10000
      confirmation_threshold: 1000
      max_active_tasks: 32
      max_concurrent_jobs: 4
      max_array_size: 1000
      output_shard_tasks: 1000
      automatic_retrieval_threshold: 1000
      max_memory_per_worker: 8GiB
      worker_pool:
        activation_threshold: 10
        default_workers: 1
        max_workers: 4
        default_task_slots_per_worker: 8
        max_task_slots_per_worker: 8
        tasks_per_lease: 10
        infrastructure_retry_limit: 1
        requeue_limit: 2
```

Replace the SSH alias, workspace, partition, wall-time, container executable,
and every execution-policy value with settings supplied or reviewed by the
cluster operator. Memory, concurrency, array-size, and worker-pool limits are
site safety policy; the values above illustrate the schema rather than universal
recommendations.

The target workspace must be writable from the login side and visible to the
scheduler compute nodes. Rundra validates backend capabilities rather than
requiring callers to parse native scheduler output.

On a new machine or agent session, audit local readiness and the selected target
definition first:

```bash
rundr doctor \
  --target cluster \
  --targets-file ~/.config/rundra/targets.yaml \
  --data-dir .rundra-runs \
  --agent codex \
  --json
```

Use `--agent codex` when working through Codex and `--agent generic` in another
environment.

If `run_store_durability.verification_argv` is returned, run that exact command
separately. Continue only when `ready` is true. In a sandbox, use a persistent
workspace path for `--data-dir`; a default home-directory store can appear
writable while being command-local.

After creating the experiment files described below, run the connected audit:

```bash
rundr doctor campaign/experiment.yaml \
  --config campaign/conf/rundra-sweep.yaml \
  --target cluster \
  --targets-file ~/.config/rundra/targets.yaml \
  --project-file campaign/rundra.yaml \
  --destination "$PWD/raw/rundra" \
  --data-dir "$PWD/.rundra-runs" \
  --connect \
  --agent codex \
  --json
```

This verifies authentication, remote backend capabilities, staging, target
workspace access, and local persistence without launching the scientific
campaign or submitting scheduler work. During initial target onboarding, an
operator can add `--scheduler-probe` to submit one bounded diagnostic job and
verify compute-side workspace access.

This experiment-specific audit applies to the explicit lifecycle in section 6,
where the generated Rundra files exist before submission. The integrated
`pogobatch cluster` command validates and plans its generated sweep internally,
but does not currently expose that temporary experiment for a separate
connected-doctor pass before execution. Complete the bootstrap audit and
validate the target configuration before using the integrated command.

## 5. Recommended: use `pogobatch cluster`

For most users, the integrated command is the shortest correct path:

```bash
python3 -m pogosim.pogobatch cluster \
  --config conf/experiment.yaml \
  --simulator-binary examples/PROGRAM/PROGRAM \
  --seeds 0:127 \
  --target cluster \
  --targets-file ~/.config/rundra/targets.yaml \
  --source-root "$PWD" \
  --data-dir "$PWD/.rundra-runs" \
  --workers 4 \
  --task-slots-per-worker 8 \
  --fetch-destination "$PWD/raw/rundra" \
  --output-dir "$PWD/results" \
  --keep-fetch \
  --json
```

For a remote scheduler, `cluster` performs a Rundra plan, submits the Run,
waits for terminal state, fetches/materializes outputs when necessary, and
invokes the same Pogobatch merge used by local batches.

Important options:

| Option | Purpose |
|---|---|
| `--target NAME` | select a configured Rundra target |
| `--profile NAME` | select a named profile from `rundra.yaml` |
| `--workers N` | request scheduler worker allocations within target policy |
| `--task-slots-per-worker N` | logical simulations run concurrently per worker |
| `--confirm-tasks N` | acknowledge an exact Task count when the Rundra safety policy requires it |
| `--fetch-destination PATH` | raw Rundra retrieval location |
| `--output-dir PATH` | merged Pogobatch result location |
| `--keep-fetch` | retain raw fetched Task outputs after a successful merge |
| `--keep-rundra-files` | retain temporary generated Rundra YAML files |
| `--keep-rundra-workspace` | retain the successful remote Rundra Run workspace |
| `--allow-partial` | permit retrieval/merge of successful Tasks from a failed campaign |
| `--offline` | prohibit Git fetches and image pulls; use only after caches are known to be warm |

By default, a fully successful integrated run removes its temporary fetched
tree and generated YAML, and purges the exact remote Rundra Run workspace. The
merged results and local Rundra RunRecord remain. Use the retention options when
raw data or diagnostics must be preserved.

### Container preparation

Remote Apptainer targets need a container. `pogobatch cluster` can build one
from a Pogosim definition when target policy permits definition builds. A cold
definition build can be expensive, so production campaigns should prefer a
versioned prebuilt image with an immutable digest:

```bash
python3 -m pogosim.pogobatch cluster \
  --config conf/experiment.yaml \
  --simulator-binary examples/PROGRAM/PROGRAM \
  --seeds 0:127 \
  --target cluster \
  --targets-file ~/.config/rundra/targets.yaml \
  --source-root "$PWD" \
  --container-image pogosim-full_vX.Y.Z.sif \
  --container-uri library://OWNER/COLLECTION/IMAGE:TAG \
  --container-sha256 EXPECTED_64_HEX_DIGEST \
  --data-dir "$PWD/.rundra-runs" \
  --output-dir "$PWD/results"
```

When the simulator path is inside `--source-root` and has an adjacent
`Makefile`, Pogobatch normally asks Rundra to compile it once during preparation
with `make -C DIRECTORY clean sim`. The prepared executable is cached using the
source, image, command, and output identity.

## 6. Explicit Rundra lifecycle

Use the explicit lifecycle when the client should detach after submission, raw
retrieval must be a distinct step, or an agent needs a stable Run ID immediately.

### 6.1 Generate a Rundra-ready sweep configuration

Rundra natively expands `_rundr.version: 1` batch markers. Pogobatch also adds
small internal markers so hierarchical choice aliases survive into Task
metadata. Generate this form with the dev helper instead of adding those markers
by hand:

```python
#!/usr/bin/env python3
from pathlib import Path
import sys

import yaml
from pogosim.pogobatch import make_rundra_sweep_config

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
document = yaml.safe_load(source.read_text(encoding="utf-8"))
destination.write_text(
    yaml.safe_dump(make_rundra_sweep_config(document), sort_keys=False),
    encoding="utf-8",
)
```

Run it as, for example:

```bash
python3 tools/make_rundra_config.py \
  conf/experiment.yaml \
  campaign/conf/rundra-sweep.yaml
```

Retain the original configuration. Pass it to `pogobatch merge --config` so the
human-authored batch definition, rather than generated markers, is embedded in
the final Feather metadata.

### 6.2 Define the atomic Rundra experiment

```yaml
# campaign/experiment.yaml
version: 1
experiment:
  name: pogosim-batch

command:
  argv:
    - python3
    - -m
    - pogosim.pogobatch
    - task
    - --config
    - "{config}"
    - --simulator-binary
    - examples/PROGRAM/PROGRAM
    - --seed
    - "{seed}"
    - --output-dir
    - /workspace/output
    - --quiet
  environment:
    # Required when the dev Python package comes from the staged checkout
    # rather than an installed wheel in the image.
    PYTHONPATH: /workspace/source/scripts

container:
  image: pogosim-full-vX.Y.Z.sif
  gpu: false

resources:
  nodes: 1
  tasks: 1
  cpus_per_task: 1
  gpus_per_task: 0
  memory: 1GiB
  walltime: "00:15:00"

outputs:
  include:
    - pogobatch_task.json
    - pogobatch_effective.yaml
    - frames/**
```

Every successful Task must return both `pogobatch_task.json` and the Feather
file referenced by its `data_file` entry. `frames/**` covers the default
`frames/data.feather` output and any additional simulator outputs placed there.

`pogobatch task` rewrites `data_filename`, `console_filename`, and `frames_name`
under its Task output directory while retaining each configured basename.

### 6.3 Define source, image, build, and worker defaults

```yaml
# campaign/rundra.yaml
version: 2
default_profile: cluster

defaults:
  config: conf/rundra-sweep.yaml
  target: cluster

profiles:
  cluster:
    target: cluster
    workers: 4
    task_slots_per_worker: 8

preparation:
  source:
    git:
      url: https://github.com/Adacoma/pogosim.git
      revision: FULL_40_CHARACTER_COMMIT

  image:
    name: pogosim-full-vX.Y.Z.sif
    uri: library://OWNER/COLLECTION/IMAGE:TAG
    sha256: EXPECTED_64_HEX_DIGEST

  build:
    argv:
      # Use a workspace-backed temporary directory on systems whose compute
      # node /tmp is small or shared.
      - env
      - TMPDIR=/workspace
      - make
      - -C
      - examples/PROGRAM
      - clean
      - sim
    outputs:
      - path: examples/PROGRAM/PROGRAM
        executable: true
    cache_scope: target
    resources:
      cpus_per_task: 1
      memory: 2GiB
      walltime: "00:15:00"
```

Pin a full Git commit and image SHA-256. A moving branch or unverified image is
not a reproducible preparation identity.

### 6.4 Validate and plan

```bash
rundr validate campaign/experiment.yaml --json

rundr plan campaign/experiment.yaml \
  --config campaign/conf/rundra-sweep.yaml \
  --seeds 0:127 \
  --target cluster \
  --targets-file ~/.config/rundra/targets.yaml \
  --project-file campaign/rundra.yaml \
  --workers 4 \
  --task-slots-per-worker 8 \
  --fetch-mode copy \
  --json
```

Review these structured fields before consuming resources:

- `plan.task_space.task_count` and seed range;
- parameter-set choices;
- `plan.resources`;
- selected partition/native options;
- `plan.scheduling.worker_count` and `task_slots_per_worker`;
- concurrent Task capacity;
- preparation source revision, image digest, and build command;
- retrieval policy;
- safety confirmation threshold.

Pass `--confirm-tasks EXACT_COUNT` to submit only when the plan says that the
Task count exceeds the confirmation threshold.

### 6.5 Submit once and retain the Run ID

```bash
rundr submit campaign/experiment.yaml \
  --config campaign/conf/rundra-sweep.yaml \
  --seeds 0:127 \
  --target cluster \
  --targets-file ~/.config/rundra/targets.yaml \
  --project-file campaign/rundra.yaml \
  --workers 4 \
  --task-slots-per-worker 8 \
  --fetch-mode copy \
  --destination "$PWD/raw/rundra" \
  --data-dir "$PWD/.rundra-runs" \
  --json
```

Store the returned `run.run_id`. Use that exact ID and the exact same
`--data-dir` for every later lifecycle command. Do not submit a duplicate if the
client is interrupted during submission; use:

```bash
rundr resume RUN_ID --data-dir "$PWD/.rundra-runs" --json
```

### 6.6 Await terminal state

For agents and unattended processes, use one blocking aggregate await rather
than repeatedly polling status:

```bash
rundr await RUN_ID \
  --fail-on-run-failure \
  --timeout 7200 \
  --data-dir "$PWD/.rundra-runs" \
  --json
```

For interactive humans, `rundr wait RUN_ID --progress` is available. Status and
logs remain inspectable independently:

```bash
rundr status RUN_ID --data-dir "$PWD/.rundra-runs" --json
rundr logs RUN_ID --preparation --data-dir "$PWD/.rundra-runs"
rundr logs RUN_ID --task 0 --data-dir "$PWD/.rundra-runs"
```

### 6.7 Fetch and materialize Task files

Worker-pool Runs can be represented by manifests or verified archives. Force a
normal file tree before invoking the generic Pogobatch merger:

```bash
rundr fetch RUN_ID \
  --destination "$PWD/raw/rundra" \
  --mode copy \
  --extract \
  --data-dir "$PWD/.rundra-runs" \
  --json
```

Fetch is idempotent. After extraction, the relevant layout is:

```text
raw/rundra/
  metadata/
    tasks.json
  output/
    task_000000/
      pogobatch_task.json
      pogobatch_effective.yaml
      frames/
        data.feather
    task_000001/
      ...
```

Before merge, verify that the numbers of Task manifests and expected Feather
files equal the successful Task count.

### 6.8 Merge

```bash
python3 -m pogosim.pogobatch merge \
  "$PWD/raw/rundra" \
  --output-dir "$PWD/results" \
  --config conf/experiment.yaml \
  --json
```

The merger recursively discovers `pogobatch_task.json` files, validates their
referenced data files and unique Task UUIDs, groups them by
`result_filename`, and writes one Feather file per group. It adds or replaces:

- `run`;
- `seed`;
- `retry_attempt`;
- every configured `pogobatch.result_new_columns` value.

It also embeds the original batch configuration and a compact Pogobatch merge
manifest in Feather schema metadata. Use `--append` only when intentionally
adding shards to an existing result; ordinary reproducible campaigns should
write a new destination.

To guarantee one merged output, use one constant
`pogobatch.result_filename_format`. If the filename contains configuration
placeholders, Pogobatch intentionally creates multiple merged files.

Keep raw retrievals separate from merged and derived outputs:

```text
raw/rundra/        immutable retrieved Task artifacts
results/           merged Pogobatch Feather files
derived/           analysis curves, summaries, figures, and statistics
```

## 7. Failure diagnosis and recovery

### Preparation failure

```bash
rundr logs RUN_ID --preparation --data-dir "$PWD/.rundra-runs"
```

Common causes include an inaccessible Git revision, image digest mismatch,
container pull failure, missing target build dependency, insufficient build
walltime/memory, or an exhausted compiler temporary directory. On nodes with a
small `/tmp`, set a writable workspace-backed `TMPDIR` in the preparation build
command.

A terminal failed Run is immutable. Correct the recipe and submit a new Run.
Use `resume` only for an interrupted or unresolved submission, not to mutate a
completed/failed experiment definition.

### Scientific Task failure

Inspect structured status and the affected Task log:

```bash
rundr tasks RUN_ID --data-dir "$PWD/.rundra-runs" --json
rundr logs RUN_ID --task TASK_ID --data-dir "$PWD/.rundra-runs"
```

Normal merge requires a successful complete campaign. `pogobatch cluster
--allow-partial` is an explicit opt-in for retrieving and merging successful
Task artifacts from a failed campaign.

### Missing manifests during merge

If merge reports that no `pogobatch_task.json` files exist:

1. confirm that the Rundra experiment declares the Task manifest and Feather
   output;
2. refetch with `--mode copy --extract`;
3. verify that `pogobatch task`, rather than the simulator directly, was the
   experiment command;
4. verify that the container used the intended Pogobatch dev package.

### Container imports the wrong Pogobatch

If a container has an older `pogosim` package installed, invoke the staged dev
package with `PYTHONPATH=/workspace/source/scripts` and
`python3 -m pogosim.pogobatch`. Record the reported Pogobatch version in the Run
provenance or experiment log.

## 8. Provenance and cleanup

After submission or completion, inspect provenance:

```bash
rundr inspect RUN_ID --data-dir "$PWD/.rundra-runs" --json
```

Record at least:

- Rundra Run ID and framework version;
- source Git revision/digest;
- container URI and verified SHA-256;
- prepared executable digest;
- target and selected partition;
- scheduler IDs;
- worker and Task-slot counts;
- actual container runtime/version;
- final Task and retrieval counts.

Preview remote workspace deletion before performing it:

```bash
rundr purge RUN_ID \
  --workspace \
  --dry-run \
  --data-dir "$PWD/.rundra-runs" \
  --json
```

After confirming that raw and merged results are safely retained, purge only
the exact Run workspace with the required Run-ID confirmation:

```bash
rundr purge RUN_ID \
  --workspace \
  --confirm RUN_ID \
  --data-dir "$PWD/.rundra-runs" \
  --json
```

The local RunRecord is separate from the remote workspace and remains available
unless explicitly purged.

## 9. Compact checklist

Before submission:

- [ ] Verify the dev Pogobatch subcommand interface.
- [ ] Use explicit seeds and calculate the full Task count.
- [ ] Enable Feather logging and restrict fields/categories where practical.
- [ ] Choose whether all conditions merge into one filename.
- [ ] Pin the Git commit and container digest.
- [ ] Run bootstrap and connected Rundra doctor checks.
- [ ] Review Rundra plan resources, partition, concurrency, preparation, and retrieval.
- [ ] Use a persistent local Rundra `--data-dir`.

After submission:

- [ ] Retain the exact Run ID.
- [ ] Await through Rundra rather than the native scheduler.
- [ ] Fetch with `--mode copy --extract` before generic merge.
- [ ] Verify Task-manifest and Feather counts.
- [ ] Merge with the original human-authored configuration.
- [ ] Keep raw, merged, and derived data in separate directories.
- [ ] Record source/image/runtime/scheduler provenance.
- [ ] Purge the exact remote workspace only after retrieval is verified.
