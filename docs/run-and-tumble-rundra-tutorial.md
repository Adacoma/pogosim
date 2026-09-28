# Worked example: run-and-tumble MSD on a cluster

This tutorial compares two run-and-tumble behaviors for **100 Pogobots**:
long runs with short tumbles, and short runs with long tumbles. It uses **128
seeds per condition** (256 simulations), retrieves the results with Rundra,
merges them into one Feather file with Pogobatch, and computes mean squared
displacement (MSD). Run the commands from the root of a Pogosim checkout.

[Watch the 51-second demonstration video](../.description/pogobatch-demo_gpt5.6-sol-xhigh-edited.mp4).
The recording shows a real cluster campaign, including a preparation-path
failure that was corrected before the successful run. Its reported numerical
results are illustrative, not reference values for this tutorial: source,
container, target, and configuration revisions affect them. GitHub strips
hand-written `<video>` elements from repository Markdown, so the link opens the
versioned MP4 rather than an inline player in this page.

For target setup, preparation details, detached submissions, and failure
recovery, use the [complete Pogobatch–Rundra guide](pogobatch-rundra-guide.md).

## 1. Prepare the executable and target

Build and install Pogosim as described in the [README](../README.md), then
install this checkout's Python package and the Rundra command. In particular,
the simulator executable must be built for the **cluster's** environment; the
integrated `cluster` command normally arranges this during Rundra preparation.
Locally, check that the controller builds and the two CLI interfaces exist:

```bash
make -C examples/run_and_tumble sim
python3 -m pip install -e ./scripts
python3 -m pogosim.pogobatch --version
rundr version
```

The commands below use a Rundra target named `isircluster`. This is an example
name, not a target definition shipped with Pogosim. Configure a target for your
site in `~/.config/rundra/targets.yaml`, with its scheduler, shared workspace,
container policy, and SSH authentication. Replace `isircluster` throughout if
your target has another name. Do not put credentials in the experiment YAML.
Before submitting work, inspect the target and confirm that the doctor reports
`ready: true`:

```bash
rundr doctor \
  --target isircluster \
  --targets-file ~/.config/rundra/targets.yaml \
  --data-dir "$PWD/.rundra-runs" \
  --agent generic \
  --json
```

The [full guide's target audit](pogobatch-rundra-guide.md#4-define-and-audit-a-rundra-target)
explains the connected checks and site-specific policies. No cluster jobs are
submitted by this doctor command.

## 2. Inspect the experiment

The complete, runnable configuration is
[`conf/batch/run_and_tumble_msd.yaml`](../conf/batch/run_and_tumble_msd.yaml).
It uses the existing [run-and-tumble controller](../examples/run_and_tumble/main.c)
and keeps robot count, arena, dynamics, logging, and time horizon fixed. Only
the correlated duration settings change:

| Condition | Run duration | Tumble duration |
| --- | --- | --- |
| `long_runs` | 20–30 s | 0.1–0.2 s |
| `long_tumbles` | 0.2–0.4 s | 8–12 s |

The controller's configuration durations are in **milliseconds**. The
hierarchical Pogobatch choice keeps each run/tumble pair together; four
independent `batch_options` would instead generate unwanted combinations.
The YAML requests positions every 1 s for 120 s, logs only the fields needed
for MSD, and disables PNG frames and the GUI.

Planning is local and does not use cluster resources:

```bash
python3 -m pogosim.pogobatch plan \
  --config conf/batch/run_and_tumble_msd.yaml \
  --json
```

Check `combinations: 2`, `task_count: 256`, seeds `0` through `127`, and the
single output name `run_and_tumble_msd.feather`. The seed range is inclusive.
For a cheap end-to-end check before cluster submission, run just seed zero
locally; this still executes both conditions:

```bash
SDL_VIDEODRIVER=dummy python3 -m pogosim.pogobatch run \
  --config conf/batch/run_and_tumble_msd.yaml \
  --simulator-binary ./examples/run_and_tumble/run_and_tumble \
  --seed 0 --jobs 2 --retries 0 \
  --output-dir results/run_and_tumble_smoke

python3 docs/analyze_run_and_tumble_msd.py \
  results/run_and_tumble_smoke/run_and_tumble_msd.feather \
  --expected-runs 1 \
  --output-dir derived/run_and_tumble_smoke
```

The one-seed plot has no meaningful between-seed uncertainty band. It only
checks build, config expansion, data logging, merge, and analysis. The SDL
setting is useful on headless Linux hosts with no display server; remote
container images must likewise provide a working headless SDL driver.

## 3. Launch, retrieve, and merge

Once the local check and Rundra target audit pass, the integrated command
performs the Rundra plan, source/container preparation, submission, wait,
retrieval, and Pogobatch merge:

```bash
python3 -m pogosim.pogobatch cluster \
  --config conf/batch/run_and_tumble_msd.yaml \
  --simulator-binary examples/run_and_tumble/run_and_tumble \
  --seeds 0:127 \
  --target isircluster \
  --targets-file ~/.config/rundra/targets.yaml \
  --source-root "$PWD" \
  --data-dir "$PWD/.rundra-runs" \
  --fetch-destination "$PWD/raw/run_and_tumble_msd" \
  --output-dir "$PWD/results/run_and_tumble_msd" \
  --keep-fetch \
  --json
```

Review the Rundra resource plan and target's concurrency policy before
launching 256 Tasks. If that policy demands explicit confirmation, add
`--confirm-tasks 256`. A remote Apptainer target also needs a suitable image or
an allowed build path; see [container preparation](pogobatch-rundra-guide.md#container-preparation).
Do not assume a local binary can run unchanged on a compute node.

On success, raw Task output remains under `raw/run_and_tumble_msd/` because
`--keep-fetch` is set, and the merged file is
`results/run_and_tumble_msd/run_and_tumble_msd.feather`. Its `condition`,
`run`, and `seed` columns preserve the choice and repeat identity; Pogobatch
also embeds the batch configuration and merge manifest in Feather metadata.
For a long campaign that must detach rather than wait in one command, follow
the [explicit Rundra lifecycle](pogobatch-rundra-guide.md#6-explicit-rundra-lifecycle):
submit once, retain the Run ID, await, fetch, then `pogobatch merge`.

## 4. Calculate the MSD

Run the accompanying [analysis script](analyze_run_and_tumble_msd.py):

```bash
python3 docs/analyze_run_and_tumble_msd.py \
  results/run_and_tumble_msd/run_and_tumble_msd.feather \
  --expected-runs 128 \
  --output-dir derived/run_and_tumble_msd
```

It writes `derived/run_and_tumble_msd/msd_curves.csv` and `msd.png` and prints
the final measured MSD for each condition. For robot *i* in seed *s*, it uses
that robot's **first logged position** as the origin and calculates
`(x(t) - x(0))² + (y(t) - y(0))²` in mm². It averages robots within each run,
then averages the 128 run-level curves. The shaded 95% intervals are a normal
approximation across seeds, not across 12,800 nominally independent robots.
The script checks both conditions, 100 robots per run, complete sample counts,
matching seed sets, and compatible logged time grids before plotting.

This experiment uses **solid boundaries**. Late-time MSD is consequently
bounded by the disk arena and must not be interpreted as unbounded-space
diffusion. Keep the configuration, source revision, container identity, target
plan, merged Feather, and derived outputs together when reporting results.
