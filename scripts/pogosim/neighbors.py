#!/usr/bin/env python3
"""
@file        neighbors.py
@brief       Plotting routines for k‑NN distances with optional explicit neighbour lists.
@details
    This module computes and plots the mean distance to neighbours for each
    time‑step of multiple simulation runs.

    The heavy work for each run is executed in parallel worker processes
    created via ``multiprocessing.get_context("spawn").Pool``.
"""

from __future__ import annotations

import os
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from pogosim import utils
from pogosim import __version__



def _compute_knn_single_run(
    run_df: pd.DataFrame,
    *,
    k: int,
    communication_radius: float,
    neighbors_col: str | None,
    aggregate_by: str,
) -> pd.DataFrame | None:
    """
    @brief  Compute k-NN stats for one simulation run.

    @param[in]  run_df             Slice containing one run.
    @param[in]  k                  Number of nearest neighbours (k-NN).
    @param[in]  communication_radius
                                   Ignore distances strictly larger than this.
    @param[in]  neighbors_col      Column that stores a comma-separated list of
                                   neighbouring robot IDs **or** *None*.
    @param[in]  aggregate_by       `"run"` or `"robot"` (see module doc).

    @return     DataFrame `{time, mean_distance, std_distance, run}` or *None*.
    """
    if k <= 0:
        raise ValueError("k must be ≥ 1")

    use_precomputed = neighbors_col and neighbors_col in run_df.columns
    run_id = run_df["run"].iat[0]

    if aggregate_by == "robot":
        rows: list[tuple[float, int, float]] = []     # time, robot_id, dist
    else:                                             # aggregate_by == "run"
        rows: list[tuple[float, float, float]] = []   # time, μ, σ

    # ------------------------------------------------------------------ loop
    for t, time_df in run_df.groupby("time"):
        # ---------- 1. distances ------------------------------------------------
        if use_precomputed:
            pos_map = (
                time_df[["robot_id", "x", "y"]]
                .set_index("robot_id")
                .apply(np.asarray, axis=1)
                .to_dict()
            )

            # per-robot container (we always need it now)
            robot_means: list[tuple[int, float]] = []

            for _, r in time_df.iterrows():
                nbr_ids = (
                    str(r[neighbors_col]).split(",")
                    if pd.notna(r[neighbors_col])
                    else []
                )
                nbr_ids = [int(n) for n in nbr_ids if str(n).strip()]
                if not nbr_ids:
                    robot_means.append((r.robot_id, np.nan))
                    continue

                dists = [
                    np.linalg.norm(pos_map[n] - np.asarray([r.x, r.y]))
                    for n in nbr_ids
                    if n in pos_map
                ]
                dists = sorted(dists)[:k]
                dists = [d for d in dists if d <= communication_radius]

                robot_means.append(
                    (r.robot_id, np.nanmean(dists) if dists else np.nan)
                )

        else:
            positions = time_df[["x", "y"]].to_numpy()
            robot_ids = time_df["robot_id"].to_numpy()

            dists, _ = KDTree(positions).query(positions, k=k + 1)
            knn = dists[:, 1:]                                # shape (n, k)
            masked = np.where(
                knn > communication_radius, np.nan, knn
            )                                                 # same shape
            valid_any = ~np.isnan(masked).all(axis=1)                      # at least 1 value
            means = np.full(masked.shape[0], np.nan, dtype=float)          # default NaN
            if valid_any.any():                                            # avoid empty slice
                means[valid_any] = np.nanmean(masked[valid_any], axis=1)
            robot_means = list(zip(robot_ids, means))   # (robot_id, μ_i)

        # ---------- 2. aggregation ---------------------------------------------
        if aggregate_by == "robot":
            for rid, μ in robot_means:
                rows.append((t, rid, μ))
        else:  # "run"
            vals = [μ for _, μ in robot_means if not np.isnan(μ)]
            rows.append(
                (t, np.mean(vals) if vals else np.nan, np.std(vals) if vals else np.nan)
            )

    # ---------- 3. build DataFrame --------------------------------------------
    if not rows:
        return None

    if aggregate_by == "robot":
        out = pd.DataFrame(rows, columns=["time", "robot_id", "distance"])
    else:
        out = pd.DataFrame(rows, columns=["time", "mean_distance", "std_distance"])

    out["run"] = run_id
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Public API – parallel orchestration
def plot_interindividual_distance_knn_over_time(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    k: int = 5,
    communication_radius: float = 133.0,
    neighbors_col: str = "neighbors_list",
    aggregate_by: str = "run",
    n_jobs: int | None = None,
) -> pd.DataFrame | None:
    """
    @brief  Driver that launches worker processes and draws the plot.

    All parameters are identical to the legacy version, with two additions:

    @param[in]  neighbors_col      Column name that contains neighbour IDs
                                   (comma-separated).  If missing, the KD-tree
                                   fallback is used automatically.

    @param[in]  aggregate_by       `"run"` or `"robot"`.  Default: `"run"`.
    """
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_jobs) as pool:
        worker = partial(
            _compute_knn_single_run,
            k=k,
            communication_radius=communication_radius,
            neighbors_col=neighbors_col,
            aggregate_by=aggregate_by,
        )
        results = pool.map(worker, (g for _, g in df.groupby("run")))

    frames = [r for r in results if r is not None]
    if not frames:
        print("No valid k-NN distances computed.")
        return None

    result_df = pd.concat(frames, ignore_index=True)

    if aggregate_by == "robot":
        grouped = result_df.groupby("time")
        global_mean = grouped["distance"].mean()
        global_std  = grouped["distance"].std()
        y_label = f"Mean distance to the {k} neighbours (per robot)"
    else:  # "run"
        grouped = result_df.groupby("time")
        global_mean = grouped["mean_distance"].mean()
        global_std  = grouped["mean_distance"].std()
        y_label = f"Mean distance to the {k} neighbours (per run)"

    # ---------- plot ----------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(global_mean.index, global_mean.values, label="Mean")
    plt.fill_between(
        global_mean.index,
        global_mean - global_std,
        global_mean + global_std,
        alpha=0.3,
        label="± 1 σ",
    )

    plt.xlabel("Time")
    plt.ylabel(y_label)
    plt.title(
        f"k-NN metric – {aggregate_by} basis (k={k}, r={communication_radius})"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    utils.save_figure(Path(output_path))

    return result_df


# ──────────────────────────────────────────────────────────────────────────────
# Front-end helpers
def create_all_neighbors_plots(
    input_file: str | Path,
    output_dir: str | Path,
    *,
    neighbors_col: str = "neighbors_list",
    stat_basis: str = "both",  # "run", "robot", "both"
) -> None:
    """
    @brief  Load data, compute communication radius heuristically and emit plots.

    @param[in]  stat_basis  Choose `"run"`, `"robot"` or `"both"`.
    """
    os.makedirs(output_dir, exist_ok=True)

    df, meta = utils.load_dataframe(input_file)
    config = meta.get("configuration", {})

    if "run" not in df.columns:
        df["run"] = 0

    # crude comm-radius heuristic identical to the legacy version
    cfg_rbt = config.get("objects", {}).get("robots", {})
    radius = float(cfg_rbt.get("radius", 26.5))
    comm_radius = float(cfg_rbt.get("communication_radius", 80)) + radius * 2

    if stat_basis in ("run", "both"):
        plot_interindividual_distance_knn_over_time(
            df,
            Path(output_dir) / "knn_run.pdf",
            communication_radius=comm_radius,
            neighbors_col=neighbors_col,
            aggregate_by="run",
        )

    if stat_basis in ("robot", "both"):
        plot_interindividual_distance_knn_over_time(
            df,
            Path(output_dir) / "knn_robot.pdf",
            communication_radius=comm_radius,
            neighbors_col=neighbors_col,
            aggregate_by="robot",
        )


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create k-NN plots (per-run and/or per-robot)."
    )
    parser.add_argument(
        "-i", "--inputFile", default="results/result.feather", help="Input feather file"
    )
    parser.add_argument(
        "-o", "--outputDir", default=".", help="Destination directory for plots"
    )
    parser.add_argument(
        "--neighborsCol",
        default="neighbors_list",
        help="Column containing neighbour IDs (comma-separated)",
    )
    parser.add_argument(
        "--statBasis",
        choices=["run", "robot", "both"],
        default="both",
        help='Aggregate statistic "run", "robot" or "both" (default)',
    )
    args = parser.parse_args()

    create_all_neighbors_plots(
        args.inputFile,
        args.outputDir,
        neighbors_col=args.neighborsCol,
        stat_basis=args.statBasis,
    )

# MODELINE "{{{1"
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
