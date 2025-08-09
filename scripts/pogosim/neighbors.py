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
    k: int = 3,
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
# Degree/Fano metrics (positions-only; KDTree-based)

def _degrees_within_radius(points: np.ndarray, radius: float) -> np.ndarray:
    """
    @brief  Return the neighbour degree (count within @p radius, excl. self)
            for each point.
    """
    if points.shape[0] < 2:
        return np.empty(0, dtype=int)
    tree = KDTree(points)
    neigh = tree.query_ball_point(points, r=radius)
    return np.asarray([len(lst) - 1 for lst in neigh], dtype=int)


def compute_fano_over_time_corrected(
    df: pd.DataFrame,
    figure_folder: str | Path,
    *,
    communication_radius: float = 133.0,
    run_id: int = 0,
    plot: bool = True,
) -> pd.DataFrame:
    """
    @brief  Fano factor of the per-agent degree distribution over time
            (single run).

    @return DataFrame with columns {time, mean_degree, variance, fano_factor}.
    """
    if "run" not in df.columns:
        df = df.copy()
        df["run"] = 0

    run_data = df[df["run"] == run_id]
    rows: list[tuple[float, float, float, float]] = []

    for t, grp in run_data.groupby("time"):
        pts = grp[["x", "y"]].to_numpy()
        deg = _degrees_within_radius(pts, communication_radius)
        if deg.size == 0:
            continue
        mu = float(deg.mean())
        var = float(deg.var())
        fano = (var / mu) if mu > 0 else np.nan
        rows.append((float(t), mu, var, fano))

    fano_df = pd.DataFrame(rows, columns=["time", "mean_degree", "variance", "fano_factor"])

    if plot and not fano_df.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(fano_df["time"], fano_df["fano_factor"], label="Fano Factor")
        plt.xlabel("Time")
        plt.ylabel("Fano Factor (σ² / μ)")
        plt.title("Fano Factor of Degree Distribution Over Time")
        plt.grid(True)
        plt.tight_layout()
        utils.save_figure(Path(figure_folder) / "fano_factor_over_time_corrected.pdf")

    return fano_df


def compute_overall_neighbor_degree_histogram(
    df: pd.DataFrame,
    figure_folder: str | Path,
    *,
    communication_radius: float = 133.0,
    run_id: int = 0,
    plot: bool = True,
) -> dict | None:
    """
    @brief  Aggregate all per-agent degrees across all times for a single run
            and (optionally) plot the histogram.
    """
    if "run" not in df.columns:
        df = df.copy()
        df["run"] = 0

    run_data = df[df["run"] == run_id]
    all_degrees: list[int] = []

    for _, grp in run_data.groupby("time"):
        pts = grp[["x", "y"]].to_numpy()
        deg = _degrees_within_radius(pts, communication_radius)
        if deg.size:
            all_degrees.extend(deg.tolist())

    if not all_degrees:
        print("No neighbor degrees found.")
        return None

    arr = np.asarray(all_degrees, dtype=int)
    mu = float(arr.mean())
    var = float(arr.var())
    fano = (var / mu) if mu > 0 else np.nan

    print("\n[Overall Degree Stats]")
    print(f"Mean degree: {mu:.2f}")
    print(f"Variance: {var:.2f}")
    print(f"Fano factor: {fano:.2f}")
    print(f"Total samples: {arr.size}")

    if plot:
        plt.figure(figsize=(8, 5))
        bins = np.arange(arr.min(), arr.max() + 2) - 0.5  # centered bins
        plt.hist(arr, bins=bins, edgecolor="black")
        plt.xlabel("Number of neighbors (degree)")
        plt.ylabel("Total agent-time samples")
        plt.title("Overall Neighbor Degree Distribution (All Time Points)")
        plt.grid(True)
        plt.tight_layout()
        utils.save_figure(Path(figure_folder) / "overall_neighbor_degree_histogram.pdf")

    return {"degrees": arr, "mean": mu, "variance": var, "fano_factor": fano}


def compute_degree_histogram_first_last(
    df: pd.DataFrame,
    figure_folder: str | Path,
    *,
    communication_radius: float = 133.0,
    plot: bool = True,
) -> dict:
    """
    @brief  Compare degree distributions at the first and last time step of
            each run; aggregate across runs and (optionally) plot side-by-side.
    """
    if "run" not in df.columns:
        df = df.copy()
        df["run"] = 0

    first_deg: list[int] = []
    last_deg: list[int] = []

    for _, run_df in df.groupby("run"):
        times = sorted(run_df["time"].unique())
        if len(times) < 2:
            continue
        for t, sink in ((times[0], first_deg), (times[-1], last_deg)):
            pts = run_df[run_df["time"] == t][["x", "y"]].to_numpy()
            deg = _degrees_within_radius(pts, communication_radius)
            if deg.size:
                sink.extend(deg.tolist())

    first = np.asarray(first_deg, dtype=int)
    last  = np.asarray(last_deg, dtype=int)

    def _stats(a: np.ndarray) -> dict:
        if a.size == 0:
            return {"mean": np.nan, "var": np.nan, "fano": np.nan, "samples": 0}
        mu = float(a.mean()); var = float(a.var())
        return {"mean": mu, "var": var, "fano": (var / mu) if mu > 0 else np.nan, "samples": int(a.size)}

    stats = {"first": _stats(first), "last": _stats(last)}

    if plot:
        plt.figure(figsize=(12, 5))

        # First time step
        plt.subplot(1, 2, 1)
        if first.size:
            bins = np.arange(first.min(), first.max() + 2) - 0.5
            plt.hist(first, bins=bins, edgecolor="black")
            plt.xlabel("Number of neighbors (degree)")
            plt.ylabel("Count (across all agents & runs)")
            plt.title("First time step")
        else:
            plt.text(0.5, 0.5, "No data", transform=plt.gca().transAxes,
                     ha="center", va="center")

        # Last time step
        plt.subplot(1, 2, 2)
        if last.size:
            bins = np.arange(last.min(), last.max() + 2) - 0.5
            plt.hist(last, bins=bins, edgecolor="black")
            plt.xlabel("Number of neighbors (degree)")
            plt.ylabel("Count (across all agents & runs)")
            plt.title("Last time step")
        else:
            plt.text(0.5, 0.5, "No data", transform=plt.gca().transAxes,
                     ha="center", va="center")

        plt.tight_layout()
        utils.save_figure(Path(figure_folder) / "degree_histogram_first_last.pdf")

    return stats


def compute_fano_over_time_all_runs(
    df: pd.DataFrame,
    figure_folder: str | Path,
    *,
    communication_radius: float = 133.0,
    plot: bool = True,
) -> pd.DataFrame:
    """
    @brief  Fano factor over time after pooling all runs at each time instant.

    @return DataFrame with columns {time, mean_degree, variance, fano_factor}.
    """
    if "run" not in df.columns:
        df = df.copy()
        df["run"] = 0

    rows: list[tuple[float, float, float, float]] = []

    for t, tdf in df.groupby("time"):
        deg_all: list[int] = []
        for _, run_df in tdf.groupby("run"):
            pts = run_df[["x", "y"]].to_numpy()
            deg = _degrees_within_radius(pts, communication_radius)
            if deg.size:
                deg_all.extend(deg.tolist())
        if not deg_all:
            continue
        arr = np.asarray(deg_all, dtype=int)
        mu = float(arr.mean())
        var = float(arr.var())
        fano = (var / mu) if mu > 0 else np.nan
        rows.append((float(t), mu, var, fano))

    fano_df = pd.DataFrame(rows, columns=["time", "mean_degree", "variance", "fano_factor"])

    if plot and not fano_df.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(fano_df["time"], fano_df["fano_factor"], label="Fano Factor")
        plt.xlabel("Time")
        plt.ylabel("Fano Factor (σ² / μ)")
        plt.title("Fano Factor of Degree Distribution Over Time (All Runs)")
        plt.grid(True)
        plt.tight_layout()
        utils.save_figure(Path(figure_folder) / "fano_factor_over_time_all_runs.pdf")

    return fano_df


def compute_overall_neighbor_degree_histogram_all_runs(
    df: pd.DataFrame,
    figure_folder: str | Path,
    *,
    communication_radius: float = 133.0,
    plot: bool = True,
) -> dict | None:
    """
    @brief  Aggregate all per-agent degrees across all runs and times; plot a
            single histogram (optional).
    """
    if "run" not in df.columns:
        df = df.copy()
        df["run"] = 0

    all_deg: list[int] = []
    for (_, _), grp in df.groupby(["run", "time"]):
        pts = grp[["x", "y"]].to_numpy()
        deg = _degrees_within_radius(pts, communication_radius)
        if deg.size:
            all_deg.extend(deg.tolist())

    if not all_deg:
        print("No neighbor degrees found.")
        return None

    arr = np.asarray(all_deg, dtype=int)
    mu = float(arr.mean())
    var = float(arr.var())
    fano = (var / mu) if mu > 0 else np.nan

    print("\n[Overall Degree Stats - All Runs]")
    print(f"Mean degree: {mu:.2f}")
    print(f"Variance: {var:.2f}")
    print(f"Fano factor: {fano:.2f}")
    print(f"Total agent-time samples: {arr.size}")

    if plot:
        plt.figure(figsize=(8, 5))
        bins = np.arange(arr.min(), arr.max() + 2) - 0.5
        plt.hist(arr, bins=bins, edgecolor="black")
        plt.xlabel("Number of neighbors (degree)")
        plt.ylabel("Total agent-time samples")
        plt.title("Neighbor Degree Distribution (All Runs, All Time Points)")
        plt.grid(True)
        plt.tight_layout()
        utils.save_figure(Path(figure_folder) / "overall_neighbor_degree_histogram_all_runs.pdf")

    return {"degrees": arr, "mean": mu, "variance": var, "fano_factor": fano}


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
            Also runs degree/Fano analyses and serializes all stats to a pickle.

    @param[in]  stat_basis  Choose `"run"`, `"robot"` or `"both"`.
    """
    import pickle

    os.makedirs(output_dir, exist_ok=True)

    df, meta = utils.load_dataframe(input_file)
    config = meta.get("configuration", {})

    if "run" not in df.columns:
        df["run"] = 0

    # crude comm-radius heuristic identical to the legacy version
    cfg_rbt = config.get("objects", {}).get("robots", {})
    radius = float(cfg_rbt.get("radius", 26.5))
    comm_radius = float(cfg_rbt.get("communication_radius", 80)) + radius * 2

    # Choose a canonical run for single-run stats: first available run id
    run_ids = sorted(df["run"].unique().tolist())
    first_run_id = int(run_ids[0]) if run_ids else 0

    # ── k-NN distance plots (controlled by stat_basis) ─────────────────────────
    knn_run_df   = None
    knn_robot_df = None

    if stat_basis in ("run", "both"):
        knn_run_df = plot_interindividual_distance_knn_over_time(
            df,
            Path(output_dir) / "knn_run.pdf",
            communication_radius=comm_radius,
            neighbors_col=neighbors_col,
            aggregate_by="run",
        )

    if stat_basis in ("robot", "both"):
        knn_robot_df = plot_interindividual_distance_knn_over_time(
            df,
            Path(output_dir) / "knn_robot.pdf",
            communication_radius=comm_radius,
            neighbors_col=neighbors_col,
            aggregate_by="robot",
        )

    # ── New degree/Fano analyses (always run) ──────────────────────────────────
    fano_time_single_run_df = compute_fano_over_time_corrected(
        df,
        output_dir,
        communication_radius=comm_radius,
        run_id=first_run_id,
        plot=True,
    )

    overall_degree_single_run = compute_overall_neighbor_degree_histogram(
        df,
        output_dir,
        communication_radius=comm_radius,
        run_id=first_run_id,
        plot=True,
    )

    first_last_degree_stats = compute_degree_histogram_first_last(
        df,
        output_dir,
        communication_radius=comm_radius,
        plot=True,
    )

    fano_time_all_runs_df = compute_fano_over_time_all_runs(
        df,
        output_dir,
        communication_radius=comm_radius,
        plot=True,
    )

    overall_degree_all_runs = compute_overall_neighbor_degree_histogram_all_runs(
        df,
        output_dir,
        communication_radius=comm_radius,
        plot=True,
    )

    # ── Serialize everything for later programmatic use ────────────────────────
    stats = {
        "metadata": {
            "comm_radius": comm_radius,
            "first_run_id": first_run_id,
            "input_file": str(input_file),
            "version": __version__,
        },
        "knn": {
            "run_df": knn_run_df,
            "robot_df": knn_robot_df,
        },
        "degree_fano": {
            "fano_time_single_run_df": fano_time_single_run_df,
            "overall_degree_single_run": overall_degree_single_run,  # dict | None
            "first_last_degree_stats": first_last_degree_stats,      # dict
            "fano_time_all_runs_df": fano_time_all_runs_df,
            "overall_degree_all_runs": overall_degree_all_runs,      # dict | None
        },
    }

    with open(Path(output_dir) / "neighbors_stats.pkl", "wb") as fh:
        pickle.dump(stats, fh)


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
