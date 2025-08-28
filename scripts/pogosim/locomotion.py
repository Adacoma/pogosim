#!/usr/bin/env python3

from __future__ import annotations

import os
import shutil
import subprocess
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.collections import LineCollection
import numpy as np
import seaborn as sns
import pandas as pd

from pogosim import utils
from pogosim import __version__


############### MSD ############### {{{1

def compute_msd_per_agent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-agent Mean Squared Displacement (MSD) within each group.
    MSD is the mean over time of ||r(t) - r(t0)||^2, where r(t0) is the first
    recorded position (x0, y0) for the agent within that group.

    Required columns: ['time','robot_category','robot_id','x','y','run'].
    Optional column:  ['arena_file'] — if present, it is included in grouping/output.
    Uses 'pogobot_ticks' to determine the first sample if available (ties broken by 'time').

    Returns a DataFrame with columns:
        If 'arena_file' present:
            ['arena_file','run','robot_category','robot_id','MSD']
        Otherwise:
            ['run','robot_category','robot_id','MSD']
    """
    # Validate required columns (arena_file is optional)
    required = ['time', 'robot_category', 'robot_id', 'x', 'y', 'run']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe missing required columns: {missing}")

    has_arena = 'arena_file' in df.columns

    # Keep only rows with finite positions
    work = df.loc[np.isfinite(df['x']) & np.isfinite(df['y'])].copy()

    # Sort so groupby('first') is well-defined: prefer ticks if present, then time
    sort_cols = (['arena_file'] if has_arena else []) + ['run', 'robot_category', 'robot_id']
    if 'pogobot_ticks' in work.columns:
        sort_cols += ['pogobot_ticks', 'time']
    else:
        sort_cols += ['time']
    work = work.sort_values(sort_cols, kind='mergesort')  # stable

    # Grouping keys
    group_keys = (['arena_file'] if has_arena else []) + ['run', 'robot_category', 'robot_id']

    # First position per agent group (after sorting)
    x0 = work.groupby(group_keys, sort=False)['x'].transform('first')
    y0 = work.groupby(group_keys, sort=False)['y'].transform('first')

    # Instantaneous squared displacement relative to first sample
    dx = work['x'] - x0
    dy = work['y'] - y0
    work['__msd__'] = dx * dx + dy * dy

    # Mean over time within each group
    out = (
        work.groupby(group_keys, sort=False)['__msd__']
            .mean()
            .reset_index()
            .rename(columns={'__msd__': 'MSD'})
    )

    # Order columns nicely
    ordered_cols = group_keys + ['MSD']
    out = out[ordered_cols]

    return out


############### HEATMAP ############### {{{1

def plot_arena_heatmaps(
    df: pd.DataFrame,
    *,
    x_col: str = "x",
    y_col: str = "y",
    arena_col: str = "arena_file",
    bins: int | Tuple[int, int] = 150,
    pdf_path: str | Path = "arena_heatmaps.pdf",
    virtual_arena_name: str = "unnamed_arena",
    points_threshold: int = 50_000,
    dpi: int = 300,
    cmap: str = "magma",
    log_norm: bool = False,
    base_panel_width: float = 4.0,
    base_panel_height: float = 4.0,
    # ── Display knobs ─────────────────────────────────────────────────── #
    use_kde: bool = False,
    show_grid_lines: bool = False,
    show_axes_labels: bool = True,
    bin_value: str = "count",             # "count" | "density"
    cbar_shrink: float = 0.80,            # colour-bar height factor
) -> None:
    """
    Render one landscape PDF page that holds a left-to-right row of heat-maps,
    **with X and Y axes in real coordinates**.
    """

    sns.set_context("paper", font_scale=1.5)      # (3) bigger everything
    plt.rcParams["axes.titlesize"] = "x-large"
    plt.rcParams["axes.labelsize"] = "large"

    # ── 1. basic validation ───────────────────────────────────────────── #
    missing = {x_col, y_col} - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing column(s): {', '.join(missing)}")
    if bin_value not in {"count", "density"}:
        raise ValueError("bin_value must be 'count' or 'density'")

    # ── 2. make sure an arena identifier exists ───────────────────────── #
    if arena_col not in df.columns:
        df = df.copy()
        df[arena_col] = virtual_arena_name

    arenas = df[arena_col].unique()
    n_arenas = len(arenas)

    # ── 3. build a single-row figure ──────────────────────────────────── #
    fig_w, fig_h = 2.0 + base_panel_width * n_arenas, 1.0 + base_panel_height
    fig, axes = plt.subplots(1, n_arenas, figsize=(fig_w, fig_h), squeeze=False)

    # ── 4. loop through arenas ────────────────────────────────────────── #
    for ax, arena in zip(axes.flatten(), arenas):
        subset = df.loc[df[arena_col] == arena, [x_col, y_col]]
        if len(subset) > points_threshold:
            subset = subset.sample(points_threshold, random_state=0)

        # ---------------------------------------------------------------- #
        #  KDE branch – seaborn already plots in coordinate space          #
        # ---------------------------------------------------------------- #
        if use_kde:
            sns.kdeplot(
                data=subset,
                x=x_col,
                y=y_col,
                fill=True,
                cmap=cmap,
                levels=100,
                thresh=0.0,
                ax=ax,
                cbar=True,
                cbar_kws={"label": "Density", "shrink": cbar_shrink, "pad": 0.10},
            )

        # ---------------------------------------------------------------- #
        #  Histogram branch – build 2-D counts/density then draw with      #
        #  imshow(extent=…) so the axes show real coordinates              #
        # ---------------------------------------------------------------- #
        else:
            hist, y_edges, x_edges = np.histogram2d(
                subset[y_col], subset[x_col], bins=bins
            )

            if bin_value == "density":
                area = np.outer(np.diff(y_edges), np.diff(x_edges))
                hist = hist / area / len(subset)

            # choose linear vs log norm
            norm = plt.LogNorm() if (log_norm and bin_value == "count") else None

            mappable = ax.imshow(
                hist,
                origin="upper",                        # row 0 at top …
                extent=[x_edges[0], x_edges[-1], y_edges[-1], y_edges[0]],
                cmap=cmap,
                norm=norm,
                aspect="equal",
            )

            # (optional) draw white grid lines at bin edges
            if show_grid_lines:
                for x in x_edges:
                    ax.axvline(x, color="white", lw=0.3)
                for y in y_edges:
                    ax.axhline(y, color="white", lw=0.3)

            # colour-bar
            cbar = fig.colorbar(
                mappable,
                ax=ax,
                shrink=cbar_shrink,
                pad=0.10,
                label="Count" if bin_value == "count" else "Density",
            )

        # ---- common cosmetics ----------------------------------------- #
        ax.set_title(f"{arena}   (n={len(subset):,})", pad=6)

        if show_axes_labels:
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
        else:
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])

        # (1) invert Y so (0, 0) is top-left, matching earlier request
        ax.invert_yaxis()

        # (5) fewer ticks for cleanliness
        ax.locator_params(axis="both", nbins=5)

        ax.set_box_aspect(1)

    plt.subplots_adjust(left=0.15, bottom=0.10, right=0.93, top=0.94)

    # ── 5. save a single-page PDF ─────────────────────────────────────── #
    utils.save_figure(pdf_path, dpi=dpi)


############### TRACES ############### {{{1

# ──────────────────────────────────────────────────────────────────────
#  gifski helper
# ──────────────────────────────────────────────────────────────────────
def _compile_gif(
    frame_paths: List[str],
    gif_path: Path,
    fps: int,
    gifski_bin: str = "gifski",
) -> bool:
    gifski_exe = shutil.which(gifski_bin)
    if gifski_exe is None:
        print(f"[WARNING] gifski binary not found ('{gifski_bin}'). GIF skipped.")
        return False
    try:
        subprocess.run(
            [gifski_exe, "-q", "-r", str(fps), "--output", str(gif_path), *frame_paths],
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] gifski failed ({e}). GIF not produced for {gif_path}.")
        return False


# ──────────────────────────────────────────────────────────────────────
#  single-run renderer
# ──────────────────────────────────────────────────────────────────────
def _render_single_run(
    run_df: pd.DataFrame,
    run_output_dir: Path,
    *,
    k_steps: int,
    robot_cmap_name: str,
    point_size: int,
    line_width: float,
    fade_min_alpha: float,
    dpi: int,
    make_gif: bool,
    gif_fps: int,
    gif_name: str,
    gifski_bin: str,
    margin_frac: float = 0.03,          # ← added: % margin around arena
) -> List[str]:
    """
    Render one run.  Axis limits are fixed from the full run extent,
    so early frames no longer have "smaller borders".
    """
    run_df = run_df.sort_values(["time", "robot_id"], ignore_index=True)

    # ── arena bounds (fixed for every frame) ────────────────────────────
    x_min, x_max = run_df["x"].min(), run_df["x"].max()
    y_min, y_max = run_df["y"].min(), run_df["y"].max()
    # add a small margin so dots aren’t exactly on the edge
    dx, dy = x_max - x_min, y_max - y_min
    x_min -= dx * margin_frac
    x_max += dx * margin_frac
    y_min -= dy * margin_frac
    y_max += dy * margin_frac

    times      = run_df["time"].unique()
    robot_ids  = np.sort(run_df["robot_id"].unique())
    cmap       = get_cmap(robot_cmap_name)
    colour_map = {rid: cmap(i % cmap.N)[:3] for i, rid in enumerate(robot_ids)}

    tail_times: List[float] = []
    frame_paths: List[str]  = []

    run_output_dir.mkdir(parents=True, exist_ok=True)

    for current_time in times:
        tail_times.append(current_time)
        if len(tail_times) > k_steps:
            tail_times.pop(0)

        window_df = run_df[run_df["time"].isin(tail_times)]
        t_old, t_new = tail_times[0], tail_times[-1]
        age_den = (t_new - t_old) or 1.0

        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("white")
        ax.set_xticks([]); ax.set_yticks([])

        # ← NEW: keep arena size constant
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        for rid, group in window_df.groupby("robot_id", sort=False):
            g = group.sort_values("time")
            xs, ys, ts = g["x"].to_numpy(), g["y"].to_numpy(), g["time"].to_numpy()

            if len(xs) > 1:
                segs = np.stack(
                    [np.column_stack([xs[:-1], ys[:-1]]),
                     np.column_stack([xs[1:],  ys[1: ]])],
                    axis=1
                )
                seg_ages   = (ts[1:] - t_old) / age_den
                seg_alphas = fade_min_alpha + (1 - fade_min_alpha) * seg_ages
                seg_rgba   = [(*colour_map[rid], a) for a in seg_alphas]

                ax.add_collection(LineCollection(
                    segs,
                    colors     = seg_rgba,
                    linewidths = line_width,
                    capstyle   = "round",
                    joinstyle  = "round",
                ))

            ax.scatter(xs[-1], ys[-1],
                       s = point_size,
                       c = [colour_map[rid]],
                       edgecolors = "none")

        ax.set_title(f"time = {current_time:.3f}   (tail = {len(tail_times)} steps)")
        fig.tight_layout()

        fname = run_output_dir / f"trace_{current_time:.6f}.png"
        fig.savefig(fname, dpi=dpi)
        plt.close(fig)
        frame_paths.append(str(fname.resolve()))

    if make_gif and frame_paths:
        _compile_gif(frame_paths,
                     run_output_dir / gif_name,
                     fps = gif_fps,
                     gifski_bin = gifski_bin)

    return frame_paths


# ──────────────────────────────────────────────────────────────────────
#  worker wrapper (needed for Pool.map)
# ──────────────────────────────────────────────────────────────────────
def _process_run(args: Tuple[int, pd.DataFrame, str, dict]) -> Tuple[int, List[str]]:
    run_val, run_df, out_dir_str, kw = args
    paths = _render_single_run(run_df,
                               Path(out_dir_str),
                               **kw)
    return run_val, paths


# ──────────────────────────────────────────────────────────────────────
#  public API
# ──────────────────────────────────────────────────────────────────────
def generate_trace_images(
    df: pd.DataFrame,
    *,
    k_steps: int = 20,
    output_dir: str | os.PathLike = "trace_frames",
    run_id: int | None = None,
    robot_cmap_name: str = "tab20",
    point_size: int = 30,
    line_width: float = 2.0,
    fade_min_alpha: float = 0.1,
    dpi: int = 150,
    run_fmt: str = "run_{run}",
    # GIF options
    make_gif: bool = False,
    gif_fps: int = 20,
    gif_name: str = "trace.gif",
    gifski_bin: str = "gifski",
    # Parallelism
    n_jobs: int | None = None,
) -> Union[List[str], Dict[int, List[str]]]:
    """
    Render fading-trail PNGs (and optional GIFs) from a robot-trace dataframe.

    Parallelisation:
        • If `run_id` is None and the dataframe has a 'run' column, individual runs
          are processed in **parallel** with a multiprocessing pool (`n_jobs` workers).
        • Set `n_jobs=1` to disable the pool (sequential).
    """
    df = df.copy()

    # ------------ single-run request -------------------------------------
    if run_id is not None:
        if "run" in df.columns:
            df = df[df["run"] == run_id]
        return _render_single_run(
            df,
            Path(output_dir),
            k_steps=k_steps,
            robot_cmap_name=robot_cmap_name,
            point_size=point_size,
            line_width=line_width,
            fade_min_alpha=fade_min_alpha,
            dpi=dpi,
            make_gif=make_gif,
            gif_fps=gif_fps,
            gif_name=gif_name,
            gifski_bin=gifski_bin,
        )

    # ------------ automatic per-run processing ---------------------------
    if "run" in df.columns:
        runs = sorted(df["run"].unique())
        base_dir = Path(output_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        # common kwargs for _render_single_run
        common_kw = dict(
            k_steps=k_steps,
            robot_cmap_name=robot_cmap_name,
            point_size=point_size,
            line_width=line_width,
            fade_min_alpha=fade_min_alpha,
            dpi=dpi,
            make_gif=make_gif,
            gif_fps=gif_fps,
            gif_name=gif_name,
            gifski_bin=gifski_bin,
        )

        # prepare task list (arg tuple per run)
        tasks: List[Tuple[int, pd.DataFrame, str, dict]] = [
            (
                r,
                df[df["run"] == r],
                str(base_dir / run_fmt.format(run=r)),
                common_kw,
            )
            for r in runs
        ]

        # sequential if n_jobs == 1
        if n_jobs == 1:
            results = [_process_run(t) for t in tasks]
        else:
            workers = n_jobs or os.cpu_count() or 1
            ctx = mp.get_context("spawn")   # safest across platforms
            with ctx.Pool(processes=workers) as pool:
                results = pool.map(_process_run, tasks)

        return {run_val: paths for run_val, paths in results}

    # ------------ dataframe without 'run' column → treat as single run ---
    return _render_single_run(
        df,
        Path(output_dir),
        k_steps=k_steps,
        robot_cmap_name=robot_cmap_name,
        point_size=point_size,
        line_width=line_width,
        fade_min_alpha=fade_min_alpha,
        dpi=dpi,
        make_gif=make_gif,
        gif_fps=gif_fps,
        gif_name=gif_name,
        gifski_bin=gifski_bin,
    )


############### MAIN ############### {{{1

def create_all_locomotion_plots(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df, meta = utils.load_dataframe(input_file)
    config = meta.get("configuration", {})

    # Insert a run column, might be needed for some plotting functions
    if "run" not in df.columns:
        df["run"] = 0

    # Create Heatmaps
    print("Creating heatmaps...")
    plot_arena_heatmaps(df, bins=30, pdf_path=os.path.join(output_dir, "arena_heatmaps.pdf"), use_kde=False, bin_value="density", show_grid_lines=False)
    plot_arena_heatmaps(df, bins=30, pdf_path=os.path.join(output_dir, "arena_heatmaps_kde.pdf"), use_kde=True, bin_value="density", show_grid_lines=False)

    # Create trace plots
    print("Creating trace plots...")
    trace_path = os.path.join(output_dir, "traces")
    shutil.rmtree(trace_path, ignore_errors=True)
    os.makedirs(trace_path, exist_ok=True)
    generate_trace_images(df, k_steps=20, output_dir=trace_path, make_gif=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputFile', type=str, default='results/result.feather', help = "Path of the input feather file")
    parser.add_argument('-o', '--outputDir', type=str, default=".", help = "Directory of the resulting plot files")
    args = parser.parse_args()

    input_file = args.inputFile
    output_dir = args.outputDir
    create_all_locomotion_plots(input_file, output_dir)


# MODELINE "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
