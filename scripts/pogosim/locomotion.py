#!/usr/bin/env python3

from __future__ import annotations

import os
import shutil
import subprocess
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Tuple

from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.collections import LineCollection
import numpy as np
from numpy.typing import ArrayLike
import seaborn as sns
import pandas as pd

from pogosim import utils
from pogosim import __version__

# Aggregate across agents at each time stamp without .apply
def _sem(s: pd.Series) -> float:
    n = s.size
    return (s.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0


############### SPEEDS / STATS / PECLET ############### {{{1
# All functions respect the optional 'arena_file' column.
# Groups are (arena_file?), run?, robot_category, robot_id

@dataclass
class WindowSpec:
    """
    Sliding window specification:
      • Use either window_steps (int) OR window_s (float seconds).
      • If window_s is given, we estimate steps per second from the per-agent median Δt.
    """
    window_steps: int | None = None
    window_s: float | None = None

def _group_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    if 'arena_file' in df.columns: cols.append('arena_file')
    if 'run' in df.columns:        cols.append('run')
    cols += ['robot_category', 'robot_id']
    return cols

def _sort_cols(df: pd.DataFrame) -> list[str]:
    cols = _group_cols(df)
    if 'pogobot_ticks' in df.columns:
        cols += ['pogobot_ticks', 'time']
    else:
        cols += ['time']
    return cols

def _unwrap(angles: np.ndarray) -> np.ndarray:
    # Unwrap to avoid 2π jumps then re-center to (-π, π] only for diffs
    return np.unwrap(angles)

def compute_instantaneous_speeds(
    df: pd.DataFrame,
    *,
    center_time: bool = True,
) -> pd.DataFrame:
    """
    Add instantaneous linear speed v [mm/s] and angular speed omega [rad/s]
    computed per agent between consecutive samples.

    Returns a new DataFrame with original columns + ['v','omega'] and:
        • 'time' = next time if center_time=False
        • 'time_mid' column added if center_time=True (midpoint of the segment)
    Notes:
        - First sample of each agent has NaN speeds (no previous point).
        - Requires columns: time, robot_category, robot_id, x, y, angle.
          (run/arena_file optional; pogobot_ticks optional)
    """
    required = ['time','robot_category','robot_id','x','y','angle']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe missing required columns: {missing}")

    work = df.copy()
    work = work.sort_values(_sort_cols(work), kind='mergesort')

    gcols = _group_cols(work)
    # per-group diffs
    for col in ['x','y','angle','time']:
        work[f'__d_{col}'] = work.groupby(gcols, sort=False)[col].diff()

    # unwrap angle before taking the difference to avoid jumps
    # (we need to recompute d_angle using unwrapped angle)
    work['__angle_unwrap'] = work.groupby(gcols, sort=False)['angle'].transform(_unwrap)
    work['__d_angle']      = work.groupby(gcols, sort=False)['__angle_unwrap'].diff()

    dt = work['__d_time'].to_numpy()
    dx = work['__d_x'].to_numpy()
    dy = work['__d_y'].to_numpy()
    dth = work['__d_angle'].to_numpy()

    with np.errstate(divide='ignore', invalid='ignore'):
        v = np.hypot(dx, dy) / dt
        omega = dth / dt

    work['v'] = v
    work['omega'] = omega

    if center_time:
        # midpoint of the segment for nicer time-series aggregation
        t_prev = work.groupby(gcols, sort=False)['time'].shift(1)
        work['time_mid'] = (t_prev + work['time']) / 2.0
    # Clean up helper cols (keep time_mid if present)
    to_drop = [c for c in work.columns if c.startswith('__d_') or c == '__angle_unwrap']
    work = work.drop(columns=to_drop)

    return work


def summarize_speed_stats(
    df: pd.DataFrame,
    *,
    window: WindowSpec | None = None,
    aggregate: bool = True,
    # plotting
    plot: bool = False,
    plot_kind: str = "time",   # "time" (mean ± SEM) or "box" (per-agent summary)
    figsize: tuple[float,float] = (7,4),
    savepath: str | None = None,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Compute per-agent mean/variance for linear speed v and angular speed ω.
    Optionally compute sliding-window stats and plot mean ± error bars.

    Returns:
      agent_stats, time_stats
        - agent_stats: one row per agent group with columns:
              [group-cols..., 'v_mean','v_var','omega_mean','omega_var','n_segments']
        - time_stats (if window or plot_kind=='time'): time series aggregated over agents:
              columns: [time, mean_v, sem_v, mean_omega, sem_omega, n_agents]
          Otherwise None.

    Notes:
      • If *window* is provided, we:
          1) compute instantaneous speeds with mid-times,
          2) within each agent, do a rolling mean over window (steps or seconds),
          3) aggregate across agents to mean ± SEM per time bin.
      • If *window* is None and plot_kind=='time', we still aggregate by time_mid
        (no smoothing) for mean ± SEM ribbons.
    """
    # 1) instantaneous speeds
    speeds = compute_instantaneous_speeds(df, center_time=True)

    # per-agent summary (drop NaNs from first diff)
    valid = speeds.dropna(subset=['v','omega'])
    gcols = _group_cols(valid)

    agent_stats = (
        valid.groupby(gcols, sort=False)
             .agg(v_mean=('v','mean'),
                  v_var =('v','var'),
                  omega_mean=('omega','mean'),
                  omega_var =('omega','var'),
                  n_segments=('v','size'))
             .reset_index()
    )

    time_stats: Optional[pd.DataFrame] = None

    # 2) optional sliding window
    ts_col = 'time_mid' if 'time_mid' in valid.columns else 'time'
    if plot or window is not None or plot_kind == "time":
        # Prepare per-agent rolling
        def _estimate_steps_per_sec(g: pd.DataFrame) -> float:
            dt = g[ts_col].diff().dropna()
            if dt.empty:
                return 1.0
            return 1.0 / float(np.median(dt))

        def _apply_window(g: pd.DataFrame) -> pd.DataFrame:
            g = g.sort_values(ts_col)
            if window is None:
                g['v_roll'] = g['v']
                g['omega_roll'] = g['omega']
                return g
            if window.window_steps is not None:
                w = int(max(1, window.window_steps))
            elif window.window_s is not None:
                sps = _estimate_steps_per_sec(g)
                w = int(max(1, round(sps * window.window_s)))
            else:
                w = 1
            g['v_roll'] = g['v'].rolling(window=w, min_periods=max(1, w//3)).mean()
            g['omega_roll'] = g['omega'].rolling(window=w, min_periods=max(1, w//3)).mean()
            return g

        needed = [ts_col, 'v', 'omega']
        rolled = (
            valid.groupby(gcols, sort=False, group_keys=False)[needed]
                 .apply(_apply_window)   # returns frame with v_roll/omega_roll added
                 .dropna(subset=['v_roll', 'omega_roll'])
        )

        # Aggregate over agents at each time stamp (align by time_mid)
        def _agg_over_agents(h: pd.DataFrame) -> pd.Series:
            return pd.Series({
                'mean_v':  h['v_roll'].mean(),
                'sem_v':   h['v_roll'].std(ddof=1) / np.sqrt(len(h)) if len(h)>1 else 0.0,
                'mean_om': h['omega_roll'].mean(),
                'sem_om':  h['omega_roll'].std(ddof=1) / np.sqrt(len(h)) if len(h)>1 else 0.0,
                'n_agents': h.shape[0],
            })

        time_stats = (
            rolled.groupby(ts_col, sort=True)
                  .agg(mean_v=('v_roll', 'mean'),
                       sem_v =('v_roll', _sem),
                       mean_om=('omega_roll', 'mean'),
                       sem_om =('omega_roll', _sem),
                       n_agents=('v_roll', 'size'))
                  .reset_index()
                  .rename(columns={ts_col: 'time'})
        )


        # 3) plotting
        if plot:
            import matplotlib.pyplot as plt
            sns.set_context("paper", font_scale=1.4)

            if plot_kind == "time":
                fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
                ax1, ax2 = axes

                # v
                ax1.plot(time_stats['time'], time_stats['mean_v'], label='⟨v⟩')
                ax1.fill_between(time_stats['time'],
                                 time_stats['mean_v']-time_stats['sem_v'],
                                 time_stats['mean_v']+time_stats['sem_v'],
                                 alpha=0.25, label='± SEM')
                ax1.set_ylabel('speed ⟨v⟩')
                ax1.legend(loc='best')

                # omega
                ax2.plot(time_stats['time'], time_stats['mean_om'], label='⟨ω⟩')
                ax2.fill_between(time_stats['time'],
                                 time_stats['mean_om']-time_stats['sem_om'],
                                 time_stats['mean_om']+time_stats['sem_om'],
                                 alpha=0.25, label='± SEM')
                ax2.set_ylabel('angular speed ⟨ω⟩ [rad/s]')
                ax2.set_xlabel('time [s]')
                ax2.legend(loc='best')

                fig.tight_layout()
                if savepath:
                    utils.save_figure(savepath)
                else:
                    plt.show()

            elif plot_kind == "box":
                # box/violin per-agent summary
                fig, axes = plt.subplots(1, 2, figsize=figsize)
                sns.boxplot(data=agent_stats, y='v_mean', ax=axes[0])
                axes[0].set_title('Per-agent ⟨v⟩')
                sns.boxplot(data=agent_stats, y='omega_mean', ax=axes[1])
                axes[1].set_title('Per-agent ⟨ω⟩')
                fig.tight_layout()
                if savepath:
                    utils.save_figure(savepath)
                else:
                    plt.show()

    return agent_stats, time_stats


def compute_peclet(
    df: pd.DataFrame,
    *,
    characteristic_length: float,
    t_fit_min: float | None = None,
    t_fit_max: float | None = None,
    window: WindowSpec | None = None,
    return_diffusivity: bool = False,
) -> pd.DataFrame:
    """
    Compute a per-agent Péclet number:
        Pe = (mean linear speed) * L / D
    where:
        L = characteristic_length (you provide, in the same units as x,y, i.e. mm)
        D = translational diffusivity estimated from MSD(t) ≈ 4 D t (2D)

    We estimate D by:
      1) Aligning each agent's trajectory to its first sample (elapsed time τ=t−t0).
      2) Computing ⟨|r(τ)−r(0)|²⟩_τ as a function of τ.
      3) Fitting a straight line in [t_fit_min, t_fit_max] to get slope ≈ 4D.

    If *window* is provided, we compute mean speed ⟨v⟩ using the same sliding window
    strategy as in summarize_speed_stats (median-Δt → steps-per-second).

    Returns a DataFrame with columns:
        [group-cols..., 'v_mean', 'D', 'Pe'] (+ 'D_r2','D_npts' for fit diagnostics)
    """
    required = ['time','robot_category','robot_id','x','y','angle']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe missing required columns: {missing}")
    if characteristic_length <= 0:
        raise ValueError("characteristic_length must be > 0.")

    # 1) mean speed (optionally windowed) per agent
    agent_stats, time_stats = summarize_speed_stats(
        df, window=window, plot=False, aggregate=True, plot_kind='time'
    )
    v_means = agent_stats.set_index(_group_cols(agent_stats))['v_mean']

    # 2) build (elapsed_time, disp2) per agent
    work = df.copy().sort_values(_sort_cols(df), kind='mergesort')
    gcols = _group_cols(work)

    # First sample per group
    x0 = work.groupby(gcols, sort=False)['x'].transform('first')
    y0 = work.groupby(gcols, sort=False)['y'].transform('first')
    t0 = work.groupby(gcols, sort=False)['time'].transform('first')
    work['__tau']   = work['time'] - t0
    work['__disp2'] = (work['x'] - x0)**2 + (work['y'] - y0)**2

    # We will fit MSD(τ) over requested τ window, per agent
    def _fit_D(gr: pd.DataFrame) -> pd.Series:
        g = gr[['__tau','__disp2']].dropna()
        g = g[g['__tau'] > 0].sort_values('__tau')

        if t_fit_min is not None:
            g = g[g['__tau'] >= t_fit_min]
        if t_fit_max is not None:
            g = g[g['__tau'] <= t_fit_max]
        if len(g) < 3:
            return pd.Series({'D': np.nan, 'D_r2': np.nan, 'D_npts': len(g)})

        # simple least-squares fit: disp2 = a * τ + b  (expect b≈0)
        x = g['__tau'].to_numpy()
        y = g['__disp2'].to_numpy()
        A = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]

        # R^2
        yhat = a*x + b
        ss_res = np.sum((y - yhat)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = 1.0 - (ss_res/ss_tot if ss_tot > 0 else np.nan)

        D = a / 4.0   # 2D: MSD ≈ 4 D τ
        return pd.Series({'D': float(D), 'D_r2': float(r2), 'D_npts': len(g)})

    diff_stats = (
        work.groupby(gcols, sort=False)[['__tau', '__disp2']]
            .apply(_fit_D)          # _fit_D only needs these two cols
            .reset_index()
    )

    # 3) assemble Pe = v_mean * L / D
    diff_stats = diff_stats.set_index(gcols)
    common_index = diff_stats.index.intersection(v_means.index)

    out = diff_stats.loc[common_index].copy()
    out['v_mean'] = v_means.loc[common_index]
    out['Pe'] = (out['v_mean'] * float(characteristic_length)) / out['D']
    out = out.reset_index()

    if not return_diffusivity:
        out = out[_group_cols(out) + ['v_mean','Pe']]
    else:
        out = out[_group_cols(out) + ['v_mean','D','D_r2','D_npts','Pe']]

    return out


############### MSD ############### {{{1

def compute_msd_per_agent(
    df: pd.DataFrame,
    *,
    per_lag: bool = False,
) -> pd.DataFrame:
    """
    Compute Mean Squared Displacement (MSD).

    Modes
    -----
    per_lag = False (default, backward-compatible):
        Returns one row per agent group (…,'MSD'), where MSD is the mean over time
        of ||r(t) - r(t0)||^2 with t0 = first sample of that agent group.

    per_lag = True:
        Returns many rows per agent group with columns:
            [... group-cols ...,'tau','disp2']
        where tau = t - t0 and disp2 = ||r(t) - r(t0)||^2, for each sample.

    Required columns: ['time','robot_category','robot_id','x','y','run'].
    Optional column:  ['arena_file'] — included in grouping/output if present.
    If 'pogobot_ticks' exists, it is used (with 'time') to define the first sample.
    """
    required = ['time', 'robot_category', 'robot_id', 'x', 'y', 'run']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe missing required columns: {missing}")

    has_arena = 'arena_file' in df.columns

    work = df.loc[np.isfinite(df['x']) & np.isfinite(df['y'])].copy()

    # Stable sort so the "first" per group is well-defined.
    sort_cols = (['arena_file'] if has_arena else []) + ['run', 'robot_category', 'robot_id']
    if 'pogobot_ticks' in work.columns:
        sort_cols += ['pogobot_ticks', 'time']
    else:
        sort_cols += ['time']
    work = work.sort_values(sort_cols, kind='mergesort')

    group_keys = (['arena_file'] if has_arena else []) + ['run', 'robot_category', 'robot_id']

    # First sample per group
    x0 = work.groupby(group_keys, sort=False)['x'].transform('first')
    y0 = work.groupby(group_keys, sort=False)['y'].transform('first')
    t0 = work.groupby(group_keys, sort=False)['time'].transform('first')

    dx = work['x'] - x0
    dy = work['y'] - y0
    disp2 = dx * dx + dy * dy

    if per_lag:
        out = work.copy()
        out['tau'] = work['time'] - t0
        out['disp2'] = disp2
        # keep only non-negative tau; sort for nice downstream grouping
        out = out.loc[out['tau'] >= 0, group_keys + ['tau', 'disp2']]
        out = out.sort_values(group_keys + ['tau'], kind='mergesort').reset_index(drop=True)
        return out

    # Backward-compatible: one mean "MSD" per agent group
    out = (
        pd.DataFrame({'__msd__': disp2})
        .join(work[group_keys])
        .groupby(group_keys, sort=False)['__msd__']
        .mean()
        .reset_index()
        .rename(columns={'__msd__': 'MSD'})
    )
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
        #ax.set_title(f"{arena}   (n={len(subset):,})", pad=6)

        if show_axes_labels:
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
        else:
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])

        # Make (0,0) top-left ONLY for KDE; histogram already flipped via imshow()
        if use_kde:
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

def _fit_power_law_alpha(msd_time: pd.DataFrame, *, frac: float = 0.25) -> tuple[float | None, float | None]:
    """
    Estimate MSD ~ K * tau^alpha by OLS on log–log for two windows:
      - early: first 'frac' fraction of positive points
      - late:  last  'frac' fraction of positive points
    Returns (alpha_early, alpha_late).
    """
    g = msd_time[(msd_time['tau'] > 0) & (msd_time['mean_msd'] > 0)]
    if len(g) < 6:
        return None, None
    n = len(g)
    k = max(3, int(round(frac * n)))
    def _slope(df: pd.DataFrame) -> float | None:
        x = np.log(df['tau'].to_numpy())
        y = np.log(df['mean_msd'].to_numpy())
        A = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(a)
    early = _slope(g.iloc[:k])
    late  = _slope(g.iloc[-k:])
    return early, late


def _fmt_float(x: float | None, prec: int = 3) -> str:
    return "n/a" if (x is None or not np.isfinite(x)) else f"{x:.{prec}f}"


def _summarize_pe(pe: np.ndarray) -> dict:
    pe = pe[np.isfinite(pe)]
    if pe.size == 0:
        return {}
    return {
        "count": pe.size,
        "mean": float(np.mean(pe)),
        "std": float(np.std(pe, ddof=1)) if pe.size > 1 else 0.0,
        "pct10": float(np.percentile(pe, 10)),
        "pct50": float(np.percentile(pe, 50)),
        "pct90": float(np.percentile(pe, 90)),
        "frac_gt_1": float(np.mean(pe > 1.0)),
        "frac_gt_10": float(np.mean(pe > 10.0)),
    }


def _regime_hint(alpha_early: float | None, alpha_late: float | None, pe90: float | None, mean_omega: float | None, mean_v: float | None) -> list[str]:
    hints: list[str] = []
    # MSD regime heuristics
    if alpha_early is not None:
        if alpha_early > 1.5: hints.append("Short-time motion looks ballistic (α_early > 1.5).")
        elif alpha_early < 0.8: hints.append("Short-time subdiffusion/caging (α_early < 0.8).")
        else: hints.append("Short-time near-diffusive (α_early ≈ 1).")
    if alpha_late is not None:
        if alpha_late > 1.2: hints.append("Long-time superdiffusive/persistent transport (α_late > 1.2).")
        elif alpha_late < 0.8: hints.append("Long-time subdiffusion/possible confinement (α_late < 0.8).")
        else: hints.append("Long-time near-diffusive (α_late ≈ 1).")
    # Péclet
    if pe90 is not None:
        if pe90 > 10:
            hints.append("High activity (Pe_90 > 10): activity-dominated transport; clustering/MIPS may be possible depending on density.")
        elif pe90 > 1:
            hints.append("Moderate activity (Pe_90 > 1): advection competes with diffusion.")
        else:
            hints.append("Low activity (Pe mostly ≤ 1): diffusion-dominated.")
    # Rotation cue
    if mean_omega is not None and mean_v is not None:
        # crude non-dimensional cue: compare rotation period to translation scale
        # If rotation period << observation window and v small → vortexy/spinning.
        if abs(mean_omega) > 0.5 and mean_v < 0.1:
            hints.append("Strong mean rotation with low translation: possible vortex/spinning behavior.")
    return hints


def write_text_summary(
    output_dir: str,
    *,
    agent_stats: pd.DataFrame | None,
    time_stats: pd.DataFrame | None,
    peclet_df: pd.DataFrame | None,
    msd_time: pd.DataFrame | None,
) -> None:
    """
    Create SUMMARY.txt with key scalars + regime hints and also print to stdout.
    """
    lines: list[str] = []
    lines.append("# Locomotion analysis — textual summary\n")

    # Basic counts
    n_agents = None
    if agent_stats is not None and not agent_stats.empty:
        gcols = [c for c in agent_stats.columns if c in ('arena_file','run','robot_category','robot_id')]
        n_agents = agent_stats[gcols].drop_duplicates().shape[0]
    lines.append(f"- Agents analyzed: {n_agents if n_agents is not None else 'n/a'}")

    # Speed / rotation (per-agent means and time-agg)
    if agent_stats is not None and not agent_stats.empty:
        v_mean_mean = float(agent_stats['v_mean'].mean())
        om_mean_mean = float(agent_stats['omega_mean'].mean())
        lines.append(f"- ⟨⟨v⟩_agent⟩ = {_fmt_float(v_mean_mean)} (mm/s)")
        lines.append(f"- ⟨⟨ω⟩_agent⟩ = {_fmt_float(om_mean_mean)} (rad/s)")
    else:
        v_mean_mean = None
        om_mean_mean = None

    # Péclet summary
    pe_stats = {}
    if peclet_df is not None and not peclet_df.empty and 'Pe' in peclet_df.columns:
        pe_vals = peclet_df['Pe'].to_numpy()
        pe_stats = _summarize_pe(pe_vals)
        lines.append("- Péclet (per-agent):")
        if pe_stats:
            lines.append(f"  • mean={_fmt_float(pe_stats['mean'])}, std={_fmt_float(pe_stats['std'])}, "
                         f"p10={_fmt_float(pe_stats['pct10'])}, p50={_fmt_float(pe_stats['pct50'])}, p90={_fmt_float(pe_stats['pct90'])}")
            lines.append(f"  • frac(Pe>1)={_fmt_float(pe_stats['frac_gt_1'])}, frac(Pe>10)={_fmt_float(pe_stats['frac_gt_10'])}")
        else:
            lines.append("  • n/a")
    else:
        lines.append("- Péclet: n/a")

    # MSD exponents (α)
    alpha_early, alpha_late = (None, None)
    if msd_time is not None and not msd_time.empty:
        alpha_early, alpha_late = _fit_power_law_alpha(msd_time, frac=0.25)
        lines.append(f"- MSD scaling exponents: α_early={_fmt_float(alpha_early)}, α_late={_fmt_float(alpha_late)}")
    else:
        lines.append("- MSD scaling exponents: n/a")

    # Regime hints
    pe90 = pe_stats.get("pct90") if pe_stats else None
    hints = _regime_hint(alpha_early, alpha_late, pe90, om_mean_mean, v_mean_mean)
    if hints:
        lines.append("- Regime hints:")
        for h in hints:
            lines.append(f"  • {h}")

    text = "\n".join(lines) + "\n"

    # Write + print
    out_path = os.path.join(output_dir, "SUMMARY.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)



def create_all_locomotion_plots(input_file, output_dir,
        *,
        characteristic_length: float | None = None,  # robot diameter or other L (same units as x,y, i.e. mm)
        t_fit_min: float | None = None,              # MSD fit window, e.g. 10.0
        t_fit_max: float | None = None,              # MSD fit window, e.g. 60.0
        speed_window_s: float | None = 2.0,          # sliding window (seconds) for ⟨v⟩,⟨ω⟩ time-series
        speed_window_steps: int | None = None,       # alternative: window in steps (overrides seconds if set)
        peclet_outlier_method: str | None = "iqr",   # None | "iqr" | "zscore" | "percentile"
        peclet_iqr_k: float = 1.5,                  # Tukey fence width for IQR
        peclet_z: float = 3.0,                      # abs z-score cutoff
        peclet_percentiles: tuple[float, float] = (1.0, 99.0),  # lower/upper percent
        ):
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

    # ---------- speed stats & Pe exports ----------
    stats_dir = os.path.join(output_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)

    # Try to infer L from metadata if not provided
    # (feel free to adapt the keys to your meta structure)
    L = characteristic_length
    if L is None:
        L = (
            config.get("characteristic_length")
            or config.get("objects", {}).get("robots", {}).get("radius", 26.5) * 2.0
            or 53.0  # fallback: Pogobot default diameter
        )

    # Build sliding window spec if requested
    win = None
    if speed_window_steps is not None:
        win = WindowSpec(window_steps=int(speed_window_steps))
    elif speed_window_s is not None:
        win = WindowSpec(window_s=float(speed_window_s))

    # 1) Speed stats (per-agent + time-aggregated) + plots
    print("Computing speed stats (⟨v⟩, Var v, ⟨ω⟩, Var ω)…")
    agent_stats, time_stats = summarize_speed_stats(
        df,
        window=win,
        plot=True,
        plot_kind="time",
        figsize=(8, 5),
        savepath=os.path.join(output_dir, "speed_time.pdf"),
    )
    # also a per-agent summary boxplot (no smoothing)
    summarize_speed_stats(
        df,
        window=None,
        plot=True,
        plot_kind="box",
        figsize=(8, 4),
        savepath=os.path.join(output_dir, "speed_box.pdf"),
    )

    # Save CSVs
    agent_stats.to_csv(os.path.join(stats_dir, "speed_agent_stats.csv"), index=False)
    if time_stats is not None:
        time_stats.to_csv(os.path.join(stats_dir, "speed_time_stats.csv"), index=False)

    # 2) Péclet numbers (fit D from MSD vs time)
    print("Computing Péclet numbers…")
    peclet_df = compute_peclet(
        df,
        characteristic_length=float(L),
        t_fit_min=t_fit_min,
        t_fit_max=t_fit_max,
        window=win,                  # use the same smoothing for ⟨v⟩ if desired
        return_diffusivity=True,     # keep D and fit diagnostics
    )
    peclet_df.to_csv(os.path.join(stats_dir, "peclet.csv"), index=False)

    # Quick Pe histogram plot with optional outlier removal (plot only)
    pe_all = peclet_df["Pe"].dropna().to_numpy()

    def _filter_outliers(pe: np.ndarray) -> np.ndarray:
        method = (peclet_outlier_method or "").lower()
        if method in ("", "none", None):
            return pe
        if method == "iqr":
            q1, q3 = np.percentile(pe, [25, 75])
            iqr = q3 - q1
            lo = q1 - peclet_iqr_k * iqr
            hi = q3 + peclet_iqr_k * iqr
            return pe[(pe >= lo) & (pe <= hi)]
        if method == "zscore":
            mu, sd = np.mean(pe), np.std(pe, ddof=1)
            if sd == 0 or not np.isfinite(sd):
                return pe
            z = (pe - mu) / sd
            return pe[np.abs(z) <= peclet_z]
        if method == "percentile":
            lo_p, hi_p = peclet_percentiles
            lo, hi = np.percentile(pe, [lo_p, hi_p])
            return pe[(pe >= lo) & (pe <= hi)]
        # fallback: no filtering if unknown method
        return pe

    pe_plot = _filter_outliers(pe_all)
    note = ""
    if peclet_outlier_method:
        note = f" (filtered: {peclet_outlier_method})"

    plt.figure(figsize=(6, 4))
    sns.histplot(pd.Series(pe_plot), kde=True)
    plt.xlabel(f"Péclet number (Pe){note}")
    plt.ylabel("count")
    plt.tight_layout()
    utils.save_figure(os.path.join(output_dir, "peclet_hist.pdf"))


    # ---------- MSD CSVs + plots ----------
    print("Computing per-agent MSD...")
    msd_agent = compute_msd_per_agent(df, per_lag=True)

    # Expect columns: group cols + ['tau', 'msd'].
    # Be robust to possible naming variants.
    col_tau = 'tau' if 'tau' in msd_agent.columns else ('lag' if 'lag' in msd_agent.columns else None)
    col_msd = 'msd' if 'msd' in msd_agent.columns else ('disp2' if 'disp2' in msd_agent.columns else None)
    if col_tau is None or col_msd is None:
        raise ValueError("compute_msd_per_agent must return a 'tau' (or 'lag') column and an 'msd' (or 'disp2') column.")
    if col_tau != 'tau':
        msd_agent = msd_agent.rename(columns={col_tau: 'tau'})
    if col_msd != 'msd':
        msd_agent = msd_agent.rename(columns={col_msd: 'msd'})

    # Save per-agent MSD (one row per agent×tau)
    stats_dir = os.path.join(output_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    msd_agent.to_csv(os.path.join(stats_dir, "msd_per_agent.csv"), index=False)

    # Aggregate across agents at each tau (no pandas .apply → no FutureWarning)
    msd_time = (
        msd_agent.groupby('tau', sort=True)
                 .agg(mean_msd=('msd', 'mean'),
                      sem_msd =('msd', _sem),
                      n_agents=('msd', 'size'))
                 .reset_index()
    )
    msd_time.to_csv(os.path.join(stats_dir, "msd_time_stats.csv"), index=False)

    # Plot 1: linear axes
    plt.figure(figsize=(7, 4))
    plt.plot(msd_time['tau'], msd_time['mean_msd'], label='⟨MSD(τ)⟩')
    plt.fill_between(msd_time['tau'],
                     msd_time['mean_msd'] - msd_time['sem_msd'],
                     msd_time['mean_msd'] + msd_time['sem_msd'],
                     alpha=0.25, label='± SEM')
    plt.xlabel('lag time τ [s]')
    plt.ylabel('MSD [mm²]')
    plt.legend(loc='best')
    plt.tight_layout()
    utils.save_figure(os.path.join(output_dir, "msd_linear.pdf"))

    # Plot 2: log–log (use only positive τ and MSD)
    msd_pos = msd_time[(msd_time['tau'] > 0) & (msd_time['mean_msd'] > 0)]
    if not msd_pos.empty:
        plt.figure(figsize=(7, 4))
        plt.plot(msd_pos['tau'], msd_pos['mean_msd'], label='⟨MSD(τ)⟩')
        plt.fill_between(msd_pos['tau'],
                         (msd_pos['mean_msd'] - msd_pos['sem_msd']).clip(lower=1e-30),
                         msd_pos['mean_msd'] + msd_pos['sem_msd'],
                         alpha=0.25, label='± SEM')
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('lag time τ [s] (log)')
        plt.ylabel('MSD [mm²] (log)')
        plt.legend(loc='best')
        plt.tight_layout()
        utils.save_figure(os.path.join(output_dir, "msd_loglog.pdf"))

    # ---------- Textual summary ----------
    write_text_summary(
        output_dir,
        agent_stats=agent_stats,
        time_stats=time_stats,
        peclet_df=peclet_df,
        msd_time=msd_time,
    )


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
