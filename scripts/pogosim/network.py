#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Non-zero Laplacian eigenvalues for Pogosim snapshots and time-averaged graphs.

Outputs:
  - instantaneous_eigs.csv
  - timeavg_eigs_tau_c.csv          (if τ_c is computable)
  - timeavg_eigs_10dt.csv           (always, if time available)

Each CSV contains, for both "swarm" (entire graph) and "lcc" (largest connected component):
  - lambda_nz_1 .. lambda_nz_5   (smallest NON-ZERO eigenvalues)
  - lambda_max
  - n_nodes, n_edges (swarm_* and lcc_* prefixed)

Notes:
  - Velocities are always estimated from (x,y,time) first and used only for τ_c.
  - Graphs are ε-symmetric; zero eigenvalues are filtered with a tolerance.
"""

from __future__ import annotations
import os
import sys
import argparse
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import cKDTree
from scipy.sparse.linalg import eigsh
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mticker
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages


#sns.set(font_scale=1.4)
sns.set_context("talk")
sns.set_style("white")
sns.set_palette("colorblind")
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r''.join([
        r'\usepackage{amsmath}',
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{helvet}",
        r"\renewcommand{\familydefault}{\sfdefault}",
        r"\usepackage[helvet]{sfmath}",
        r"\everymath={\sf}",
        r'\centering',
        ]))

# ----------------------------- Utilities ----------------------------------- #

def _safe_mkdir(p):
    os.makedirs(p, exist_ok=True)

def ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    preferred = ['arena_file', 'run', 'robot_category', 'robot_id', 'time']
    sort_cols = [c for c in preferred if c in df.columns]
    return df.sort_values(sort_cols, kind='mergesort').reset_index(drop=True) if sort_cols else df.reset_index(drop=True)

def estimate_velocities(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_sorted(df)
    gcols = [c for c in ['arena_file', 'run', 'robot_category', 'robot_id'] if c in df.columns]
    out = df.copy()
    for _, g in out.groupby(gcols, sort=False):
        idx = g.index.to_numpy()
        t = g['time'].to_numpy()
        x = g['x'].to_numpy()
        y = g['y'].to_numpy()
        dt = np.diff(t, prepend=np.nan)
        dt[0] = dt[1] if dt.size > 1 else 1.0
        out.loc[idx, 'vx'] = np.diff(x, prepend=x[0]) / dt
        out.loc[idx, 'vy'] = np.diff(y, prepend=y[0]) / dt
    return out

def snapshot_key_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in ['arena_file', 'run', 'time'] if c in df.columns]

def _median_dt(df: pd.DataFrame) -> Optional[float]:
    if 'time' not in df.columns:
        return None
    keys = [c for c in ['arena_file','run'] if c in df.columns]
    dts = []
    if keys:
        for _, g in df.groupby(keys, sort=False):
            t = g['time'].drop_duplicates().sort_values().to_numpy()
            if t.size >= 2:
                dts.append(float(np.median(np.diff(t))))
    else:
        t = df['time'].drop_duplicates().sort_values().to_numpy()
        if t.size >= 2:
            dts.append(float(np.median(np.diff(t))))
    if dts:
        return float(np.median(dts))
    return None

def _tau_c_rc_over_vbar(df: pd.DataFrame, rc: float) -> Optional[float]:
    sp = np.hypot(df['vx'].to_numpy(), df['vy'].to_numpy())
    vbar = float(np.nanmean(sp)) if np.isfinite(sp).any() else 0.0
    if vbar > 1e-9:
        return float(rc / vbar)
    return None

def _windowize_times(times: np.ndarray, w: float) -> List[Tuple[float,float,float]]:
    if times.size == 0:
        return []
    t_min, t_max = float(times.min()), float(times.max())
    if w <= 0:
        return [(t_min, t_max, 0.5*(t_min+t_max))]
    starts = np.arange(t_min, t_max + 1e-12, w)
    out = []
    for s in starts:
        e = min(s + w, t_max + 1e-12)
        if e > s + 1e-9:
            out.append((float(s), float(e), float(0.5*(s+e))))
    return out

def _kdtree_neighbors(pts: np.ndarray, rc: float) -> List[List[int]]:
    tree = cKDTree(pts)
    return tree.query_ball_point(pts, rc)

def _largest_cc_nodes(G: nx.Graph) -> Optional[set]:
    if G.number_of_nodes() == 0:
        return None
    comps = list(nx.connected_components(G))
    if not comps:
        return None
    return set(max(comps, key=len))

def _nanquantile_safe(x, q):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.nanquantile(x, q))


def _unique_combinations_across_tables(tables: list[pd.DataFrame], cols: list[str]) -> list[dict]:
    """Collect unique dicts of {col: value} across given tables (skip missing cols gracefully)."""
    combos = set()
    for tab in tables:
        if tab is None or tab.empty:
            continue
        present = [c for c in cols if c in tab.columns]
        if not present:
            continue
        for vals in tab[present].drop_duplicates().itertuples(index=False, name=None):
            combos.add(tuple(zip(present, vals)))
    # normalize to full set (missing columns -> None) so file names are consistent
    allcols = cols[:]
    out = []
    for combo in combos:
        d = {k: v for k, v in combo}
        for c in allcols:
            d.setdefault(c, None)
        out.append(d)
    # stable sort by stringified tuple
    out.sort(key=lambda d: tuple((c, str(d[c])) for c in allcols))
    return out

def _fmt_combo_suffix(d: dict) -> str:
    parts = [f"{k}={os.path.basename(str(v)) if v is not None else 'ALL'}" for k, v in d.items()]
    return "__".join(parts)

def _subset_by_combo(tab: Optional[pd.DataFrame], combo: dict) -> Optional[pd.DataFrame]:
    if tab is None or tab.empty:
        return tab
    mask = pd.Series(True, index=tab.index)
    for k, v in combo.items():
        if k not in tab.columns or v is None:
            continue
        mask &= (tab[k] == v)
    sub = tab[mask].copy()
    return sub if not sub.empty else None


def _write_lines_pdf_for_combo(inst: Optional[pd.DataFrame],
                               tauc: Optional[pd.DataFrame],
                               ten_dt: Optional[pd.DataFrame],
                               k_first: int,
                               out_dir: str,
                               combo_suffix: str):
    outpdf = os.path.join(out_dir, f"{combo_suffix}.pdf")
    pages = 0
    with PdfPages(outpdf) as pdf:
        for tag, tab, time_col, title in [
            ('instantaneous', inst, 'time', 'Instantaneous Laplacian (run-aggregated)'),
            ('timeavg_tau_c', tauc, 'time_center', 'Time-averaged (tau_c) (run-aggregated)'),
            ('timeavg_10dt', ten_dt, 'time_center', r'Time-averaged $(10\times \Delta t)$ (run-aggregated)'),
        ]:
            if tab is None or tab.empty:
                continue
            ylim_log = _global_log_ylim_timeseries(tab, k_first, time_col)
            arenas = sorted(tab['arena_file'].dropna().unique()) if 'arena_file' in tab.columns else [None]
            for arena in arenas:
                g = tab if arena is None else tab[tab['arena_file'] == arena]
                agg = {
                    'swarm': _aggregate_runs_for_arena(g, time_col, 'swarm', k_first),
                    'lcc':   _aggregate_runs_for_arena(g, time_col, 'lcc',   k_first),
                }
                fig = make_combined_timeseries_figure(
                    agg_by_prefix=agg,
                    arena_name=str(arena),
                    time_col=time_col,
                    k_first=k_first,
                    title_tag=title,
                    ylim_log=ylim_log
                )
                if fig is None:
                    continue
                pdf.savefig(fig)
                plt.close(fig)
                pages += 1
    # optional: remove empty file if no pages
    if pages == 0 and os.path.exists(outpdf):
        os.remove(outpdf)


def write_lines_per_arena_all_sets(inst: Optional[pd.DataFrame],
                                   tauc: Optional[pd.DataFrame],
                                   ten_dt: Optional[pd.DataFrame],
                                   k_first: int,
                                   out_dir: str):
    # collect arenas across available tables
    arena_sets = []
    for tab in (inst, tauc, ten_dt):
        if tab is not None and not tab.empty and 'arena_file' in tab.columns:
            arena_sets.append(set(tab['arena_file'].dropna().unique()))
    arenas = sorted(set.union(*arena_sets) if arena_sets else {None})

    for arena in arenas:
        outpdf = os.path.join(out_dir, f'lines__arena={os.path.basename(str(arena)) if arena is not None else "ALL"}.pdf')
        pages = 0
        with PdfPages(outpdf) as pdf:
            for tag, tab, time_col, title in [
                ('instantaneous', inst, 'time', 'Instantaneous Laplacian (run-aggregated)'),
                ('timeavg_tau_c', tauc, 'time_center', 'Time-averaged (tau_c) (run-aggregated)'),
                ('timeavg_10dt', ten_dt, 'time_center', r'Time-averaged $(10 \times \Delta t)$ (run-aggregated)'),
            ]:
                if tab is None or tab.empty:
                    continue
                g = tab if arena is None else tab[tab['arena_file'] == arena]
                if g.empty:
                    continue
                ylim_log = _global_log_ylim_timeseries(g, k_first, time_col)
                agg = {
                    'swarm': _aggregate_runs_for_arena(g, time_col, 'swarm', k_first),
                    'lcc':   _aggregate_runs_for_arena(g, time_col, 'lcc',   k_first),
                }
                fig = make_combined_timeseries_figure(
                    agg_by_prefix=agg,
                    arena_name=str(arena),
                    time_col=time_col,
                    k_first=k_first,
                    title_tag=title,
                    ylim_log=ylim_log
                )
                if fig is None:
                    continue
                pdf.savefig(fig)
                plt.close(fig)
                pages += 1
        if pages == 0 and os.path.exists(outpdf):
            os.remove(outpdf)


def _eig_to_latex_label(label):
    return r"$" + label.replace("lambda", r"\lambda").replace("_nz", "^+").replace("_max", r'_\texttt{max}') + r"$"


# ------------------------- Laplacian / eigen helpers ------------------------ #

def _laplacian_matrix(G: nx.Graph, weight: Optional[str]=None):
    L = nx.laplacian_matrix(G, weight=weight)
    return L.astype(float)

def _smallest_nonzero_k_and_lammax(L: sparse.spmatrix, k: int, zero_tol: float=1e-10) -> Tuple[np.ndarray, float]:
    """
    Return the smallest k NON-ZERO eigenvalues of L and the largest eigenvalue.
    If fewer than k non-zero eigenvalues exist, pad with NaN.
    """
    n = L.shape[0]
    want = min(max(k, 1), max(n-1, 1))  # cannot exceed n-1 non-zero values (at least one zero)
    out = np.full(k, np.nan, dtype=float)

    # Strategy: ask ARPACK for m smallest (m > k), filter zeros, top-up if needed
    # choose m heuristically
    m = min(max(want + 8, 12), max(n-1, 1))
    try:
        vals = eigsh(L, k=m, which='SA', return_eigenvectors=False, tol=1e-7, maxiter=20000)
        vals = np.sort(np.real(vals))
    except Exception:
        # dense fallback for small n
        if n <= 1200:
            vals = np.linalg.eigvalsh(L.toarray())
            vals = np.sort(np.real(vals))
        else:
            vals = np.array([])

    if vals.size:
        nz = vals[vals > zero_tol]
        out[:min(k, nz.size)] = nz[:min(k, nz.size)]

    # λ_max
    try:
        lmax_arr = eigsh(L, k=1, which='LA', return_eigenvectors=False, tol=1e-7, maxiter=20000)
        lam_max = float(np.real(lmax_arr[0]))
    except Exception:
        if n <= 1200:
            lam_max = float(np.linalg.eigvalsh(L.toarray())[-1])
        else:
            lam_max = np.nan

    return out, lam_max

def _graph_from_points(pts: np.ndarray, rc: float, weights: Optional[np.ndarray]=None) -> nx.Graph:
    """
    If weights is None: unweighted edges (0/1 within rc).
    If weights is provided: it's an NxN matrix; edges are where weights>0 with 'weight'.
    """
    N = pts.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(N))
    if weights is None:
        neighs = _kdtree_neighbors(pts, rc)
        for i, nbrs in enumerate(neighs):
            for j in nbrs:
                if j > i:
                    G.add_edge(i, j)
    else:
        nz = np.where(weights > 0)
        for i, j in zip(*nz):
            if j > i:
                G.add_edge(int(i), int(j), weight=float(weights[i, j]))
    return G

# ---------------------------- Core computations ----------------------------- #

def _compute_eigs_for_graph(G: nx.Graph, k_first: int, weighted: bool=False, zero_tol: float=1e-10) -> Dict[str, float]:
    """
    Compute non-zero eigenvalues for the whole graph ('swarm_*') and for the LCC ('lcc_*').
    Returns a dict of metrics.
    """
    out: Dict[str, float] = {}
    weight_key = 'weight' if weighted else None

    # Entire swarm
    L = _laplacian_matrix(G, weight=weight_key)
    nz_vals, lam_max = _smallest_nonzero_k_and_lammax(L, k_first, zero_tol)
    for i in range(k_first):
        out[f'swarm_lambda_nz_{i+1}'] = nz_vals[i]
    out['swarm_lambda_max'] = lam_max
    out['swarm_n_nodes'] = G.number_of_nodes()
    out['swarm_n_edges'] = G.number_of_edges()

    # LCC
    cc = _largest_cc_nodes(G)
    if cc is not None and len(cc) >= 1:
        Glcc = G.subgraph(cc).copy()
        Llcc = _laplacian_matrix(Glcc, weight=weight_key)
        nz_vals_lcc, lam_max_lcc = _smallest_nonzero_k_and_lammax(Llcc, k_first, zero_tol)
        for i in range(k_first):
            out[f'lcc_lambda_nz_{i+1}'] = nz_vals_lcc[i]
        out['lcc_lambda_max'] = lam_max_lcc
        out['lcc_n_nodes'] = Glcc.number_of_nodes()
        out['lcc_n_edges'] = Glcc.number_of_edges()
    else:
        for i in range(k_first):
            out[f'lcc_lambda_nz_{i+1}'] = np.nan
        out['lcc_lambda_max'] = np.nan
        out['lcc_n_nodes'] = 0
        out['lcc_n_edges'] = 0

    return out

def instantaneous_eigs(df: pd.DataFrame, rc: float, k_first: int=5, zero_tol: float=1e-10) -> pd.DataFrame:
    need = ['time','x','y']
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe missing columns: {missing}")

    metakeys = (
            [f'swarm_lambda_nz_{i+1}' for i in range(k_first)] +
            ['swarm_lambda_max', 'swarm_n_nodes', 'swarm_n_edges'] +
            [f'lcc_lambda_nz_{i+1}' for i in range(k_first)] +
            ['lcc_lambda_max', 'lcc_n_nodes', 'lcc_n_edges']
            )

    df = ensure_sorted(df)
    keys = snapshot_key_columns(df)
    rows = []
    for key, snap in df.groupby(keys, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        pts = snap[['x','y']].to_numpy()
        G = _graph_from_points(pts, rc)
        metrics = _compute_eigs_for_graph(G, k_first, weighted=False, zero_tol=zero_tol)
        rows.append(tuple(list(key) + [metrics.get(c, np.nan) for c in metakeys]))
    cols = keys + metakeys
    return pd.DataFrame(rows, columns=cols)

def time_averaged_eigs(df: pd.DataFrame, rc: float, window_s: float, k_first: int=5, zero_tol: float=1e-10) -> pd.DataFrame:
    df = ensure_sorted(df)
    keys0 = [c for c in ['arena_file','run'] if c in df.columns]
    results = []
    group_iter = df.groupby(keys0, sort=False) if keys0 else [((), df)]

    metakeys = (
            [f'swarm_lambda_nz_{i+1}' for i in range(k_first)] +
            ['swarm_lambda_max', 'swarm_n_nodes', 'swarm_n_edges'] +
            [f'lcc_lambda_nz_{i+1}' for i in range(k_first)] +
            ['lcc_lambda_max', 'lcc_n_nodes', 'lcc_n_edges']
            )

    for key0, g in group_iter:
        if not isinstance(key0, tuple):
            key0 = (key0,)
        times = g['time'].drop_duplicates().sort_values().to_numpy()
        if times.size == 0:
            continue
        windows = _windowize_times(times, window_s)

        # consistent node order across snapshots
        ids = np.sort(g['robot_id'].drop_duplicates().to_numpy()) if 'robot_id' in g.columns else np.arange(len(g['time'].unique()))
        N = len(ids)
        id_index = {rid: i for i, rid in enumerate(ids)}

        # build adjacency per snapshot
        adj_by_time: Dict[float, np.ndarray] = {}
        for t in times:
            snap = g[g['time'] == t].copy()
            if 'robot_id' in snap.columns:
                snap = snap.set_index('robot_id').reindex(ids).reset_index()
            pts = snap[['x','y']].to_numpy()
            Gt = _graph_from_points(pts, rc)
            A = np.zeros((N, N), dtype=np.float32)
            for (u, v) in Gt.edges():
                A[u, v] = 1.0
                A[v, u] = 1.0
            adj_by_time[float(t)] = A

        for (t0, t1, tc) in windows:
            sel = [float(t) for t in times if (t >= t0 and t < t1)]
            if not sel:
                continue
            Abar = np.zeros((N, N), dtype=np.float64)
            for t in sel:
                Abar += adj_by_time[t]
            Abar /= float(len(sel))

            # time-averaged weighted graph
            Gw = _graph_from_points(np.zeros((N,2)), rc, weights=Abar)  # pts unused when weights provided
            metrics = _compute_eigs_for_graph(Gw, k_first, weighted=True, zero_tol=zero_tol)

            row = list(key0) + [tc, t0, t1]
            row.extend([metrics.get(m, np.nan) for m in metakeys])
            results.append(tuple(row))

    cols = keys0 + ['time_center', 't_start', 't_end'] + metakeys
    return pd.DataFrame(results, columns=cols)


# -------------------- Y-limits, log-safe utilities -------------------------- #

def _log_safe_minmax(values: np.ndarray, positive_floor: float = 1e-12):
    """Return (ymin, ymax) for log axes from possibly nonpositive data."""
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    pos = v[v > 0.0]
    if pos.size == 0:
        # fallback to [1, 1] (will be expanded by caller)
        return positive_floor, 1.0
    ymin = float(np.min(pos))
    ymax = float(np.max(pos))
    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 1.0
    # gentle padding
    return max(ymin * 0.9, positive_floor), ymax * 1.1

def _collect_cols_if_exist(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]

def _global_log_ylim_timeseries(df: pd.DataFrame, k_first: int, time_col: str) -> tuple[float, float]:
    """
    Compute global y-lims for timeseries across both prefixes (swarm,lcc) and
    all eigen columns present in df.
    """
    cols = []
    for prefix in ('swarm', 'lcc'):
        cols += [f'{prefix}_lambda_nz_{i+1}' for i in range(k_first)]
        cols += [f'{prefix}_lambda_max']
    cols = _collect_cols_if_exist(df, cols)
    if not cols:
        return (1e-12, 1.0)
    vals = df[cols].to_numpy().ravel()
    return _log_safe_minmax(vals)

def _global_log_ylim_violins(*long_dfs: pd.DataFrame) -> tuple[float, float]:
    """Global y-lims from one or more long-form dataframes with a 'value' column."""
    vals = []
    for d in long_dfs:
        if d is not None and not d.empty and 'value' in d.columns:
            vals.append(d['value'].to_numpy())
    if not vals:
        return (1e-12, 1.0)
    return _log_safe_minmax(np.concatenate(vals))



# --------------------------------- Plots (run-aggregated) ------------------- #

def _line_style(i):
    styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    return styles[i % len(styles)]

def _infer_time_bin_width(df: pd.DataFrame, time_col: str) -> float:
    """Heuristic: use median Δt as bin width; fallback to 1.0 if missing."""
    if time_col not in df.columns:
        return 1.0
    t = df[time_col].dropna().drop_duplicates().sort_values().to_numpy()
    if t.size >= 2:
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size:
            return float(np.median(dt))
    return 1.0

def _bin_time_column(df: pd.DataFrame, time_col: str, bin_w: float) -> pd.Series:
    """Round to nearest bin center (robust to tiny FP jitter)."""
    return np.round(df[time_col].to_numpy() / bin_w) * bin_w

def _aggregate_runs_for_arena(df: pd.DataFrame, time_col: str, prefix: str, k_first: int,
                              q_low: float = 0.25, q_high: float = 0.75) -> pd.DataFrame:
    """
    For a single arena (df already filtered), aggregate across runs at each time bin:
    - median line
    - IQR (q_low..q_high)
    - min/max envelope
    Returns wide table with columns:
      time, <metric>, <metric>_q25, <metric>_q75, <metric>_min, <metric>_max
    for metrics in {prefix}_lambda_nz_1..k and {prefix}_lambda_max.
    """
    cols_nz = [f'{prefix}_lambda_nz_{i+1}' for i in range(k_first)]
    needed = [time_col] + cols_nz + [f'{prefix}_lambda_max']
    d = df[needed].dropna(subset=[time_col]).copy()
    if d.empty:
        return pd.DataFrame()

    # Bin time to align slightly-misaligned runs
    bin_w = _infer_time_bin_width(d, time_col)
    d['_tbin'] = _bin_time_column(d, time_col, bin_w)

    # Aggregate per bin
    agg_dict = {}
    for col in cols_nz + [f'{prefix}_lambda_max']:
        agg_dict[col] = [
            'median',
            lambda x: _nanquantile_safe(x, 0.25),
            lambda x: _nanquantile_safe(x, 0.75),
            'min', 'max'
        ]
    g = d.groupby('_tbin', sort=True).agg(agg_dict)

    # flatten columns
    g.columns = pd.Index([
        f"{m}_{stat}" for m, stat in zip(
            np.repeat(cols_nz + [f'{prefix}_lambda_max'], 5),
            ['median', 'q25', 'q75', 'min', 'max'] * (len(cols_nz) + 1)
        )
    ])
    g = g.reset_index().rename(columns={'_tbin': time_col})
    return g


def _plot_aggregated_timeseries_for_arena(df: pd.DataFrame, arena_name: str, time_col: str,
                                          prefix: str, k_first: int, title_tag: str,
                                          out_path_base: str,
                                          ylim_log: tuple[float,float] | None = None):
    if df.empty:
        return

    cols_nz = [f'{prefix}_lambda_nz_{i+1}' for i in range(k_first)]
    metrics = cols_nz + [f'{prefix}_lambda_max']
    have = [m for m in metrics if f"{m}_median" in df.columns]
    if not have:
        return

    fig, ax = plt.subplots(figsize=(9.8, 5.6), constrained_layout=True)
    t = df[time_col].to_numpy()

    # color cycle (same for line and fills)
    cycle_colors = plt.rcParams.get('axes.prop_cycle', None)
    colors = cycle_colors.by_key()['color'] if cycle_colors else [f"C{i}" for i in range(10)]

    for i, m in enumerate(have):
        med  = df[f"{m}_median"].to_numpy()
        q25  = df[f"{m}_q25"].to_numpy()
        q75  = df[f"{m}_q75"].to_numpy()
        vmin = df[f"{m}_min"].to_numpy()
        vmax = df[f"{m}_max"].to_numpy()

        color = colors[i % len(colors)]

        # min–max (light alpha)
        ax.fill_between(t, vmin, vmax, color=color, alpha=0.10, linewidth=0.0, zorder=1)
        # IQR (stronger alpha)
        ax.fill_between(t, q25, q75, color=color, alpha=0.25, linewidth=0.0, zorder=2)
        # mean/median line
        ax.plot(t, med, linestyle=_line_style(i), linewidth=2.2, color=color, zorder=3,
                label=rf'${m.split("_")[-1].replace("nz","\\lambda^+") if "lambda" in m else m}$')

    ax.set_title(f'{title_tag} – {prefix.upper()} (arena="{arena_name}")')
    ax.set_xlabel('time (s)' if time_col in ('time', 'time_center') else time_col)
    ax.set_ylabel('eigenvalue')
    ax.grid(True, alpha=0.35, which='major')
    ax.set_yscale('log')

    # Ticks: major on X; major+minor on Y (log)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=9))  # more major ticks on X
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=12))  # more labels on Y
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)*0.1, numticks=90))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    if ylim_log is not None:
        ax.set_ylim(ylim_log)

    # Larger legend font, more columns if many curves
    ncurves = len(have)
    ax.legend(ncol=min(4, ncurves), fontsize=12, framealpha=0.9)

    pdf = out_path_base + f'_{prefix}.pdf'
    fig.savefig(pdf)
    plt.close(fig)


def plot_instantaneous_across_runs(inst_df: pd.DataFrame, k_first: int, out_dir: str):
    if inst_df is None or inst_df.empty:
        return
    _safe_mkdir(out_dir)
    time_col = 'time'
    ylim_log = _global_log_ylim_timeseries(inst_df, k_first, time_col)

    arenas = sorted(inst_df['arena_file'].dropna().unique()) if 'arena_file' in inst_df.columns else [None]
    for arena in arenas:
        g = inst_df if arena is None else inst_df[inst_df['arena_file'] == arena]
        agg = {
            'swarm': _aggregate_runs_for_arena(g, time_col, 'swarm', k_first),
            'lcc':   _aggregate_runs_for_arena(g, time_col, 'lcc',   k_first),
        }
        outpdf = os.path.join(out_dir, f'instantaneous_aggregated_{os.path.basename(str(arena)) if arena is not None else "all"}.pdf')
        _plot_combined_timeseries_per_arena(
            agg_by_prefix=agg,
            arena_name=str(arena),
            time_col=time_col,
            k_first=k_first,
            title_tag='Instantaneous Laplacian (run-aggregated)',
            out_pdf_path=outpdf,
            ylim_log=ylim_log
        )


def plot_timeavg_across_runs(tab: pd.DataFrame, k_first: int, out_dir: str, tag: str):
    if tab is None or tab.empty:
        return
    _safe_mkdir(out_dir)
    time_col = 'time_center'
    ylim_log = _global_log_ylim_timeseries(tab, k_first, time_col)

    arenas = sorted(tab['arena_file'].dropna().unique()) if 'arena_file' in tab.columns else [None]
    for arena in arenas:
        g = tab if arena is None else tab[tab['arena_file'] == arena]
        agg = {
            'swarm': _aggregate_runs_for_arena(g, time_col, 'swarm', k_first),
            'lcc':   _aggregate_runs_for_arena(g, time_col, 'lcc',   k_first),
        }
        outpdf = os.path.join(out_dir, f'timeavg_{tag}_aggregated_{os.path.basename(str(arena)) if arena is not None else "all"}.pdf')
        _plot_combined_timeseries_per_arena(
            agg_by_prefix=agg,
            arena_name=str(arena),
            time_col=time_col,
            k_first=k_first,
            title_tag=f'Time-averaged ({tag}) (run-aggregated)',
            out_pdf_path=outpdf,
            ylim_log=ylim_log
        )


def _plot_combined_timeseries_per_arena(
        agg_by_prefix, arena_name, time_col, k_first, title_tag, out_pdf_path, ylim_log=None):
    fig = make_combined_timeseries_figure(
        agg_by_prefix, arena_name, time_col, k_first, title_tag, ylim_log
    )
    if fig is None:
        return
    fig.savefig(out_pdf_path)
    plt.close(fig)


def make_combined_timeseries_figure(
        agg_by_prefix: dict[str, pd.DataFrame],
        arena_name: str,
        time_col: str,
        k_first: int,
        title_tag: str,
        ylim_log: tuple[float, float] | None = None):

    prefixes = [p for p in ('swarm', 'lcc') if p in agg_by_prefix and not agg_by_prefix[p].empty]
    if not prefixes:
        return None

    ncols = len(prefixes)

    # Bigger/taller figure; we'll manage layout manually to reserve room for title + bottom legend
    fig, axes = plt.subplots(
        nrows=1, ncols=ncols,
        figsize=(11.0, 7.0),
        sharey=True,
        constrained_layout=False
    )
    if ncols == 1:
        axes = [axes]

    cycle_colors = plt.rcParams.get('axes.prop_cycle', None)
    colors = cycle_colors.by_key()['color'] if cycle_colors else [f"C{i}" for i in range(10)]

    for ax, prefix in zip(axes, prefixes):
        df = agg_by_prefix[prefix]
        if df.empty:
            ax.set_visible(False); continue

        cols_nz = [f'{prefix}_lambda_nz_{i+1}' for i in range(k_first)]
        metrics = cols_nz + [f'{prefix}_lambda_max']
        have = [m for m in metrics if f"{m}_median" in df.columns]
        if not have:
            ax.set_visible(False); continue

        t = df[time_col].to_numpy()
        for i, m in enumerate(have):
            med  = df[f"{m}_median"].to_numpy()
            q25  = df[f"{m}_q25"].to_numpy()
            q75  = df[f"{m}_q75"].to_numpy()
            vmin = df[f"{m}_min"].to_numpy()
            vmax = df[f"{m}_max"].to_numpy()
            color = colors[i % len(colors)]
            ax.fill_between(t, vmin, vmax, color=color, alpha=0.10, linewidth=0.0, zorder=1)
            ax.fill_between(t, q25, q75, color=color, alpha=0.25, linewidth=0.0, zorder=2)
            ax.plot(t, med, linestyle=_line_style(i), linewidth=2.2, color=color, zorder=3,
                    label=m.replace(f"{prefix}_", ""))

        ax.set_title(prefix.upper())
        ax.set_xlabel('time (s)' if time_col in ('time', 'time_center') else time_col)
        ax.grid(True, alpha=0.35, which='major')
        ax.set_yscale('log')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=9))
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=12))
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2,10)*0.1, numticks=90))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        if ylim_log is not None:
            ax.set_ylim(ylim_log)

    # Suptitle at the top; we'll reserve head/foot space with tight_layout(rect=...)
    fig.suptitle(f'{title_tag} – arena="{arena_name}"', fontsize=18)

    # Gather legend entries from the first visible axes
    handles, labels = [], []
    for ax in axes:
        if ax.get_visible():
            handles, labels = ax.get_legend_handles_labels()
            break
    labels = [_eig_to_latex_label(s) for s in labels]

    # Lay out axes, leaving room for suptitle (top ≈ 0.90) and bottom legend (bottom ≈ 0.10)
    # rect = [left, bottom, right, top]
    fig.tight_layout(rect=[0.00, 0.09, 1.01, 1.01])

    if handles and labels:
        # Bottom, centered legend; wide ncol so it’s compact
        fig.legend(
            handles, labels,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(6, len(labels)),
            fontsize=18,
            framealpha=0.95
        )

    return fig



# ------------------------------ Violin plots -------------------------------- #

def _per_run_time_means_with_prefix(tab: pd.DataFrame, time_col: str, prefix: str, k_first: int) -> pd.DataFrame:
    cols = [f'{prefix}_lambda_nz_{i+1}' for i in range(k_first)] + [f'{prefix}_lambda_max']
    keep = [c for c in ['arena_file','run',time_col] if c in tab.columns] + cols
    d = tab[keep].dropna(subset=[time_col]).copy()
    if d.empty:
        return pd.DataFrame(columns=['arena_file','run','eigen','value','prefix'])
    grp = d.groupby(['arena_file','run'], sort=False)[cols].mean().reset_index()
    long = grp.melt(id_vars=['arena_file','run'], var_name='eigen', value_name='value')
    long['prefix'] = prefix
    return long


def violins_combined_arenas_prefixes_all_eigs(tab: pd.DataFrame, time_col: str, tag: str,
                                              k_first: int, out_dir: str,
                                              global_ylim: tuple[float,float] | None = None):
    if tab is None or tab.empty:
        return
    _safe_mkdir(out_dir)

    long_swarm = _per_run_time_means_with_prefix(tab, time_col, 'swarm', k_first)
    long_lcc   = _per_run_time_means_with_prefix(tab, time_col, 'lcc',   k_first)
    long_all   = pd.concat([long_swarm, long_lcc], ignore_index=True)

    if long_all.empty:
        return

    # keep only finite and strictly positive values (log-scale safety)
    long_all = long_all[np.isfinite(long_all['value']) & (long_all['value'] > 0)]

    # if a whole eigen is empty after filtering, it simply won't get plotted
    long_all['eigen_clean'] = long_all['eigen'].str.replace(r'^(swarm|lcc)_', '', regex=True)

    if global_ylim is None:
        global_ylim = _global_log_ylim_violins(long_all)

    eigs = [f'lambda_nz_{i+1}' for i in range(k_first)] + ['lambda_max']
    num_eigs = len(eigs)
    ncols = min(4, num_eigs)
    nrows = int(np.ceil(num_eigs / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(4.2*ncols, 4.6*nrows),
                             sharey=True, constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    arena_order = sorted(long_all['arena_file'].dropna().unique())

    for i, eig in enumerate(eigs):
        ax = axes[i]
        # match clean label
        sub = long_all[long_all['eigen_clean'] == f'lambda_{eig.split("_",1)[1]}']
        if sub.empty:
            ax.axis('off')
            continue

        sns.violinplot(
            data=sub, x='arena_file', y='value', hue='prefix',
            inner='box', cut=0, ax=ax, order=arena_order, linewidth=1.0,
            density_norm='width'   # <- replaces deprecated scale='width'
        )
        ax.set_yscale('log')
        ax.set_ylim(global_ylim)
        ax.set_xlabel('arena' if i >= (num_eigs - ncols) else '')
        ax.set_ylabel('eigenvalue (per-run mean)')
        ax.set_title(eig.replace('_', ' '))
        ax.grid(True, axis='y', alpha=0.25, which='major')
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2,10)*0.1, numticks=90))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

        if i == 0:
            ax.legend(title='graph', loc='upper right', fontsize=11, title_fontsize=12, framealpha=0.9)
        else:
            ax.legend_.remove()

    for j in range(num_eigs, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f'{tag}: violins of per-run means – ALL eigenvalues (hue: swarm vs lcc)', fontsize=18)
    out = os.path.join(out_dir, f'violin_{tag}_all_eigs_combined.pdf')
    fig.savefig(out)
    plt.close(fig)

    return global_ylim


def violins_three_sets_one_figure(inst: Optional[pd.DataFrame],
                                  tauc: Optional[pd.DataFrame],
                                  ten_dt: Optional[pd.DataFrame],
                                  k_first: int,
                                  out_dir: str,
                                  filename: str = 'violins_all_sets.pdf'):
    sets = []
    if inst is not None and not inst.empty:
        sets.append(('instantaneous', inst, 'time'))
    if tauc is not None and not tauc.empty:
        sets.append(('timeavg_tau_c', tauc, 'time_center'))
    if ten_dt is not None and not ten_dt.empty:
        sets.append(('timeavg_10dt', ten_dt, 'time_center'))
    if not sets:
        return

    # Prepare long-form per set
    prepared = []
    for tag, tab, tcol in sets:
        long_swarm = _per_run_time_means_with_prefix(tab, tcol, 'swarm', k_first)
        long_lcc   = _per_run_time_means_with_prefix(tab, tcol, 'lcc',   k_first)
        d = pd.concat([long_swarm, long_lcc], ignore_index=True)
        d = d[np.isfinite(d['value']) & (d['value'] > 0)]
        d['eigen_clean'] = d['eigen'].str.replace(r'^(swarm|lcc)_', '', regex=True)
        prepared.append((tag, d))

    # Global y-lims across all sets
    global_ylim = _global_log_ylim_violins(*[d for _, d in prepared])

    eigs = [f'lambda_nz_{i+1}' for i in range(k_first)] + ['lambda_max']
    num_eigs = len(eigs)
    ncols = min(4, num_eigs)
    nrows_per_set = int(np.ceil(num_eigs / ncols))
    total_rows = nrows_per_set * len(prepared)

    fig, axes = plt.subplots(nrows=total_rows, ncols=ncols,
                             figsize=(4.2*ncols, 4.6*total_rows),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_2d(axes)
    axes = axes.reshape(total_rows, ncols)

    for si, (tag, long_all) in enumerate(prepared):
        arena_order = sorted(long_all['arena_file'].dropna().unique())
        for j, eig in enumerate(eigs):
            r = si * nrows_per_set + (j // ncols)
            c = j % ncols
            ax = axes[r, c]
            sub = long_all[long_all['eigen_clean'] == f'lambda_{eig.split("_",1)[1]}']
            if sub.empty:
                ax.axis('off'); continue
            sns.violinplot(
                data=sub, x='arena_file', y='value', hue='prefix',
                inner='box', cut=0, ax=ax, order=arena_order, linewidth=1.0,
                density_norm='width'
            )
            ax.set_yscale('log')
            ax.set_ylim(global_ylim)
            ax.set_xlabel('arena' if (j // ncols) == (nrows_per_set - 1) else '')
            ax.set_ylabel(f'{tag}\nvalue' if c == 0 else '')
            ax.set_title(f'{tag} – {eig.replace("_", " ")}')
            #ax.set_title(f'{tag} – {_eig_to_latex_label(eig)}')
            ax.grid(True, axis='y', alpha=0.25, which='major')
            ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2,10)*0.1, numticks=90))
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
            if (j == 0):
                ax.legend(title='graph', loc='upper right', fontsize=11, title_fontsize=12, framealpha=0.9)
            else:
                ax.legend_.remove()

        # turn off any extra panels for this set row-block
        for jj in range(len(eigs), nrows_per_set * ncols):
            r = si * nrows_per_set + (jj // ncols)
            c = jj % ncols
            axes[r, c].axis('off')

    fig.suptitle('Violins of per-run means – ALL sets (rows) × eigenvalues (columns)', fontsize=18)
    out = os.path.join(out_dir, filename)
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------- I/O ------------------------------------ #

def load_df_and_config(input_file: str):
    try:
        from pogosim import utils  # type: ignore
    except Exception:
        try:
            import utils  # type: ignore
        except Exception as e:
            raise ImportError("Cannot import pogosim.utils.load_dataframe; ensure PYTHONPATH includes your project.") from e

    df, meta = utils.load_dataframe(input_file)  # must return (DataFrame, meta)
    config = meta.get("configuration", {}) if isinstance(meta, dict) else {}
    objects = config.get("objects", {})
    robots = objects.get("robots", {})
    communication_radius_border = float(robots.get("communication_radius", 80.0))
    robot_radius = float(robots.get("radius", 26.5))
    rc = communication_radius_border + 2.0 * robot_radius  # center-to-center cutoff in mm
    return df, rc, meta

def _prepare_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# --------------------------------- Main ------------------------------------ #

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Compute NON-ZERO Laplacian eigenvalues (swarm & LCC) for Pogosim data.")
    p.add_argument("-i", "--input", dest="input_file", required=True,
                   help="Input Feather/CSV produced by Pogosim")
    p.add_argument("-o", "--output", dest="output_dir", required=True,
                   help="Directory to write PDFs, summary and pickle")
    p.add_argument("--rc", type=float, default=0.0, required=False, help="Neighbor cutoff rc (same units as x,y; e.g., mm)")
    p.add_argument("--k-first", type=int, default=5, help="How many smallest NON-ZERO eigenvalues (default: 5)")
    p.add_argument("--avg-window-s", type=str, default="auto",
                   help='Float (seconds) for a fixed window or "auto" for τ_c and 10×median Δt.')
    p.add_argument("--zero-tol", type=float, default=1e-10, help="Zero-eigenvalue tolerance (default: 1e-10)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    _prepare_output_dir(args.output_dir)

    # Load & clean
    df, rc_neighbor, meta = load_df_and_config(args.input_file)
    if args.rc > 0.0:
        rc_neighbor = args.rc
    df = df.dropna(subset=['x','y','time']).reset_index(drop=True)
    # Estimate velocities FIRST (for τ_c)
    df = estimate_velocities(df)

    # Extract optional grouping columns for result slicing
    config = meta.get("configuration", {}) if isinstance(meta, dict) else {}
    slice_cols = config.get("result_new_columns", None)
    if isinstance(slice_cols, list):
        slice_cols = [c for c in slice_cols if isinstance(c, str)]
    else:
        slice_cols = None

    # --- Instantaneous ---
    inst = instantaneous_eigs(df, rc=rc_neighbor, k_first=args.k_first, zero_tol=args.zero_tol)
    if not inst.empty:
        inst.to_csv(os.path.join(args.output_dir, "instantaneous_eigs.csv"), index=False)
    print(f"[instantaneous] snapshots: {len(inst)}")

    # --- Time-averaged (τ_c and 10×Δt or fixed) ---
    tavg_tau = None
    tavg_10dt = None

    if args.avg_window_s.lower() == "auto":
        win_tau = _tau_c_rc_over_vbar(df, rc_neighbor)
        if win_tau is not None:
            print(f"[auto] τ_c ≈ rc/⟨|v|⟩ = {win_tau:.3f} s")
            tavg_tau = time_averaged_eigs(df, rc=rc_neighbor, window_s=win_tau, k_first=args.k_first, zero_tol=args.zero_tol)
            if not tavg_tau.empty:
                tavg_tau.to_csv(os.path.join(args.output_dir, "timeavg_eigs_tau_c.csv"), index=False)
        else:
            print("[auto] τ_c not computable (v̄ too small or invalid) → skipping τ_c")

        md = _median_dt(df)
        if md is not None and np.isfinite(md) and md > 0:
            win_10dt = 10.0 * md
            print(f"[auto] 10× median Δt = {win_10dt:.3f} s (median Δt={md:.3f} s)")
            tavg_10dt = time_averaged_eigs(df, rc=rc_neighbor, window_s=win_10dt, k_first=args.k_first, zero_tol=args.zero_tol)
            if not tavg_10dt.empty:
                tavg_10dt.to_csv(os.path.join(args.output_dir, "timeavg_eigs_10dt.csv"), index=False)
        else:
            print("[auto] could not compute median Δt → skipping 10×Δt")
    else:
        win = float(args.avg_window_s)
        print(f"[fixed] time-averaging window = {win:.3f} s")
        tavg_10dt = time_averaged_eigs(df, rc=rc_neighbor, window_s=win, k_first=args.k_first, zero_tol=args.zero_tol)
        if not tavg_10dt.empty:
            tavg_10dt.to_csv(os.path.join(args.output_dir, "timeavg_eigs.csv"), index=False)

    # ---- Plots ----
    plots_dir = args.output_dir
    _safe_mkdir(plots_dir)

#    # ---------- run-aggregated timeseries (shared log-ylim per set) ----------
#    if not inst.empty:
#        plot_instantaneous_across_runs(inst, args.k_first, plots_dir)
#
#    if tavg_tau is not None and not tavg_tau.empty:
#        plot_timeavg_across_runs(tavg_tau, args.k_first, plots_dir, tag='tau_c')
#
#    if tavg_10dt is not None and not tavg_10dt.empty:
#        tag = 'fixed' if args.avg_window_s.lower() != 'auto' else '10dt'
#        plot_timeavg_across_runs(tavg_10dt, args.k_first, plots_dir, tag=tag)

#    # ---------- combined violins (one figure per set; all eigs; hue=swarm/lcc) ----------
#    global_violin_ylim = None
#    if not inst.empty:
#        global_violin_ylim = violins_combined_arenas_prefixes_all_eigs(
#            inst, time_col='time', tag='instantaneous',
#            k_first=args.k_first, out_dir=plots_dir, global_ylim=global_violin_ylim
#        )
#
#    if tavg_tau is not None and not tavg_tau.empty:
#        global_violin_ylim = violins_combined_arenas_prefixes_all_eigs(
#            tavg_tau, time_col='time_center', tag='timeavg_tau_c',
#            k_first=args.k_first, out_dir=plots_dir, global_ylim=global_violin_ylim
#        )
#
#    if tavg_10dt is not None and not tavg_10dt.empty:
#        vtag = 'fixed' if args.avg_window_s.lower() != 'auto' else 'timeavg_10dt'
#        global_violin_ylim = violins_combined_arenas_prefixes_all_eigs(
#            tavg_10dt, time_col='time_center', tag=vtag,
#            k_first=args.k_first, out_dir=plots_dir, global_ylim=global_violin_ylim
#        )


    # 1) Keep per-set, per-arena single-PDF outputs (one page/figure per arena)
    plot_instantaneous_across_runs(inst, args.k_first, plots_dir)
    plot_timeavg_across_runs(tavg_tau,  args.k_first, plots_dir, tag='tau_c')
    plot_timeavg_across_runs(tavg_10dt, args.k_first, plots_dir, tag='10dt')

    # 2) per-arena multipage PDF that combines ALL sets (Instantaneous, tau_c, 10dt)
    write_lines_per_arena_all_sets(inst, tavg_tau, tavg_10dt, args.k_first, plots_dir)

    # 3) Combined violins (already added previously)
    violins_three_sets_one_figure(inst, tavg_tau, tavg_10dt, args.k_first, plots_dir,
                                  filename='violins_all_sets_combined.pdf')

    # 4) Per-combination outputs if configured
    if slice_cols:
        combos = _unique_combinations_across_tables([inst, tavg_tau, tavg_10dt], slice_cols)
        for combo in combos:
            inst_sub   = _subset_by_combo(inst, combo)
            tauc_sub   = _subset_by_combo(tavg_tau, combo)
            ten_dt_sub = _subset_by_combo(tavg_10dt, combo)
            if all(x is None or x.empty for x in (inst_sub, tauc_sub, ten_dt_sub)):
                continue
            suffix = _fmt_combo_suffix(combo)
            combo_dir = os.path.join(plots_dir, f"combination__{suffix}")
            _safe_mkdir(combo_dir)

            # (A) one multipage PDF containing ALL related line plots (arenas × sets)
            _write_lines_pdf_for_combo(inst_sub, tauc_sub, ten_dt_sub, args.k_first, combo_dir,
                                       combo_suffix="lines_lineplots")

            # (B) one PDF regrouping ALL violins
            violins_three_sets_one_figure(inst_sub, tauc_sub, ten_dt_sub, args.k_first, combo_dir,
                                          filename="violins_all_sets.pdf")



    # --- Console summary (quick sanity check) ---
    def _summ(name, tab: Optional[pd.DataFrame], col='swarm_lambda_nz_1'):
        if tab is None or tab.empty or col not in tab.columns:
            print(f"[summary] {name}: n/a")
            return
        arr = tab[col].to_numpy()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            print(f"[summary] {name}: n/a")
            return
        print(f"[summary] {name}: {col} mean={np.mean(arr):.4f} ± {np.std(arr):.4f} (n={arr.size})")

    _summ("instantaneous (λ₂≡first non-zero, swarm)", inst, 'swarm_lambda_nz_1')
    _summ("instantaneous (λ_max, swarm)", inst, 'swarm_lambda_max')
    if tavg_tau is not None:
        _summ("tavg τ_c (λ₂ swarm)", tavg_tau, 'swarm_lambda_nz_1')
    if tavg_10dt is not None:
        _summ("tavg 10×Δt (λ₂ swarm)", tavg_10dt, 'swarm_lambda_nz_1')

    print(f"\nDone. Outputs in: {args.output_dir}\n")

if __name__ == "__main__":
    sys.exit(main())

