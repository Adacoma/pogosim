#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Active-matter global metrics for Pogosim — executable script.

- Non-periodic boundaries (matching your sims).
- All distances assumed in millimetres (mm).
- Neighbor cutoff rc derived from config:
      rc = communication_radius_border_to_border + 2*robot_radius
      (which equals center-to-center communication radius)

Outputs:
- CSVs for each metric (optional; easy to add).
- PDFs: one plot per file.
- SUMMARY_global.txt (human-friendly).
- metrics.pkl (pickle dict of all DataFrames).
"""

from __future__ import annotations

import os
import sys
import argparse
import pickle
from dataclasses import dataclass
from typing import Iterable, Tuple, Optional, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None  # type: ignore

# --------------------------------------------------------------------------- #
#                               Core utilities                                #
# --------------------------------------------------------------------------- #

@dataclass
class Box:
    x0: float
    x1: float
    y0: float
    y1: float
    periodic: bool = False  # kept for API compatibility; we never enable it here

    @property
    def Lx(self) -> float: return float(self.x1 - self.x0)
    @property
    def Ly(self) -> float: return float(self.y1 - self.y0)

    def wrap(self, pts: np.ndarray) -> np.ndarray:
        return pts  # non-periodic in this project

    def delta(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a - b  # non-periodic minimal image = plain difference


def infer_box(df: pd.DataFrame, margin: float = 0.0, periodic: bool = False) -> Box:
    x0 = float(df['x'].min()) - margin
    x1 = float(df['x'].max()) + margin
    y0 = float(df['y'].min()) - margin
    y1 = float(df['y'].max()) + margin
    return Box(x0, x1, y0, y1, periodic=False)


def ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    preferred = ['arena_file', 'run', 'robot_category', 'robot_id', 'time']
    sort_cols = [c for c in preferred if c in df.columns]
    if not sort_cols:
        return df.reset_index(drop=True)
    return df.sort_values(sort_cols, kind='mergesort').reset_index(drop=True)


def estimate_velocities(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_sorted(df)
    # group columns: include only those present
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


def snapshot_key_columns(df: pd.DataFrame) -> list[str]:
    # Only include keys that actually exist in df
    return [c for c in ['arena_file', 'run', 'time'] if c in df.columns]

def snapshot_groups(df: pd.DataFrame):
    """
    Yield (key_tuple, snapshot_df), where key_tuple is ALWAYS a tuple
    (even if there's only one key, e.g., ('time',) ).
    """
    need = ['time','robot_category','robot_id','x','y']
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe missing columns: {missing}")

    keys = snapshot_key_columns(df)
    if not keys:
        # No grouping keys: treat whole df as one snapshot
        yield ((), df)
        return

    for key, snap in df.groupby(keys, sort=False):
        if not isinstance(key, tuple):
            key = (key,)  # normalize scalar -> tuple
        yield key, snap


def _kdtree_neighbors(pts: np.ndarray, rc: float, box: Box) -> List[List[int]]:
    if cKDTree is None:
        raise ImportError("scipy is required for neighbor queries (scipy.spatial.cKDTree).")
    tree = cKDTree(pts)
    return tree.query_ball_point(pts, rc)

# --------------------------------------------------------------------------- #
#                                  Metrics                                    #
# --------------------------------------------------------------------------- #

def vicsek_polar_order_snapshot(pts: np.ndarray, vel: np.ndarray) -> float:
    speeds = np.linalg.norm(vel, axis=1)
    mask = speeds > 1e-12
    if not np.any(mask):
        return np.nan
    vhat = vel[mask] / speeds[mask, None]
    Pvec = vhat.mean(axis=0)
    return float(np.linalg.norm(Pvec))


def vicsek_polar_order(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if not {'vx','vy'}.issubset(work.columns):
        work = estimate_velocities(work)

    keys = snapshot_key_columns(work)
    rows = []
    for key, snap in snapshot_groups(work):
        vel = snap[['vx','vy']].to_numpy()
        pts = snap[['x','y']].to_numpy()
        P = vicsek_polar_order_snapshot(pts, vel)
        rows.append((*key, P))

    cols = keys + ['P_polar']
    return pd.DataFrame(rows, columns=cols)


def vicsek_local_polar_order_snapshot(pts, vel, rc, box=None):
    if box is None:
        box = infer_box(pd.DataFrame({'x': pts[:,0], 'y': pts[:,1]}))
    speeds = np.linalg.norm(vel, axis=1)
    vhat = np.zeros_like(vel, dtype=float)
    nonzero = speeds > 1e-12
    vhat[nonzero] = vel[nonzero] / speeds[nonzero, None]

    neighs = _kdtree_neighbors(pts, rc, box)
    N = len(pts)
    P_local = np.full(N, np.nan, dtype=float)
    n_nbrs = np.zeros(N, dtype=int)

    for i, nbrs in enumerate(neighs):
        # KEEP self in the set (KDTree already includes i)
        # Filter to neighbors (including self) with non-zero speed
        valid = [j for j in nbrs if nonzero[j]]
        n_nbrs[i] = max(0, len(valid) - 1)  # if you still want “neighbors (excl. self)” count
        if len(valid) == 0:
            continue
        m = np.mean(vhat[valid], axis=0)
        P_local[i] = float(np.linalg.norm(m))
    return P_local, n_nbrs



def vicsek_local_polar_order(
    df: pd.DataFrame,
    rc: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      summary_df: per-snapshot stats (mean, std, median, q25, q75, frac_isolated, mean_n)
      dist_df:    per-agent values (one row per agent per snapshot) with P_local and n_nbrs
    """
    work = df.copy()
    if not {'vx','vy'}.issubset(work.columns):
        work = estimate_velocities(work)

    keys = snapshot_key_columns(work)
    id_cols = [c for c in ['robot_category','robot_id'] if c in work.columns]

    sum_rows = []
    dist_rows = []
    for key, snap in snapshot_groups(work):
        pts = snap[['x','y']].to_numpy()
        vel = snap[['vx','vy']].to_numpy()
        box = infer_box(snap)
        P_local, n_nbrs = vicsek_local_polar_order_snapshot(pts, vel, rc=rc, box=box)

        # per-agent distribution rows
        base_key = list(key)
        rep = len(snap)
        for (idx, (p, n)), (_, agent_row) in zip(enumerate(zip(P_local, n_nbrs)), snap.iterrows()):
            dist_rows.append(tuple(base_key + [*(agent_row[c] for c in id_cols), float(p), int(n)]))

        # per-snapshot summary (ignore NaNs for agents with no neighbors)
        finite = np.isfinite(P_local)
        vals = P_local[finite]
        # weights: valid neighbor count + 1 (self). If you didn't keep it, re-query here as shown.
        w = (n_nbrs[finite] + 1).astype(float)
        if vals.size == 0:
            stats = dict(mean=np.nan, std=np.nan, median=np.nan, q25=np.nan, q75=np.nan)
        else:
            wsum = np.sum(w)
            wmean = float(np.sum(w * vals) / (wsum + 1e-12))
            stats = dict(
                mean=wmean,
                std=float(np.std(vals)),       # leave unweighted for spread, or make it w-std
                median=float(np.median(vals)),
                q25=float(np.quantile(vals, 0.25)),
                q75=float(np.quantile(vals, 0.75)),
            )
        frac_isolated = 1.0 - (np.count_nonzero(finite) / max(len(P_local), 1))
        mean_n = float(np.mean(n_nbrs)) if len(n_nbrs) else np.nan
        sum_rows.append(tuple(list(key) + [
            stats['mean'], stats['std'], stats['median'], stats['q25'], stats['q75'],
            frac_isolated, mean_n
        ]))

    summary_cols = keys + ['P_local_mean','P_local_std','P_local_median','P_local_q25','P_local_q75','frac_isolated','mean_n_neighbors']
    dist_cols    = keys + id_cols + ['P_local','n_neighbors']
    return pd.DataFrame(sum_rows, columns=summary_cols), pd.DataFrame(dist_rows, columns=dist_cols)


def local_pairwise_alignment_snapshot(
    vel: np.ndarray,
    neighs: List[List[int]],
    *,
    exclude_zero_speed: bool = True,
) -> Tuple[np.ndarray, float, int, np.ndarray]:
    """
    Compute node-wise pairwise alignment A_i and edge-weighted mean per snapshot.

    For agent i with neighbors N_i (excluding self), define:
        A_i = (1/|N_i|) * sum_{j in N_i} cos(theta_i - theta_j).
    Random headings -> E[A_i] = 0. Perfect local alignment -> A_i = 1.

    Returns:
      A_local : (N,) array with A_i (NaN if no valid neighbors)
      A_edge_mean : scalar, edge-weighted mean across all pairs (i,j)
      num_edges : total number of directed edges counted (sum_i |N_i|)
      n_nbrs : (N,) neighbor counts used for A_i
    """
    N = vel.shape[0]
    th = np.arctan2(vel[:, 1], vel[:, 0])
    speed = np.hypot(vel[:, 0], vel[:, 1])
    nonzero = (speed > 1e-12) if exclude_zero_speed else np.ones(N, dtype=bool)

    A_local = np.full(N, np.nan, dtype=float)
    n_nbrs = np.zeros(N, dtype=int)
    sum_cos = 0.0
    num_edges = 0

    for i, nbrs in enumerate(neighs):
        # exclude self (KDTree includes it)
        nbrs = [j for j in nbrs if j != i]
        if not nonzero[i]:
            continue
        # keep neighbors with nonzero speed if requested
        nbrs = [j for j in nbrs if nonzero[j]]
        n = len(nbrs)
        n_nbrs[i] = n
        if n == 0:
            continue
        dth = th[i] - th[np.array(nbrs)]
        ci = np.cos(dth)
        A_local[i] = float(np.mean(ci))
        sum_cos += float(np.sum(ci))
        num_edges += n

    A_edge_mean = (sum_cos / num_edges) if num_edges > 0 else np.nan
    return A_local, A_edge_mean, num_edges, n_nbrs


def local_pairwise_alignment(
    df: pd.DataFrame,
    rc: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      summary_df: per-snapshot stats with:
          - A_edge_mean (edge-weighted mean of cos differences; zero baseline)
          - A_node_mean, A_node_std, A_node_median, A_node_q25, A_node_q75
          - frac_isolated (fraction with n_nbrs == 0 w.r.t. valid-speed filter)
          - mean_n_neighbors
      dist_df: per-agent rows with A_local and n_neighbors (one row per agent per snapshot)
    """
    work = df.copy()
    if not {'vx','vy'}.issubset(work.columns):
        work = estimate_velocities(work)

    keys = snapshot_key_columns(work)
    id_cols = [c for c in ['robot_category','robot_id'] if c in work.columns]

    sum_rows = []
    dist_rows = []

    for key, snap in snapshot_groups(work):
        pts = snap[['x','y']].to_numpy()
        vel = snap[['vx','vy']].to_numpy()
        box = infer_box(snap)
        neighs = _kdtree_neighbors(pts, rc, box)

        A_local, A_edge_mean, num_edges, n_nbrs = local_pairwise_alignment_snapshot(vel, neighs)

        # per-agent distribution rows
        base_key = list(key)
        for (Ai, n_i), (_, agent_row) in zip(zip(A_local, n_nbrs), snap.iterrows()):
            dist_rows.append(tuple(base_key + [*(agent_row[c] for c in id_cols), float(Ai), int(n_i)]))

        # per-snapshot summary (ignore NaNs from isolated/invalid)
        finite = np.isfinite(A_local)
        vals = A_local[finite]
        frac_isolated = 1.0 - (np.count_nonzero(finite) / max(len(A_local), 1))
        mean_n = float(np.mean(n_nbrs)) if len(n_nbrs) else np.nan
        if vals.size == 0:
            stats = dict(A_node_mean=np.nan, A_node_std=np.nan, A_node_median=np.nan, A_node_q25=np.nan, A_node_q75=np.nan)
        else:
            stats = dict(
                A_node_mean=float(np.mean(vals)),
                A_node_std=float(np.std(vals)),
                A_node_median=float(np.median(vals)),
                A_node_q25=float(np.quantile(vals, 0.25)),
                A_node_q75=float(np.quantile(vals, 0.75)),
            )

        sum_rows.append(tuple(list(key) + [
            float(A_edge_mean),               # edge-weighted mean (zero baseline)
            stats['A_node_mean'], stats['A_node_std'], stats['A_node_median'], stats['A_node_q25'], stats['A_node_q75'],
            frac_isolated, mean_n
        ]))

    summary_cols = keys + [
        'A_edge_mean',
        'A_node_mean','A_node_std','A_node_median','A_node_q25','A_node_q75',
        'frac_isolated','mean_n_neighbors'
    ]
    dist_cols    = keys + id_cols + ['A_local','n_neighbors']
    return pd.DataFrame(sum_rows, columns=summary_cols), pd.DataFrame(dist_rows, columns=dist_cols)



def bond_orientational_order_snapshot(pts: np.ndarray, rc: float, k: int = 6, box: Optional[Box] = None) -> float:
    if box is None:
        box = Box(pts[:,0].min(), pts[:,0].max(), pts[:,1].min(), pts[:,1].max(), periodic=False)
    neighs = _kdtree_neighbors(pts, rc, box)
    vals = []
    for i, nbrs in enumerate(neighs):
        nbrs = [j for j in nbrs if j != i]
        if not nbrs:
            continue
        rij = pts[nbrs] - pts[i]
        angles = np.arctan2(rij[:,1], rij[:,0])
        psi_i = np.exp(1j * k * angles).mean()
        vals.append(psi_i)
    if not vals:
        return np.nan
    psi_k = np.mean(vals)
    return float(np.abs(psi_k))


def bond_orientational_order(df: pd.DataFrame, rc: float, k: int = 6) -> pd.DataFrame:
    work = df.copy()
    keys = snapshot_key_columns(work)
    rows = []
    for key, snap in snapshot_groups(work):
        pts = snap[['x','y']].to_numpy()
        box = infer_box(snap)
        val = bond_orientational_order_snapshot(pts, rc=rc, k=k, box=box)
        rows.append((*key, val))
    cols = keys + [f'psi_{k}']
    return pd.DataFrame(rows, columns=cols)


def largest_cluster_fraction_snapshot(pts: np.ndarray, rc: float, box: Optional[Box] = None) -> float:
    if box is None:
        box = infer_box(pd.DataFrame({'x': pts[:,0], 'y': pts[:,1]}))
    neighs = _kdtree_neighbors(pts, rc, box)
    N = len(pts)
    seen = np.zeros(N, dtype=bool)
    best = 0
    for i in range(N):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        count = 0
        while stack:
            u = stack.pop()
            count += 1
            for v in neighs[u]:
                if v == u or seen[v]:
                    continue
                seen[v] = True
                stack.append(v)
        best = max(best, count)
    return best / max(1, N)


def largest_cluster_fraction(df: pd.DataFrame, rc: float) -> pd.DataFrame:
    work = df.copy()
    keys = snapshot_key_columns(work)
    rows = []
    for key, snap in snapshot_groups(work):
        pts = snap[['x','y']].to_numpy()
        box = infer_box(snap)
        fmax = largest_cluster_fraction_snapshot(pts, rc=rc, box=box)
        rows.append((*key, fmax))
    cols = keys + ['largest_cluster_frac']
    return pd.DataFrame(rows, columns=cols)


def radial_distribution_function_snapshot(
    pts: np.ndarray, box: Box, r_max: float, dr: float
) -> Tuple[np.ndarray, np.ndarray]:
    N = len(pts)
    r = np.arange(dr/2, r_max, dr)
    if N < 2:
        return r, np.full_like(r, np.nan, dtype=float)
    if cKDTree is None:
        raise ImportError("scipy is required for g(r) (scipy.spatial.cKDTree).")

    tree = cKDTree(pts)
    pairs = tree.query_pairs(r_max)
    if not pairs:
        return r, np.full_like(r, np.nan, dtype=float)
    dists = np.array([np.linalg.norm(pts[i]-pts[j]) for i, j in pairs], dtype=float)

    bins = np.arange(0.0, r_max + dr, dr)
    counts, edges = np.histogram(dists, bins=bins)
    r = 0.5 * (edges[:-1] + edges[1:])

    area = box.Lx * box.Ly
    rho = N / area if area > 0 else np.nan
    shell_area = 2 * np.pi * r * dr
    g_r = (2.0 * counts / max(N, 1)) / (rho * shell_area + 1e-12)
    return r, g_r


def radial_distribution_function(df: pd.DataFrame, r_max: float, dr: float) -> pd.DataFrame:
    keys = snapshot_key_columns(df)
    rows = []
    for key, snap in snapshot_groups(df):
        pts = snap[['x','y']].to_numpy()
        box = infer_box(snap)
        r, g = radial_distribution_function_snapshot(pts, box, r_max=r_max, dr=dr)
        for ri, gi in zip(r, g):
            rows.append((*key, ri, gi))
    cols = keys + ['r','g_r']
    return pd.DataFrame(rows, columns=cols)


def structure_factor_snapshot(
    pts: np.ndarray, box: Box, grid_nx: int = 128, grid_ny: int = 128
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    H, xedges, yedges = np.histogram2d(
        pts[:,0], pts[:,1],
        bins=[grid_nx, grid_ny],
        range=[[box.x0, box.x1], [box.y0, box.y1]],
        density=False
    )
    rho = H.astype(float)
    rho_k = np.fft.fftshift(np.fft.fft2(rho))
    S = (rho_k * np.conj(rho_k)).real
    kx = np.fft.fftshift(np.fft.fftfreq(grid_nx, d=(box.Lx / grid_nx))) * 2*np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(grid_ny, d=(box.Ly / grid_ny))) * 2*np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)

    k_max = float(np.max(K))
    nbins = min(grid_nx, grid_ny) // 2
    k_edges = np.linspace(0.0, k_max, nbins+1)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])
    S_k = np.zeros(nbins, dtype=float)
    counts = np.zeros(nbins, dtype=int)

    flat_K = K.ravel()
    flat_S = S.ravel()
    inds = np.digitize(flat_K, k_edges) - 1
    mask = (inds >= 0) & (inds < nbins)
    for b in range(nbins):
        sel = (inds == b) & mask
        if np.any(sel):
            S_k[b] = float(np.mean(flat_S[sel]))
            counts[b] = int(np.sum(sel))
        else:
            S_k[b] = np.nan
            counts[b] = 0
    return k_centers, S_k, counts


def structure_factor(df: pd.DataFrame, grid_nx: int = 128, grid_ny: int = 128) -> pd.DataFrame:
    keys = snapshot_key_columns(df)
    rows = []
    for key, snap in snapshot_groups(df):
        pts = snap[['x','y']].to_numpy()
        box = infer_box(snap)
        k, S_k, cnt = structure_factor_snapshot(pts, box, grid_nx=grid_nx, grid_ny=grid_ny)
        for ki, Si, ci in zip(k, S_k, cnt):
            rows.append((*key, ki, Si, ci))
    cols = keys + ['k','S_k','bin_count']
    return pd.DataFrame(rows, columns=cols)


def number_fluctuation_exponent_snapshot(
    pts: np.ndarray, box: Box, n_scales: int = 6, samples_per_scale: int = 128, rng: Optional[np.random.Generator] = None
) -> Tuple[float, pd.DataFrame]:
    if rng is None:
        rng = np.random.default_rng(12345)
    N = len(pts)
    if N < 2 or box.Lx <= 0 or box.Ly <= 0:
        return np.nan, pd.DataFrame(columns=['N_bar','sigma','scale_idx'])
    min_side = min(box.Lx, box.Ly)
    side_fracs = np.geomspace(0.05, 0.5, num=n_scales)
    rows = []
    for si, frac in enumerate(side_fracs):
        side = frac * min_side
        for _ in range(samples_per_scale):
            x0 = rng.uniform(box.x0, box.x1 - side)
            y0 = rng.uniform(box.y0, box.y1 - side)
            x1 = x0 + side
            y1 = y0 + side
            inside = (pts[:,0] >= x0) & (pts[:,0] < x1) & (pts[:,1] >= y0) & (pts[:,1] < y1)
            n = int(np.sum(inside))
            rows.append((si, n, side*side))
    tab = pd.DataFrame(rows, columns=['scale_idx','N','area'])
    grouped = tab.groupby('scale_idx', as_index=False).agg(N_bar=('N','mean'), sigma=('N','std'), area=('area','first'))
    g = grouped.replace(0, np.nan).dropna()
    if len(g) < 2:
        return np.nan, grouped.assign(alpha=np.nan)
    x = np.log(g['N_bar'].to_numpy())
    y = np.log(g['sigma'].to_numpy() + 1e-12)
    A = np.vstack([x, np.ones_like(x)]).T
    alpha, _inter = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(alpha), grouped.assign(alpha=alpha)


def number_fluctuation_exponent(df: pd.DataFrame, n_scales: int = 6, samples_per_scale: int = 128) -> pd.DataFrame:
    keys = snapshot_key_columns(df)
    rows = []
    for key, snap in snapshot_groups(df):
        pts = snap[['x','y']].to_numpy()
        box = infer_box(snap)
        alpha, _ = number_fluctuation_exponent_snapshot(pts, box, n_scales=n_scales, samples_per_scale=samples_per_scale)
        rows.append((*key, alpha))
    cols = keys + ['alpha_number_fluct']
    return pd.DataFrame(rows, columns=cols)


def effective_temperature_ratio(df: pd.DataFrame, v2_ref: float) -> pd.DataFrame:
    work = df.copy()
    if not {'vx','vy'}.issubset(work.columns):
        work = estimate_velocities(work)
    keys = snapshot_key_columns(work)
    rows = []
    for key, snap in snapshot_groups(work):
        v2 = (snap['vx'].to_numpy()**2 + snap['vy'].to_numpy()**2)
        ratio = float(np.nanmean(v2) / (v2_ref + 1e-12))
        rows.append((*key, ratio))
    cols = keys + ['Teff_over_Tref']
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
#                            Convenience + Summary                            #
# --------------------------------------------------------------------------- #

def compute_all_metrics(
    df: pd.DataFrame,
    *,
    rc_bond: float,
    rc_cluster: float,
    gr_rmax: float,
    gr_dr: float,
    S_grid_nx: int,
    S_grid_ny: int,
    v2_ref: Optional[float] = None,
    psi_k: int = 6,
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    out['polar_order']    = vicsek_polar_order(df)
    out['local_polar'], out['local_polar_dist'] = vicsek_local_polar_order(df, rc=rc_bond)
    out['local_pair_align'], out['local_pair_align_dist'] = local_pairwise_alignment(df, rc=rc_bond)
    out['psi_k']          = bond_orientational_order(df, rc=rc_bond, k=psi_k)
    out['largest_cluster']= largest_cluster_fraction(df, rc=rc_cluster)
    out['g_r']            = radial_distribution_function(df, r_max=gr_rmax, dr=gr_dr)
    out['S_k']            = structure_factor(df, grid_nx=S_grid_nx, grid_ny=S_grid_ny)
    out['number_fluct']   = number_fluctuation_exponent(df)
    if v2_ref is not None:
        out['Teff_ratio'] = effective_temperature_ratio(df, v2_ref=v2_ref)
    return out


def _fmt(x, p=3):
    return "n/a" if (x is None or not np.isfinite(x)) else f"{x:.{p}f}"


def write_global_summary(output_dir: str, results: Dict[str, pd.DataFrame]) -> None:
    lines = ["# Global structure metrics — summary\n"]
    def add_line(label, value): lines.append(f"- {label}: {value}")

    P = results.get('polar_order', pd.DataFrame())
    if not P.empty:
        add_line("Polar order ⟨P⟩ over time", _fmt(P['P_polar'].mean()))
    LP = results.get('local_polar', pd.DataFrame())
    if not LP.empty:
        add_line("Local polar order ⟨P_local⟩ over time", _fmt(LP['P_local_mean'].mean()))
        add_line("Mean fraction isolated (no neighbors)", _fmt(LP['frac_isolated'].mean()))
    LPA = results.get('local_pair_align', pd.DataFrame())
    if not LPA.empty:
        add_line("Local pairwise alignment ⟨A⟩ over time (edge-weighted)", _fmt(LPA['A_edge_mean'].mean()))
    psi_any = [k for k in results if k == 'psi_k']
    if psi_any:
        psi_df = results['psi_k']
        psi_col = [c for c in psi_df.columns if c.startswith('psi_')][0] if not psi_df.empty else None
        add_line(f"Bond-orientational ⟨|{psi_col}|⟩", _fmt(psi_df[psi_col].mean() if psi_col else np.nan))
    L = results.get('largest_cluster', pd.DataFrame())
    if not L.empty:
        add_line("Largest cluster fraction ⟨f_max⟩", _fmt(L['largest_cluster_frac'].mean()))
    GR = results.get('g_r', pd.DataFrame())
    if not GR.empty:
        # crude peak height near first neighbor shell
        gr_agg = GR.groupby('r', as_index=False)['g_r'].mean()
        peak = gr_agg['g_r'].max()
        r_at_peak = gr_agg.loc[gr_agg['g_r'].idxmax(), 'r']
        add_line("g(r) first peak height", f"{_fmt(peak)} at r ≈ {_fmt(r_at_peak)} mm")
    SK = results.get('S_k', pd.DataFrame())
    if not SK.empty:
        Sk_agg = SK.groupby('k', as_index=False)['S_k'].mean()
        add_line("S(k) max", f"{_fmt(Sk_agg['S_k'].max())} at k ≈ {_fmt(Sk_agg.loc[Sk_agg['S_k'].idxmax(),'k'])} mm⁻¹")
    NF = results.get('number_fluct', pd.DataFrame())
    if not NF.empty:
        add_line("Number-fluctuation exponent ⟨α⟩", _fmt(NF['alpha_number_fluct'].mean()))
    TE = results.get('Teff_ratio', pd.DataFrame())
    if TE is not None and not TE.empty:
        add_line("⟨T_eff/T_ref⟩", _fmt(TE['Teff_over_Tref'].mean()))

    text = "\n".join(lines) + "\n"
    with open(os.path.join(output_dir, "SUMMARY_global.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)

# --------------------------------------------------------------------------- #
#                                    Plots                                    #
# --------------------------------------------------------------------------- #

def _prepare_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _save_pdf(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)

def plot_time_series_with_band(df: pd.DataFrame, y_mean: str, y_std: str, ylabel: str, title: str, out_pdf: str) -> None:
    fig = plt.figure(figsize=(7, 4.2))
    ax = fig.add_subplot(111)
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
    else:
        if 'run' in df.columns:
            # faint per-run mean if available
            for r, g in df.groupby('run', sort=False):
                if {y_mean}.issubset(g.columns):
                    ax.plot(g['time'], g[y_mean], alpha=0.25, linewidth=1)
        agg = df.groupby('time', as_index=False).agg(mu=(y_mean,'mean'), sig=(y_std,'mean'))
        ax.plot(agg['time'], agg['mu'], linewidth=2.2)
        if np.isfinite(agg['sig']).any():
            ax.fill_between(agg['time'], agg['mu']-agg['sig'], agg['mu']+agg['sig'], alpha=0.2, linewidth=0)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    _save_pdf(fig, out_pdf)


def plot_time_series(df: pd.DataFrame, y_col: str, ylabel: str, title: str, out_pdf: str) -> None:
    fig = plt.figure(figsize=(7, 4.2))
    ax = fig.add_subplot(111)
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
    else:
        # average per time (across runs if multiple); also show faint runs
        if 'run' in df.columns:
            for r, g in df.groupby('run', sort=False):
                ax.plot(g['time'], g[y_col], alpha=0.25, linewidth=1)
            agg = df.groupby('time', as_index=False)[y_col].mean()
            ax.plot(agg['time'], agg[y_col], linewidth=2.2)
        else:
            ax.plot(df['time'], df[y_col], linewidth=2.2)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    _save_pdf(fig, out_pdf)

def plot_histogram(df: pd.DataFrame, col: str, bins: int, title: str, out_pdf: str) -> None:
    fig = plt.figure(figsize=(6.2, 4.2))
    ax = fig.add_subplot(111)
    if df.empty or col not in df.columns:
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
    else:
        vals = df[col].to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            ax.text(0.5, 0.5, "No finite values", ha='center', va='center')
        else:
            ax.hist(vals, bins=bins, density=True)
            ax.set_xlabel(col)
            ax.set_ylabel("PDF")
            ax.set_title(title)
    _save_pdf(fig, out_pdf)

def plot_gr(gr_df: pd.DataFrame, out_pdf: str) -> None:
    fig = plt.figure(figsize=(6.2, 4.2))
    ax = fig.add_subplot(111)
    if gr_df.empty:
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
    else:
        # mean ± std over time
        agg = gr_df.groupby('r', as_index=False).agg(g_mean=('g_r','mean'), g_std=('g_r','std'))
        ax.plot(agg['r'], agg['g_mean'], linewidth=2.0)
        if np.isfinite(agg['g_std']).any():
            ax.fill_between(agg['r'], agg['g_mean']-agg['g_std'], agg['g_mean']+agg['g_std'], alpha=0.2, linewidth=0)
        ax.set_xlabel("r [mm]")
        ax.set_ylabel("g(r)")
        ax.set_title("Radial distribution function")
    _save_pdf(fig, out_pdf)

def plot_sk(sk_df: pd.DataFrame, out_pdf: str) -> None:
    fig = plt.figure(figsize=(6.2, 4.2))
    ax = fig.add_subplot(111)
    if sk_df.empty:
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
    else:
        agg = sk_df.groupby('k', as_index=False).agg(S_mean=('S_k','mean'), S_std=('S_k','std'))
        ax.plot(agg['k'], agg['S_mean'], linewidth=2.0)
        if np.isfinite(agg['S_std']).any():
            ax.fill_between(agg['k'], agg['S_mean']-agg['S_std'], agg['S_mean']+agg['S_std'], alpha=0.2, linewidth=0)
        ax.set_xlabel("k [mm⁻¹]")
        ax.set_ylabel("S(k)")
        ax.set_title("Static structure factor")
    _save_pdf(fig, out_pdf)

# --------------------------------------------------------------------------- #
#                                    Main                                     #
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Compute global active-matter metrics (non-periodic, mm units)."
    )
    p.add_argument("-i", "--input", dest="input_file", required=True,
                   help="Input Feather/CSV produced by Pogosim")
    p.add_argument("-o", "--output", dest="output_dir", required=True,
                   help="Directory to write PDFs, summary and pickle")
    p.add_argument("--psi-k", type=int, default=6, help="k for bond-orientational order (default: 6)")
    p.add_argument("--gr-rmax", type=float, default=200.0, help="g(r) max radius in mm (default: 200)")
    p.add_argument("--gr-dr", type=float, default=2.0, help="g(r) bin width in mm (default: 2)")
    p.add_argument("--sk-nx", type=int, default=128, help="S(k) grid size Nx (default: 128)")
    p.add_argument("--sk-ny", type=int, default=128, help="S(k) grid size Ny (default: 128)")
    p.add_argument("--v2-ref", type=float, default=None, help="Reference <v^2> for T_eff/T_ref (optional)")
    p.add_argument("--no-csv", action="store_true", help="Skip CSV export (PDF+pickle+summary only)")
    return p.parse_args(argv)


def load_df_and_config(input_file: str):
    # Try the same import style as locomotion.py uses
    try:
        from scripts.pogosim import utils  # type: ignore
    except Exception:
        try:
            import utils  # type: ignore
        except Exception as e:
            raise ImportError("Cannot import utils.load_dataframe; ensure PYTHONPATH includes your project.") from e

    df, meta = utils.load_dataframe(input_file)  # must return (DataFrame, meta)
    config = meta.get("configuration", {}) if isinstance(meta, dict) else {}
    objects = config.get("objects", {})
    robots = objects.get("robots", {})
    communication_radius_border = float(robots.get("communication_radius", 80.0))
    robot_radius = float(robots.get("radius", 26.5))
    rc = communication_radius_border + 2.0 * robot_radius  # center-to-center cutoff in mm
    return df, rc, meta


def main(argv=None):
    args = parse_args(argv)
    _prepare_output_dir(args.output_dir)

    df, rc_neighbor, meta = load_df_and_config(args.input_file)

    # Remove raws with NaN poses
    df = df.dropna(subset=["x", "y", "angle"]).reset_index(drop=True)

    # Compute metrics (non-periodic)
    results = compute_all_metrics(
        df,
        rc_bond=rc_neighbor,
        rc_cluster=rc_neighbor,
        gr_rmax=args.gr_rmax,
        gr_dr=args.gr_dr,
        S_grid_nx=args.sk_nx,
        S_grid_ny=args.sk_ny,
        v2_ref=args.v2_ref,
        psi_k=args.psi_k,
    )

    # Optional CSV export
    if not args.no_csv:
        for key, tdf in results.items():
            if tdf is None or tdf.empty:
                continue
            tdf.to_csv(os.path.join(args.output_dir, f"{key}.csv"), index=False)

    # Plots (PDF one per file)
    if 'polar_order' in results:
        plot_time_series(
            results['polar_order'], 'P_polar', "Polar order P", "Vicsek polar order vs time",
            os.path.join(args.output_dir, "polar_order_time.pdf")
        )
    if 'local_polar' in results:
        plot_time_series_with_band(
            results['local_polar'],
            'P_local_mean', 'P_local_std',
            "Local polar order ⟨P_local⟩",
            "Local Vicsek order (mean ± std) vs time",
            os.path.join(args.output_dir, "local_polar_time.pdf")
        )
    if 'local_polar_dist' in results:
        # pooled histogram across all times/runs/agents (quick diagnostic)
        plot_histogram(
            results['local_polar_dist'],
            'P_local', 40,
            "Distribution of local Vicsek order P_local",
            os.path.join(args.output_dir, "local_polar_hist.pdf")
        )
    if 'local_pair_align' in results:
        plot_time_series(
            results['local_pair_align'],
            'A_edge_mean',
            "Local pairwise alignment ⟨A⟩ (edge-weighted; zero baseline)",
            "Local pairwise alignment vs time",
            os.path.join(args.output_dir, "local_pair_align_time.pdf")
        )
    if 'local_pair_align_dist' in results:
        plot_histogram(
            results['local_pair_align_dist'],
            'A_local',
            40,
            "Distribution of local pairwise alignment A_i",
            os.path.join(args.output_dir, "local_pair_align_hist.pdf")
        )
    if 'psi_k' in results and not results['psi_k'].empty:
        psi_col = [c for c in results['psi_k'].columns if c.startswith('psi_')][0]
        plot_time_series(
            results['psi_k'], psi_col, f"|{psi_col}|", f"Bond-orientational {psi_col} vs time",
            os.path.join(args.output_dir, f"{psi_col}_time.pdf")
        )
    if 'largest_cluster' in results:
        plot_time_series(
            results['largest_cluster'], 'largest_cluster_frac', "Largest cluster fraction",
            "Largest connected fraction vs time",
            os.path.join(args.output_dir, "largest_cluster_time.pdf")
        )
    if 'g_r' in results:
        plot_gr(results['g_r'], os.path.join(args.output_dir, "g_r.pdf"))
    if 'S_k' in results:
        plot_sk(results['S_k'], os.path.join(args.output_dir, "S_k.pdf"))
    if 'number_fluct' in results:
        plot_time_series(
            results['number_fluct'], 'alpha_number_fluct', "α (number fluctuations)",
            "Number fluctuation exponent vs time",
            os.path.join(args.output_dir, "number_fluct_alpha_time.pdf")
        )
    if 'Teff_ratio' in results:
        plot_time_series(
            results['Teff_ratio'], 'Teff_over_Tref', "T_eff/T_ref",
            "Effective temperature ratio vs time",
            os.path.join(args.output_dir, "teff_ratio_time.pdf")
        )

    # Text summary (printed + file)
    write_global_summary(args.output_dir, results)

    # Pickle (all metrics together)
    with open(os.path.join(args.output_dir, "metrics.pkl"), "wb") as f:
        pickle.dump({"meta": meta, "results": results}, f)

    print(f"\nDone. Outputs in: {args.output_dir}\n")


if __name__ == "__main__":
    sys.exit(main())

