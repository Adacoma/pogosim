#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Active-matter global metrics for Pogosim — executable script.

- Non-periodic boundaries
- All distances assumed in millimetres (mm).
- Neighbor cutoff rc derived from config:
      rc = communication_radius_border_to_border + 2*robot_radius
      (which equals center-to-center communication radius)

Outputs:
- CSVs for each metric
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
from scipy.ndimage import gaussian_filter

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


def nan_gaussian_smooth(A, sigma=1.2):
    mask = np.isfinite(A).astype(float)
    A0 = np.nan_to_num(A, nan=0.0)
    num = gaussian_filter(A0, sigma=sigma, mode='nearest')
    den = gaussian_filter(mask, sigma=sigma, mode='nearest') + 1e-12
    out = num / den
    return out


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
    """
    Per-snapshot local Vicsek order with Rayleigh small-sample correction.

    Returns:
      P_local : (N,) raw local mean resultant length (includes self in average)
      n_nbrs  : (N,) neighbor count EXCLUDING self, for backward-compat
      Also used by caller to derive bias-free and Rayleigh-Z summaries.
    """
    if box is None:
        box = infer_box(pd.DataFrame({'x': pts[:,0], 'y': pts[:,1]}))

    speeds = np.linalg.norm(vel, axis=1)
    vhat = np.zeros_like(vel, dtype=float)
    nonzero = speeds > 1e-12
    vhat[nonzero] = vel[nonzero] / speeds[nonzero, None]

    neighs = _kdtree_neighbors(pts, rc, box)
    N = len(pts)

    P_local = np.full(N, np.nan, dtype=float)
    n_nbrs  = np.zeros(N, dtype=int)  # neighbors EXCLUDING self

    for i, nbrs in enumerate(neighs):
        # keep neighbors (INCLUDING self) with nonzero speed
        valid = [j for j in nbrs if nonzero[j]]
        m = len(valid)  # sample size including self (if nonzero)
        n_nbrs[i] = max(0, m - 1)

        if m == 0:
            continue
        mvec = np.mean(vhat[valid], axis=0)
        P_local[i] = float(np.linalg.norm(mvec))

    return P_local, n_nbrs


def vicsek_local_polar_order(
    df: pd.DataFrame,
    rc: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-snapshot summary + per-agent distribution with bias-corrected local order.

    Adds:
      - P_local_biasfree  = max(0, P_local - E0), E0 = sqrt(pi)/(2*sqrt(m))
      - Rayleigh_Z        = 2*m*P_local^2  (NaN for m < 2)
      - m_incl_self       = neighbor count INCLUDING self used in the average

    Keeps existing columns so downstream plotting remains compatible.
    """
    work = df.copy()
    if not {'vx','vy'}.issubset(work.columns):
        work = estimate_velocities(work)

    keys = snapshot_key_columns(work)
    id_cols = [c for c in ['robot_category','robot_id'] if c in work.columns]

    sum_rows: list = []
    dist_rows: list = []

    for key, snap in snapshot_groups(work):
        pts = snap[['x','y']].to_numpy()
        vel = snap[['vx','vy']].to_numpy()
        box = infer_box(snap)

        # raw local order + neighbor counts (excl. self)
        P_local, n_nbrs = vicsek_local_polar_order_snapshot(pts, vel, rc=rc, box=box)

        # derive m (incl. self) consistent with snapshot function
        speeds = np.linalg.norm(vel, axis=1)
        nonzero = speeds > 1e-12
        neighs = _kdtree_neighbors(pts, rc, box)
        m_incl_self = np.array([sum(nonzero[nbrs]) for nbrs in neighs], dtype=int)

        # Rayleigh small-sample baseline and Z
        # E0 = sqrt(pi)/(2*sqrt(m)) for m>=1, else 0
        with np.errstate(divide='ignore', invalid='ignore'):
            E0 = np.sqrt(np.pi) / (2.0 * np.sqrt(np.maximum(m_incl_self.astype(float), 1.0)))
        E0[m_incl_self < 1] = 0.0

        P_biasfree = np.maximum(0.0, P_local - E0)

        Rayleigh_Z = np.full_like(P_local, np.nan, dtype=float)
        mask_m2 = (m_incl_self >= 2) & np.isfinite(P_local)
        Rayleigh_Z[mask_m2] = 2.0 * m_incl_self[mask_m2] * (P_local[mask_m2] ** 2)

        # per-agent distribution rows (keep old cols, add new ones)
        base_key = list(key)
        for (p, pb, z, n, m), (_, agent_row) in zip(
            zip(P_local, P_biasfree, Rayleigh_Z, n_nbrs, m_incl_self),
            snap.iterrows()
        ):
            dist_rows.append(tuple(base_key + [
                *(agent_row[c] for c in id_cols),
                float(p), float(pb), float(z) if np.isfinite(z) else np.nan,
                int(n), int(m)
            ]))

        # per-snapshot summaries
        def _stats(vals: np.ndarray) -> dict:
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return dict(mean=np.nan, std=np.nan, median=np.nan, q25=np.nan, q75=np.nan)
            return dict(
                mean=float(np.mean(vals)),
                std=float(np.std(vals)),
                median=float(np.median(vals)),
                q25=float(np.quantile(vals, 0.25)),
                q75=float(np.quantile(vals, 0.75)),
            )

        # Original weighted mean for P_local (by m = n_nbrs+1), keep for compatibility
        finite = np.isfinite(P_local)
        vals_P = P_local[finite]
        w = (n_nbrs[finite] + 1).astype(float)
        if vals_P.size == 0:
            P_summary = dict(mean=np.nan, std=np.nan, median=np.nan, q25=np.nan, q75=np.nan)
        else:
            wsum = np.sum(w)
            wmean = float(np.sum(w * vals_P) / (wsum + 1e-12))
            P_summary = dict(
                mean=wmean,
                std=float(np.std(vals_P)),
                median=float(np.median(vals_P)),
                q25=float(np.quantile(vals_P, 0.25)),
                q75=float(np.quantile(vals_P, 0.75)),
            )

        PB_summary = _stats(P_biasfree)
        Z_summary  = _stats(Rayleigh_Z[np.isfinite(Rayleigh_Z)])

        frac_isolated = 1.0 - (np.count_nonzero(np.isfinite(P_local)) / max(len(P_local), 1))
        mean_n = float(np.mean(n_nbrs)) if len(n_nbrs) else np.nan
        mean_m = float(np.mean(m_incl_self)) if len(m_incl_self) else np.nan

        sum_rows.append(tuple(list(key) + [
            P_summary['mean'], P_summary['std'], P_summary['median'], P_summary['q25'], P_summary['q75'],
            PB_summary['mean'], PB_summary['std'], PB_summary['median'], PB_summary['q25'], PB_summary['q75'],
            Z_summary['mean'], frac_isolated, mean_n, mean_m
        ]))

    summary_cols = keys + [
        'P_local_mean','P_local_std','P_local_median','P_local_q25','P_local_q75',
        'P_local_biasfree_mean','P_local_biasfree_std','P_local_biasfree_median', 'P_local_biasfree_q25','P_local_biasfree_q75',
        'Rayleigh_Z_mean', 'frac_isolated','mean_n_neighbors','mean_m_incl_self'
    ]
    dist_cols    = keys + id_cols + [
        'P_local','P_local_biasfree','Rayleigh_Z','n_neighbors','m_incl_self'
    ]

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


def grid_velocity_field_snapshot(
    pts: np.ndarray,
    vel: np.ndarray,
    box: Box,
    nx: int = 64,
    ny: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin particle velocities onto a regular grid (PIV-like).
    Returns:
      X, Y : (ny,nx) meshgrid of cell centers
      U, V : (ny,nx) mean velocity per cell (np.nan where empty)
      C    : (ny,nx) counts per cell
    """
    # Grid edges and centers
    x_edges = np.linspace(box.x0, box.x1, nx+1)
    y_edges = np.linspace(box.y0, box.y1, ny+1)
    x_cent = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_cent = 0.5 * (y_edges[:-1] + y_edges[1:])
    X, Y = np.meshgrid(x_cent, y_cent)

    # Digitize points
    ix = np.digitize(pts[:,0], x_edges) - 1
    iy = np.digitize(pts[:,1], y_edges) - 1
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

    U = np.full((ny, nx), np.nan, dtype=float)
    V = np.full((ny, nx), np.nan, dtype=float)
    C = np.zeros((ny, nx), dtype=int)

    if np.any(valid):
        # accumulate sums and counts
        sumU = np.zeros((ny, nx), dtype=float)
        sumV = np.zeros((ny, nx), dtype=float)
        for i in np.where(valid)[0]:
            C[iy[i], ix[i]] += 1
            sumU[iy[i], ix[i]] += vel[i,0]
            sumV[iy[i], ix[i]] += vel[i,1]
        nz = C > 0
        U[nz] = sumU[nz] / C[nz]
        V[nz] = sumV[nz] / C[nz]

    U = nan_gaussian_smooth(U, sigma=1.2)
    V = nan_gaussian_smooth(V, sigma=1.2)

    return X, Y, U, V, C


def vorticity_enstrophy_snapshot(
    U: np.ndarray, V: np.ndarray, box: Box
) -> Tuple[np.ndarray, float, float]:
    """
    Compute scalar vorticity ω = ∂v/∂x - ∂u/∂y on the grid and enstrophy ⟨ω^2⟩.
    Returns:
      omega_grid, enstrophy (mean ω^2 over finite cells), mean_abs_omega
    """
    ny, nx = U.shape
    if nx < 2 or ny < 2:
        return np.full_like(U, np.nan), np.nan, np.nan

    dx = (box.Lx) / nx
    dy = (box.Ly) / ny

    # np.gradient returns derivatives along axis with given spacing
    dVdx = np.gradient(V, dx, axis=1)
    dUdy = np.gradient(U, dy, axis=0)

    omega = dVdx - dUdy  # shape (ny,nx)
    finite = np.isfinite(omega)
    if not np.any(finite):
        return omega, np.nan, np.nan
    enst = float(np.nanmean(omega[finite]**2))
    mean_abs = float(np.nanmean(np.abs(omega[finite])))
    return omega, enst, mean_abs


def swirling_strength_okubo_weiss_snapshot(
    U: np.ndarray, V: np.ndarray, box: Box
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute swirling strength λ_ci and Okubo–Weiss Q on the grid.
    λ_ci = sqrt(max(0, -Δ)) / 2 where Δ is the eigen discriminant of ∇u.
    Q = s_n^2 + s_s^2 - ω^2 (vortex-dominated regions have Q < 0).
    Returns:
      lambda_ci_grid, Q_grid, mean_lambda_ci, frac_Q_negative
    """
    ny, nx = U.shape
    if nx < 2 or ny < 2:
        shp = U.shape
        return (np.full(shp, np.nan), np.full(shp, np.nan), np.nan, np.nan)

    dx = (box.Lx) / nx
    dy = (box.Ly) / ny

    dudx = np.gradient(U, dx, axis=1)
    dudy = np.gradient(U, dy, axis=0)
    dvdx = np.gradient(V, dx, axis=1)
    dvdy = np.gradient(V, dy, axis=0)

    # Vorticity and strain (2D)
    omega = dvdx - dudy
    s_n = dudx - dvdy            # normal strain
    s_s = dudy + dvdx            # shear strain

    # Okubo–Weiss parameter
    Q = s_n**2 + s_s**2 - omega**2

    # Swirling strength λ_ci = Im(λ) for complex eigenvalues of J
    # discriminant Δ of 2x2 Jacobian eigenvalues:
    tr = dudx + dvdy
    det = dudx*dvdy - dudy*dvdx
    # eigenvalues: (tr ± sqrt(tr^2 - 4 det))/2
    disc = tr**2 - 4.0*det
    lambda_ci = np.sqrt(np.maximum(0.0, -disc)) / 2.0  # non-zero where disc < 0

    finite_ci = np.isfinite(lambda_ci)
    mean_ci = float(np.nanmean(lambda_ci[finite_ci])) if np.any(finite_ci) else np.nan
    frac_Q_neg = float(np.nanmean(Q < 0)) if np.isfinite(Q).any() else np.nan

    return lambda_ci, Q, mean_ci, frac_Q_neg


def milling_order_snapshot(
    pts: np.ndarray, vel: np.ndarray, center: Optional[Tuple[float,float]] = None
) -> Dict[str, float]:
    """
    Global milling/vortex order parameters based on angular momentum and tangential alignment.

    Returns dict with:
      Lz          : mean angular momentum z-component per agent
      Lz_norm     : |Lz| normalized by (mean r * mean speed) for scale invariance
      tau_mean    : ⟨t̂·v̂⟩ signed tangential alignment (CCW positive)
      tau_abs     : ⟨|t̂·v̂|⟩ milling index (near 1 if all move tangentially)
      radial_mean : ⟨r̂·v̂⟩ (radial motion; ~0 in a mill)
    """
    # center of rotation: center-of-mass by default
    if center is None:
        c = pts.mean(axis=0)
    else:
        c = np.array(center, dtype=float)

    r = pts - c[None, :]
    v = vel.copy()

    r_norm = np.linalg.norm(r, axis=1)
    v_norm = np.linalg.norm(v, axis=1)
    vhat = np.zeros_like(v)
    nzv = v_norm > 1e-12
    vhat[nzv] = v[nzv] / v_norm[nzv, None]

    # tangential and radial unit vectors
    rhat = np.zeros_like(r)
    nzr = r_norm > 1e-12
    rhat[nzr] = r[nzr] / r_norm[nzr, None]
    that = np.zeros_like(r)
    # ẑ × r̂ = (-r_y, r_x) for 2D tangential (CCW)
    that[nzr, 0] = -rhat[nzr, 1]
    that[nzr, 1] =  rhat[nzr, 0]

    # angular momentum per agent (scalar z-component): Lz_i = (r_i × v_i)_z = x_i*v_y - y_i*v_x
    Lz_i = r[:,0]*v[:,1] - r[:,1]*v[:,0]
    Lz = float(np.nanmean(Lz_i))

    # normalized by typical scales
    mean_r = float(np.nanmean(r_norm))
    mean_v = float(np.nanmean(v_norm))
    Lz_norm = Lz / ((mean_r*mean_v) + 1e-12)

    # tangential and radial alignment (using unit velocity)
    tau = np.sum(that * vhat, axis=1)       # t̂·v̂
    rho = np.sum(rhat * vhat, axis=1)       # r̂·v̂
    finite = nzv & nzr
    tau_mean = float(np.nanmean(tau[finite])) if np.any(finite) else np.nan
    tau_abs  = float(np.nanmean(np.abs(tau[finite]))) if np.any(finite) else np.nan
    radial_mean = float(np.nanmean(rho[finite])) if np.any(finite) else np.nan

    return dict(Lz=Lz, Lz_norm=Lz_norm, tau_mean=tau_mean, tau_abs=tau_abs, radial_mean=radial_mean)


def vortex_field_metrics(
    df: pd.DataFrame, nx: int = 64, ny: int = 64
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds grid velocity fields per snapshot, then computes:
      - enstrophy: mean(ω^2)
      - mean_abs_vorticity: mean(|ω|)
      - mean_lambda_ci: mean swirling strength
      - frac_Q_negative: fraction of grid cells with Q < 0 (vorticity dominated)
    Returns:
      summary_df (per snapshot)
      dist_df    (optional per-cell distributions; here we store ω and λ_ci)
    """
    work = df.copy()
    if not {'vx','vy'}.issubset(work.columns):
        work = estimate_velocities(work)

    keys = snapshot_key_columns(work)
    sum_rows = []
    dist_rows = []
    for key, snap in snapshot_groups(work):
        box = infer_box(snap)
        pts = snap[['x','y']].to_numpy()
        vel = snap[['vx','vy']].to_numpy()
        _, _, U, V, C = grid_velocity_field_snapshot(pts, vel, box, nx=nx, ny=ny)

        omega, enst, mean_abs = vorticity_enstrophy_snapshot(U, V, box)
        lam_ci, Q, mean_ci, frac_Q_neg = swirling_strength_okubo_weiss_snapshot(U, V, box)

        sum_rows.append(tuple(list(key) + [enst, mean_abs, mean_ci, frac_Q_neg]))

        # optional: store per-cell values for diagnostics
        for j in range(omega.shape[0]):
            for i in range(omega.shape[1]):
                if np.isfinite(omega[j,i]) or np.isfinite(lam_ci[j,i]):
                    dist_rows.append(tuple(list(key) + [float(omega[j,i]), float(lam_ci[j,i]), int(C[j,i])]))

    summary_cols = keys + ['enstrophy', 'mean_abs_vorticity', 'mean_lambda_ci', 'frac_Q_negative']
    dist_cols    = keys + ['omega', 'lambda_ci', 'cell_count']
    return pd.DataFrame(sum_rows, columns=summary_cols), pd.DataFrame(dist_rows, columns=dist_cols)


def milling_metrics(
    df: pd.DataFrame, *, center_mode: str = "cm", arena_center: Optional[Tuple[float,float]] = None
) -> pd.DataFrame:
    """
    Global, center-based milling/vortex order parameters per snapshot.
    center_mode: "cm" (center-of-mass) or "arena"
    """
    work = df.copy()
    if not {'vx','vy'}.issubset(work.columns):
        work = estimate_velocities(work)

    keys = snapshot_key_columns(work)
    rows = []
    for key, snap in snapshot_groups(work):
        pts = snap[['x','y']].to_numpy()
        vel = snap[['vx','vy']].to_numpy()
        if center_mode == "arena" and arena_center is not None:
            center = arena_center
        else:
            center = None
        stats = milling_order_snapshot(pts, vel, center=center)
        rows.append(tuple(list(key) + [stats['Lz'], stats['Lz_norm'], stats['tau_mean'], stats['tau_abs'], stats['radial_mean']]))
    cols = keys + ['Lz', 'Lz_norm', 'tau_mean', 'tau_abs', 'radial_mean']
    return pd.DataFrame(rows, columns=cols)


def alignment_correlation_snapshot(
    pts: np.ndarray,
    vel: np.ndarray,
    *,
    r_max: float,
    dr: float,
    box: Optional[Box] = None,
    exclude_zero_speed: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Alignment correlation C(r) between unit velocities as a function of pair distance.
    For each pair (i,j) with separation r_ij in bin b:
        C(r_b) = ⟨ v̂_i · v̂_j ⟩_{|r_ij| in bin b}
    where v̂ = v / ||v|| (pairs with zero speed can be excluded).

    Returns:
      r_centers : (B,) bin centers
      C_r       : (B,) correlation values (NaN if no pairs in bin)
      counts    : (B,) number of pairs in each bin

    Notes:
      - Non-periodic distances (consistent with this script).
      - We do NOT include self pairs; only i<j unique pairs are used.
      - Expected baseline for random headings is ~0.
    """
    if box is None:
        box = infer_box(pd.DataFrame({'x': pts[:,0], 'y': pts[:,1]}))

    speed = np.linalg.norm(vel, axis=1)
    nonzero = (speed > 1e-12) if exclude_zero_speed else np.ones(len(pts), dtype=bool)

    vhat = np.zeros_like(vel, dtype=float)
    ok = nonzero
    vhat[ok] = vel[ok] / speed[ok, None]

    if cKDTree is None:
        raise ImportError("scipy is required for C(r) (scipy.spatial.cKDTree).")
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r_max)
    if not pairs:
        # set up bins anyway
        bins = np.arange(0.0, r_max + dr, dr)
        r = 0.5 * (bins[:-1] + bins[1:])
        return r, np.full_like(r, np.nan, dtype=float), np.zeros_like(r, dtype=int)

    # distances and dot-products for valid-speed pairs only
    pairs = np.array(list(pairs), dtype=int)
    i = pairs[:,0]; j = pairs[:,1]
    # keep pairs where both speeds are valid if exclude_zero_speed
    if exclude_zero_speed:
        mask = ok[i] & ok[j]
        i, j = i[mask], j[mask]
        if i.size == 0:
            bins = np.arange(0.0, r_max + dr, dr)
            r = 0.5 * (bins[:-1] + bins[1:])
            return r, np.full_like(r, np.nan, dtype=float), np.zeros_like(r, dtype=int)

    dij = np.linalg.norm(pts[i] - pts[j], axis=1)
    dot = np.sum(vhat[i] * vhat[j], axis=1)

    # bin by distance
    bins = np.arange(0.0, r_max + dr, dr)
    r = 0.5 * (bins[:-1] + bins[1:])
    which = np.digitize(dij, bins) - 1
    valid = (which >= 0) & (which < r.size)

    C = np.full(r.shape, np.nan, dtype=float)
    cnt = np.zeros(r.shape, dtype=int)

    if np.any(valid):
        # accumulate sum and counts per bin
        for b in np.unique(which[valid]):
            sel = (which == b) & valid
            cnt[b] = int(np.sum(sel))
            if cnt[b] > 0:
                C[b] = float(np.mean(dot[sel]))

    return r, C, cnt


def alignment_correlation(
    df: pd.DataFrame,
    *,
    r_max: float,
    dr: float,
) -> pd.DataFrame:
    """
    Compute C(r) per snapshot and return a long DataFrame with columns:
      keys + ['r','C_r','pair_count'].

    We keep one row per (snapshot, r-bin). Aggregation (mean ± std across time)
    is handled in the plotting function just like g(r).
    """
    work = df.copy()
    if not {'vx','vy'}.issubset(work.columns):
        work = estimate_velocities(work)

    keys = snapshot_key_columns(work)
    rows = []
    for key, snap in snapshot_groups(work):
        pts = snap[['x','y']].to_numpy()
        vel = snap[['vx','vy']].to_numpy()
        box = infer_box(snap)
        r, C, cnt = alignment_correlation_snapshot(pts, vel, r_max=r_max, dr=dr, box=box)
        for ri, Ci, ni in zip(r, C, cnt):
            rows.append((*key, float(ri), float(Ci) if np.isfinite(Ci) else np.nan, int(ni)))
    cols = keys + ['r','C_r','pair_count']
    return pd.DataFrame(rows, columns=cols)


def alignment_correlation_length(
    Cr_df: pd.DataFrame,
    *,
    dr: float,
    positive_only: bool = True
) -> pd.DataFrame:
    """
    Derive a per-snapshot scalar correlation length ξ from C(r):
       ξ = ∑_r max(C(r),0)*dr        (default; integral of positive lobe)
    Other definitions (e.g., first zero-crossing) could be added later.

    Returns: keys + ['xi_align']
    """
    if Cr_df.empty:
        keys = [c for c in ['arena_file','run','time'] if c in Cr_df.columns]
        return pd.DataFrame(columns=keys + ['xi_align'])

    keys = snapshot_key_columns(Cr_df)
    rows = []
    for key, snap in Cr_df.groupby(keys, sort=False):
        r = snap['r'].to_numpy()
        c = snap['C_r'].to_numpy()
        m = np.isfinite(r) & np.isfinite(c)
        if not np.any(m):
            xi = np.nan
        else:
            cc = c[m]
            if positive_only:
                cc = np.maximum(0.0, cc)
            # simple Riemann sum (uniform bins)
            xi = float(np.sum(cc) * dr)
        # normalize by C(0)? We skip; C(0) is not defined (no self-pairs).
        rows.append((*key, xi))
    return pd.DataFrame(rows, columns=keys + ['xi_align'])



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
    vort_nx: int = 64,
    vort_ny: int = 64,
    milling_center: str = "cm",
    aligncorr_rmax: float = 200.0,
    aligncorr_dr: float = 2.0,
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    out['polar_order']    = vicsek_polar_order(df)
    out['local_polar'], out['local_polar_dist'] = vicsek_local_polar_order(df, rc=rc_bond)
    out['local_pair_align'], out['local_pair_align_dist'] = local_pairwise_alignment(df, rc=rc_bond)
    out['align_corr']     = alignment_correlation(df, r_max=aligncorr_rmax, dr=aligncorr_dr)
    out['align_corr_len'] = alignment_correlation_length(out['align_corr'], dr=aligncorr_dr)
    out['psi_k']          = bond_orientational_order(df, rc=rc_bond, k=psi_k)
    out['largest_cluster']= largest_cluster_fraction(df, rc=rc_cluster)
    out['g_r']            = radial_distribution_function(df, r_max=gr_rmax, dr=gr_dr)
    out['S_k']            = structure_factor(df, grid_nx=S_grid_nx, grid_ny=S_grid_ny)
    out['number_fluct']   = number_fluctuation_exponent(df)
    if v2_ref is not None:
        out['Teff_ratio'] = effective_temperature_ratio(df, v2_ref=v2_ref)
    out['vortex_summary'], out['vortex_field_dist'] = vortex_field_metrics(df, nx=vort_nx, ny=vort_ny)
    out['milling'] = milling_metrics(df, center_mode=milling_center)
    return out


def _fmt(x, p=3):
    return "n/a" if (x is None or not np.isfinite(x)) else f"{x:.{p}f}"


def write_global_summary(output_dir: str, results: Dict[str, pd.DataFrame]) -> None:
    lines = ["# Global structure metrics — summary\n"]
    def add_line(label, value): lines.append(f"- {label}: {value}")

    P = results.get('polar_order', pd.DataFrame())
    if not P.empty:
        add_line("Global Polar order ⟨P⟩ over time", _fmt(P['P_polar'].mean()))
    LP = results.get('local_polar', pd.DataFrame())
    if not LP.empty:
        add_line("Local polar order ⟨P_local⟩ over time, with Rayleigh small-sample correction", _fmt(LP['P_local_biasfree_mean'].mean()))
    LPA = results.get('local_pair_align', pd.DataFrame())
    if not LPA.empty:
        add_line("Local pairwise alignment ⟨A⟩ over time (edge-weighted)", _fmt(LPA['A_edge_mean'].mean()))
    AC = results.get('align_corr_len', pd.DataFrame())
    if not AC.empty:
        add_line("Alignment correlation length ⟨ξ⟩", _fmt(AC['xi_align'].mean()))
    psi_any = [k for k in results if k == 'psi_k']
    if psi_any:
        psi_df = results['psi_k']
        psi_col = [c for c in psi_df.columns if c.startswith('psi_')][0] if not psi_df.empty else None
        add_line(f"Bond-orientational ⟨|{psi_col}|⟩", _fmt(psi_df[psi_col].mean() if psi_col else np.nan))
    if not LP.empty:
        add_line("Mean fraction isolated (no neighbors)", _fmt(LP['frac_isolated'].mean()))
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

    VOR = results.get('vortex_summary', pd.DataFrame())
    if not VOR.empty:
        add_line("Mean enstrophy ⟨ω²⟩", _fmt(VOR['enstrophy'].mean()))
        add_line("Mean |ω|", _fmt(VOR['mean_abs_vorticity'].mean()))
        add_line("Mean swirling strength ⟨λ_ci⟩", _fmt(VOR['mean_lambda_ci'].mean()))
        add_line("Okubo–Weiss vortex area ⟨Frac(Q<0)⟩", _fmt(VOR['frac_Q_negative'].mean()))

    MIL = results.get('milling', pd.DataFrame())
    if not MIL.empty:
        add_line("Milling index ⟨|t̂·v̂|⟩", _fmt(MIL['tau_abs'].mean()))
        add_line("Signed tangential ⟨t̂·v̂⟩", _fmt(MIL['tau_mean'].mean()))
        add_line("Normalized angular momentum ⟨Lz_norm⟩", _fmt(MIL['Lz_norm'].mean()))


    text = "\n".join(lines) + "\n"
    with open(os.path.join(output_dir, "SUMMARY_global.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


def write_metric_descriptions(output_dir: str, results: Dict[str, pd.DataFrame]) -> None:
    """
    Create a long, human-readable description of each summarized metric:
    - What it represents + informal definition
    - Domain and typical ranges
    - Useful literature references (short canonical pointers)
    - Filenames of associated plots produced by this script

    The text is printed to stdout and saved to SUMMARY_metric_details.txt.
    """
    lines: list[str] = []
    add = lines.append

    add("# Global structure metrics — detailed descriptions\n")

    # Helper: resolve psi_k column name if present
    psi_col = None
    if 'psi_k' in results and not results['psi_k'].empty:
        psi_col = [c for c in results['psi_k'].columns if c.startswith('psi_')][0]

    # ---- Polar order P ------------------------------------------------------
    if 'polar_order' in results:
        add("## Vicsek polar order ⟨P⟩")
        add("**Meaning.** Global alignment of headings: P = ||⟨v̂_i⟩|| (1 = everyone aligned; 0 ~ random).")
        add("**Domain.** 0 ≤ P ≤ 1.")
        add("**Typical values.** Disordered ~0–0.2; weak order 0.2–0.6; strong flocking >0.6 (ballpark).")
        add("**Refs.** Vicsek et al., *Phys. Rev. Lett.* 75, 1226 (1995). Toner & Tu, *Phys. Rev. Lett.* 75, 4326 (1995).")
        add(f"**Plots.** `polar_order_time.pdf`")
        add("")

    # ---- Local polar order P_local (bias-free mean) ------------------------
    if 'local_polar' in results:
        add("## Local Vicsek order ⟨P_local⟩ (bias-corrected)")
        add("**Meaning.** Local alignment inside a neighborhood of radius rc. We report a Rayleigh small-sample "
            "bias-corrected mean of per-agent local resultants.")
        add("**Domain.** 0 ≤ P_local ≤ 1 (mean and distribution).")
        add("**Typical values.** Disordered ~0–0.3; mesoscopic order 0.3–0.7; strong crystalline/laned >0.7.")
        add("**Refs.** Rayleigh resultant length bias: Berens, *CircStat* notes; Vicsek et al. 1995 (local variants).")
        add("**Plots.** `local_polar_time.pdf`, `local_polar_hist.pdf`")
        add("")

    # ---- Local pairwise alignment A ----------------------------------------
    if 'local_pair_align' in results:
        add("## Local pairwise alignment ⟨A⟩ (edge-weighted)")
        add("**Meaning.** Average cos(θ_i−θ_j) over neighbor pairs; zero for random headings, +1 for perfect local "
            "alignment, negative when locally anti-aligned.")
        add("**Domain.** −1 ≤ A ≤ 1 (edge-weighted mean reported).")
        add("**Typical values.** Random ~0; locally ordered 0.3–0.9; anti-aligned patterns <0.")
        add("**Refs.** Standard XY-like alignment statistics; used widely in active matter + flocking.")
        add("**Plots.** `local_pair_align_time.pdf`, `local_pair_align_hist.pdf`")
        add("")

    # ---- Alignment correlation C(r) and ξ ----------------------------------
    if 'align_corr' in results or 'align_corr_len' in results:
        add("## Velocity alignment correlation C(r) and correlation length ξ")
        add("**Meaning.** C(r) = ⟨v̂_i·v̂_j⟩ for pairs at separation r. Correlation length ξ is the integral of the "
            "positive part of C(r) (Riemann sum), giving a scalar range of orientational order.")
        add("**Domain.** −1 ≤ C(r) ≤ 1; ξ ≥ 0 (units of length, here mm).")
        add("**Typical values.** Disordered: short-range C(r)≈0; collectively moving phases show a positive lobe and "
            "finite ξ; near flocking/milling, ξ grows with system coherence.")
        add("**Refs.** E.g., Cavagna et al., *PNAS* 107, 11865 (2010) on correlation in flocks; general spin/active "
            "matter correlation formalism.")
        add("**Plots.** `align_corr_Cr.pdf`, `align_corr_xi_time.pdf`")
        add("")

    # ---- Bond-orientational order psi_k ------------------------------------
    if psi_col is not None:
        add(f"## Bond-orientational order ⟨|{psi_col}|⟩")
        add("**Meaning.** k-fold bond orientation (here typically k=6): averages e^{ikθ} over neighbor bonds; measures "
            "local hexatic/positional order.")
        add("**Domain.** 0 ≤ |ψ_k| ≤ 1.")
        add("**Typical values.** Liquids ~0–0.3; hexatic/locally crystalline regions 0.3–0.8+; perfect lattice → 1.")
        add("**Refs.** Nelson & Halperin, *Phys. Rev. B* 19, 2457 (1979); Steinhardt et al., *Phys. Rev. B* 28, 784 (1983).")
        add(f"**Plots.** `{psi_col}_time.pdf`")
        add("")

    # ---- Largest cluster fraction ------------------------------------------
    if 'largest_cluster' in results:
        add("## Largest cluster fraction ⟨f_max⟩")
        add("**Meaning.** Size of the largest connected component (rc-graph) divided by N.")
        add("**Domain.** 0 < f_max ≤ 1.")
        add("**Typical values.** Gas-like: 0.1–0.4; percolated/single-cluster: 0.6–1.0 (depends on density & rc).")
        add("**Refs.** Standard connectivity/percolation diagnostics for proximity graphs.")
        add("**Plots.** `largest_cluster_time.pdf`")
        add("")

    # ---- g(r) ---------------------------------------------------------------
    if 'g_r' in results:
        add("## Radial distribution function g(r)")
        add("**Meaning.** Pair-density relative to a Poisson fluid. Peaks mark shell distances; g(r)→1 at large r.")
        add("**Domain.** g(r) ≥ 0 (theoretically).")
        add("**Typical values.** Liquids: pronounced first peak near typical spacing; gases: flatter near 1.")
        add("**Refs.** Hansen & McDonald, *Theory of Simple Liquids*.")
        add("**Plots.** `g_r.pdf`")
        add("")

    # ---- S(k) ---------------------------------------------------------------
    if 'S_k' in results:
        add("## Static structure factor S(k)")
        add("**Meaning.** Power spectrum of density fluctuations; peaks signal dominant length-scales λ≈2π/k.")
        add("**Domain.** S(k) ≥ 0.")
        add("**Typical values.** Disordered fluids: broad low-k; emerging patterns/clusters: clear peak at k*.")
        add("**Refs.** Chaikin & Lubensky, *Principles of Condensed Matter Physics*.")
        add("**Plots.** `S_k.pdf`")
        add("")

    # ---- Number fluctuations alpha -----------------------------------------
    if 'number_fluct' in results:
        add("## Number-fluctuation exponent ⟨α⟩")
        add("**Meaning.** From σ_N ∝ ⟨N⟩^{α}: α=0.5 is Poisson; α>0.5 indicates giant number fluctuations (active matter hallmark).")
        add("**Domain.** Typically 0.3–1.0 in practice.")
        add("**Typical values.** Passive-like ~0.5; active/collective >0.5; strong clustering can approach ~0.8–1.0.")
        add("**Refs.** Ramaswamy et al., *EPL* 62, 196 (2003); Narayan et al., *Science* 317, 105 (2007).")
        add("**Plots.** `number_fluct_alpha_time.pdf`")
        add("")

    # ---- Teff/Tref ----------------------------------------------------------
    if 'Teff_ratio' in results:
        add("## Effective temperature ratio ⟨T_eff/T_ref⟩")
        add("**Meaning.** Proxy from mean squared speed ⟨v²⟩ relative to a reference; compares agitation levels.")
        add("**Domain.** > 0 (dimensionless).")
        add("**Typical values.** Depends on actuation; often O(0.5–5) in lab/sim settings.")
        add("**Refs.** Effective temperature notions in athermal active systems (various reviews).")
        add("**Plots.** `teff_ratio_time.pdf`")
        add("")

    # ---- Vorticity / Enstrophy / Swirling / Okubo–Weiss --------------------
    if 'vortex_summary' in results:
        add("## Vorticity & enstrophy, swirling strength λ_ci, Okubo–Weiss Q")
        add("**Meaning.** From gridded velocity fields: ω = ∂v/∂x − ∂u/∂y; enstrophy ⟨ω²⟩ measures rotational intensity; "
            "λ_ci measures local swirling (imaginary part of Jacobian eigenvalues); Okubo–Weiss Q<0 marks vortex-dominated regions.")
        add("**Domain.** ⟨ω²⟩ ≥ 0; ⟨|ω|⟩ ≥ 0; ⟨λ_ci⟩ ≥ 0; 0 ≤ Frac(Q<0) ≤ 1. Units of ω, λ_ci depend on space/time units.")
        add("**Typical values.** Depend strongly on speed scales and grid; compare runs relatively.")
        add("**Refs.** Zhou et al., *J. Fluid Mech.* 387, 353 (1999) (swirling strength). Okubo (1970), Weiss (1991) (Q).")
        add("**Plots.** `enstrophy_time.pdf`, `vorticity_abs_time.pdf`, `swirling_strength_time.pdf`, "
            "`okubo_weiss_frac_time.pdf`, `omega_hist.pdf`, `lambda_ci_hist.pdf`, plus `vorticity_heatmap_*.pdf`")
        add("")

    # ---- Milling metrics ----------------------------------------------------
    if 'milling' in results:
        add("## Milling / vortex order parameters (τ, |τ|, Lz_norm)")
        add("**Meaning.** Tangential alignment τ = ⟨t̂·v̂⟩ (signed), milling index |τ|, and normalized angular momentum Lz_norm; "
            "capture global rotation/milling around a center.")
        add("**Domain.** −1 ≤ τ ≤ 1; 0 ≤ |τ| ≤ 1; Lz_norm typically in [−1,1].")
        add("**Typical values.** Non-milling: |τ| ≈ 0–0.3; clear mills: |τ| ≳ 0.6 with consistent τ sign and sizable Lz_norm.")
        add("**Refs.** Vortex/milling order parameters used in swarm & collective motion literature (e.g., Couzin et al.).")
        add("**Plots.** `milling_index_time.pdf`, `milling_signed_time.pdf`, `angular_momentum_time.pdf`")
        add("")

    text = "\n".join(lines) + "\n"
    out_path = os.path.join(output_dir, "SUMMARY_metric_details.txt")
    with open(out_path, "w", encoding="utf-8") as f:
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

def plot_Cr(Cr_df: pd.DataFrame, out_pdf: str) -> None:
    fig = plt.figure(figsize=(6.2, 4.2))
    ax = fig.add_subplot(111)
    if Cr_df.empty:
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
    else:
        agg = Cr_df.groupby('r', as_index=False).agg(C_mean=('C_r','mean'), C_std=('C_r','std'))
        ax.plot(agg['r'], agg['C_mean'], linewidth=2.0)
        if np.isfinite(agg['C_std']).any():
            ax.fill_between(agg['r'], agg['C_mean']-agg['C_std'], agg['C_mean']+agg['C_std'], alpha=0.2, linewidth=0)
        ax.set_xlabel("r [mm]")
        ax.set_ylabel("C(r) = ⟨v̂ᵢ·v̂ⱼ⟩")
        ax.set_title("Velocity alignment correlation C(r)")
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

def plot_vorticity_heatmap(omega: np.ndarray, out_pdf: str, title: str = "Vorticity field"):
    fig = plt.figure(figsize=(6.0, 5.2))
    ax = fig.add_subplot(111)
    im = ax.imshow(omega, origin='lower', aspect='auto')
    fig.colorbar(im, ax=ax, label='ω')
    ax.set_title(title)
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
    p.add_argument("--vort-nx", type=int, default=64, help="Grid Nx for vorticity/swirling metrics")
    p.add_argument("--vort-ny", type=int, default=64, help="Grid Ny for vorticity/swirling metrics")
    p.add_argument("--milling-center", type=str, default="cm", choices=["cm","arena"],
                   help="Center for milling order parameters: center-of-mass or arena center")
    p.add_argument("--aligncorr-rmax", type=float, default=200.0,
                   help="C(r) max radius in mm (default: 200)")
    p.add_argument("--aligncorr-dr", type=float, default=2.0,
                   help="C(r) bin width in mm (default: 2)")
    p.add_argument('-N', "--no-extended-infos", action="store_true", help="Skip Extended infos on metrics, just print a summary of metric values")

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
        vort_nx=args.vort_nx,
        vort_ny=args.vort_ny,
        milling_center=args.milling_center,
        aligncorr_rmax=args.aligncorr_rmax,
        aligncorr_dr=args.aligncorr_dr,
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
            'P_local_biasfree_mean', 'P_local_biasfree_std',
            "Local polar order ⟨P_local⟩",
            "Local Vicsek order (mean ± std) vs time",
            os.path.join(args.output_dir, "local_polar_time.pdf")
        )
    if 'local_polar_dist' in results:
        # pooled histogram across all times/runs/agents (quick diagnostic)
        plot_histogram(
            results['local_polar_dist'],
            'P_local_biasfree', 40,
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
    if 'align_corr' in results:
        plot_Cr(results['align_corr'], os.path.join(args.output_dir, "align_corr_Cr.pdf"))

    if 'align_corr_len' in results:
        plot_time_series(
            results['align_corr_len'], 'xi_align', "Correlation length ξ",
            "Velocity alignment correlation length vs time",
            os.path.join(args.output_dir, "align_corr_xi_time.pdf")
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

    if 'vortex_summary' in results:
        plot_time_series(
            results['vortex_summary'], 'enstrophy', "Enstrophy ⟨ω²⟩",
            "Enstrophy vs time",
            os.path.join(args.output_dir, "enstrophy_time.pdf")
        )
        plot_time_series(
            results['vortex_summary'], 'mean_abs_vorticity', "Mean |ω|",
            "Mean absolute vorticity vs time",
            os.path.join(args.output_dir, "vorticity_abs_time.pdf")
        )
        plot_time_series(
            results['vortex_summary'], 'mean_lambda_ci', "⟨λ_ci⟩",
            "Swirling strength (mean λ_ci) vs time",
            os.path.join(args.output_dir, "swirling_strength_time.pdf")
        )
        plot_time_series(
            results['vortex_summary'], 'frac_Q_negative', "Frac(Q<0)",
            "Okubo–Weiss vortex area fraction vs time",
            os.path.join(args.output_dir, "okubo_weiss_frac_time.pdf")
        )

    if 'vortex_field_dist' in results and not results['vortex_field_dist'].empty:
        plot_histogram(
            results['vortex_field_dist'], 'omega', 60,
            "Distribution of grid vorticity ω",
            os.path.join(args.output_dir, "omega_hist.pdf")
        )
        plot_histogram(
            results['vortex_field_dist'], 'lambda_ci', 60,
            "Distribution of swirling strength λ_ci",
            os.path.join(args.output_dir, "lambda_ci_hist.pdf")
        )

    if 'milling' in results:
        plot_time_series(
            results['milling'], 'tau_abs', "Milling index ⟨|t̂·v̂|⟩",
            "Global milling index vs time",
            os.path.join(args.output_dir, "milling_index_time.pdf")
        )
        plot_time_series(
            results['milling'], 'tau_mean', "Signed tangential ⟨t̂·v̂⟩",
            "Signed tangential alignment vs time",
            os.path.join(args.output_dir, "milling_signed_time.pdf")
        )
        plot_time_series(
            results['milling'], 'Lz_norm', "Normalized Lz",
            "Normalized angular momentum vs time",
            os.path.join(args.output_dir, "angular_momentum_time.pdf")
        )

    # --- Vorticity heatmap (robust selection) ---
    if 'vortex_summary' in results and not results['vortex_summary'].empty:
        vor = results['vortex_summary']
        key_cols = [c for c in vor.columns
                    if c not in ('enstrophy','mean_abs_vorticity','mean_lambda_ci','frac_Q_negative')]

        # 1) Prefer snapshot with max finite enstrophy
        cand = vor[np.isfinite(vor.get('enstrophy'))]
        metric_used = 'enstrophy'

        # 2) Fallbacks if needed
        if cand.empty and 'mean_lambda_ci' in vor:
            cand = vor[np.isfinite(vor['mean_lambda_ci'])]
            metric_used = 'mean_lambda_ci'
        if cand.empty and 'mean_abs_vorticity' in vor:
            cand = vor[np.isfinite(vor['mean_abs_vorticity'])]
            metric_used = 'mean_abs_vorticity'

        if cand.empty:
            print("[vorticity heatmap] No finite vorticity/swirling metrics; skipping heatmap.")
        else:
            imax = cand[metric_used].idxmax()
            key_vals = vor.loc[imax, key_cols].to_dict()

            # Isolate that snapshot in the original df
            snap = df.copy()
            for k, v in key_vals.items():
                snap = snap[snap[k] == v]

            if not {'vx','vy'}.issubset(snap.columns):
                snap = estimate_velocities(snap)  # uses your helper

            box = infer_box(snap)
            pts = snap[['x','y']].to_numpy()
            vel = snap[['vx','vy']].to_numpy()

            _, _, U, V, _ = grid_velocity_field_snapshot(
                pts, vel, box, nx=getattr(args, 'vort_nx', 64), ny=getattr(args, 'vort_ny', 64)
            )
            omega, _, _ = vorticity_enstrophy_snapshot(U, V, box)

            nice_keys = ", ".join(f"{k}={key_vals[k]}" for k in key_cols)
            out_pdf = os.path.join(args.output_dir, f"vorticity_heatmap_{metric_used}.pdf")
            plot_vorticity_heatmap(omega, out_pdf, title=f"Vorticity field ({metric_used}; {nice_keys})")


    # Text summary (printed + file)
    write_global_summary(args.output_dir, results)
    if not args.no_extended_infos:
        write_metric_descriptions(args.output_dir, results)

    # Pickle (all metrics together)
    with open(os.path.join(args.output_dir, "metrics.pkl"), "wb") as f:
        pickle.dump({"meta": meta, "results": results}, f)

    print(f"\nDone. Outputs in: {args.output_dir}\n")


if __name__ == "__main__":
    sys.exit(main())

# MODELINE "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
