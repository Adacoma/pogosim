#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classify Toner–Tu simulation phases from metrics.csv and explain why.

Inputs
------
A CSV produced by the stabilized simulator (frames + metrics), with columns like:
  frame_id,t,mean_speed,polarization,dP_dt,rho_mean,rho_std,rho_contrast,
  vort_rms,vort_kurt,ow_neg_frac,anisotropy_A,band_theta_deg,streamness_S,
  dom_kx,dom_ky,dom_k_mag,dom_wavelength,render_field

Usage
-----
  python tt_classify.py -i /path/to/metrics.csv
  python tt_classify.py -i metrics.csv --json out.json

Notes
-----
- We aggregate metrics over the **last 20%** of saved frames (min 5 frames).
- Thresholds are heuristic; see the RULES section below to tweak for your grids.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ------------------------------- Utils --------------------------------

def rolling_slope(t: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope dy/dt for the given points (guarded)."""
    if len(t) < 3:
        return 0.0
    t0 = t - t.mean()
    den = float((t0 * t0).sum()) + 1e-15
    return float((t0 * y).sum() / den)

def tail_slice(n: int, frac: float = 0.2, min_n: int = 5) -> slice:
    """Return a slice for the last 'frac' of samples, at least min_n."""
    k = max(min_n, int(np.ceil(frac * n)))
    k = min(k, n)
    return slice(n - k, n)

def get(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    """Column accessor with default if missing."""
    return df[col].to_numpy(dtype=float) if col in df.columns else np.full(len(df), default, dtype=float)

@dataclass
class Features:
    # Tail-averaged (over last 20% frames)
    P: float                 # polarization
    dP_dt: float             # trend of P (slope)
    Crho: float              # rho_contrast
    vort_rms: float
    vort_kurt: float
    ow_neg_frac: float
    A: float                 # anisotropy_A
    S: float                 # streamness_S
    k_dom: float             # dom_k_mag
    lambda_dom: float        # dom_wavelength
    field: str               # render_field in CSV (for context)
    # Whole-run stats (fallbacks)
    P0: float                # polarization at first saved frame
    Pend: float              # polarization at last saved frame

def compute_features(csv_path: str) -> Features:
    df = pd.read_csv(csv_path)
    if len(df) < 2:
        raise ValueError("metrics.csv has fewer than 2 rows; need at least initial and one saved frame.")

    sl = tail_slice(len(df), frac=0.2, min_n=5)
    tail = df.iloc[sl]

    # Pull columns (with guards if older CSV)
    P_all   = get(df, "polarization")
    P_tail  = get(tail, "polarization")
    t_tail  = get(tail, "t")
    dP_col  = get(tail, "dP_dt", default=np.nan)
    dP_dt   = np.nanmean(dP_col) if np.isfinite(dP_col).any() else rolling_slope(t_tail, P_tail)

    Feat = Features(
        P=float(np.nanmean(P_tail)),
        dP_dt=float(dP_dt),
        Crho=float(np.nanmean(get(tail, "rho_contrast"))),
        vort_rms=float(np.nanmean(get(tail, "vort_rms"))),
        vort_kurt=float(np.nanmean(get(tail, "vort_kurt", default=3.0))),
        ow_neg_frac=float(np.nanmean(get(tail, "ow_neg_frac"))),
        A=float(np.nanmean(get(tail, "anisotropy_A"))),
        S=float(np.nanmean(get(tail, "streamness_S"))),
        k_dom=float(np.nanmean(get(tail, "dom_k_mag"))),
        lambda_dom=float(np.nanmean(get(tail, "dom_wavelength"))),
        field=str(tail["render_field"].iloc[-1]) if "render_field" in tail.columns else "unknown",
        P0=float(P_all[0]),
        Pend=float(P_all[-1]),
    )
    return Feat


# ------------------------------- RULES --------------------------------
# Thresholds (tune ±20% for your resolution and parameters).
TH = dict(
    P_low=0.2,          # gas / clusters
    P_mid=0.7,          # mid-range
    P_high=0.8,         # flock
    P_very_high=0.9,    # crystal
    Crho_tiny=0.01,     # ultra-uniform density
    Crho_small=0.03,    # nearly uniform
    Crho_high=0.10,     # clustered
    A_high=0.60,        # strong anisotropy (bands/lanes)
    A_low=0.30,
    S_band=0.70,        # flow aligned with k_max (⊥ stripes) → bands
    S_lane=0.30,        # flow ∥ stripes → lanes/streams
    vort_low=0.05,      # depends on grid; adjust if needed
    vort_high=0.15,
    ow_frac_high=0.25,  # many coherent vortices (eddies)
    dP_pos=0.001,       # positive trend
)

def classify(feat: Features) -> Tuple[str, float, List[str]]:
    """
    Return (label, confidence 0..1, reasons[])
    """
    r = []  # reasons
    T = TH
    P, C, A, S = feat.P, feat.Crho, feat.A, feat.S
    vort, owf = feat.vort_rms, feat.ow_neg_frac

    # 1) Gas (disordered)
    if P < T["P_low"] and C < 0.05 and A < T["A_low"] and vort < T["vort_high"]:
        r += [f"P={P:.2f} low", f"ρ-contrast={C:.3f} small", f"anisotropy A={A:.2f} low"]
        return "Disordered gas", 0.9, r

    # 2) Patchy clusters
    if P < T["P_mid"] and C > T["Crho_high"] and A < 0.5:
        r += [f"P={P:.2f} low-mid", f"ρ-contrast={C:.3f} high → clustering", f"A={A:.2f} not strongly anisotropic"]
        return "Patchy clusters", 0.8, r

    # 3) Bands vs Lanes/Streams (need anisotropy high and a spectral peak)
    if A > T["A_high"] and feat.k_dom > 0.0:
        if S >= T["S_band"]:
            r += [f"A={A:.2f} high", f"S={S:.2f} (flow ⟂ stripes)"]
            return "Traveling bands", 0.9, r
        if S <= T["S_lane"]:
            r += [f"A={A:.2f} high", f"S={S:.2f} (flow ∥ stripes)"]
            return "Lanes / streams", 0.85, r

    # 4) Flocks (homogeneous vs polarized crystal)
    if P >= T["P_high"]:
        if C < T["Crho_tiny"] and vort < TH["vort_low"]:
            r += [f"P={P:.2f} very high", f"ρ-contrast={C:.3f} tiny", f"vort_rms={vort:.3f} very low"]
            return "Polarized crystal (stiff medium)", 0.9, r
        else:
            r += [f"P={P:.2f} high", f"ρ-contrast={C:.3f} small", f"vort_rms={vort:.3f} small–moderate"]
            return "Homogeneous flock", 0.8, r

    # 5) Defect-rich flock vs Active eddies (both mid P, high vorticity)
    if T["P_low"] <= P < T["P_high"] and vort >= T["vort_high"]:
        if owf >= T["ow_frac_high"] or feat.vort_kurt >= 4.0:
            r += [f"P={P:.2f} mid", f"vort_rms={vort:.3f} high", f"OW<0 frac={owf:.2f} high"]
            return "Active eddies (coherent vortices)", 0.8, r
        else:
            r += [f"P={P:.2f} mid", f"vort_rms={vort:.3f} high", f"OW<0 frac={owf:.2f} moderate"]
            return "Defect-rich flock (active turbulence)", 0.75, r

    # 6) Noise-induced flocking (trend-based)
    if feat.P0 < T["P_low"] and feat.Pend >= T["P_mid"] and feat.dP_dt > T["dP_pos"]:
        r += [f"P rose from {feat.P0:.2f} → {feat.Pend:.2f}", f"dP/dt={feat.dP_dt:.3g} > 0"]
        return "Noise-induced flocking (near-threshold)", 0.7, r

    # Fallbacks:
    if A > T["A_high"]:
        r += [f"A={A:.2f} high but streamness S={S:.2f} mid → ambiguous stripes"]
        return "Striped coexistence (ambiguous bands/lanes)", 0.55, r

    # If nothing matches strongly:
    r += [f"Unclear regime: P={P:.2f}, ρ-contrast={C:.3f}, A={A:.2f}, vort={vort:.3f}"]
    return "Unclassified (tune thresholds)", 0.4, r


# ------------------------------- CLI ----------------------------------

def main():
    ap = argparse.ArgumentParser(description="Classify Toner–Tu phase from metrics.csv")
    ap.add_argument("-i", "--input", required=True, help="Path to metrics.csv")
    ap.add_argument("--json", help="Optional path to write a JSON report")
    ap.add_argument("--show", action="store_true", help="Print the tail-averaged features used")
    args = ap.parse_args()

    try:
        feat = compute_features(args.input)
        label, conf, reasons = classify(feat)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)

    # Console report
    print(f"\n=== Toner–Tu Phase Classification ===")
    print(f"File: {args.input}")
    print(f"Phase: {label}")
    print(f"Confidence: {conf:.2f}")
    print("Why:")
    for s in reasons:
        print(f"  - {s}")

    if args.show:
        print("\nTail-averaged features (last ~20% frames):")
        print(f"  P={feat.P:.3f}, dP/dt={feat.dP_dt:.4g}, ρ-contrast={feat.Crho:.4f}")
        print(f"  A={feat.A:.3f}, streamness S={feat.S:.3f}")
        print(f"  vort_rms={feat.vort_rms:.4f}, vort_kurt={feat.vort_kurt:.3f}, OW<0 frac={feat.ow_neg_frac:.3f}")
        print(f"  k_dom={feat.k_dom:.4g}, λ_dom={feat.lambda_dom:.4g}, field={feat.field}")
        print(f"  P0={feat.P0:.3f} → Pend={feat.Pend:.3f}")

    # Optional JSON
    if args.json:
        out = dict(
            file=args.input,
            phase=label,
            confidence=conf,
            reasons=reasons,
            features=dict(
                P=feat.P, dP_dt=feat.dP_dt, rho_contrast=feat.Crho,
                anisotropy_A=feat.A, streamness_S=feat.S,
                vort_rms=feat.vort_rms, vort_kurt=feat.vort_kurt, ow_neg_frac=feat.ow_neg_frac,
                k_dom=feat.k_dom, lambda_dom=feat.lambda_dom,
                render_field=feat.field, P0=feat.P0, Pend=feat.Pend
            )
        )
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nJSON report written to: {args.json}")

if __name__ == "__main__":
    main()

