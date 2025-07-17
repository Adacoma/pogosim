#!/usr/bin/env python3
"""Swarm‑analytics pipeline – one plot per metric **and per arena**
===============================================================

Each PDF now overlays **all runs of the same arena in a single figure**, so
comparisons are immediate:

* `arena_disk_novelty_curve.pdf`  → many faint run‑wise curves + bold mean.
* `arena_disk_neighbor_curve.pdf` → idem for neighbour degree, etc.

Scalar CSV outputs and the printed summary remain unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def auc(t: np.ndarray, y: np.ndarray) -> float:
    """Trapezoidal integration."""
    return float(np.trapz(y=y, x=t))


def norm_novelty(series: pd.Series, tau: float) -> pd.Series:
    return (1.0 - series.astype(float) / tau).clip(0, 1)


def mean_neighbor_degree(g: pd.DataFrame) -> pd.Series:
    n = g["robot_id"].nunique()
    return g.groupby("time", sort=True)["total_neighbors"].mean().sort_index() / max(1, n - 1)


def isolation_fraction(g: pd.DataFrame) -> pd.Series:
    return (
        g.assign(_iso=(g["total_neighbors"] == 0).astype(float))
         .groupby("time", sort=True)["_iso"].mean()
         .sort_index()
    )


def swarm_mean(g: pd.DataFrame, col: str) -> pd.Series:
    return g.groupby("time", sort=True)[col].mean().sort_index()


def unique_contact_rate(g: pd.DataFrame) -> float:
    dt = g["time"].max() - g["time"].min()
    return float(g.groupby("robot_id")["nov_unique"].apply(lambda s: (s.max() - s.min()) / dt).mean())


def convex_hull_area(pts: np.ndarray) -> float:
    if pts.shape[0] < 3:
        return 0.0
    return float(ConvexHull(pts).volume)


def hull_curve(g: pd.DataFrame) -> pd.Series:
    areas = (
        g.groupby("time", sort=True)[["x", "y"]]
         .apply(lambda xy: convex_hull_area(xy.values))
         .sort_index()
    )
    return areas / areas.max() if areas.max() > 0 else areas


def state_histogram(g: pd.DataFrame) -> dict[int, float]:
    c = Counter(g["state"].astype(int))
    total = sum(c.values())
    return {k: v / total for k, v in c.items()}

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def process(inp: Path, outdir: Path, tau: float):
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_feather(inp)

    needed = {
        "time", "nov_cnt", "total_neighbors", "nov_unique", "surprise",
        "state", "x", "y", "run", "arena_file", "robot_id",
    }
    if not needed.issubset(df.columns):
        raise ValueError(f"Missing cols: {needed.difference(df.columns)}")

    df["norm_novelty"] = norm_novelty(df["nov_cnt"], tau)

    # Collect scalar KPIs
    novelty_rows, surprise_rows, uniq_rows, state_rows = [], [], [], []

    # ---------- iterate arenas ----------
    for arena, g_arena in df.groupby("arena_file"):
        # for plotting: metric -> list of (series, label)
        metric_series: dict[str, list[tuple[pd.Series, str]]] = defaultdict(list)

        for run, g_run in g_arena.groupby("run"):
            label = f"run {run}"

            curves = {
                "novelty": g_run.groupby("time", sort=True)["norm_novelty"].mean().sort_index(),
                "neighbor": mean_neighbor_degree(g_run),
                "isolation": isolation_fraction(g_run),
                "surprise": swarm_mean(g_run, "surprise"),
                "hull": hull_curve(g_run),
            }

            for key, ser in curves.items():
                metric_series[key].append((ser, label))

            # Scalars
            t = curves["novelty"].index.to_numpy(float)
            novelty_rows.append({"arena_file": arena, "run": run, "auc": auc(t, curves["novelty"].to_numpy(float))})
            surprise_rows.append({"arena_file": arena, "run": run, "auc": auc(t, curves["surprise"].reindex_like(curves["novelty"]).to_numpy(float))})
            uniq_rows.append({"arena_file": arena, "run": run, "rate_contacts_per_s": unique_contact_rate(g_run)})
            state_rows.append({"arena_file": arena, "run": run, **state_histogram(g_run)})

        # ---------- make one PDF per metric with *all* runs ----------
        ylabels = {
            "novelty": "Normalised novelty",
            "neighbor": "Mean neighbour degree / (N−1)",
            "isolation": "Isolation fraction",
            "surprise": "Surprise (a.u.)",
            "hull": "Convex‑hull area / max",
        }

        for key, series_list in metric_series.items():
            plt.figure()
            for ser, lbl in series_list:
                plt.plot(ser.index, ser.values, label=lbl, alpha=0.35)
            # add bold mean
            mean_curve = pd.concat([s for s, _ in series_list], axis=1).mean(axis=1)
            plt.plot(mean_curve.index, mean_curve.values, color="black", linewidth=2, label="mean")
            plt.xlabel("Time [s]")
            plt.ylabel(ylabels[key])
            plt.title(f"{ylabels[key]} vs time (arena {arena})")
            plt.legend(fontsize="x-small", frameon=False)
            plt.tight_layout()
            pdf = outdir / f"{arena}_{key}_curve.pdf"
            plt.savefig(pdf, format="pdf")
            plt.close()
            print(f"[✓] Plot saved → {pdf}")

    # ---------- save scalar CSVs ----------
    def save(rows, fname):
        df_out = pd.DataFrame(rows)
        path = outdir / fname
        df_out.to_csv(path, index=False)
        print(f"[✓] Saved → {path}")
        return df_out

    novelty_df = save(novelty_rows, "novelty_auc.csv")
    surprise_df = save(surprise_rows, "surprise_auc.csv")
    uniq_df = save(uniq_rows, "unique_contact_rate.csv")

    state_df = pd.DataFrame(state_rows).fillna(0.0)
    state_df.to_csv(outdir / "state_histograms.csv", index=False)
    print(f"[✓] Saved → {outdir / 'state_histograms.csv'}\n")

    # ---------- print summary ----------
    def stats(df_metric, name, col):
        print(f"=== {name} (mean ± std) ===")
        for arena, grp in df_metric.groupby("arena_file"):
            print(f"{arena:<20s}: {grp[col].mean():.3f} ± {grp[col].std():.3f}")
        print("------------------------------")
        print(f"{'Overall':<20s}: {df_metric[col].mean():.3f} ± {df_metric[col].std():.3f}\n")

    stats(novelty_df, "Novelty AUC", "auc")
    stats(surprise_df, "Surprise AUC", "auc")
    stats(uniq_df, "Unique‑contact rate (cts/s)", "rate_contacts_per_s")

    # ---------- behaviour‑state bar chart per arena ----------
    for arena, grp in state_df.groupby("arena_file"):
        mean_hist = grp.drop(columns=["arena_file", "run"]).mean()
        plt.figure()
        plt.bar(mean_hist.index.astype(int), mean_hist.values)
        plt.xlabel("State ID")
        plt.ylabel("Fraction of time")
        plt.title(f"Behaviour state distribution (arena {arena})")
        plt.tight_layout()
        pdf = outdir / f"{arena}_state_histogram.pdf"
        plt.savefig(pdf, format="pdf")
        plt.close()
        print(f"[✓] Plot saved → {pdf}")

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Swarm metrics with one plot per arena per metric (runs overlaid).")
    p.add_argument("input", type=Path, help="Feather dataframe path")
    p.add_argument("output", type=Path, help="Output directory")
    p.add_argument("--tau", type=float, default=4.0, help="decay_tau used in simulation [default: 4.0]")
    args = p.parse_args()

    process(args.input, args.output, args.tau)


# MODELINE "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
