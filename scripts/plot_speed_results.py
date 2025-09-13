#!/usr/bin/env python3
"""
Plot speed benchmark results (one line per condition) from a CSV produced by bench_speed.sh.

Adds a "real-time" reference bar at y = --sim-seconds (default 60 s): below it is faster than real-time.

CSV columns expected:
  - condition (str)
  - robots (int)
  - real_per_run, user_per_run, sys_per_run  (preferred)  OR  real, user, sys (fallback)

Usage:
  python plot_speed_results.py path/to/speed_results.csv \
      --metric real \
      --output results/speed_benchmark/speed_plot.pdf \
      --sim-seconds 60 \
      --xlog \
      --ylog
"""
import argparse
from pathlib import Path
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="Input CSV file (from bench_speed.sh).")
    p.add_argument(
        "--metric",
        choices=["real", "user", "sys"],
        default="real",
        help="Which timing to plot on Y (default: real).",
    )
    p.add_argument(
        "--output",
        default="speed_plot.pdf",
        help="Output PDF filename (default: speed_plot.pdf).",
    )
    p.add_argument(
        "--xlog",
        action="store_true",
        help="Use log scale on X (robots). Recommended for 1..1000.",
    )
    p.add_argument(
        "--ylog",
        action="store_true",
        help="Use log scale on Y (time). Falls back to symlog if nonpositive values are present.",
    )
    p.add_argument(
        "--sim-seconds",
        type=float,
        default=60.0,
        help="Simulated seconds per run to mark the real-time threshold (default: 60).",
    )
    return p.parse_args()


def pick_metric_column(df: pd.DataFrame, base: str) -> str:
    # Prefer per-run columns; otherwise fall back to total columns
    per_run = f"{base}_per_run"
    if per_run in df.columns:
        return per_run
    if base in df.columns:
        return base
    raise ValueError(
        f"Could not find column '{per_run}' or '{base}' in CSV. "
        f"Available: {list(df.columns)}"
    )


def main():
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "condition" not in df or "robots" not in df:
        raise ValueError("CSV must contain columns: 'condition' and 'robots'.")

    df["condition"] = df["condition"].astype(str).str.strip()
    df["robots"] = pd.to_numeric(df["robots"], errors="coerce").astype(int)

    metric_col = pick_metric_column(df, args.metric)

    # Aggregate in case multiple rows per (condition, robots) exist
    df_agg = (
        df.groupby(["condition", "robots"], as_index=False)[metric_col]
        .mean()
        .sort_values(["condition", "robots"])
    )

    # Preferred condition order
    preferred_order = ["noGUI_noExport", "noGUI_export1s", "GUI_noExport", "GUI_export1s"]
    present = [c for c in preferred_order if c in df_agg["condition"].unique()]
    extras = [c for c in df_agg["condition"].unique() if c not in present]
    cond_order = present + extras

    # Okabe–Ito colorblind-safe palette
    okabe_ito = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#000000", "#E69F00"]
    color_map = {cond: okabe_ito[i % len(okabe_ito)] for i, cond in enumerate(cond_order)}
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    marker_map = {cond: markers[i % len(markers)] for i, cond in enumerate(cond_order)}

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=150)

    # Plot lines
    for cond in cond_order:
        sub = df_agg[df_agg["condition"] == cond].sort_values("robots")
        if sub.empty:
            continue
        ax.plot(
            sub["robots"].values,
            sub[metric_col].values,
            label=cond,
            marker=marker_map[cond],
            linewidth=1.8,
            markersize=5,
            color=color_map[cond],
        )

    # X axis
    if args.xlog:
        ax.set_xscale("log")
        xticks = [1, 5, 25, 50, 100, 250, 500, 1000]
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(v) for v in xticks])

    # Y axis (log / symlog handling with dense ticks)
    if args.ylog:
        yvals = df_agg[metric_col].to_numpy()
        if np.all(yvals > 0):
            ax.set_yscale("log")
            # More readable major ticks: 1, 2, 5 × 10^k
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            # Minor ticks at every 1–9 × 10^k (no labels)
            ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=tuple(range(1, 10)), numticks=100))
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())

            # Label majors as plain numbers (no scientific notation)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(
                lambda y, _: f"{y:.0f}" if y >= 1 else f"{y:g}"
            ))
        else:
            # Fallback for any nonpositive values: keep symlog, but avoid cluttered ticks
            ax.set_yscale("symlog", linthresh=1e-2)
            # Reasonable major ticks on symlog
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
            ax.yaxis.set_minor_locator(mticker.NullLocator())


    ax.set_xlabel("Robots")
    metric_label = {
        "real": "Wall time (s) per run",
        "user": "User CPU time (s) per run",
        "sys": "Sys CPU time (s) per run",
    }[args.metric]
    ax.set_ylabel(metric_label)
    ax.set_title("Simulation speed vs number of robots")

    # Real-time reference bar (only meaningful for wall time)
    if args.metric == "real":
        thr = float(args.sim_seconds)
        ax.axhline(thr, color="#595959", linestyle="--", linewidth=1.5, label=f"real-time ({int(thr)} s)")
#        # Annotate faster/slower on the right side
#        ax.text(
#            0.99, 0.51, "slower", transform=ax.transAxes,
#            ha="right", va="bottom", fontsize=9, color="#595959"
#        )
#        ax.text(
#            0.99, 0.49, "faster", transform=ax.transAxes,
#            ha="right", va="top", fontsize=9, color="#595959"
#        )

    # Cosmetics
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(title="Condition", frameon=False)
    ax.set_ylim(0.2, 70)
    fig.tight_layout()

    fig.savefig(out, bbox_inches="tight")
    # No plt.show() for batch use


if __name__ == "__main__":
    main()

