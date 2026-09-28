#!/usr/bin/env python3
"""Compute the worked tutorial's origin-relative MSD from merged Pogobatch data."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CONDITIONS = {"long_runs", "long_tumbles"}
COLUMNS = ["condition", "run", "seed", "robot_category", "robot_id", "time", "x", "y"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Merged run_and_tumble_msd.feather")
    parser.add_argument("--output-dir", type=Path, default=Path("derived/run_and_tumble_msd"))
    parser.add_argument("--expected-runs", type=int, default=128,
                        help="Expected independent seeds per condition (use 1 for a smoke test)")
    args = parser.parse_args()
    if args.expected_runs < 1:
        parser.error("--expected-runs must be positive")

    # Read only the columns needed for the analysis: a full campaign has about
    # three million logged robot positions at the tutorial's one-second cadence.
    data = pd.read_feather(args.input, columns=COLUMNS)
    data = data.loc[data["robot_category"] == "robots"].drop(columns="robot_category")
    if data.empty or data.isna().any().any():
        raise ValueError("Missing robot rows or null identifiers, times, or coordinates")
    if set(data["condition"].unique()) != CONDITIONS:
        observed = sorted(data["condition"].unique())
        raise ValueError(f"Expected conditions {sorted(CONDITIONS)}, got {observed}")

    runs = data.groupby("condition")["run"].nunique()
    if not runs.eq(args.expected_runs).all():
        raise ValueError(f"Expected {args.expected_runs} runs per condition, got {runs.to_dict()}")
    robots = data.groupby(["condition", "run"])["robot_id"].nunique()
    if not robots.eq(100).all():
        raise ValueError("Expected 100 distinct robots in every run")
    seeds = {
        condition: set(group["seed"].unique())
        for condition, group in data.groupby("condition")
    }
    if seeds["long_runs"] != seeds["long_tumbles"]:
        raise ValueError(
            "Conditions do not have the same seed set; check failed or retried Tasks"
        )

    trajectory = ["condition", "run", "robot_id"]
    data = data.sort_values(trajectory + ["time"], kind="mergesort")
    # A sample index avoids equating timestamp differences caused only by
    # floating-point representation; unequal trajectory lengths are rejected.
    sample_counts = data.groupby(trajectory).size()
    if sample_counts.nunique() != 1:
        raise ValueError("Trajectories have unequal numbers of logged samples")
    data["sample"] = data.groupby(trajectory).cumcount()
    origin = data.groupby(trajectory)[["time", "x", "y"]].transform("first")
    data["elapsed_s"] = data["time"] - origin["time"]
    data["squared_displacement_mm2"] = (
        (data["x"] - origin["x"]) ** 2 + (data["y"] - origin["y"]) ** 2
    )

    # Average robots within each independent run first. The across-run spread
    # below therefore measures seed-to-seed variation, not pseudo-replication
    # from treating all 100 robots as independent experimental repeats.
    per_run = data.groupby(["condition", "run", "sample"]).agg(
        elapsed_s=("elapsed_s", "mean"),
        msd_mm2=("squared_displacement_mm2", "mean"),
        robot_count=("robot_id", "nunique"),
    ).reset_index()
    if not per_run["robot_count"].eq(100).all():
        raise ValueError("One or more runs are missing robot observations")
    time_spread = per_run.groupby(["condition", "sample"])["elapsed_s"].agg(
        lambda values: values.max() - values.min()
    )
    if time_spread.gt(0.05).any():
        raise ValueError(
            "Logged time grids differ by more than 0.05 s; align before averaging"
        )

    curve = per_run.groupby(["condition", "sample"]).agg(
        elapsed_s=("elapsed_s", "mean"),
        mean_msd_mm2=("msd_mm2", "mean"),
        sd_msd_mm2=("msd_mm2", "std"),
        n_runs=("run", "nunique"),
    ).reset_index()
    # A descriptive normal-approximation CI across seeds; no CI is claimed for
    # a one-seed smoke test. Solid arena walls limit late-time displacement.
    curve["ci95_half_width_mm2"] = 1.96 * curve["sd_msd_mm2"] / curve["n_runs"].pow(0.5)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve.to_csv(args.output_dir / "msd_curves.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for condition, group in curve.groupby("condition"):
        x = group["elapsed_s"].to_numpy(dtype=float)
        y = group["mean_msd_mm2"].to_numpy(dtype=float)
        ax.plot(x, y, label=condition.replace("_", " "))
        if args.expected_runs > 1:
            half_width = group["ci95_half_width_mm2"].to_numpy(dtype=float)
            ax.fill_between(x, y - half_width, y + half_width, alpha=0.2)
        final = group.iloc[-1]
        print(f"{condition}: MSD at {final['elapsed_s']:.2f} s = {final['mean_msd_mm2']:.2f} mm^2")
    ax.set(xlabel="Elapsed simulated time (s)", ylabel="Mean squared displacement (mm²)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "msd.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
