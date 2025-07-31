#!/usr/bin/env python3
"""Analyse lighthouse localisation accuracy.

This script reads a Feather dataframe exported by the swarm simulator, produces
three diagnostic scatter plots, and prints accuracy metrics after estimating a
single global scale factor that maps the predicted pose (pred_x, pred_y) to the
ground‑truth coordinates (x, y).

Usage
-----
$ python analyse_lighthouse.py frames/data.feather out_plots/
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def parse_args():
    """Return parsed command‑line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyse lighthouse localisation accuracy from a Feather file.")
    parser.add_argument("feather_file", type=Path,
                        help="Input Feather file (e.g., frames/data.feather)")
    parser.add_argument("output_dir", type=Path,
                        help="Directory where PDF scatter plots will be saved")
    return parser.parse_args()


def estimate_scale(x: np.ndarray, y: np.ndarray,
                   pred_x: np.ndarray, pred_y: np.ndarray) -> float:
    """Return the scalar *s* that minimises the least‑squares error between
    the ground‑truth vectors (x, y) and the scaled predictions s·(pred_x, pred_y).

    The closed‑form solution comes from d/ds Σ‖[x y]^T − s·[pred_x pred_y]^T‖² = 0.
    """
    numerator = np.sum(x * pred_x + y * pred_y)
    denominator = np.sum(pred_x ** 2 + pred_y ** 2)
    return numerator / denominator if denominator != 0 else 0.0


def compute_errors(x: np.ndarray, y: np.ndarray,
                   pred_x: np.ndarray, pred_y: np.ndarray,
                   scale: float):
    """Return (rmse_x, rmse_y, rmse_total) after applying *scale* to predictions."""
    dx = x - scale * pred_x
    dy = y - scale * pred_y
    rmse_x = float(np.sqrt(np.mean(dx ** 2)))
    rmse_y = float(np.sqrt(np.mean(dy ** 2)))
    rmse_total = float(np.sqrt(np.mean(dx ** 2 + dy ** 2)))
    return rmse_x, rmse_y, rmse_total


def make_plots(df: pd.DataFrame, out_dir: Path):
    """Generate and save the three requested scatter plots."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) pred_x vs pred_y (unit disc)
    ax = sns.scatterplot(x="pred_x", y="pred_y", hue="robot_id", data=df, s=20)
    #ax.set(xlim=(-1, 1), ylim=(-1, 1), title="Predicted position (unit circle)")
    ax.set( title="Predicted position (unit circle)")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(out_dir / "pred_xy.pdf", dpi=300)
    plt.close()

    # 2) x vs pred_x
    ax = sns.scatterplot(x="x", y="pred_x", hue="robot_id", data=df, s=20)
    ax.set(title="Ground truth X vs predicted X", xlabel="x [m]", ylabel="pred_x [arb]")
    #plt.ylim([-1, 1])
    plt.tight_layout()
    plt.savefig(out_dir / "x_vs_pred_x.pdf", dpi=300)
    plt.close()

    # 3) y vs pred_y
    ax = sns.scatterplot(x="y", y="pred_y", hue="robot_id", data=df, s=20)
    ax.set(title="Ground truth Y vs predicted Y", xlabel="y [m]", ylabel="pred_y [arb]")
    #plt.ylim([-1, 1])
    plt.tight_layout()
    plt.savefig(out_dir / "y_vs_pred_y.pdf", dpi=300)
    plt.close()


def main():
    args = parse_args()

    # Load dataframe
    df = pd.read_feather(args.feather_file)
    df = df[df['pose_valid'] == True]

    # Estimate global scale factor
    scale = estimate_scale(df["x"].values, df["y"].values,
                           df["pred_x"].values, df["pred_y"].values)

    # Accuracy statistics
    rmse_x, rmse_y, rmse_total = compute_errors(
        df["x"].values, df["y"].values,
        df["pred_x"].values, df["pred_y"].values,
        scale)

    # Pretty print
    print("=== Lighthouse localisation accuracy ===")
    print(f"Scale factor (s):        {scale: .6f}")
    print(f"RMSE x (m):              {rmse_x: .6f}")
    print(f"RMSE y (m):              {rmse_y: .6f}")
    print(f"RMSE 2‑D (m):            {rmse_total: .6f}")

    # Generate scatter plots
    make_plots(df, args.output_dir)


if __name__ == "__main__":
    main()

