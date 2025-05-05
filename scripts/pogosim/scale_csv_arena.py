#!/usr/bin/env python3
"""
Scale a polygon to 1.5 × 10⁵ mm² and export as CSV or SVG.

If <output> ends in .csv → write scaled CSV.
If <output> ends in .svg → write scaled SVG.

Usage
-----
python scale_and_export.py <input.csv> <output.{csv|svg}>
"""

import argparse
import csv
import math
import os
from typing import List, Tuple

DESIRED_AREA_MM2 = 150_000.0  # 1.5 × 10⁵ mm²


# ---------- geometry helpers -------------------------------------------------
def read_points(path: str) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            try:
                pts.append((float(row[0]), float(row[1])))
            except ValueError:  # skip header/bad rows
                continue
    if len(pts) < 3:
        raise ValueError("Need at least three points for a polygon.")
    return pts


def shoelace_area(pts: List[Tuple[float, float]]) -> float:
    area = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def scale_points(pts: List[Tuple[float, float]],
                 target_area: float) -> List[Tuple[float, float]]:
    current = shoelace_area(pts)
    if current == 0:
        raise ValueError("Polygon area is zero; cannot scale.")
    s = math.sqrt(target_area / current)
    return [(x * s, y * s) for x, y in pts], s


# ---------- writers ----------------------------------------------------------
def write_csv(path: str, pts: List[Tuple[float, float]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        #w.writerow(["X_mm", "Y_mm"])
        for x, y in pts:
            w.writerow([f"{x:.6f}", f"{y:.6f}"])


def write_svg(path: str, pts: List[Tuple[float, float]]) -> None:
    xs, ys = zip(*pts)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width, height = max_x - min_x, max_y - min_y
    shifted = [(x - min_x, y - min_y) for x, y in pts]
    poly_str = " ".join(f"{x:.3f},{y:.3f}" for x, y in shifted)

    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width:.3f}mm" height="{height:.3f}mm"
     viewBox="0 0 {width:.3f} {height:.3f}">
  <polygon points="{poly_str}"
           fill="none"
           stroke="black"
           stroke-width="0.5"/>
</svg>
'''
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)


# ---------- CLI --------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scale a CSV polygon to 1.5×10⁵ mm² and export to CSV or SVG."
    )
    p.add_argument("input", help="Input CSV with X,Y coordinates")
    p.add_argument("output", help="Output file (.csv or .svg)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pts_in = read_points(args.input)
    pts_scaled, s = scale_points(pts_in, DESIRED_AREA_MM2)

    ext = os.path.splitext(args.output)[1].lower()
    if ext == ".csv":
        write_csv(args.output, pts_scaled)
    elif ext == ".svg":
        write_svg(args.output, pts_scaled)
    else:
        raise ValueError("Output extension must be .csv or .svg")

    print(f"Scaled by factor {s:.6f} and wrote {len(pts_scaled)} points to "
          f"'{args.output}'.")


if __name__ == "__main__":
    main()

