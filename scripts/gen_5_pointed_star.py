#!/usr/bin/env python3
"""
Generate a 5-pointed equilateral star with total area 1.5×10⁵ mm².
Outputs: star_mm.csv and star_mm.svg
"""

import csv
import math
from pathlib import Path
from typing import List, Tuple

# ─────────── parameters ───────────
A_TARGET = 150_000.0                         # mm²
K = math.sin(math.pi / 10) / math.sin(3 * math.pi / 10)  # r/R for regular 5-star
Point = Tuple[float, float]


# ─────────── geometry helpers ───────────
def star_points(R: float, k: float = K) -> List[Point]:
    """Return the 10 vertices of a regular 5-point star centred at (0,0)."""
    pts = []
    for i in range(5):
        t_outer = 2 * math.pi * i / 5
        t_inner = t_outer + math.pi / 5
        pts.append((R * math.cos(t_outer), R * math.sin(t_outer)))   # outer tip
        pts.append((R * k * math.cos(t_inner), R * k * math.sin(t_inner)))  # inner
    return pts


def poly_area(pts: List[Point]) -> float:
    """Shoelace formula (unsigned)."""
    return abs(
        sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]))
    ) / 2.0


# ─────────── compute scale ───────────
A_unit = poly_area(star_points(1.0))
R_mm = math.sqrt(A_TARGET / A_unit)          # outer radius in mm
pts_mm = star_points(R_mm)
edge_len = math.dist(pts_mm[0], pts_mm[1])

print(f"Outer radius R  = {R_mm:.3f} mm")
print(f"Edge length L   = {edge_len:.3f} mm")
print(f"Area (check)    = {poly_area(pts_mm):.0f} mm²")

# ─────────── write CSV (fixed) ───────────
csv_path = Path("star_mm.csv")
with csv_path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["X_mm", "Y_mm"])
    w.writerows((f"{x:.6f}", f"{y:.6f}") for x, y in pts_mm)
print(f"Wrote CSV   → {csv_path}")

# ─────────── write SVG ───────────
xs, ys = zip(*pts_mm)
min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
width, height = max_x - min_x, max_y - min_y
shifted = [(x - min_x, y - min_y) for x, y in pts_mm]

svg_path = Path("star_mm.svg")
svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width:.3f}mm" height="{height:.3f}mm"
     viewBox="0 0 {width:.3f} {height:.3f}">
  <polygon points="{' '.join(f'{x:.3f},{y:.3f}' for x, y in shifted)}"
           fill="none"
           stroke="black"
           stroke-width="0.5"/>
</svg>
'''
svg_path.write_text(svg, encoding="utf-8")
print(f"Wrote SVG   → {svg_path}")

