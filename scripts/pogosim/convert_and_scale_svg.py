#!/usr/bin/env python3
"""
Extract polygon points from an SVG <path>, save them to CSV, then
scale the polygon so its area is 1.5 × 10⁵ mm² and write a new SVG.

Example
-------
python convert_and_scale_svg.py input.svg points.csv scaled.svg
"""

import argparse
import csv
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

TARGET_AREA_MM2 = 150_000.0  # 1.5 × 10⁵ mm²


# ───────────────────────── SVG PATH  →  POINT LIST ──────────────────────────
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

def tokenize_path(path_d: str) -> List[str]:
    """Return a list of command letters and number tokens."""
    # split so each letter or number is a separate token
    tokens = re.findall(r"[A-Za-z]|-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", path_d)
    return tokens


def parse_path_points(path_d: str) -> List[Tuple[float, float]]:
    """
    Very small subset of an SVG path parser that understands:
    m/M, l/L, h/H, v/V, z/Z  (enough for the supplied star).
    Returns a list of vertex coordinates – the polygon is closed, but
    the final closing point is *not* repeated.
    """
    tokens = tokenize_path(path_d)
    idx = 0
    cmd = None
    current: Tuple[float, float] = (0.0, 0.0)
    start_point: Tuple[float, float] | None = None
    points: List[Tuple[float, float]] = []

    while idx < len(tokens):
        tok = tokens[idx]
        if tok.isalpha():
            cmd = tok
            idx += 1
            continue

        def take_number() -> float:
            nonlocal idx
            n = float(tokens[idx])
            idx += 1
            return n

        if cmd in ("m", "M"):                       # moveto
            dx, dy = take_number(), take_number()
            current = (
                current[0] + dx if cmd == "m" else dx,
                current[1] + dy if cmd == "m" else dy,
            )
            start_point = current
            points.append(current)
            # after a moveto, numbers without a command repeat ⇒ lineto
            cmd = "l" if cmd == "m" else "L"
        elif cmd in ("l", "L"):                     # lineto
            dx, dy = take_number(), take_number()
            current = (
                current[0] + dx if cmd == "l" else dx,
                current[1] + dy if cmd == "l" else dy,
            )
            points.append(current)
        elif cmd in ("h", "H"):                     # horizontal lineto
            dx = take_number()
            current = (
                current[0] + dx if cmd == "h" else dx,
                current[1],
            )
            points.append(current)
        elif cmd in ("v", "V"):                     # vertical lineto
            dy = take_number()
            current = (
                current[0],
                current[1] + dy if cmd == "v" else dy,
            )
            points.append(current)
        elif cmd in ("z", "Z"):                     # closepath
            if start_point and current != start_point:
                points.append(start_point)
            idx += 1
        else:
            raise ValueError(f"Unsupported SVG path command: {cmd}")

    # Drop redundant final point if it duplicates the first
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    return points


# ───────────────────────── GEOMETRY UTILS ──────────────────────────
def polygon_area(points: List[Tuple[float, float]]) -> float:
    """Unsigned area via the shoelace formula."""
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def scale_points(points: List[Tuple[float, float]], target_area_mm2: float
                 ) -> Tuple[List[Tuple[float, float]], float]:
    """Return (scaled_points, scale_factor)."""
    current_area = polygon_area(points)
    if current_area == 0:
        raise ValueError("Original polygon has zero area.")
    s = math.sqrt(target_area_mm2 / current_area)
    scaled = [(x * s, y * s) for x, y in points]
    return scaled, s


# ───────────────────────── I/O ──────────────────────────
def write_csv(path: Path, points: List[Tuple[float, float]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        #w.writerow(["X_mm", "Y_mm"])
        for x, y in points:
            w.writerow([f"{x:.6f}", f"{y:.6f}"])


def write_svg(path: Path, points: List[Tuple[float, float]]) -> None:
    xs, ys = zip(*points)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width, height = max_x - min_x, max_y - min_y
    shifted = [(x - min_x, y - min_y) for x, y in points]
    pts_str = " ".join(f"{x:.3f},{y:.3f}" for x, y in shifted)
    svg_out = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width:.3f}mm" height="{height:.3f}mm"
     viewBox="0 0 {width:.3f} {height:.3f}">
  <polygon points="{pts_str}"
           fill="none"
           stroke="black"
           stroke-width="0.5"/>
</svg>
'''
    path.write_text(svg_out, encoding="utf-8")


def extract_first_path_d(svg_path: Path) -> str:
    tree = ET.parse(svg_path)
    root = tree.getroot()
    # SVG elements may have namespace; strip it
    for elem in root.iter():
        tag = elem.tag.split("}", maxsplit=1)[-1]  # drop namespace
        if tag == "path":
            d_attr = elem.get("d")
            if not d_attr:
                raise ValueError("Path element has no 'd' attribute.")
            return d_attr
    raise ValueError("No <path> element found in SVG.")


# ───────────────────────── CLI ──────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert an SVG star to CSV + scaled SVG (1.5e5 mm²).")
    p.add_argument("input_svg", type=Path, help="Original SVG file")
    p.add_argument("output_csv", type=Path, help="CSV to write points to")
    p.add_argument("scaled_svg", type=Path, help="SVG to write scaled shape to")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1. get path string from the original SVG
    d_attr = extract_first_path_d(args.input_svg)

    # 2. parse path → points
    raw_points = parse_path_points(d_attr)

    # 3. write CSV of the original points
    write_csv(args.output_csv, raw_points)
    print(f"Saved {len(raw_points)} points to '{args.output_csv}'")

    # 4. scale to target area and write SVG
    scaled_points, scale = scale_points(raw_points, TARGET_AREA_MM2)
    write_svg(args.scaled_svg, scaled_points)
    area_scaled = polygon_area(scaled_points)
    print(
        f"Scaled by factor {scale:.6f}; new area ≈ {area_scaled:.0f} mm²; "
        f"wrote '{args.scaled_svg}'."
    )


if __name__ == "__main__":
    main()

