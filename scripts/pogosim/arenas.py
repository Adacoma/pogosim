#!/usr/bin/env python3
"""
Arena polygon loader/scaler/plotter.

- Reads an arena polygon from a CSV where loops are separated by blank lines.
- Scales the polygon to match a target surface (area) from YAML (or CLI).
- Translates the polygon so that its min-x/min-y bound sits at the origin.
- Prints geometry stats and saves a figure via utils.save_figure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import yaml
from shapely.geometry import Polygon
from shapely import affinity

import utils  # ← use project plotting helper


# ──────────────────────────────────────────────────────────────────────────────
# I/O utilities

def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """
    @brief Load a YAML configuration file.

    Expected keys:
      - arena_file: path to CSV polygon description
      - arena_surface: target area in mm^2
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def load_arena_polygon_from_csv(arena_file_path: str | Path) -> Polygon:
    """
    @brief Load a polygon (with optional holes) from a CSV file.

    CSV format:
      - One coordinate per line: "x,y"
      - Loops (outer shell + holes) are separated by a blank line
      - First loop is the outer boundary; subsequent loops are holes
    """
    arena_file_path = Path(arena_file_path)
    with arena_file_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    loops: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                loops.append(current)
                current = []
            continue
        x_str, y_str = stripped.split(",")
        current.append((float(x_str), float(y_str)))

    if current:
        loops.append(current)

    if not loops:
        raise ValueError("No valid loops found in arena file")

    shell = loops[0]
    holes = loops[1:] if len(loops) > 1 else []
    polygon = Polygon(shell, holes=holes)

    if not polygon.is_valid:
        raise ValueError("Loaded polygon is invalid")

    return polygon


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers

def get_bounds_from_polygon(polygon: Polygon) -> Tuple[float, float, float, float]:
    """
    @brief Return (minx, miny, maxx, maxy) bounds of the polygon.
    """
    return polygon.bounds  # type: ignore[return-value]


def translate_to_origin(polygon: Polygon) -> Polygon:
    """
    @brief Translate so that minx/miny is at (0, 0).
    """
    minx, miny, _, _ = polygon.bounds
    return affinity.translate(polygon, xoff=-minx, yoff=-miny)


def scale_polygon_to_area(polygon: Polygon, target_area: float) -> Polygon:
    """
    @brief Uniformly scale @p polygon so that its area equals @p target_area.

    Scales around the origin (0, 0). If your polygon isn't at the origin,
    call translate_to_origin() before scaling.
    """
    if target_area <= 0:
        raise ValueError("target_area must be > 0")

    src_area = polygon.area
    if src_area <= 0:
        raise ValueError("source polygon has non-positive area")

    scale = (target_area / src_area) ** 0.5
    return affinity.scale(polygon, xfact=scale, yfact=scale, origin=(0, 0))


# ──────────────────────────────────────────────────────────────────────────────
# Plotting

def plot_arena_polygon(polygon: Polygon, out_path: str | Path | None) -> None:
    """
    @brief Plot the polygon (with holes) and optionally save to @p out_path
           using utils.save_figure.
    """
    plt.figure(figsize=(8, 6))

    # Outer boundary
    x, y = polygon.exterior.xy
    plt.plot(x, y)
    plt.fill(x, y, alpha=0.3, label="Outer boundary")

    # Holes
    hole_labeled = False
    for interior in polygon.interiors:
        xh, yh = interior.xy
        plt.plot(xh, yh)
        plt.fill(xh, yh, color="white", label=("Hole" if not hole_labeled else None))
        hole_labeled = True

    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Scaled Arena Polygon")
    plt.grid(True)
    plt.legend()

    if out_path is not None:
        utils.save_figure(Path(out_path))  # ← delegate to project helper


# ──────────────────────────────────────────────────────────────────────────────
# Orchestration

def build_scaled_arena_polygon(
    arena_file: str | Path,
    arena_surface: float,
) -> Polygon:
    """
    @brief Load, normalize to origin, then scale a polygon to target area.
    """
    poly = load_arena_polygon_from_csv(arena_file)
    poly = translate_to_origin(poly)
    poly = scale_polygon_to_area(poly, arena_surface)
    return poly


def summarize_polygon(poly: Polygon) -> Dict[str, Any]:
    """
    @brief Compute a summary of polygon metrics useful for logging.
    """
    minx, miny, maxx, maxy = get_bounds_from_polygon(poly)
    return {
        "bounds": (minx, miny, maxx, maxy),
        "area": poly.area,
        "is_valid": bool(poly.is_valid),
        "is_simple": bool(poly.is_simple),
        "n_exterior_points": len(poly.exterior.coords),
        "n_holes": len(poly.interiors),
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Load, scale, and plot an arena polygon from CSV + YAML.",
    )
    p.add_argument("--config", type=Path, default=Path("conf/simple.yaml"),
                   help="YAML config with 'arena_file' and 'arena_surface'.")
    p.add_argument("--arena-file", type=Path, default=None,
                   help="Override arena_file from YAML.")
    p.add_argument("--arena-surface", type=float, default=None,
                   help="Override arena_surface (mm^2) from YAML.")
    p.add_argument("--output-dir", type=Path, default=Path("results"),
                   help="Directory to write outputs.")
    p.add_argument("--outfile", type=str, default="arena_polygon.pdf",
                   help="Filename for the saved figure (use .pdf/.png, etc.).")
    p.add_argument("--no-save", action="store_true",
                   help="Do not save the figure to disk.")
    p.add_argument("--show", action="store_true",
                   help="Show the plot interactively.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Load config
    cfg = load_yaml_config(args.config)

    # Resolve parameters (CLI overrides YAML)
    arena_file = args.arena_file or Path(cfg.get("arena_file", ""))
    arena_surface = args.arena_surface if args.arena_surface is not None else float(cfg.get("arena_surface", 0.0))

    if not arena_file:
        print("ERROR: 'arena_file' not provided (YAML or --arena-file).", file=sys.stderr)
        return 2
    if arena_surface <= 0:
        print("ERROR: 'arena_surface' must be > 0 (YAML or --arena-surface).", file=sys.stderr)
        return 2

    # Build polygon
    poly = build_scaled_arena_polygon(arena_file, arena_surface)

    # Summaries / logs
    summary = summarize_polygon(poly)
    print(f"Arena file: {arena_file}")
    print(f"Arena surface (from config/CLI): {arena_surface} mm²")
    print("Polygon bounds (minx, miny, maxx, maxy):", summary["bounds"])
    print("Polygon area (scaled):", summary["area"], "mm²")
    print("Is polygon valid?", summary["is_valid"])
    print("Is polygon simple?", summary["is_simple"])
    print("Number of points in polygon:", summary["n_exterior_points"])
    print("Number of holes:", summary["n_holes"])

    # Plot
    out_path = None if args.no_save else (args.output_dir / args.outfile)
    plot_arena_polygon(poly, out_path)

    if args.show:
        plt.show()
    else:
        plt.close("all")

    # Echo bounds at the end to mirror the original script
    print(summary["bounds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# MODELINE "{{{1"
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
