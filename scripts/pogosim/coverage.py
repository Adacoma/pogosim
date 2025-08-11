#!/usr/bin/env python3
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
from scipy.spatial import Voronoi

from pogosim import utils
from pogosim import __version__


# ──────────────────────────────────────────────────────────────────────────────
# Config & arena loading (prefers config_path over meta['configuration'])

def _safe_get(d: Dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _load_yaml(path: str | Path) -> Dict[str, Any]:
    import yaml
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _extract_arena_csv_path(value: Any) -> str | None:
    """
    Accept either:
      - dict with 'default_option' => absolute/relative CSV path (must exist)
      - string => CSV path (must exist)
    Return normalized string path or None if value unusable.
    """
    if isinstance(value, dict):
        opt = value.get("default_option")
        if isinstance(opt, str) and opt.strip():
            p = Path(opt)
            if not p.exists():
                raise FileNotFoundError(f"arena_file.default_option does not exist: {p}")
            if p.suffix.lower() != ".csv":
                raise ValueError(f"arena_file.default_option must point to a .csv file: {p}")
            return str(p)
        # If dict is present but lacks a proper default_option, it's invalid per spec
        raise ValueError("arena_file is a dict but missing a valid 'default_option' CSV path.")
    if isinstance(value, str) and value.strip():
        p = Path(value)
        if not p.exists():
            raise FileNotFoundError(f"arena_file path does not exist: {p}")
        if p.suffix.lower() != ".csv":
            raise ValueError(f"arena_file must point to a .csv file: {p}")
        return str(p)
    return None

def _resolve_params(config_path: str | Path | None, meta: Dict[str, Any] | None) -> dict:
    """
    Merge parameters from YAML and meta with priority to YAML when provided.
    Returns dict with keys:
      arena_surface, default_arena_file (CSV path or None), radius, communication_radius.
    """
    out = {
        "arena_surface": None,
        "default_arena_file": None,  # validated CSV path if available
        "radius": 26.5,
        "communication_radius": 80.0,
    }

    if config_path is not None:
        cfg = _load_yaml(config_path)
        out["arena_surface"] = float(cfg.get("arena_surface", out["arena_surface"] or 0.0) or 0.0)
        # arena_file can be dict with default_option or a string path
        out["default_arena_file"] = _extract_arena_csv_path(cfg.get("arena_file"))
        out["radius"] = float(_safe_get(cfg, ["objects", "robots", "radius"], out["radius"]))
        out["communication_radius"] = float(_safe_get(cfg, ["objects", "robots", "communication_radius"], out["communication_radius"]))
    elif meta is not None:
        cfg = meta.get("configuration", {})
        out["arena_surface"] = float(cfg.get("arena_surface", out["arena_surface"] or 0.0) or 0.0)
        out["default_arena_file"] = _extract_arena_csv_path(cfg.get("arena_file"))
        out["radius"] = float(_safe_get(cfg, ["objects", "robots", "radius"], out["radius"]))
        out["communication_radius"] = float(_safe_get(cfg, ["objects", "robots", "communication_radius"], out["communication_radius"]))

    if out["arena_surface"] is None or out["arena_surface"] <= 0:
        raise ValueError("arena_surface must be provided and > 0 (via --config or meta['configuration']).")

    return out


# Prefer the previously developed arena helpers; fallback to minimal local impl.
def _build_scaled_polygon_from_csv(csv_path: str | Path, arena_surface: float) -> Polygon:
    try:
        # Reuse your earlier module if available
        from arena_polygon_cli import build_scaled_arena_polygon  # type: ignore
        return build_scaled_arena_polygon(csv_path, arena_surface)
    except Exception:
        from shapely import affinity
        loops: list[list[tuple[float, float]]] = []
        cur: list[tuple[float, float]] = []
        with Path(csv_path).open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    if cur:
                        loops.append(cur); cur = []
                    continue
                x_str, y_str = s.split(",")
                cur.append((float(x_str), float(y_str)))
        if cur:
            loops.append(cur)
        if not loops:
            raise ValueError(f"No valid loops in arena CSV: {csv_path}")
        shell = loops[0]
        holes = loops[1:] if len(loops) > 1 else []
        poly = Polygon(shell, holes=holes)
        if not poly.is_valid:
            raise ValueError(f"Invalid polygon in CSV: {csv_path}")
        minx, miny, _, _ = poly.bounds
        poly = affinity.translate(poly, xoff=-minx, yoff=-miny)
        scale = (arena_surface / poly.area) ** 0.5
        poly = affinity.scale(poly, xfact=scale, yfact=scale, origin=(0, 0))
        return poly

def _load_arena_by_name(
    name: str,
    arena_surface: float,
    *,
    arenas_dir: str | Path = "arenas",
) -> tuple[Polygon, Tuple[float, float, float, float]]:
    """
    Load an arena polygon by short name using 'arenas/<name>.csv',
    translate to origin and scale to @p arena_surface.
    """
    csv_path = Path(arenas_dir) / f"{name}.csv"
    poly = _build_scaled_polygon_from_csv(csv_path, arena_surface)
    return poly, poly.bounds  # (minx, miny, maxx, maxy)

def _load_default_arena(
    default_arena_file: str | Path,
    arena_surface: float,
) -> tuple[Polygon, Tuple[float, float, float, float]]:
    poly = _build_scaled_polygon_from_csv(default_arena_file, arena_surface)
    return poly, poly.bounds


# ──────────────────────────────────────────────────────────────────────────────
# Voronoi helpers

def voronoi_finite_polygons_2d(vor: Voronoi, radius: float | None = None):
    """
    Reconstruct infinite Voronoi regions into finite polygons (2D only).
    Return (regions, vertices).
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions: list[list[int]] = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max()

    all_ridges: dict[int, list[tuple[int, int, int]]] = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        ang = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(ang)]
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics

def compute_voronoi_metrics(
    df: pd.DataFrame,
    arena_polygon: Polygon,
    arena_bounds: Tuple[float, float, float, float],
    arena_surface: float,
    *,
    communication_radius: float = 133.0,
) -> pd.DataFrame:
    """
    Compute per-time Voronoi cell area stats and coverage ratio.
    """
    bbox = box(*arena_bounds)
    results: list[tuple[int, float, float, float, float, float, int, float]] = []

    if "run" not in df.columns:
        df = df.copy()
        df["run"] = 0

    for run_id, run_df in df.groupby("run"):
        for t, tdf in run_df.groupby("time"):
            pts = tdf[["x", "y"]].to_numpy()
            if pts.shape[0] < 2:
                continue

            vor = Voronoi(pts)
            regions, vertices = voronoi_finite_polygons_2d(
                vor, radius=2 * max(arena_bounds[2] - arena_bounds[0], arena_bounds[3] - arena_bounds[1])
            )

            areas: list[float] = []
            for region in regions:
                poly = Polygon(vertices[region])
                if not poly.is_valid or poly.is_empty:
                    continue
                clipped = poly.intersection(arena_polygon)
                if not clipped.is_empty:
                    geoms = [clipped] if isinstance(clipped, Polygon) else list(clipped.geoms)
                    for g in geoms:
                        if not g.is_empty and g.is_valid:
                            areas.append(g.area)

            # coverage via union of clipped circles
            coverage_geoms = []
            for p in pts:
                circ = Point(p).buffer(communication_radius)
                clipped = circ.intersection(arena_polygon)
                if not clipped.is_empty:
                    coverage_geoms.append(clipped)
            covered_area = unary_union(coverage_geoms).area if coverage_geoms else 0.0

            if areas:
                mean_area = float(np.mean(areas))
                std_area  = float(np.std(areas))
                var_area  = float(np.var(areas))
                cv_area   = (std_area / mean_area) if mean_area > 0 else np.nan
                coverage_ratio = float(covered_area / arena_surface)
                results.append((int(run_id), float(t), mean_area, std_area, var_area, cv_area, len(areas), coverage_ratio))

    return pd.DataFrame(
        results,
        columns=["run", "time", "mean_area", "std_area", "var_area", "cv_area", "n_cells", "coverage_ratio"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Plots (per-arena; matplotlib only; save via utils.save_figure)

def _suffix_for(arena_name: str) -> str:
    return f"__arena_{arena_name}"

def _sanitize_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "arena"

def _suffix_for(label: str) -> str:
    return f"__arena_{_sanitize_label(label)}"

def _label_from_csv(csv_path: str | Path) -> str:
    return Path(csv_path).stem

def plot_voronoi_variance(df_voronoi: pd.DataFrame, out_dir: str | Path, arena_name: str) -> None:
    plt.figure(figsize=(10, 6))
    for run_id, group in df_voronoi.groupby("run"):
        plt.plot(group["time"], group["var_area"], label=f"Run {run_id}")
    plt.xlabel("Time")
    plt.ylabel("Voronoi Cell Area Variance")
    plt.title(f"Voronoi Cell Area Variance Over Time ({arena_name})")
    plt.grid(True); plt.legend(); plt.tight_layout()
    utils.save_figure(Path(out_dir) / f"voronoi_variance_over_time{_suffix_for(arena_name)}.pdf")

def plot_voronoi_std(df_voronoi: pd.DataFrame, out_dir: str | Path, arena_name: str) -> None:
    plt.figure(figsize=(10, 6))
    for run_id, group in df_voronoi.groupby("run"):
        plt.plot(group["time"], group["std_area"], label=f"Run {run_id}")
    plt.xlabel("Time"); plt.ylabel("Std-dev of cell area (mm²)")
    plt.title(f"Std-dev of Voronoi Cell Area Over Time ({arena_name})")
    plt.grid(True); plt.legend(); plt.tight_layout()
    utils.save_figure(Path(out_dir) / f"voronoi_std_over_time{_suffix_for(arena_name)}.pdf")

def plot_cv(df_voronoi: pd.DataFrame, out_dir: str | Path, arena_name: str) -> None:
    plt.figure(figsize=(10, 6))
    for run_id, group in df_voronoi.groupby("run"):
        plt.plot(group["time"], group["cv_area"], label=f"Run {run_id}")
    plt.xlabel("Time"); plt.ylabel("Coefficient of Variation (CV)")
    plt.title(f"Voronoi Cell Area CV Over Time ({arena_name})")
    plt.grid(True); plt.legend(); plt.tight_layout()
    utils.save_figure(Path(out_dir) / f"voronoi_cv_over_time{_suffix_for(arena_name)}.pdf")

def plot_voronoi_global_variance(df_voronoi: pd.DataFrame, out_dir: str | Path, arena_name: str) -> None:
    grouped = df_voronoi.groupby("time")
    gmean = grouped["var_area"].mean()
    gstd  = grouped["var_area"].std()

    plt.figure(figsize=(10, 6))
    plt.plot(gmean.index, gmean.values, label="Global mean")
    plt.fill_between(gmean.index, gmean - gstd, gmean + gstd, alpha=0.3, label="±1 stddev")
    plt.xlabel("Time"); plt.ylabel("Voronoi Cell Area Variance")
    plt.title(f"Global Average of Voronoi Cell Area Variance Over Time ({arena_name})")
    plt.grid(True); plt.legend(); plt.tight_layout()
    utils.save_figure(Path(out_dir) / f"voronoi_global_var_over_time{_suffix_for(arena_name)}.pdf")

def plot_coverage_ratio(df_voronoi: pd.DataFrame, out_dir: str | Path, arena_name: str) -> None:
    grouped = df_voronoi.groupby("time")
    gmean = grouped["coverage_ratio"].mean()
    gstd  = grouped["coverage_ratio"].std()

    ymin = float(df_voronoi["coverage_ratio"].min()) if not df_voronoi.empty else 0.0
    ymax = float(df_voronoi["coverage_ratio"].max()) if not df_voronoi.empty else 1.0
    pad  = 0.01 * max(1.0, abs(ymax - ymin))

    plt.figure(figsize=(10, 6))
    plt.plot(gmean.index, gmean.values, label="Coverage ratio mean")
    plt.fill_between(gmean.index, gmean - gstd, gmean + gstd, alpha=0.3, label="±1 stddev")
    plt.ylim(ymin - pad, ymax + pad)
    plt.xlabel("Time"); plt.ylabel("Coverage ratio")
    plt.title(f"Communication Coverage Over Time ({arena_name})")
    plt.grid(True); plt.legend(); plt.tight_layout()
    utils.save_figure(Path(out_dir) / f"coverage_ratio_over_time{_suffix_for(arena_name)}.pdf")

def plot_voronoi_diagram(
    df: pd.DataFrame,
    out_dir: str | Path,
    arena_polygon: Polygon,
    arena_bounds: Tuple[float, float, float, float],
    time_point: float,
    *,
    run_id: int = 0,
    communication_radius: float = 133.0,
    figsize: tuple[int, int] = (10, 10),
    arena_name: str = "default",
):
    """
    Single time-slice Voronoi diagram, clipped to the arena.
    """
    if "run" not in df.columns:
        df = df.copy(); df["run"] = 0
    run_data = df[df["run"] == run_id]
    pts = run_data[run_data["time"] == time_point][["x", "y"]].to_numpy()
    if pts.shape[0] < 3:
        print(f"Not enough points at time {time_point} for run {run_id} ({arena_name})")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    ax.fill(*arena_polygon.exterior.xy, alpha=0.2, label="Arena")
    ax.plot(*arena_polygon.exterior.xy)

    vor = Voronoi(pts)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    colors = plt.cm.Set3(np.linspace(0, 1, len(pts)))
    for region, color in zip(regions, colors):
        poly = Polygon(vertices[region])
        if not poly.is_valid or poly.is_empty:
            continue
        clipped = poly.intersection(arena_polygon)
        if clipped.is_empty:
            continue
        geoms = [clipped] if isinstance(clipped, Polygon) else list(clipped.geoms)
        for g in geoms:
            if g.is_empty or not g.is_valid:
                continue
            x, y = g.exterior.xy
            ax.fill(x, y, alpha=0.6, edgecolor="black", linewidth=0.8, color=color)

    for p in pts:
        circ = Point(p).buffer(communication_radius).intersection(arena_polygon)
        geoms = [circ] if isinstance(circ, Polygon) else list(circ.geoms)
        for g in geoms:
            x, y = g.exterior.xy
            ax.plot(x, y, linestyle="--", linewidth=0.8, alpha=0.6)

    ax.scatter(pts[:, 0], pts[:, 1], zorder=10, linewidth=0.8)

    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    ax.set_title(f"Voronoi Diagram at t={time_point:.2f} (Run {run_id}) — {arena_name}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    utils.save_figure(Path(out_dir) / f"voronoi_diagram_run{run_id}_t{time_point:.2f}{_suffix_for(arena_name)}.pdf")
    return fig, ax

def plot_voronoi_evolution(
    df: pd.DataFrame,
    out_dir: str | Path,
    arena_polygon: Polygon,
    arena_bounds: Tuple[float, float, float, float],
    time_points: list[float],
    *,
    run_id: int = 0,
    communication_radius: float = 133.0,
    figsize: tuple[int, int] = (16, 12),
    arena_name: str = "default",
):
    """
    Grid of Voronoi diagrams at multiple time instants.
    """
    if "run" not in df.columns:
        df = df.copy(); df["run"] = 0
    run_df = df[df["run"] == run_id]

    n = len(time_points)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n == 1:
        axes = np.array([axes])
    axes = axes.reshape(rows, cols)

    for i, t in enumerate(time_points):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        pts = run_df[run_df["time"] == t][["x", "y"]].to_numpy()
        if pts.shape[0] < 3:
            ax.text(0.5, 0.5, f"Not enough points\nat time {t:.2f}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Time {t:.2f}")
            ax.axis("off")
            continue

        ax.fill(*arena_polygon.exterior.xy, alpha=0.2)
        ax.plot(*arena_polygon.exterior.xy)

        vor = Voronoi(pts)
        regions, vertices = voronoi_finite_polygons_2d(vor)
        colors = plt.cm.Set3(np.linspace(0, 1, len(pts)))
        for region, color in zip(regions, colors):
            poly = Polygon(vertices[region])
            if not poly.is_valid or poly.is_empty:
                continue
            clipped = poly.intersection(arena_polygon)
            if clipped.is_empty:
                continue
            geoms = [clipped] if isinstance(clipped, Polygon) else list(clipped.geoms)
            for g in geoms:
                if g.is_empty or not g.is_valid:
                    continue
                x, y = g.exterior.xy
                ax.fill(x, y, alpha=0.6, edgecolor="black", linewidth=0.8, color=color)

        ax.scatter(pts[:, 0], pts[:, 1], s=20, linewidth=0.6)
        ax.set_aspect("equal")
        ax.set_title(f"Time {t:.1f}  (n={len(pts)})")
        ax.grid(True, alpha=0.3)

        if r == rows - 1:
            ax.set_xlabel("X (mm)")
        if c == 0:
            ax.set_ylabel("Y (mm)")

    # Hide unused axes
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].set_visible(False)

    plt.tight_layout()
    utils.save_figure(Path(out_dir) / f"voronoi_diagram_evolution_run{run_id}{_suffix_for(arena_name)}.pdf")
    return fig, axes


# ──────────────────────────────────────────────────────────────────────────────
# Orchestration

def create_all_coverage_plots(
    input_file: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    arenas_dir: str | Path = "arenas",
    run_for_single_figs: int | None = None,
) -> None:
    """
    Load data & arenas, compute Voronoi metrics per-arena, save all figures.

    If the dataframe has an 'arena_file' column, each distinct non-empty value 'X'
    loads 'arenas/X.csv' (scaled to the unique arena_surface from config) and plots
    are emitted with a '__arena_X' suffix. Rows with blank/NaN arena_file fall
    back to the default arena path resolved from configuration.

    If there is **no 'arena_file' column**, use the default arena path resolved
    from configuration (priority to config_path, else meta['configuration']),
    where 'arena_file' may be:
      - dict with 'default_option' → valid CSV path
      - string → valid CSV path
    """
    os.makedirs(output_dir, exist_ok=True)

    # Data
    df, meta = utils.load_dataframe(input_file)
    if "run" not in df.columns:
        df["run"] = 0

    # Resolve parameters (YAML first if provided)
    params = _resolve_params(config_path, meta)
    arena_surface = float(params["arena_surface"])
    radius = float(params["radius"])
    comm_r = float(params["communication_radius"]) + 2.0 * radius  # same heuristic as neighbors

    # Build arena lookup (+ display labels)
    arena_map: dict[str, tuple[Polygon, Tuple[float, float, float, float]]] = {}
    arena_labels: dict[str, str] = {}

    if "arena_file" in df.columns:
        arena_series = df["arena_file"].astype("string")
        has_blank = arena_series.isna() | (arena_series.str.strip() == "")

        # Named arenas
        arena_names = sorted(set(arena_series[~has_blank].dropna().tolist()))
        for name in arena_names:
            poly, bounds = _load_arena_by_name(str(name), arena_surface, arenas_dir=arenas_dir)
            arena_map[str(name)] = (poly, bounds)
            arena_labels[str(name)] = str(name)  # display exactly the provided name

        # Default only if needed
        if has_blank.any():
            if params["default_arena_file"] is None:
                raise ValueError(
                    "Some rows have empty arena_file but configuration has no valid default CSV "
                    "(arena_file as dict with 'default_option' or a string path)."
                )
            poly_def, bounds_def = _load_default_arena(params["default_arena_file"], arena_surface)
            arena_map["_default_"] = (poly_def, bounds_def)
            arena_labels["_default_"] = _label_from_csv(params["default_arena_file"])

        # Normalize key on dataframe
        arena_series = arena_series.fillna("_default_").replace({"": "_default_"})
        df = df.assign(_arena_key=arena_series)

    else:
        # No column; single default is required
        if params["default_arena_file"] is None:
            raise ValueError(
                "No 'arena_file' column and configuration has no valid default CSV "
                "(arena_file as dict with 'default_option' or a string path)."
            )
        poly, bounds = _load_default_arena(params["default_arena_file"], arena_surface)
        arena_map["_default_"] = (poly, bounds)
        arena_labels["_default_"] = _label_from_csv(params["default_arena_file"])
        df = df.assign(_arena_key="_default_")

    # Compute metrics per arena and plot
    results: list[pd.DataFrame] = []

    for arena_key, gdf in df.groupby("_arena_key"):
        if arena_key not in arena_map:
            raise ValueError(f"Arena '{arena_key}' referenced in dataframe has no loaded geometry.")
        poly, bounds = arena_map[arena_key]
        label = arena_labels.get(arena_key, str(arena_key))  # e.g., "disk" for default

        df_vor = compute_voronoi_metrics(gdf, poly, bounds, arena_surface, communication_radius=comm_r)
        df_vor = df_vor.assign(arena_name=str(label))
        results.append(df_vor)

        if df_vor.empty:
            continue

        # Plots for this arena (use the label, not the internal key)
        plot_voronoi_variance(df_vor, output_dir, str(label))
        plot_voronoi_std(df_vor, output_dir, str(label))
        plot_cv(df_vor, output_dir, str(label))
        plot_voronoi_global_variance(df_vor, output_dir, str(label))
        plot_coverage_ratio(df_vor, output_dir, str(label))

        # Single & multi-time diagrams
        times = sorted(gdf["time"].unique().tolist())
        single_run = run_for_single_figs if run_for_single_figs is not None else int(gdf["run"].iloc[0])
        if times:
            mid_t = times[len(times) // 2]
            plot_voronoi_diagram(
                gdf, output_dir, poly, bounds, mid_t,
                run_id=single_run, communication_radius=comm_r, arena_name=str(label),
            )
            sample_ids = [0, len(times)//4, len(times)//2, 3*len(times)//4, len(times)-1]
            sample_times = [times[i] for i in sorted(set(sample_ids)) if 0 <= i < len(times)]
            plot_voronoi_evolution(
                gdf, output_dir, poly, bounds, sample_times,
                run_id=single_run, communication_radius=comm_r, arena_name=str(label),
            )


    # Persist combined metrics
    if results:
        combined = pd.concat(results, ignore_index=True)
        combined.to_csv(Path(output_dir) / "voronoi_metrics_per_arena.csv", index=False)


# ──────────────────────────────────────────────────────────────────────────────
# CLI

def parse_args(argv: list[str] | None = None):
    import argparse
    p = argparse.ArgumentParser(description="Coverage/Voronoi analysis & figures (per-arena).")
    p.add_argument("-i", "--inputFile", type=Path, default=Path("results/result.feather"),
                   help="Path of the input feather/parquet file")
    p.add_argument("-o", "--outputDir", type=Path, default=Path("."),
                   help="Directory for output figures")
    p.add_argument("-c", "--config", type=Path, default=None,
                   help="YAML with arena_surface and arena_file (preferred over meta)")
    p.add_argument("--arenas-dir", type=Path, default=Path("arenas"),
                   help="Directory that contains <name>.csv arena files for arena_file column")
    p.add_argument("--run", type=int, default=None,
                   help="Run id to use for single-time diagrams")
    return p.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    create_all_coverage_plots(
        args.inputFile,
        args.outputDir,
        config_path=args.config,
        arenas_dir=args.arenas_dir,
        run_for_single_figs=args.run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# MODELINE "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
