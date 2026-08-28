#!/usr/bin/env python3
"""
Reproduce the main magnetometer-calibration plots from the Pogobot August 2026
report using Pogosim outputs.

Inputs
------
- Pogosim stdout/stderr dump from the cooperative magnetometer-calibration
  firmware. The dump must contain:
    CALPT,...
    CALIB_MEAN,...
    CALIB_E1,...
    CALIB_E2,...
    CALIB_C2D,...
    CALIB_W2,...
    CALIB_AXES,...
    live ANGLE,... / MAG,... pairs
- Pogosim data.feather with at least:
    time, robot_id, angle

Outputs
-------
The output directory contains:
- calibration_summary.png
- calibration_summary.pdf
- 01_angle_comparison.png
- 02_error_vs_iteration.png
- 03_error_vs_true_angle.png
- 04_raw_3d_plane_axes.png
- 05_projected_ellipse.png
- 06_corrected_circle.png
- mission_alignment.csv

The six-panel summary follows the same logic as Figure 1 of the real-Pogobot
report:
1. calibrated magnetometer angle vs ground-truth angle,
2. signed angular error vs measurement index,
3. signed angular error vs true angle,
4. raw 3-D calibration cloud + fitted plane/basis,
5. projected cloud + fitted ellipse,
6. ellipse-corrected circle, colored by out-of-plane residual.

The firmware heading has an arbitrary constant phase and may also use the
opposite angular sign from the simulator robot orientation. For the top-row
comparisons, this script finds the best mapping

    robot_angle ~= sign * firmware_angle + phase

with sign in {-1, +1}, then reports errors after this single global phase/sign
alignment. The bottom-row plots use the actual CALIB_* parameters dumped by
the firmware.

Dependencies
------------
Python 3, numpy, pandas, matplotlib, plus either:
- pyarrow (for pandas.read_feather), or
- an installed pogosim package providing pogosim.utils.load_dataframe.

Example
-------
./scripts/plot_magnetometer_simulation_report.py \
    --dump tmp_magneto/dump_pogosim2.txt \
    --data tmp_magneto/data2.feather \
    --robot-id 0 \
    -o tmp_magneto/sim_report
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PRINTF_RE = re.compile(
    r"\[ROBOT #(?P<robot_id>\d+)\]\s*"
    r"\[PRINTF\]\s*(?P<payload>.*)$"
)


@dataclass
class MagSample:
    firmware_angle_deg: float
    mag_x: int
    mag_y: int
    mag_z: int


@dataclass
class StationaryPlateau:
    start_time_s: float
    end_time_s: float
    angle_deg: float

    @property
    def center_time_s(self) -> float:
        return 0.5 * (self.start_time_s + self.end_time_s)

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s


@dataclass
class Calibration:
    mean: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    u0: float
    v0: float
    w2: np.ndarray
    el1: float
    el2: float
    ev1x: float
    ev1y: float
    k_norm: float
    s_norm: float


@dataclass
class HeadingAlignment:
    sign: int
    phase_deg: float
    adjusted_angle_deg: np.ndarray
    signed_error_deg: np.ndarray


def wrap_deg(value: np.ndarray | float) -> np.ndarray | float:
    return np.mod(value, 360.0)


def signed_circular_difference_deg(
    angle_a: np.ndarray | float,
    angle_b: np.ndarray | float,
) -> np.ndarray:
    a = np.asarray(angle_a, dtype=float)
    b = np.asarray(angle_b, dtype=float)
    return (a - b + 180.0) % 360.0 - 180.0


def circular_distance_deg(
    angle_a: np.ndarray | float,
    angle_b: np.ndarray | float,
) -> np.ndarray:
    return np.abs(
        signed_circular_difference_deg(angle_a, angle_b)
    )


def circular_mean_deg(values_deg: np.ndarray) -> float:
    values_rad = np.deg2rad(
        np.asarray(values_deg, dtype=float)
    )
    vector = np.mean(np.exp(1j * values_rad))
    if abs(vector) < 1.0e-12:
        raise ValueError("circular mean is undefined")
    return float(np.rad2deg(np.angle(vector)) % 360.0)


def read_robot_payloads(
    dump_path: Path,
    robot_id: int,
) -> list[str]:
    payloads: list[str] = []

    for raw_line in dump_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        match = PRINTF_RE.search(raw_line)
        if match is None:
            continue

        if int(match.group("robot_id")) != robot_id:
            continue

        payloads.append(match.group("payload").strip())

    if not payloads:
        raise ValueError(
            f"no [ROBOT #{robot_id}] [PRINTF] lines found in "
            f"{dump_path}"
        )

    return payloads


def parse_calibration_points(
    payloads: list[str],
) -> np.ndarray:
    try:
        first_calib_mean = next(
            index
            for index, payload in enumerate(payloads)
            if payload.startswith("CALIB_MEAN,")
        )
    except StopIteration as exc:
        raise ValueError(
            "CALIB_MEAN was not found in the Pogosim dump"
        ) from exc

    points: list[list[float]] = []

    for payload in payloads[:first_calib_mean]:
        if not payload.startswith("CALPT,"):
            continue

        fields = payload.split(",")
        if len(fields) != 4:
            continue

        points.append(
            [
                float(fields[1]),
                float(fields[2]),
                float(fields[3]),
            ]
        )

    if not points:
        raise ValueError("no CALPT records found before CALIB_MEAN")

    return np.asarray(points, dtype=float)


def decode_float_words(words: list[int]) -> list[float]:
    if len(words) % 2 != 0:
        raise ValueError(
            "encoded float record contains an odd number of uint16 words"
        )

    decoded: list[float] = []

    for index in range(0, len(words), 2):
        packed = struct.pack(
            "<HH",
            words[index],
            words[index + 1],
        )
        decoded.append(struct.unpack("<f", packed)[0])

    return decoded


def first_tag_values(
    payloads: list[str],
    tag: str,
) -> list[float]:
    prefix = f"{tag},"

    for payload in payloads:
        if not payload.startswith(prefix):
            continue

        words = [
            int(value)
            for value in payload.split(",")[1:]
        ]
        return decode_float_words(words)

    raise ValueError(f"{tag} not found in dump")


def parse_calibration(
    payloads: list[str],
) -> Calibration:
    mean = np.asarray(
        first_tag_values(payloads, "CALIB_MEAN"),
        dtype=float,
    )
    e1 = np.asarray(
        first_tag_values(payloads, "CALIB_E1"),
        dtype=float,
    )
    e2 = np.asarray(
        first_tag_values(payloads, "CALIB_E2"),
        dtype=float,
    )
    c2d = first_tag_values(payloads, "CALIB_C2D")
    w2_values = first_tag_values(payloads, "CALIB_W2")
    axes = first_tag_values(payloads, "CALIB_AXES")

    if len(mean) != 3 or len(e1) != 3 or len(e2) != 3:
        raise ValueError("invalid CALIB_MEAN/E1/E2 record length")
    if len(c2d) != 2:
        raise ValueError("invalid CALIB_C2D record length")
    if len(w2_values) != 4:
        raise ValueError("invalid CALIB_W2 record length")
    if len(axes) != 6:
        raise ValueError("invalid CALIB_AXES record length")

    return Calibration(
        mean=mean,
        e1=e1,
        e2=e2,
        u0=float(c2d[0]),
        v0=float(c2d[1]),
        w2=np.asarray(w2_values, dtype=float).reshape(2, 2),
        el1=float(axes[0]),
        el2=float(axes[1]),
        ev1x=float(axes[2]),
        ev1y=float(axes[3]),
        k_norm=float(axes[4]),
        s_norm=float(axes[5]),
    )


def parse_live_mission_samples(
    payloads: list[str],
) -> list[MagSample]:
    calib_mean_indices = [
        index
        for index, payload in enumerate(payloads)
        if payload.startswith("CALIB_MEAN,")
    ]

    if not calib_mean_indices:
        raise ValueError("CALIB_MEAN not found")

    start = calib_mean_indices[0]

    if len(calib_mean_indices) >= 2:
        stop = calib_mean_indices[1]
    else:
        stop = len(payloads)

    mission_payloads = payloads[start:stop]

    samples: list[MagSample] = []
    index = 0

    while index + 1 < len(mission_payloads):
        angle_payload = mission_payloads[index]
        mag_payload = mission_payloads[index + 1]

        if (
            angle_payload.startswith("ANGLE,")
            and mag_payload.startswith("MAG,")
        ):
            angle_fields = angle_payload.split(",")
            mag_fields = mag_payload.split(",")

            if len(angle_fields) == 2 and len(mag_fields) == 4:
                samples.append(
                    MagSample(
                        firmware_angle_deg=float(angle_fields[1]),
                        mag_x=int(mag_fields[1]),
                        mag_y=int(mag_fields[2]),
                        mag_z=int(mag_fields[3]),
                    )
                )
                index += 2
                continue

        index += 1

    if not samples:
        raise ValueError(
            "no live ANGLE/MAG mission pairs found between the "
            "live calibration block and final flash dump"
        )

    return samples


def load_feather(path: Path) -> tuple[Any, dict[str, Any]]:
    try:
        import pogosim.utils as pogosim_utils

        loaded = pogosim_utils.load_dataframe(str(path))
        if (
            isinstance(loaded, tuple)
            and len(loaded) == 2
        ):
            frame, metadata = loaded
            if not isinstance(metadata, dict):
                metadata = {}
            return frame, metadata
        return loaded, {}
    except Exception:
        pass

    import pandas as pd

    try:
        return pd.read_feather(path), {}
    except ImportError as exc:
        raise RuntimeError(
            "could not load the Feather file. Install pyarrow or "
            "run inside an environment with pogosim.utils.load_dataframe"
        ) from exc


def infer_angle_unit(
    values: np.ndarray,
    requested_unit: str,
) -> str:
    if requested_unit != "auto":
        return requested_unit

    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        raise ValueError("Feather angle column is empty")

    if float(np.max(finite)) <= 2.0 * math.pi + 0.25:
        return "radians"

    return "degrees"


def detect_stationary_plateaus(
    times_s: np.ndarray,
    angles_deg: np.ndarray,
    speed_threshold_deg_s: float,
    min_duration_s: float,
) -> list[StationaryPlateau]:
    if len(times_s) != len(angles_deg):
        raise ValueError("time/angle length mismatch")
    if len(times_s) < 2:
        return []

    dt = np.diff(times_s)

    if np.any(dt <= 0.0):
        raise ValueError(
            "Feather time values must be strictly increasing"
        )

    angular_steps = circular_distance_deg(
        angles_deg[1:],
        angles_deg[:-1],
    )
    angular_speed = angular_steps / dt
    stationary_edges = (
        angular_speed <= speed_threshold_deg_s
    )

    plateaus: list[StationaryPlateau] = []

    edge = 0
    while edge < len(stationary_edges):
        if not stationary_edges[edge]:
            edge += 1
            continue

        first_edge = edge

        while (
            edge + 1 < len(stationary_edges)
            and stationary_edges[edge + 1]
        ):
            edge += 1

        last_edge = edge
        first_row = first_edge
        last_row = last_edge + 1

        start_time_s = float(times_s[first_row])
        end_time_s = float(times_s[last_row])

        if end_time_s - start_time_s >= min_duration_s:
            plateau_angle = circular_mean_deg(
                angles_deg[first_row : last_row + 1]
            )
            plateaus.append(
                StationaryPlateau(
                    start_time_s=start_time_s,
                    end_time_s=end_time_s,
                    angle_deg=plateau_angle,
                )
            )

        edge += 1

    return plateaus


def get_mission_ground_truth(
    feather_path: Path,
    robot_id: int,
    calibration_point_count: int,
    mission_count: int,
    data_angle_unit: str,
    stationary_speed_threshold_deg_s: float,
    min_plateau_duration_s: float,
    mission_plateau_start: int | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[StationaryPlateau],
    int,
]:
    frame, _ = load_feather(feather_path)

    required = {"time", "robot_id", "angle"}
    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"{feather_path} is missing columns: "
            f"{sorted(missing)}"
        )

    robot_frame = frame.loc[
        frame["robot_id"] == robot_id,
        ["time", "angle"],
    ].copy()

    robot_frame = robot_frame.dropna().sort_values("time")

    if len(robot_frame) < 3:
        raise ValueError(
            f"not enough Feather rows for robot {robot_id}"
        )

    times_s = robot_frame["time"].to_numpy(dtype=float)
    raw_angles = robot_frame["angle"].to_numpy(dtype=float)

    angle_unit = infer_angle_unit(
        raw_angles,
        data_angle_unit,
    )

    if angle_unit == "radians":
        angles_deg = np.rad2deg(raw_angles)
    else:
        angles_deg = raw_angles.copy()

    angles_deg = wrap_deg(angles_deg)

    plateaus = detect_stationary_plateaus(
        times_s=times_s,
        angles_deg=angles_deg,
        speed_threshold_deg_s=(
            stationary_speed_threshold_deg_s
        ),
        min_duration_s=min_plateau_duration_s,
    )

    if mission_plateau_start is None:
        # In the current calibration firmware, mission point 0 is
        # measured without an extra motor step after the final CALPT.
        # It therefore shares the final calibration plateau.
        start_index = calibration_point_count - 1
    else:
        start_index = mission_plateau_start

    stop_index = start_index + mission_count

    if start_index < 0 or stop_index > len(plateaus):
        raise ValueError(
            f"mission requires plateaus [{start_index}, "
            f"{stop_index - 1}], but only {len(plateaus)} "
            "stationary plateaus were detected. Adjust "
            "--stationary-speed-threshold-deg-s, "
            "--min-plateau-duration-s, or "
            "--mission-plateau-start."
        )

    selected = plateaus[start_index:stop_index]

    true_angles_deg = np.asarray(
        [plateau.angle_deg for plateau in selected],
        dtype=float,
    )
    sample_times_s = np.asarray(
        [plateau.center_time_s for plateau in selected],
        dtype=float,
    )

    return (
        true_angles_deg,
        sample_times_s,
        plateaus,
        start_index,
    )


def fit_heading_alignment(
    firmware_angles_deg: np.ndarray,
    true_angles_deg: np.ndarray,
) -> HeadingAlignment:
    if len(firmware_angles_deg) != len(true_angles_deg):
        raise ValueError(
            "firmware-angle / true-angle length mismatch"
        )

    best: tuple[
        float,
        float,
        int,
        float,
        np.ndarray,
        np.ndarray,
    ] | None = None

    for sign in (-1, 1):
        signed_firmware = wrap_deg(
            sign * firmware_angles_deg
        )

        phase_deg = circular_mean_deg(
            wrap_deg(true_angles_deg - signed_firmware)
        )

        adjusted = wrap_deg(
            signed_firmware + phase_deg
        )

        errors = signed_circular_difference_deg(
            adjusted,
            true_angles_deg,
        )

        median_abs = float(np.median(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors * errors)))

        candidate = (
            median_abs,
            rmse,
            sign,
            phase_deg,
            adjusted,
            errors,
        )

        if best is None or candidate[:2] < best[:2]:
            best = candidate

    assert best is not None

    _, _, sign, phase_deg, adjusted, errors = best

    if phase_deg >= 180.0:
        phase_deg -= 360.0

    return HeadingAlignment(
        sign=sign,
        phase_deg=float(phase_deg),
        adjusted_angle_deg=adjusted,
        signed_error_deg=errors,
    )


def projected_calibration_points(
    calibration_points: np.ndarray,
    calibration: Calibration,
) -> tuple[np.ndarray, np.ndarray]:
    centered = calibration_points - calibration.mean
    u = centered @ calibration.e1
    v = centered @ calibration.e2
    return u, v


def ellipse_geometry(
    calibration: Calibration,
    sample_count: int = 720,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
    if (
        calibration.el1 <= 0.0
        or calibration.el2 <= 0.0
        or calibration.k_norm <= 0.0
    ):
        raise ValueError(
            "invalid fitted ellipse eigenvalues/k_norm"
        )

    short_axis = (
        calibration.s_norm
        * math.sqrt(
            calibration.k_norm / calibration.el1
        )
    )
    long_axis = (
        calibration.s_norm
        * math.sqrt(
            calibration.k_norm / calibration.el2
        )
    )

    axis_1 = np.asarray(
        [calibration.ev1x, calibration.ev1y],
        dtype=float,
    )
    axis_1 /= np.linalg.norm(axis_1)

    axis_2 = np.asarray(
        [-axis_1[1], axis_1[0]],
        dtype=float,
    )

    t = np.linspace(
        0.0,
        2.0 * math.pi,
        sample_count,
    )

    ellipse = (
        np.asarray(
            [calibration.u0, calibration.v0],
            dtype=float,
        )[:, None]
        + short_axis
        * axis_1[:, None]
        * np.cos(t)[None, :]
        + long_axis
        * axis_2[:, None]
        * np.sin(t)[None, :]
    )

    return (
        ellipse[0],
        ellipse[1],
        axis_1,
        axis_2,
        short_axis,
        long_axis,
    )


def corrected_circle_points(
    u: np.ndarray,
    v: np.ndarray,
    calibration: Calibration,
) -> tuple[np.ndarray, np.ndarray]:
    centered = np.vstack(
        (
            u - calibration.u0,
            v - calibration.v0,
        )
    )
    corrected = calibration.w2 @ centered
    return corrected[0], corrected[1]


def equalize_3d_axes(ax: Any, points: np.ndarray) -> None:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))

    if radius <= 0.0:
        radius = 1.0

    ax.set_xlim(
        centers[0] - radius,
        centers[0] + radius,
    )
    ax.set_ylim(
        centers[1] - radius,
        centers[1] + radius,
    )
    ax.set_zlim(
        centers[2] - radius,
        centers[2] + radius,
    )


def make_plots(
    output_dir: Path,
    calibration_points: np.ndarray,
    calibration: Calibration,
    samples: list[MagSample],
    true_angles_deg: np.ndarray,
    sample_times_s: np.ndarray,
    plateaus: list[StationaryPlateau],
    plateau_start_index: int,
    dpi: int,
    save_individual: bool,
) -> dict[str, float | int]:
    import matplotlib.pyplot as plt

    firmware_angles = np.asarray(
        [sample.firmware_angle_deg for sample in samples],
        dtype=float,
    )

    alignment = fit_heading_alignment(
        firmware_angles,
        true_angles_deg,
    )

    errors = alignment.signed_error_deg
    median_abs_error = float(
        np.median(np.abs(errors))
    )
    rmse = float(
        np.sqrt(np.mean(errors * errors))
    )
    max_abs_error = float(
        np.max(np.abs(errors))
    )

    u, v = projected_calibration_points(
        calibration_points,
        calibration,
    )

    (
        ellipse_u,
        ellipse_v,
        ellipse_axis_1,
        ellipse_axis_2,
        short_axis,
        long_axis,
    ) = ellipse_geometry(calibration)

    corrected_x, corrected_y = corrected_circle_points(
        u,
        v,
        calibration,
    )

    plane_normal = np.cross(
        calibration.e1,
        calibration.e2,
    )
    normal_norm = np.linalg.norm(plane_normal)

    if normal_norm <= 1.0e-12:
        raise ValueError(
            "CALIB_E1 and CALIB_E2 do not define a valid plane"
        )

    plane_normal /= normal_norm

    out_of_plane = (
        (calibration_points - calibration.mean)
        @ plane_normal
    )

    radius = (
        calibration.s_norm
        * math.sqrt(calibration.k_norm)
    )

    ellipse_center_3d = (
        calibration.mean
        + calibration.u0 * calibration.e1
        + calibration.v0 * calibration.e2
    )

    axis_1_3d = (
        ellipse_axis_1[0] * calibration.e1
        + ellipse_axis_1[1] * calibration.e2
    )
    axis_2_3d = (
        ellipse_axis_2[0] * calibration.e1
        + ellipse_axis_2[1] * calibration.e2
    )

    figure = plt.figure(figsize=(14, 9))

    ax1 = figure.add_subplot(2, 3, 1)
    ax2 = figure.add_subplot(2, 3, 2)
    ax3 = figure.add_subplot(2, 3, 3)
    ax4 = figure.add_subplot(2, 3, 4, projection="3d")
    ax5 = figure.add_subplot(2, 3, 5)
    ax6 = figure.add_subplot(2, 3, 6)

    # 1. Ground truth vs calibrated angle.
    ax1.scatter(
        true_angles_deg,
        alignment.adjusted_angle_deg,
        s=16,
    )
    ax1.plot(
        [0.0, 360.0],
        [0.0, 360.0],
        linestyle="--",
    )
    ax1.set_xlim(0.0, 360.0)
    ax1.set_ylim(0.0, 360.0)
    ax1.set_xlabel("Ground-truth robot angle (deg)")
    ax1.set_ylabel("Calibrated magnetometer angle (deg)")
    ax1.set_title("Sensor comparison")
    ax1.grid(True, alpha=0.3)

    # 2. Error by iteration.
    indices = np.arange(len(errors))
    ax2.scatter(
        indices,
        errors,
        s=16,
    )
    ax2.axhline(0.0, linestyle="--")
    ax2.set_xlabel("Measurement index")
    ax2.set_ylabel("Signed angular error (deg)")
    ax2.set_title("Error vs measurement index")
    ax2.grid(True, alpha=0.3)

    stats_text = (
        f"Median |error| = {median_abs_error:.2f} deg\n"
        f"RMSE = {rmse:.2f} deg\n"
        f"Max |error| = {max_abs_error:.2f} deg\n"
        f"mapping: theta = "
        f"{alignment.sign:+d}*ANGLE "
        f"{alignment.phase_deg:+.2f} deg"
    )
    ax2.text(
        0.02,
        0.98,
        stats_text,
        transform=ax2.transAxes,
        va="top",
        ha="left",
        fontsize=8,
    )

    # 3. Error vs true angle.
    ax3.scatter(
        true_angles_deg,
        errors,
        s=16,
    )
    ax3.axhline(0.0, linestyle="--")
    ax3.set_xlim(0.0, 360.0)
    ax3.set_xlabel("Ground-truth robot angle (deg)")
    ax3.set_ylabel("Signed angular error (deg)")
    ax3.set_title("Error vs true angle")
    ax3.grid(True, alpha=0.3)

    # 4. Raw 3-D point cloud + fitted basis.
    ax4.scatter(
        calibration_points[:, 0],
        calibration_points[:, 1],
        calibration_points[:, 2],
        s=14,
        alpha=0.75,
        label="Calibration samples",
    )
    ax4.scatter(
        [ellipse_center_3d[0]],
        [ellipse_center_3d[1]],
        [ellipse_center_3d[2]],
        marker="*",
        s=80,
        label="Ellipse center",
    )

    for axis_vector, half_length, label in (
        (
            axis_1_3d,
            short_axis,
            "Ellipse axis 1",
        ),
        (
            axis_2_3d,
            long_axis,
            "Ellipse axis 2",
        ),
    ):
        p0 = (
            ellipse_center_3d
            - half_length * axis_vector
        )
        p1 = (
            ellipse_center_3d
            + half_length * axis_vector
        )
        ax4.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            linewidth=2,
            label=label,
        )

    normal_length = 0.75 * max(
        short_axis,
        long_axis,
    )
    normal_end = (
        ellipse_center_3d
        + normal_length * plane_normal
    )
    ax4.plot(
        [
            ellipse_center_3d[0],
            normal_end[0],
        ],
        [
            ellipse_center_3d[1],
            normal_end[1],
        ],
        [
            ellipse_center_3d[2],
            normal_end[2],
        ],
        linewidth=2,
        label="Plane normal",
    )

    plane_extent = 1.05 * max(
        short_axis,
        long_axis,
    )
    plane_coords = np.linspace(
        -plane_extent,
        plane_extent,
        2,
    )
    plane_a, plane_b = np.meshgrid(
        plane_coords,
        plane_coords,
    )
    plane_xyz = (
        ellipse_center_3d[:, None, None]
        + calibration.e1[:, None, None]
        * plane_a[None, :, :]
        + calibration.e2[:, None, None]
        * plane_b[None, :, :]
    )
    ax4.plot_surface(
        plane_xyz[0],
        plane_xyz[1],
        plane_xyz[2],
        alpha=0.08,
    )

    ax4.set_xlabel("Mag X")
    ax4.set_ylabel("Mag Y")
    ax4.set_zlabel("Mag Z")
    ax4.set_title("3-D raw cloud, fitted plane and axes")
    ax4.legend(fontsize=7, loc="best")
    equalize_3d_axes(ax4, calibration_points)

    # 5. Projected ellipse.
    ax5.scatter(
        u,
        v,
        s=14,
        alpha=0.75,
        label="Projected samples",
    )
    ax5.plot(
        ellipse_u,
        ellipse_v,
        linewidth=1.5,
        label="Fitted ellipse",
    )
    ax5.scatter(
        [calibration.u0],
        [calibration.v0],
        marker="*",
        s=80,
        label="Ellipse center",
    )

    for axis_vector, half_length in (
        (ellipse_axis_1, short_axis),
        (ellipse_axis_2, long_axis),
    ):
        p0 = (
            np.asarray(
                [calibration.u0, calibration.v0]
            )
            - half_length * axis_vector
        )
        p1 = (
            np.asarray(
                [calibration.u0, calibration.v0]
            )
            + half_length * axis_vector
        )
        ax5.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            linewidth=2,
        )

    ratio = max(long_axis, short_axis) / min(
        long_axis,
        short_axis,
    )
    ax5.set_aspect("equal", adjustable="box")
    ax5.set_xlabel("Projected u")
    ax5.set_ylabel("Projected v")
    ax5.set_title(
        f"Projected fitted ellipse (axis ratio {ratio:.3f})"
    )
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=7, loc="best")

    # 6. Corrected circle.
    corrected_scatter = ax6.scatter(
        corrected_x,
        corrected_y,
        c=out_of_plane,
        s=18,
    )

    circle_t = np.linspace(
        0.0,
        2.0 * math.pi,
        720,
    )
    ax6.plot(
        radius * np.cos(circle_t),
        radius * np.sin(circle_t),
        linestyle="--",
        label="Ideal circle",
    )

    ax6.axhline(0.0, linewidth=0.8)
    ax6.axvline(0.0, linewidth=0.8)
    ax6.set_aspect("equal", adjustable="box")
    ax6.set_xlabel("Corrected X")
    ax6.set_ylabel("Corrected Y")
    ax6.set_title(
        "Corrected circle\n"
        "(color = out-of-plane residual)"
    )
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=7, loc="best")

    colorbar = figure.colorbar(
        corrected_scatter,
        ax=ax6,
        fraction=0.046,
        pad=0.04,
    )
    colorbar.set_label(
        "Out-of-plane residual (raw units)"
    )

    figure.suptitle(
        "Pogosim magnetometer calibration diagnostics",
        fontsize=15,
    )
    figure.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.96)
    )

    summary_png = output_dir / "calibration_summary.png"
    summary_pdf = output_dir / "calibration_summary.pdf"

    figure.savefig(
        summary_png,
        dpi=dpi,
        bbox_inches="tight",
    )
    figure.savefig(
        summary_pdf,
        bbox_inches="tight",
    )

    if save_individual:
        subplot_specs = [
            (
                ax1,
                "01_angle_comparison.png",
            ),
            (
                ax2,
                "02_error_vs_iteration.png",
            ),
            (
                ax3,
                "03_error_vs_true_angle.png",
            ),
            (
                ax4,
                "04_raw_3d_plane_axes.png",
            ),
            (
                ax5,
                "05_projected_ellipse.png",
            ),
            (
                ax6,
                "06_corrected_circle.png",
            ),
        ]

        # Re-render each diagnostic independently so labels are not clipped
        # and 3-D axes keep their own layout.
        for plot_index, filename in enumerate(
            [
                "01_angle_comparison.png",
                "02_error_vs_iteration.png",
                "03_error_vs_true_angle.png",
                "04_raw_3d_plane_axes.png",
                "05_projected_ellipse.png",
                "06_corrected_circle.png",
            ],
            start=1,
        ):
            single = plt.figure(figsize=(6.5, 5.2))

            if plot_index == 1:
                ax = single.add_subplot(1, 1, 1)
                ax.scatter(
                    true_angles_deg,
                    alignment.adjusted_angle_deg,
                    s=18,
                )
                ax.plot(
                    [0.0, 360.0],
                    [0.0, 360.0],
                    linestyle="--",
                )
                ax.set_xlim(0.0, 360.0)
                ax.set_ylim(0.0, 360.0)
                ax.set_xlabel(
                    "Ground-truth robot angle (deg)"
                )
                ax.set_ylabel(
                    "Calibrated magnetometer angle (deg)"
                )
                ax.set_title("Sensor comparison")
                ax.grid(True, alpha=0.3)

            elif plot_index == 2:
                ax = single.add_subplot(1, 1, 1)
                ax.scatter(indices, errors, s=18)
                ax.axhline(0.0, linestyle="--")
                ax.set_xlabel("Measurement index")
                ax.set_ylabel(
                    "Signed angular error (deg)"
                )
                ax.set_title(
                    "Error vs measurement index"
                )
                ax.grid(True, alpha=0.3)
                ax.text(
                    0.02,
                    0.98,
                    stats_text,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=9,
                )

            elif plot_index == 3:
                ax = single.add_subplot(1, 1, 1)
                ax.scatter(
                    true_angles_deg,
                    errors,
                    s=18,
                )
                ax.axhline(0.0, linestyle="--")
                ax.set_xlim(0.0, 360.0)
                ax.set_xlabel(
                    "Ground-truth robot angle (deg)"
                )
                ax.set_ylabel(
                    "Signed angular error (deg)"
                )
                ax.set_title("Error vs true angle")
                ax.grid(True, alpha=0.3)

            elif plot_index == 4:
                ax = single.add_subplot(
                    1,
                    1,
                    1,
                    projection="3d",
                )
                ax.scatter(
                    calibration_points[:, 0],
                    calibration_points[:, 1],
                    calibration_points[:, 2],
                    s=16,
                    alpha=0.75,
                    label="Calibration samples",
                )
                ax.scatter(
                    [ellipse_center_3d[0]],
                    [ellipse_center_3d[1]],
                    [ellipse_center_3d[2]],
                    marker="*",
                    s=90,
                    label="Ellipse center",
                )

                for axis_vector, half_length, label in (
                    (
                        axis_1_3d,
                        short_axis,
                        "Ellipse axis 1",
                    ),
                    (
                        axis_2_3d,
                        long_axis,
                        "Ellipse axis 2",
                    ),
                ):
                    p0 = (
                        ellipse_center_3d
                        - half_length * axis_vector
                    )
                    p1 = (
                        ellipse_center_3d
                        + half_length * axis_vector
                    )
                    ax.plot(
                        [p0[0], p1[0]],
                        [p0[1], p1[1]],
                        [p0[2], p1[2]],
                        linewidth=2,
                        label=label,
                    )

                ax.plot(
                    [
                        ellipse_center_3d[0],
                        normal_end[0],
                    ],
                    [
                        ellipse_center_3d[1],
                        normal_end[1],
                    ],
                    [
                        ellipse_center_3d[2],
                        normal_end[2],
                    ],
                    linewidth=2,
                    label="Plane normal",
                )
                ax.plot_surface(
                    plane_xyz[0],
                    plane_xyz[1],
                    plane_xyz[2],
                    alpha=0.08,
                )
                ax.set_xlabel("Mag X")
                ax.set_ylabel("Mag Y")
                ax.set_zlabel("Mag Z")
                ax.set_title(
                    "3-D raw cloud, fitted plane and axes"
                )
                ax.legend(fontsize=8)
                equalize_3d_axes(
                    ax,
                    calibration_points,
                )

            elif plot_index == 5:
                ax = single.add_subplot(1, 1, 1)
                ax.scatter(
                    u,
                    v,
                    s=16,
                    alpha=0.75,
                    label="Projected samples",
                )
                ax.plot(
                    ellipse_u,
                    ellipse_v,
                    linewidth=1.5,
                    label="Fitted ellipse",
                )
                ax.scatter(
                    [calibration.u0],
                    [calibration.v0],
                    marker="*",
                    s=90,
                    label="Ellipse center",
                )

                for axis_vector, half_length in (
                    (
                        ellipse_axis_1,
                        short_axis,
                    ),
                    (
                        ellipse_axis_2,
                        long_axis,
                    ),
                ):
                    p0 = (
                        np.asarray(
                            [
                                calibration.u0,
                                calibration.v0,
                            ]
                        )
                        - half_length * axis_vector
                    )
                    p1 = (
                        np.asarray(
                            [
                                calibration.u0,
                                calibration.v0,
                            ]
                        )
                        + half_length * axis_vector
                    )
                    ax.plot(
                        [p0[0], p1[0]],
                        [p0[1], p1[1]],
                        linewidth=2,
                    )

                ax.set_aspect(
                    "equal",
                    adjustable="box",
                )
                ax.set_xlabel("Projected u")
                ax.set_ylabel("Projected v")
                ax.set_title(
                    f"Projected fitted ellipse "
                    f"(axis ratio {ratio:.3f})"
                )
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)

            else:
                ax = single.add_subplot(1, 1, 1)
                scatter = ax.scatter(
                    corrected_x,
                    corrected_y,
                    c=out_of_plane,
                    s=20,
                )
                ax.plot(
                    radius * np.cos(circle_t),
                    radius * np.sin(circle_t),
                    linestyle="--",
                    label="Ideal circle",
                )
                ax.axhline(0.0, linewidth=0.8)
                ax.axvline(0.0, linewidth=0.8)
                ax.set_aspect(
                    "equal",
                    adjustable="box",
                )
                ax.set_xlabel("Corrected X")
                ax.set_ylabel("Corrected Y")
                ax.set_title(
                    "Corrected circle "
                    "(color = out-of-plane residual)"
                )
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                cb = single.colorbar(
                    scatter,
                    ax=ax,
                )
                cb.set_label(
                    "Out-of-plane residual "
                    "(raw units)"
                )

            single.tight_layout()
            single.savefig(
                output_dir / filename,
                dpi=dpi,
                bbox_inches="tight",
            )
            plt.close(single)

    plt.close(figure)

    alignment_csv = (
        output_dir / "mission_alignment.csv"
    )

    with alignment_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "point",
                "plateau_center_time_s",
                "true_robot_angle_deg",
                "firmware_angle_deg",
                "phase_aligned_angle_deg",
                "signed_error_deg",
                "mag_x",
                "mag_y",
                "mag_z",
            ]
        )

        for point, sample in enumerate(samples):
            writer.writerow(
                [
                    point,
                    f"{sample_times_s[point]:.9f}",
                    f"{true_angles_deg[point]:.9f}",
                    f"{sample.firmware_angle_deg:.9f}",
                    (
                        f"{alignment.adjusted_angle_deg[point]:.9f}"
                    ),
                    f"{errors[point]:.9f}",
                    sample.mag_x,
                    sample.mag_y,
                    sample.mag_z,
                ]
            )

    return {
        "sample_count": len(samples),
        "calibration_point_count": len(
            calibration_points
        ),
        "plateau_count": len(plateaus),
        "mission_plateau_start": plateau_start_index,
        "heading_sign": alignment.sign,
        "heading_phase_deg": alignment.phase_deg,
        "median_abs_error_deg": median_abs_error,
        "rmse_deg": rmse,
        "max_abs_error_deg": max_abs_error,
        "ellipse_axis_ratio": ratio,
        "median_plateau_duration_s": float(
            np.median(
                [
                    plateau.duration_s
                    for plateau in plateaus[
                        plateau_start_index:
                        plateau_start_index + len(samples)
                    ]
                ]
            )
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Pogobot-style magnetometer calibration plots "
            "from Pogosim dump + data.feather."
        )
    )

    parser.add_argument(
        "--dump",
        type=Path,
        required=True,
        help="Pogosim console dump",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Pogosim data.feather",
    )
    parser.add_argument(
        "--robot-id",
        type=int,
        default=0,
        help="robot ID to analyze (default: 0)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="directory where plots and alignment CSV are written",
    )
    parser.add_argument(
        "--data-angle-unit",
        choices=["auto", "radians", "degrees"],
        default="auto",
        help=(
            "unit of data.feather angle column "
            "(default: auto)"
        ),
    )
    parser.add_argument(
        "--stationary-speed-threshold-deg-s",
        type=float,
        default=5.0,
        help=(
            "maximum angular speed considered stationary "
            "(default: 5 deg/s)"
        ),
    )
    parser.add_argument(
        "--min-plateau-duration-s",
        type=float,
        default=0.15,
        help=(
            "minimum stationary plateau duration "
            "(default: 0.15 s)"
        ),
    )
    parser.add_argument(
        "--mission-plateau-start",
        type=int,
        default=None,
        help=(
            "override mission plateau start index. By default "
            "the current firmware structure uses N_CAL - 1."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG resolution (default: 200)",
    )
    parser.add_argument(
        "--no-individual",
        action="store_true",
        help=(
            "only save the 2x3 summary, not six individual PNGs"
        ),
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        payloads = read_robot_payloads(
            args.dump,
            args.robot_id,
        )
        calibration_points = parse_calibration_points(
            payloads
        )
        calibration = parse_calibration(payloads)
        samples = parse_live_mission_samples(payloads)

        (
            true_angles_deg,
            sample_times_s,
            plateaus,
            plateau_start_index,
        ) = get_mission_ground_truth(
            feather_path=args.data,
            robot_id=args.robot_id,
            calibration_point_count=len(
                calibration_points
            ),
            mission_count=len(samples),
            data_angle_unit=args.data_angle_unit,
            stationary_speed_threshold_deg_s=(
                args.stationary_speed_threshold_deg_s
            ),
            min_plateau_duration_s=(
                args.min_plateau_duration_s
            ),
            mission_plateau_start=(
                args.mission_plateau_start
            ),
        )

        args.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        metrics = make_plots(
            output_dir=args.output_dir,
            calibration_points=calibration_points,
            calibration=calibration,
            samples=samples,
            true_angles_deg=true_angles_deg,
            sample_times_s=sample_times_s,
            plateaus=plateaus,
            plateau_start_index=plateau_start_index,
            dpi=args.dpi,
            save_individual=not args.no_individual,
        )

    except Exception as exc:
        parser.exit(2, f"error: {exc}\n")

    print(
        f"Wrote plots to {args.output_dir}\n"
        f"Calibration points: "
        f"{metrics['calibration_point_count']}\n"
        f"Mission samples: {metrics['sample_count']}\n"
        f"Detected stationary plateaus: "
        f"{metrics['plateau_count']}\n"
        f"Mission plateau start: "
        f"{metrics['mission_plateau_start']}\n"
        f"Median mission plateau duration: "
        f"{metrics['median_plateau_duration_s']:.3f} s\n"
        f"Heading mapping: theta = "
        f"{metrics['heading_sign']:+d} * ANGLE "
        f"{metrics['heading_phase_deg']:+.3f} deg\n"
        f"Median |angular error|: "
        f"{metrics['median_abs_error_deg']:.3f} deg\n"
        f"RMSE: {metrics['rmse_deg']:.3f} deg\n"
        f"Max |angular error|: "
        f"{metrics['max_abs_error_deg']:.3f} deg\n"
        f"Ellipse axis ratio: "
        f"{metrics['ellipse_axis_ratio']:.3f}",
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
