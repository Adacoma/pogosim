#!/usr/bin/env python3
"""
Build a Pogosim CSV magnetometer reference from either:

1. Real Pogobot experiment:
   UART dump + camera-derived ground-truth robot angles.

2. Pogosim experiment:
   Pogosim console dump + data.feather ground-truth trajectory.

By default, the generated CSV follows conf/example_magnetometer_reference.csv:
72 angles from 0 to 355 degrees in 5-degree steps, four samples per angle,
and the same metadata/comment/header layout:

    # magnetic_north_deg=...
    # noise_stddev=...
    # seed=...
    # convention: relative_angle = magnetic_north_angle - robot_angle
    angle,mag_x,mag_y,mag_z

Important convention
--------------------
The CSV ``angle`` column is the robot angle during calibration. Pogosim's
CSV fitter internally forms:

    relative_angle = calibration_field_angle - angle

At runtime the simulated sensor uses:

    relative_angle = world_field_angle - robot_theta + robot_angle_offset

Therefore the output also reports the calibration-field angle that should
be used with the generated CSV.

Dependencies
------------
Real mode:
    Python 3 + numpy + pandas

Pogosim mode:
    Python 3 + numpy + pandas + pyarrow
    (or a working ``pogosim.utils.load_dataframe`` installation)

Examples
--------
Real robot, including the old robot-only UART format:

    python3 make_magnetometer_reference.py real \
        --uart uart_angles.txt \
        --camera video_angles.txt \
        -o magnetometer_reference_real.csv

Pogosim:

    python3 make_magnetometer_reference.py pogosim \
        --dump dump_pogosim.txt \
        --data data.feather \
        --robot-id 0 \
        -o magnetometer_reference_pogosim.csv

For dense Pogosim trajectories, the default alignment detects stationary
orientation plateaus in data.feather and uses the CALPT count from the dump
to anchor the mission phase structurally. Mission point 0 shares the final
calibration plateau because there is no motor step between the last CALPT and
the first mission MAG sample. Sparse data falls back to the older periodic
time-grid alignment.

The Pogobot heading_deg() output is expressed in the fitted calibration
basis and may have a constant angular phase offset relative to the physical
magnetic-field coordinate system. Pogosim mode therefore reports both the raw
firmware/physical heading difference and a phase-corrected residual. The
ground-truth angle written to the reference CSV is never phase-corrected: it
remains the actual robot orientation from data.feather.

For exact timestamp-based pairing in arbitrary experiments, log simulation
time or pogobot_ticks together with every MAG measurement.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


POGOSIM_PRINTF_RE = re.compile(
    r"^\[(?P<timestamp>[^\]]+)\].*?"
    r"\[ROBOT #(?P<robot_id>\d+)\]\s*"
    r"\[PRINTF\]\s*(?P<payload>.*)$"
)


@dataclass
class MagSample:
    mag_x: int
    mag_y: int
    mag_z: int
    heading_deg: float | None = None


@dataclass
class AlignmentResult:
    first_time_s: float
    period_s: float
    field_angle_deg: float
    median_error_deg: float
    mean_error_deg: float
    max_error_deg: float
    sample_times_s: np.ndarray
    robot_angles_deg: np.ndarray


@dataclass
class StationaryPlateau:
    start_time_s: float
    end_time_s: float
    duration_s: float
    angle_deg: float


@dataclass
class PlateauAlignmentResult:
    field_angle_deg: float
    heading_phase_offset_deg: float
    raw_median_error_deg: float
    raw_mean_error_deg: float
    raw_max_error_deg: float
    corrected_median_error_deg: float
    corrected_mean_error_deg: float
    corrected_max_error_deg: float
    plateau_start_index: int
    robot_angles_deg: np.ndarray
    sample_times_s: np.ndarray
    plateau_durations_s: np.ndarray


def wrap_deg(angle: np.ndarray | float) -> np.ndarray | float:
    return np.mod(angle, 360.0)


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
    return np.abs(signed_circular_difference_deg(angle_a, angle_b))


def circular_mean_deg(angles_deg: np.ndarray) -> float:
    angles_rad = np.deg2rad(np.asarray(angles_deg, dtype=float))
    vector = np.mean(np.exp(1j * angles_rad))
    if abs(vector) < 1.0e-12:
        raise ValueError("circular mean is undefined for these angles")
    return float(np.rad2deg(np.angle(vector)) % 360.0)


def parse_mag_payload(payload: str) -> tuple[int, int, int] | None:
    fields = payload.strip().split(",")
    if len(fields) != 4 or fields[0] != "MAG":
        return None
    try:
        return int(fields[1]), int(fields[2]), int(fields[3])
    except ValueError:
        return None


def parse_angle_payload(payload: str) -> float | None:
    fields = payload.strip().split(",")
    if len(fields) != 2 or fields[0] != "ANGLE":
        return None
    try:
        return float(fields[1])
    except ValueError:
        return None


def extract_plain_payload(line: str) -> str:
    match = POGOSIM_PRINTF_RE.match(line.rstrip())
    if match:
        return match.group("payload").strip()
    return line.strip()


def extract_adjacent_angle_mag_pairs(
    payloads: list[str],
) -> list[MagSample]:
    samples: list[MagSample] = []
    i = 0
    while i + 1 < len(payloads):
        heading = parse_angle_payload(payloads[i])
        mag = parse_mag_payload(payloads[i + 1])
        if heading is not None and mag is not None:
            samples.append(
                MagSample(
                    mag_x=mag[0],
                    mag_y=mag[1],
                    mag_z=mag[2],
                    heading_deg=heading,
                )
            )
            i += 2
        else:
            i += 1
    return samples


def load_real_uart(path: Path, expected_count: int | None) -> list[MagSample]:
    payloads = [
        extract_plain_payload(line)
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
    ]

    # New cooperative code prints chronological ANGLE/MAG pairs live and may
    # subsequently print the page-organized flash dump. When two CALIB_MEAN
    # blocks exist, the live mission records are between them.
    calib_mean_indices = [
        index
        for index, payload in enumerate(payloads)
        if payload.startswith("CALIB_MEAN,")
    ]
    if len(calib_mean_indices) >= 2:
        live_payloads = payloads[
            calib_mean_indices[0] + 1 : calib_mean_indices[1]
        ]
    else:
        live_payloads = payloads

    live_pairs = extract_adjacent_angle_mag_pairs(live_payloads)
    if live_pairs:
        if expected_count is None or len(live_pairs) == expected_count:
            return live_pairs

    # Old robot-only dumps store MAG and ANGLE pages separately. Collecting
    # each record type independently preserves the original measurement order.
    mags: list[tuple[int, int, int]] = []
    headings: list[float] = []
    for payload in payloads:
        mag = parse_mag_payload(payload)
        if mag is not None:
            mags.append(mag)
            continue
        heading = parse_angle_payload(payload)
        if heading is not None:
            headings.append(heading)

    if not mags:
        raise ValueError(f"no MAG records found in {path}")

    if expected_count is not None and len(mags) != expected_count:
        raise ValueError(
            f"UART contains {len(mags)} MAG records but camera data contains "
            f"{expected_count} rows"
        )

    if headings and len(headings) != len(mags):
        raise ValueError(
            f"UART contains {len(mags)} MAG records but {len(headings)} "
            "ANGLE records; cannot infer calibration-field angle reliably"
        )

    samples: list[MagSample] = []
    for index, mag in enumerate(mags):
        heading = headings[index] if headings else None
        samples.append(
            MagSample(
                mag_x=mag[0],
                mag_y=mag[1],
                mag_z=mag[2],
                heading_deg=heading,
            )
        )
    return samples


def choose_camera_angle_column(columns: list[str]) -> str:
    preferred = [
        "Angle_Video(deg)",
        "angle_video_deg",
        "angle_deg",
        "angle",
    ]
    for name in preferred:
        if name in columns:
            return name

    candidates = [
        name
        for name in columns
        if "angle" in name.lower()
        and "center" not in name.lower()
    ]
    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(
        "could not determine camera angle column; use --camera-angle-column"
    )


def load_camera_angles(
    path: Path,
    requested_column: str | None,
) -> tuple[np.ndarray, str]:
    import pandas as pd

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"camera file {path} is empty")

    if "Point_ID" in frame.columns:
        frame = frame.sort_values("Point_ID")

    column = requested_column or choose_camera_angle_column(list(frame.columns))
    if column not in frame.columns:
        raise ValueError(
            f"camera angle column {column!r} does not exist; "
            f"available columns: {list(frame.columns)}"
        )

    angles = frame[column].to_numpy(dtype=float)
    if not np.all(np.isfinite(angles)):
        raise ValueError("camera angle column contains non-finite values")

    return wrap_deg(angles), column


def infer_real_angle_convention(
    samples: list[MagSample],
    camera_angles_deg: np.ndarray,
    requested_sign: str,
    requested_field_angle_deg: float | None,
) -> tuple[np.ndarray, float, int, dict[str, float]]:
    headings = np.array(
        [
            sample.heading_deg
            for sample in samples
            if sample.heading_deg is not None
        ],
        dtype=float,
    )

    if len(headings) != len(camera_angles_deg):
        if requested_field_angle_deg is None:
            raise ValueError(
                "cannot infer the real calibration-field angle because UART "
                "ANGLE records are missing; provide "
                "--calibration-field-angle-deg"
            )
        if requested_sign == "auto":
            raise ValueError(
                "cannot infer camera angle sign without UART ANGLE records; "
                "provide --camera-angle-sign 1 or -1"
            )

    candidate_signs = [1, -1] if requested_sign == "auto" else [int(requested_sign)]
    best: tuple[float, float, int, float, float] | None = None

    for sign in candidate_signs:
        robot_angles = wrap_deg(sign * camera_angles_deg)

        if requested_field_angle_deg is None:
            field_angle = circular_mean_deg(headings + robot_angles)
        else:
            field_angle = wrap_deg(requested_field_angle_deg)

        if len(headings) == len(robot_angles):
            predicted_heading = wrap_deg(field_angle - robot_angles)
            errors = circular_distance_deg(headings, predicted_heading)
            median_error = float(np.median(errors))
            mean_error = float(np.mean(errors))
        else:
            median_error = 0.0
            mean_error = 0.0

        candidate = (
            median_error,
            mean_error,
            sign,
            float(field_angle),
            float(np.max(errors)) if len(headings) else 0.0,
        )
        if best is None or candidate[:2] < best[:2]:
            best = candidate

    assert best is not None
    median_error, mean_error, sign, field_angle, max_error = best

    robot_angles = wrap_deg(sign * camera_angles_deg)
    stats = {
        "median_heading_error_deg": median_error,
        "mean_heading_error_deg": mean_error,
        "max_heading_error_deg": max_error,
    }
    return robot_angles, field_angle, sign, stats


def count_pogosim_calibration_points(path: Path, robot_id: int) -> int:
    count = 0
    for raw_line in path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        match = POGOSIM_PRINTF_RE.match(raw_line.rstrip())
        if not match:
            continue
        if int(match.group("robot_id")) != robot_id:
            continue

        payload = match.group("payload").strip()
        if payload.startswith("CALIB_MEAN,"):
            break
        if payload.startswith("CALPT,"):
            count += 1

    return count


def load_pogosim_live_samples(path: Path, robot_id: int) -> list[MagSample]:
    records: list[tuple[int, str]] = []

    for raw_line in path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        match = POGOSIM_PRINTF_RE.match(raw_line.rstrip())
        if not match:
            continue
        current_robot_id = int(match.group("robot_id"))
        if current_robot_id != robot_id:
            continue
        records.append((current_robot_id, match.group("payload").strip()))

    if not records:
        raise ValueError(
            f"no Pogosim [PRINTF] records found for robot {robot_id} in {path}"
        )

    payloads = [payload for _, payload in records]

    # The cooperative test program emits one CALIB_MEAN block before the
    # chronological mission samples, then a second CALIB_MEAN block when the
    # flash contents are dumped at the end. Restrict parsing to the interval
    # between those blocks so the page-organized final dump cannot create a
    # spurious ANGLE/MAG pair at a page boundary.
    calib_mean_indices = [
        index
        for index, payload in enumerate(payloads)
        if payload.startswith("CALIB_MEAN,")
    ]
    if calib_mean_indices:
        live_start = calib_mean_indices[0] + 1
        live_end = (
            calib_mean_indices[1]
            if len(calib_mean_indices) >= 2
            else len(payloads)
        )
        live_payloads = payloads[live_start:live_end]
    else:
        live_payloads = payloads

    samples = extract_adjacent_angle_mag_pairs(live_payloads)

    if not samples:
        raise ValueError(
            "could not find chronological live ANGLE/MAG pairs in Pogosim "
            "output. The final flash dump alone is insufficient for automatic "
            "time alignment."
        )

    return samples


def load_feather_with_metadata(path: Path) -> tuple[Any, dict[str, Any]]:
    # Preferred route: Pogosim's helper already decodes embedded metadata.
    try:
        import pogosim.utils as pogosim_utils

        frame, metadata = pogosim_utils.load_dataframe(str(path))
        if not isinstance(metadata, dict):
            metadata = {}
        return frame, metadata
    except Exception:
        pass

    import pandas as pd

    frame = pd.read_feather(path)
    metadata: dict[str, Any] = {}

    # Pandas does not expose Arrow schema metadata. Read it separately.
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc

        with pa.memory_map(str(path), "r") as source:
            reader = ipc.open_file(source)
            raw_metadata = reader.schema.metadata or {}

        configuration = raw_metadata.get(b"configuration")
        if configuration is not None:
            configuration_text = configuration.decode("utf-8", errors="replace")
            metadata["_configuration_text"] = configuration_text

            try:
                import yaml

                parsed = yaml.safe_load(configuration_text)
                if isinstance(parsed, dict):
                    metadata.update(parsed)
            except Exception:
                pass
    except Exception:
        pass

    return frame, metadata


def _decode_configuration_metadata(
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    """Return the Pogosim YAML configuration regardless of metadata shape."""
    candidates: list[Any] = [
        metadata.get("configuration"),
        metadata.get(b"configuration"),
        metadata.get("_configuration_text"),
        metadata,
    ]

    for candidate in candidates:
        if isinstance(candidate, dict):
            # The normal pogosim.utils.load_dataframe() shape is:
            # metadata["configuration"] == parsed YAML dictionary.
            if "magnetometer" in candidate:
                return candidate

        if isinstance(candidate, bytes):
            candidate = candidate.decode("utf-8", errors="replace")

        if isinstance(candidate, str):
            try:
                import yaml

                parsed = yaml.safe_load(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

    return None


def extract_world_field_angle_deg(
    metadata: dict[str, Any],
) -> float | None:
    configuration = _decode_configuration_metadata(metadata)

    if configuration is not None:
        magnetometer = configuration.get("magnetometer")
        if isinstance(magnetometer, dict):
            world_field = magnetometer.get("world_field")
            if isinstance(world_field, dict):
                try:
                    x = float(world_field["x"])
                    y = float(world_field["y"])
                    if x != 0.0 or y != 0.0:
                        return float(
                            math.degrees(math.atan2(y, x)) % 360.0
                        )
                except (KeyError, TypeError, ValueError):
                    pass

    # Last-resort text parser for unusual/older metadata encodings.
    text_candidates = [
        metadata.get("configuration"),
        metadata.get("_configuration_text"),
    ]
    for text in text_candidates:
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        if not isinstance(text, str):
            continue

        match = re.search(
            r"(?ms)^\s*world_field:\s*\n"
            r"\s*x:\s*([-+0-9.eE]+)\s*\n"
            r"\s*y:\s*([-+0-9.eE]+)",
            text,
        )
        if match:
            x = float(match.group(1))
            y = float(match.group(2))
            if x != 0.0 or y != 0.0:
                return float(math.degrees(math.atan2(y, x)) % 360.0)

    return None

def infer_angle_unit(values: np.ndarray, requested_unit: str) -> str:
    if requested_unit != "auto":
        return requested_unit

    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        raise ValueError("dataframe angle column contains no finite values")

    if float(np.max(finite)) <= 2.0 * math.pi + 0.25:
        return "radians"
    return "degrees"


def detect_stationary_tail_start(
    times_s: np.ndarray,
    robot_angles_deg: np.ndarray,
    tolerance_deg: float,
    min_rows: int = 3,
) -> float | None:
    if len(times_s) < min_rows + 1:
        return None

    final_angle = robot_angles_deg[-1]
    differences = circular_distance_deg(robot_angles_deg, final_angle)
    moving_indices = np.flatnonzero(differences > tolerance_deg)

    if moving_indices.size == 0:
        return None

    first_stationary_index = int(moving_indices[-1] + 1)
    if len(times_s) - first_stationary_index < min_rows:
        return None

    return float(times_s[first_stationary_index])


def interpolate_unwrapped_angle_deg(
    sample_times_s: np.ndarray,
    data_times_s: np.ndarray,
    unwrapped_angles_deg: np.ndarray,
) -> np.ndarray:
    interpolated = np.interp(
        sample_times_s,
        data_times_s,
        unwrapped_angles_deg,
    )
    return wrap_deg(interpolated)



def detect_stationary_plateaus(
    times_s: np.ndarray,
    robot_angles_deg: np.ndarray,
    speed_threshold_deg_s: float,
    min_duration_s: float,
) -> list[StationaryPlateau]:
    if len(times_s) != len(robot_angles_deg):
        raise ValueError("time and angle arrays have different lengths")
    if len(times_s) < 2:
        return []
    if speed_threshold_deg_s <= 0.0:
        raise ValueError("stationary speed threshold must be > 0")
    if min_duration_s < 0.0:
        raise ValueError("minimum plateau duration must be >= 0")

    dt = np.diff(times_s)
    if np.any(dt <= 0.0):
        raise ValueError("dataframe time values must be strictly increasing")

    angular_steps_deg = circular_distance_deg(
        robot_angles_deg[1:],
        robot_angles_deg[:-1],
    )
    speeds_deg_s = angular_steps_deg / dt
    stationary_edges = speeds_deg_s <= speed_threshold_deg_s

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
        duration_s = end_time_s - start_time_s

        if duration_s >= min_duration_s:
            plateau_angles = robot_angles_deg[first_row : last_row + 1]
            angle_deg = circular_mean_deg(plateau_angles)
            plateaus.append(
                StationaryPlateau(
                    start_time_s=start_time_s,
                    end_time_s=end_time_s,
                    duration_s=duration_s,
                    angle_deg=angle_deg,
                )
            )

        edge += 1

    return plateaus


def search_plateau_alignment(
    live_headings_deg: np.ndarray,
    field_angle_deg: float | None,
    plateaus: list[StationaryPlateau],
    expected_start_index: int | None = None,
    structural_search_radius: int = 2,
) -> PlateauAlignmentResult:
    sample_count = len(live_headings_deg)
    if sample_count == 0:
        raise ValueError("no live headings available for plateau alignment")
    if len(plateaus) < sample_count:
        raise ValueError(
            f"only {len(plateaus)} stationary plateaus detected for "
            f"{sample_count} samples"
        )

    if structural_search_radius < 0:
        raise ValueError("structural search radius must be >= 0")

    if expected_start_index is None:
        candidate_start_indices = list(
            range(len(plateaus) - sample_count + 1)
        )
    else:
        first = max(0, expected_start_index - structural_search_radius)
        last = min(
            len(plateaus) - sample_count,
            expected_start_index + structural_search_radius,
        )
        if last < first:
            raise ValueError(
                f"expected mission plateau start {expected_start_index} "
                f"is incompatible with {len(plateaus)} detected plateaus "
                f"and {sample_count} samples"
            )
        candidate_start_indices = list(range(first, last + 1))

    best = None

    for start_index in candidate_start_indices:
        block = plateaus[start_index : start_index + sample_count]
        robot_angles = np.array(
            [plateau.angle_deg for plateau in block],
            dtype=float,
        )

        if field_angle_deg is None:
            effective_field_angle_deg = circular_mean_deg(
                wrap_deg(live_headings_deg + robot_angles)
            )
        else:
            effective_field_angle_deg = float(wrap_deg(field_angle_deg))

        # Physical/model-relative heading expected from the known field and
        # ground-truth robot orientation.
        physical_headings = wrap_deg(
            effective_field_angle_deg - robot_angles
        )

        raw_errors = circular_distance_deg(
            live_headings_deg,
            physical_headings,
        )

        # heading_deg() in the Pogobot calibration code is expressed in the
        # fitted (e1, e2) basis. Its zero phase is therefore not guaranteed to
        # coincide with the physical magnetic-field zero. Estimate that one
        # additive phase offset, then report the residual after removing it.
        signed_offsets = signed_circular_difference_deg(
            live_headings_deg,
            physical_headings,
        )
        heading_phase_offset_deg = circular_mean_deg(signed_offsets)
        if heading_phase_offset_deg >= 180.0:
            heading_phase_offset_deg -= 360.0

        corrected_headings = wrap_deg(
            physical_headings + heading_phase_offset_deg
        )
        corrected_errors = circular_distance_deg(
            live_headings_deg,
            corrected_headings,
        )

        raw_median_error = float(np.median(raw_errors))
        raw_mean_error = float(np.mean(raw_errors))
        raw_max_error = float(np.max(raw_errors))
        corrected_median_error = float(np.median(corrected_errors))
        corrected_mean_error = float(np.mean(corrected_errors))
        corrected_max_error = float(np.max(corrected_errors))

        centers = np.array(
            [
                0.5 * (plateau.start_time_s + plateau.end_time_s)
                for plateau in block
            ],
            dtype=float,
        )
        durations = np.array(
            [plateau.duration_s for plateau in block],
            dtype=float,
        )

        structural_distance = (
            abs(start_index - expected_start_index)
            if expected_start_index is not None
            else 0
        )

        # Structure is authoritative when CALPT provides the phase boundary.
        # The phase-corrected residual is only a secondary validation signal.
        candidate = (
            structural_distance,
            corrected_median_error,
            corrected_mean_error,
            start_index,
            effective_field_angle_deg,
            heading_phase_offset_deg,
            raw_median_error,
            raw_mean_error,
            raw_max_error,
            corrected_max_error,
            robot_angles,
            centers,
            durations,
        )
        if best is None or candidate[:3] < best[:3]:
            best = candidate

    assert best is not None
    (
        _,
        corrected_median_error,
        corrected_mean_error,
        start_index,
        effective_field_angle_deg,
        heading_phase_offset_deg,
        raw_median_error,
        raw_mean_error,
        raw_max_error,
        corrected_max_error,
        robot_angles,
        centers,
        durations,
    ) = best

    return PlateauAlignmentResult(
        field_angle_deg=float(effective_field_angle_deg),
        heading_phase_offset_deg=float(heading_phase_offset_deg),
        raw_median_error_deg=float(raw_median_error),
        raw_mean_error_deg=float(raw_mean_error),
        raw_max_error_deg=float(raw_max_error),
        corrected_median_error_deg=float(corrected_median_error),
        corrected_mean_error_deg=float(corrected_mean_error),
        corrected_max_error_deg=float(corrected_max_error),
        plateau_start_index=int(start_index),
        robot_angles_deg=robot_angles,
        sample_times_s=centers,
        plateau_durations_s=durations,
    )


def score_pogosim_alignment(
    first_time_s: float,
    period_s: float,
    sample_count: int,
    live_headings_deg: np.ndarray,
    field_angle_deg: float | None,
    data_times_s: np.ndarray,
    robot_unwrapped_deg: np.ndarray,
) -> tuple[float, float, float, float, np.ndarray]:
    sample_times = first_time_s + np.arange(sample_count, dtype=float) * period_s

    if (
        sample_times[0] < data_times_s[0]
        or sample_times[-1] > data_times_s[-1]
    ):
        return math.inf, math.inf, math.inf, math.nan, sample_times

    robot_angles = interpolate_unwrapped_angle_deg(
        sample_times,
        data_times_s,
        robot_unwrapped_deg,
    )

    if field_angle_deg is None:
        # From heading = field_angle - robot_angle, each aligned sample gives
        # field_angle = heading + robot_angle. Use the circular mean so the
        # alignment can be solved even when Feather metadata is absent.
        effective_field_angle_deg = circular_mean_deg(
            wrap_deg(live_headings_deg + robot_angles)
        )
    else:
        effective_field_angle_deg = float(wrap_deg(field_angle_deg))

    expected_headings = wrap_deg(effective_field_angle_deg - robot_angles)
    errors = circular_distance_deg(live_headings_deg, expected_headings)

    return (
        float(np.median(errors)),
        float(np.mean(errors)),
        float(np.max(errors)),
        effective_field_angle_deg,
        sample_times,
    )

def search_pogosim_alignment(
    live_headings_deg: np.ndarray,
    field_angle_deg: float | None,
    data_times_s: np.ndarray,
    robot_angles_deg: np.ndarray,
    period_min_s: float,
    period_max_s: float,
    stationary_tolerance_deg: float,
    end_gap_max_s: float,
) -> AlignmentResult:
    if len(live_headings_deg) < 2:
        raise ValueError("need at least two live ANGLE/MAG samples for alignment")

    if period_min_s <= 0.0 or period_max_s <= period_min_s:
        raise ValueError("invalid Pogosim period search interval")

    robot_unwrapped_deg = np.rad2deg(
        np.unwrap(np.deg2rad(robot_angles_deg))
    )

    stationary_tail_start = detect_stationary_tail_start(
        data_times_s,
        robot_angles_deg,
        stationary_tolerance_deg,
    )

    sample_count = len(live_headings_deg)
    best: (
        tuple[
            float,
            float,
            float,
            float,
            float,
            float,
            np.ndarray,
        ]
        | None
    ) = None

    def consider(first_time_s: float, period_s: float) -> None:
        nonlocal best
        (
            median_error,
            mean_error,
            max_error,
            effective_field_angle_deg,
            sample_times,
        ) = score_pogosim_alignment(
            first_time_s=first_time_s,
            period_s=period_s,
            sample_count=sample_count,
            live_headings_deg=live_headings_deg,
            field_angle_deg=field_angle_deg,
            data_times_s=data_times_s,
            robot_unwrapped_deg=robot_unwrapped_deg,
        )
        candidate = (
            median_error,
            mean_error,
            max_error,
            first_time_s,
            period_s,
            effective_field_angle_deg,
            sample_times,
        )
        if best is None or candidate[:2] < best[:2]:
            best = candidate

    # Coarse search.
    for period_s in np.arange(
        period_min_s,
        period_max_s + 0.5 * 0.01,
        0.01,
    ):
        span_s = (sample_count - 1) * period_s

        if stationary_tail_start is not None:
            first_min = max(
                float(data_times_s[0]),
                stationary_tail_start - end_gap_max_s - span_s,
            )
            first_max = min(
                float(data_times_s[-1] - span_s),
                stationary_tail_start - span_s,
            )
        else:
            first_min = float(data_times_s[0])
            first_max = float(data_times_s[-1] - span_s)

        if first_max < first_min:
            continue

        for first_time_s in np.arange(
            first_min,
            first_max + 0.5 * 0.05,
            0.05,
        ):
            consider(float(first_time_s), float(period_s))

    if best is None or not math.isfinite(best[0]):
        raise ValueError("could not find a valid Pogosim timing alignment")

    # Fine search around the best coarse solution.
    _, _, _, coarse_first, coarse_period, _, _ = best
    fine_period_min = max(period_min_s, coarse_period - 0.03)
    fine_period_max = min(period_max_s, coarse_period + 0.03)
    fine_first_min = max(float(data_times_s[0]), coarse_first - 0.20)
    fine_first_max = coarse_first + 0.20

    for period_s in np.arange(
        fine_period_min,
        fine_period_max + 0.5 * 0.0005,
        0.0005,
    ):
        latest_first = float(
            data_times_s[-1] - (sample_count - 1) * period_s
        )
        local_first_max = min(fine_first_max, latest_first)
        if local_first_max < fine_first_min:
            continue

        for first_time_s in np.arange(
            fine_first_min,
            local_first_max + 0.5 * 0.005,
            0.005,
        ):
            consider(float(first_time_s), float(period_s))

    assert best is not None
    (
        median_error,
        mean_error,
        max_error,
        first_time_s,
        period_s,
        effective_field_angle_deg,
        sample_times_s,
    ) = best

    robot_angles = interpolate_unwrapped_angle_deg(
        sample_times_s,
        data_times_s,
        robot_unwrapped_deg,
    )

    return AlignmentResult(
        first_time_s=float(first_time_s),
        period_s=float(period_s),
        field_angle_deg=float(effective_field_angle_deg),
        median_error_deg=float(median_error),
        mean_error_deg=float(mean_error),
        max_error_deg=float(max_error),
        sample_times_s=sample_times_s,
        robot_angles_deg=robot_angles,
    )

def prepare_pogosim_reference(
    dump_path: Path,
    data_path: Path,
    robot_id: int,
    requested_field_angle_deg: float | None,
    data_angle_unit: str,
    period_min_s: float,
    period_max_s: float,
    stationary_tolerance_deg: float,
    end_gap_max_s: float,
    alignment_method: str,
    stationary_speed_threshold_deg_s: float,
    min_plateau_duration_s: float,
    max_plateau_save_period_s: float,
) -> tuple[np.ndarray, list[MagSample], float, dict[str, Any]]:
    calibration_point_count = count_pogosim_calibration_points(
        dump_path,
        robot_id,
    )
    samples = load_pogosim_live_samples(dump_path, robot_id)
    live_headings_deg = np.array(
        [sample.heading_deg for sample in samples],
        dtype=float,
    )

    frame, metadata = load_feather_with_metadata(data_path)

    required_columns = {"time", "robot_id", "angle"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{data_path} is missing required columns: {sorted(missing)}"
        )

    robot_frame = frame.loc[
        frame["robot_id"] == robot_id,
        ["time", "angle"],
    ].copy()
    robot_frame = robot_frame.dropna().sort_values("time")

    if len(robot_frame) < 3:
        raise ValueError(
            f"dataframe contains too few rows for robot {robot_id}"
        )

    data_times_s = robot_frame["time"].to_numpy(dtype=float)
    raw_angles = robot_frame["angle"].to_numpy(dtype=float)

    unit = infer_angle_unit(raw_angles, data_angle_unit)
    if unit == "radians":
        robot_angles_deg = np.rad2deg(raw_angles)
    else:
        robot_angles_deg = raw_angles.copy()
    robot_angles_deg = wrap_deg(robot_angles_deg)

    if requested_field_angle_deg is not None:
        field_angle_deg: float | None = float(
            wrap_deg(requested_field_angle_deg)
        )
        field_angle_source = "command_line"
    else:
        field_angle_deg = extract_world_field_angle_deg(metadata)
        field_angle_source = (
            "feather_metadata"
            if field_angle_deg is not None
            else "inferred_from_alignment"
        )

    data_save_period_s = float(np.median(np.diff(data_times_s)))

    requested_alignment = alignment_method
    if requested_alignment not in {"auto", "plateau", "periodic"}:
        raise ValueError(
            f"unknown Pogosim alignment method: {requested_alignment}"
        )

    use_plateau = requested_alignment == "plateau"
    if requested_alignment == "auto":
        use_plateau = data_save_period_s <= max_plateau_save_period_s

    plateau_failure: str | None = None

    if use_plateau:
        try:
            plateaus = detect_stationary_plateaus(
                times_s=data_times_s,
                robot_angles_deg=robot_angles_deg,
                speed_threshold_deg_s=stationary_speed_threshold_deg_s,
                min_duration_s=min_plateau_duration_s,
            )

            # Experiment structure:
            # - each CALPT is measured after a motor step;
            # - mission point 0 is measured without another motor step, so it
            #   shares the plateau of the final calibration point;
            # - after the final mission measurement, one last motor step is
            #   performed but no MAG sample is taken on that final plateau.
            #
            # Therefore, with N calibration points, mission sample 0 belongs
            # to stationary plateau N - 1.
            expected_mission_start = (
                calibration_point_count - 1
                if calibration_point_count > 0
                else None
            )

            plateau_alignment = search_plateau_alignment(
                live_headings_deg=live_headings_deg,
                field_angle_deg=field_angle_deg,
                plateaus=plateaus,
                expected_start_index=expected_mission_start,
                structural_search_radius=2,
            )

            stats: dict[str, Any] = {
                "alignment_method": "stationary_plateaus",
                "field_angle_source": field_angle_source,
                "stationary_plateau_count": len(plateaus),
                "calibration_point_count": calibration_point_count,
                "expected_mission_plateau_start_index": (
                    expected_mission_start
                    if expected_mission_start is not None
                    else -1
                ),
                "selected_plateau_start_index": (
                    plateau_alignment.plateau_start_index
                ),
                "median_selected_plateau_duration_s": float(
                    np.median(
                        plateau_alignment.plateau_durations_s
                    )
                ),
                "firmware_heading_phase_offset_deg": (
                    plateau_alignment.heading_phase_offset_deg
                ),
                "raw_heading_consistency_median_deg": (
                    plateau_alignment.raw_median_error_deg
                ),
                "raw_heading_consistency_mean_deg": (
                    plateau_alignment.raw_mean_error_deg
                ),
                "raw_heading_consistency_max_deg": (
                    plateau_alignment.raw_max_error_deg
                ),
                "phase_corrected_heading_residual_median_deg": (
                    plateau_alignment.corrected_median_error_deg
                ),
                "phase_corrected_heading_residual_mean_deg": (
                    plateau_alignment.corrected_mean_error_deg
                ),
                "phase_corrected_heading_residual_max_deg": (
                    plateau_alignment.corrected_max_error_deg
                ),
                "data_save_period_s": data_save_period_s,
            }

            return (
                plateau_alignment.robot_angles_deg,
                samples,
                plateau_alignment.field_angle_deg,
                stats,
            )
        except ValueError as exception:
            plateau_failure = str(exception)
            if requested_alignment == "plateau":
                raise

    alignment = search_pogosim_alignment(
        live_headings_deg=live_headings_deg,
        field_angle_deg=field_angle_deg,
        data_times_s=data_times_s,
        robot_angles_deg=robot_angles_deg,
        period_min_s=period_min_s,
        period_max_s=period_max_s,
        stationary_tolerance_deg=stationary_tolerance_deg,
        end_gap_max_s=end_gap_max_s,
    )

    stats = {
        "alignment_method": "periodic_time_grid",
        "first_sample_time_s": alignment.first_time_s,
        "sample_period_s": alignment.period_s,
        "field_angle_source": field_angle_source,
        "median_heading_consistency_error_deg": (
            alignment.median_error_deg
        ),
        "mean_heading_consistency_error_deg": (
            alignment.mean_error_deg
        ),
        "max_heading_consistency_error_deg": alignment.max_error_deg,
        "data_save_period_s": data_save_period_s,
    }
    if plateau_failure is not None:
        stats["plateau_fallback_reason"] = plateau_failure

    return (
        alignment.robot_angles_deg,
        samples,
        alignment.field_angle_deg,
        stats,
    )

def apply_optional_angle_binning(
    robot_angles_deg: np.ndarray,
    bin_deg: float,
) -> np.ndarray:
    if bin_deg <= 0.0:
        return wrap_deg(robot_angles_deg)
    return wrap_deg(np.round(robot_angles_deg / bin_deg) * bin_deg)



def harmonic_basis_deg(
    angles_deg: np.ndarray,
    harmonic_order: int,
) -> np.ndarray:
    if harmonic_order < 1:
        raise ValueError("harmonic order must be at least 1")

    angles_rad = np.deg2rad(
        np.asarray(angles_deg, dtype=float)
    )
    basis = np.ones(
        (len(angles_rad), 1 + 2 * harmonic_order),
        dtype=float,
    )

    for order in range(1, harmonic_order + 1):
        phase = order * angles_rad
        basis[:, 2 * order - 1] = np.cos(phase)
        basis[:, 2 * order] = np.sin(phase)

    return basis


def fit_reference_harmonics(
    robot_angles_deg: np.ndarray,
    samples: list[MagSample],
    harmonic_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angles = np.asarray(robot_angles_deg, dtype=float)
    measurements = np.array(
        [
            [sample.mag_x, sample.mag_y, sample.mag_z]
            for sample in samples
        ],
        dtype=float,
    )

    if len(angles) != len(measurements):
        raise ValueError(
            f"have {len(angles)} angles but "
            f"{len(measurements)} magnetometer samples"
        )

    parameter_count = 1 + 2 * harmonic_order
    if len(angles) < parameter_count:
        raise ValueError(
            f"need at least {parameter_count} samples for "
            f"harmonic order {harmonic_order}"
        )

    design = harmonic_basis_deg(angles, harmonic_order)
    coefficients, _, rank, _ = np.linalg.lstsq(
        design,
        measurements,
        rcond=None,
    )
    if rank < parameter_count:
        raise ValueError(
            "harmonic reference fit is rank-deficient; "
            "provide better angular coverage"
        )

    fitted = design @ coefficients
    residuals = measurements - fitted
    rmse_axes = np.sqrt(np.mean(residuals * residuals, axis=0))

    return coefficients, residuals, rmse_axes


def generate_regular_reference_rows(
    robot_angles_deg: np.ndarray,
    samples: list[MagSample],
    grid_step_deg: float,
    samples_per_angle: int,
    harmonic_order: int,
    seed: int,
    add_residual_noise: bool,
) -> tuple[
    list[tuple[float, int, int, int]],
    np.ndarray,
]:
    if grid_step_deg <= 0.0 or grid_step_deg > 360.0:
        raise ValueError("--grid-step-deg must be in (0, 360]")
    if samples_per_angle < 1:
        raise ValueError("--samples-per-angle must be at least 1")

    grid_count_float = 360.0 / grid_step_deg
    grid_count = int(round(grid_count_float))
    if not math.isclose(
        grid_count * grid_step_deg,
        360.0,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError(
            "--grid-step-deg must divide 360 exactly"
        )

    coefficients, residuals, rmse_axes = fit_reference_harmonics(
        robot_angles_deg=robot_angles_deg,
        samples=samples,
        harmonic_order=harmonic_order,
    )

    grid_angles = (
        np.arange(grid_count, dtype=float) * grid_step_deg
    )
    grid_design = harmonic_basis_deg(
        grid_angles,
        harmonic_order,
    )
    mean_measurements = grid_design @ coefficients

    rng = np.random.default_rng(seed)
    rows: list[tuple[float, int, int, int]] = []

    for angle_index, angle_deg in enumerate(grid_angles):
        mean_vector = mean_measurements[angle_index]

        for _ in range(samples_per_angle):
            vector = mean_vector.copy()

            if add_residual_noise and len(residuals) > 0:
                # Bootstrap one complete XYZ residual vector rather than
                # sampling each axis independently. This preserves residual
                # cross-axis correlations present in the source experiment.
                residual_index = int(
                    rng.integers(0, len(residuals))
                )
                vector += residuals[residual_index]

            quantized = np.rint(vector).astype(int)
            rows.append(
                (
                    float(angle_deg),
                    int(quantized[0]),
                    int(quantized[1]),
                    int(quantized[2]),
                )
            )

    return rows, rmse_axes


def format_compact_number(value: float) -> str:
    if math.isclose(
        value,
        round(value),
        rel_tol=0.0,
        abs_tol=1.0e-10,
    ):
        return str(int(round(value)))
    return f"{value:.8f}".rstrip("0").rstrip(".")


def write_reference_csv(
    output_path: Path,
    robot_angles_deg: np.ndarray,
    samples: list[MagSample],
    calibration_field_angle_deg: float,
    source: str,
    source_stats: dict[str, Any],
    angle_bin_deg: float,
    preserve_order: bool,
    grid_step_deg: float,
    samples_per_angle: int,
    reference_harmonic_order: int,
    reference_seed: int,
    add_residual_noise: bool,
    raw_output: bool,
) -> int:
    if len(robot_angles_deg) != len(samples):
        raise ValueError(
            f"have {len(robot_angles_deg)} robot angles but "
            f"{len(samples)} magnetometer samples"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_output:
        angles = apply_optional_angle_binning(
            np.asarray(robot_angles_deg, dtype=float),
            angle_bin_deg,
        )
        rows = [
            (
                float(angles[index]),
                int(sample.mag_x),
                int(sample.mag_y),
                int(sample.mag_z),
            )
            for index, sample in enumerate(samples)
        ]
        if not preserve_order:
            rows.sort(key=lambda row: row[0])

        rmse_axes = np.zeros(3, dtype=float)
    else:
        rows, rmse_axes = generate_regular_reference_rows(
            robot_angles_deg=np.asarray(
                robot_angles_deg,
                dtype=float,
            ),
            samples=samples,
            grid_step_deg=grid_step_deg,
            samples_per_angle=samples_per_angle,
            harmonic_order=reference_harmonic_order,
            seed=reference_seed,
            add_residual_noise=add_residual_noise,
        )

    # The example file uses a single scalar noise_stddev metadata value.
    # The Pogosim CSV reader ignores comments, but keeping this line makes
    # generated references human-readable and format-compatible.
    pooled_noise_stddev = float(
        np.sqrt(np.mean(rmse_axes * rmse_axes))
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "# magnetic_north_deg="
            f"{format_compact_number(calibration_field_angle_deg)}\n"
        )
        handle.write(
            "# noise_stddev="
            f"{format_compact_number(pooled_noise_stddev)}\n"
        )
        handle.write(f"# seed={reference_seed}\n")
        handle.write(
            "# convention: relative_angle = "
            "magnetic_north_angle - robot_angle\n"
        )

        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["angle", "mag_x", "mag_y", "mag_z"]
        )

        for angle, mag_x, mag_y, mag_z in rows:
            writer.writerow(
                [
                    format_compact_number(angle),
                    mag_x,
                    mag_y,
                    mag_z,
                ]
            )

    return len(rows)


def run_real(args: argparse.Namespace) -> None:
    camera_angles_deg, camera_column = load_camera_angles(
        args.camera,
        args.camera_angle_column,
    )
    samples = load_real_uart(args.uart, len(camera_angles_deg))

    robot_angles_deg, effective_heading_reference_deg, camera_sign, stats = (
        infer_real_angle_convention(
            samples=samples,
            camera_angles_deg=camera_angles_deg,
            requested_sign=args.camera_angle_sign,
            requested_field_angle_deg=args.calibration_field_angle_deg,
        )
    )

    # The angle inferred above comes from firmware ANGLE + camera orientation.
    # Because the Pogobot calibration basis has an arbitrary phase, this is
    # only an effective firmware-heading reference, not physical magnetic
    # north. The CSV model itself is phase-invariant: choose a convenient
    # reference field angle and let the fitted harmonic coefficients encode
    # the measured phase of the real sensor.
    reference_field_angle_deg = float(
        wrap_deg(args.reference_field_angle_deg)
    )

    stats["camera_angle_column"] = camera_column
    stats["camera_angle_sign"] = camera_sign
    stats["sample_count"] = len(samples)
    stats["effective_firmware_heading_reference_deg"] = (
        effective_heading_reference_deg
    )
    stats["reference_field_angle_deg"] = reference_field_angle_deg

    row_count = write_reference_csv(
        output_path=args.output,
        robot_angles_deg=robot_angles_deg,
        samples=samples,
        calibration_field_angle_deg=reference_field_angle_deg,
        source="real_robot_camera",
        source_stats=stats,
        angle_bin_deg=args.angle_bin_deg,
        preserve_order=args.preserve_order,
        grid_step_deg=args.grid_step_deg,
        samples_per_angle=args.samples_per_angle,
        reference_harmonic_order=args.reference_harmonic_order,
        reference_seed=args.reference_seed,
        add_residual_noise=not args.no_residual_noise,
        raw_output=args.raw_output,
    )

    print(
        f"Wrote {row_count} rows to {args.output}\n"
        f"Effective firmware heading reference (diagnostic only): "
        f"{effective_heading_reference_deg:.6f} deg\n"
        f"Reference CSV calibration-field angle: "
        f"{reference_field_angle_deg:.6f} deg\n"
        f"Set YAML magnetometer.csv.calibration_field_angle to: "
        f"{reference_field_angle_deg:.6f} deg\n"
        f"Camera angle sign: {camera_sign:+d}\n"
        f"Median UART-heading/camera consistency error: "
        f"{stats['median_heading_error_deg']:.3f} deg",
        file=sys.stderr,
    )


def run_pogosim(args: argparse.Namespace) -> None:
    robot_angles_deg, samples, field_angle_deg, stats = (
        prepare_pogosim_reference(
            dump_path=args.dump,
            data_path=args.data,
            robot_id=args.robot_id,
            requested_field_angle_deg=args.calibration_field_angle_deg,
            data_angle_unit=args.data_angle_unit,
            period_min_s=args.period_min_s,
            period_max_s=args.period_max_s,
            stationary_tolerance_deg=args.stationary_tolerance_deg,
            end_gap_max_s=args.end_gap_max_s,
            alignment_method=args.alignment,
            stationary_speed_threshold_deg_s=(
                args.stationary_speed_threshold_deg_s
            ),
            min_plateau_duration_s=args.min_plateau_duration_s,
            max_plateau_save_period_s=args.max_plateau_save_period_s,
        )
    )

    stats["robot_id"] = args.robot_id
    stats["sample_count"] = len(samples)

    row_count = write_reference_csv(
        output_path=args.output,
        robot_angles_deg=robot_angles_deg,
        samples=samples,
        calibration_field_angle_deg=field_angle_deg,
        source="pogosim",
        source_stats=stats,
        angle_bin_deg=args.angle_bin_deg,
        preserve_order=args.preserve_order,
        grid_step_deg=args.grid_step_deg,
        samples_per_angle=args.samples_per_angle,
        reference_harmonic_order=args.reference_harmonic_order,
        reference_seed=args.reference_seed,
        add_residual_noise=not args.no_residual_noise,
        raw_output=args.raw_output,
    )

    print(
        f"Wrote {row_count} rows to {args.output}\n"
        f"Calibration-field angle: {field_angle_deg:.6f} deg\n"
        f"Set YAML csv.calibration_field_angle to: "
        f"{field_angle_deg:.6f} deg\n"
        f"Alignment method: {stats['alignment_method']}",
        file=sys.stderr,
    )

    if stats["alignment_method"] == "stationary_plateaus":
        print(
            f"Detected stationary plateaus: "
            f"{stats['stationary_plateau_count']}\n"
            f"Calibration points in dump: "
            f"{stats['calibration_point_count']}\n"
            f"Expected mission plateau start: "
            f"{stats['expected_mission_plateau_start_index']}\n"
            f"Selected plateau block starts at index: "
            f"{stats['selected_plateau_start_index']}\n"
            f"Median selected plateau duration: "
            f"{stats['median_selected_plateau_duration_s']:.3f} s",
            file=sys.stderr,
        )
    else:
        print(
            f"Inferred first sample time: "
            f"{stats['first_sample_time_s']:.6f} s\n"
            f"Inferred sample period: "
            f"{stats['sample_period_s']:.6f} s",
            file=sys.stderr,
        )

    if stats["alignment_method"] == "stationary_plateaus":
        raw_error = stats["raw_heading_consistency_median_deg"]
        phase_offset = stats["firmware_heading_phase_offset_deg"]
        corrected_error = stats[
            "phase_corrected_heading_residual_median_deg"
        ]
        print(
            f"Raw firmware/physical heading median difference: "
            f"{raw_error:.3f} deg\n"
            f"Estimated firmware heading phase offset: "
            f"{phase_offset:+.3f} deg\n"
            f"Median phase-corrected heading residual: "
            f"{corrected_error:.3f} deg",
            file=sys.stderr,
        )

        if corrected_error > 20.0:
            print(
                "WARNING: even after removing the fitted constant heading "
                "phase offset, firmware ANGLE values disagree strongly with "
                "the structurally matched Pogosim orientations.",
                file=sys.stderr,
            )
    else:
        median_error = stats["median_heading_consistency_error_deg"]
        print(
            f"Median heading consistency error: "
            f"{median_error:.3f} deg",
            file=sys.stderr,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Pogosim magnetometer reference CSV from a real "
            "Pogobot+camera experiment or from Pogosim dump+Feather output."
        )
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="output reference CSV",
    )
    common.add_argument(
        "--calibration-field-angle-deg",
        type=float,
        default=None,
        help=(
            "known calibration magnetic-field angle. In Pogosim mode this "
            "overrides Feather metadata. In real mode it is used only when "
            "validating UART ANGLE against camera angles; it does not set the "
            "generated CSV phase (use --reference-field-angle-deg for that)."
        ),
    )
    common.add_argument(
        "--angle-bin-deg",
        type=float,
        default=0.0,
        help=(
            "optionally round robot angles to this bin size in degrees; "
            "0 keeps continuous angles (default: 0)"
        ),
    )
    common.add_argument(
        "--preserve-order",
        action="store_true",
        help=(
            "preserve chronological measurement order instead of sorting "
            "output rows by angle"
        ),
    )
    common.add_argument(
        "--grid-step-deg",
        type=float,
        default=5.0,
        help=(
            "angle step of the generated Pogosim reference table "
            "(default: 5 degrees, matching the example file)"
        ),
    )
    common.add_argument(
        "--samples-per-angle",
        type=int,
        default=4,
        help=(
            "number of rows generated at each reference angle "
            "(default: 4, matching the example file)"
        ),
    )
    common.add_argument(
        "--reference-harmonic-order",
        type=int,
        default=3,
        help=(
            "harmonic order used to regularize/resample the measured raw "
            "magnetometer curve (default: 3)"
        ),
    )
    common.add_argument(
        "--reference-seed",
        type=int,
        default=12345,
        help=(
            "seed used when bootstrapping source residuals into repeated "
            "reference samples (default: 12345)"
        ),
    )
    common.add_argument(
        "--no-residual-noise",
        action="store_true",
        help=(
            "do not bootstrap source fit residuals when generating repeated "
            "reference samples"
        ),
    )
    common.add_argument(
        "--raw-output",
        action="store_true",
        help=(
            "write the directly paired samples instead of the regular "
            "0..355-degree Pogosim reference grid; mainly for debugging"
        ),
    )

    real = subparsers.add_parser(
        "real",
        parents=[common],
        help="build reference from real-robot UART + camera angles",
    )
    real.add_argument(
        "--uart",
        type=Path,
        required=True,
        help="UART output from the calibration experiment",
    )
    real.add_argument(
        "--camera",
        type=Path,
        required=True,
        help="CSV containing camera-derived true robot angles",
    )
    real.add_argument(
        "--camera-angle-column",
        default=None,
        help="camera angle column name; inferred by default",
    )
    real.add_argument(
        "--camera-angle-sign",
        choices=["auto", "1", "-1"],
        default="auto",
        help=(
            "multiply camera angles by this sign before writing them; "
            "auto chooses the convention best matching UART ANGLE records"
        ),
    )
    real.add_argument(
        "--reference-field-angle-deg",
        type=float,
        default=0.0,
        help=(
            "phase convention to write into the generated Pogosim reference. "
            "This does not need to equal physical magnetic north; it must "
            "match magnetometer.csv.calibration_field_angle (and normally "
            "the simulated world-field angle). Default: 0 degrees."
        ),
    )
    real.set_defaults(function=run_real)

    pogosim = subparsers.add_parser(
        "pogosim",
        parents=[common],
        help="build reference from Pogosim dump + data.feather",
    )
    pogosim.add_argument(
        "--dump",
        type=Path,
        required=True,
        help="Pogosim console output containing live ANGLE/MAG records",
    )
    pogosim.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Pogosim data.feather",
    )
    pogosim.add_argument(
        "--robot-id",
        type=int,
        default=0,
        help="robot ID to extract (default: 0)",
    )
    pogosim.add_argument(
        "--data-angle-unit",
        choices=["auto", "radians", "degrees"],
        default="auto",
        help="unit of data.feather angle column (default: auto)",
    )
    pogosim.add_argument(
        "--period-min-s",
        type=float,
        default=0.5,
        help="minimum candidate MAG sample period in simulation seconds",
    )
    pogosim.add_argument(
        "--period-max-s",
        type=float,
        default=3.0,
        help="maximum candidate MAG sample period in simulation seconds",
    )
    pogosim.add_argument(
        "--stationary-tolerance-deg",
        type=float,
        default=0.05,
        help="angle tolerance used to detect the stationary tail",
    )
    pogosim.add_argument(
        "--end-gap-max-s",
        type=float,
        default=10.0,
        help=(
            "maximum allowed interval between last inferred MAG sample and "
            "start of final stationary tail"
        ),
    )
    pogosim.add_argument(
        "--alignment",
        choices=["auto", "plateau", "periodic"],
        default="auto",
        help=(
            "Pogosim MAG/ground-truth pairing method. 'auto' uses stationary "
            "plateaus for dense data and falls back to the old periodic "
            "time-grid search (default: auto)"
        ),
    )
    pogosim.add_argument(
        "--stationary-speed-threshold-deg-s",
        type=float,
        default=5.0,
        help=(
            "maximum angular speed considered stationary for plateau "
            "detection (default: 5 deg/s)"
        ),
    )
    pogosim.add_argument(
        "--min-plateau-duration-s",
        type=float,
        default=0.15,
        help=(
            "minimum duration of a stationary orientation plateau "
            "(default: 0.15 s)"
        ),
    )
    pogosim.add_argument(
        "--max-plateau-save-period-s",
        type=float,
        default=0.20,
        help=(
            "in auto mode, use plateau alignment when the median Feather "
            "logging period is at most this value (default: 0.20 s)"
        ),
    )
    pogosim.set_defaults(function=run_pogosim)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        args.function(args)
    except Exception as exception:
        parser.exit(2, f"error: {exception}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
