#!/usr/bin/env python3
"""Generate a synthetic raw-magnetometer calibration CSV.

The generated data follows the same convention as raw_magnetometer_model:

    relative_angle = magnetic_north_angle - robot_theta

Each output axis is a Fourier series of that relative angle. The first
harmonic creates the offset ellipse, while the second harmonic creates a
head/tail distortion near robot headings of 0 and 180 degrees.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Vector3:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Harmonic:
    order: int
    cos_coefficient: Vector3
    sin_coefficient: Vector3


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic raw-magnetometer reference CSV."
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="magnetometer_reference.csv",
        type=Path,
        help="output CSV path (default: magnetometer_reference.csv)",
    )
    parser.add_argument(
        "--angle-step-deg",
        type=float,
        default=5.0,
        help="robot-angle spacing in degrees (default: 5)",
    )
    parser.add_argument(
        "--samples-per-angle",
        type=int,
        default=4,
        help="number of noisy samples at each angle (default: 4)",
    )
    parser.add_argument(
        "--magnetic-north-deg",
        type=float,
        default=25.0,
        help="world magnetic-north angle in degrees (default: 25)",
    )
    parser.add_argument(
        "--noise-stddev",
        type=float,
        default=3.0,
        help="independent Gaussian noise standard deviation (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="random seed (default: 12345)",
    )
    parser.add_argument(
        "--floating-point",
        action="store_true",
        help="write floating-point values instead of integer-like raw readings",
    )
    parser.add_argument(
        "--no-clamp",
        action="store_true",
        help="do not clamp generated readings to [250, 600]",
    )
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    if not math.isfinite(arguments.angle_step_deg) or arguments.angle_step_deg <= 0.0:
        raise ValueError("--angle-step-deg must be finite and positive")
    if arguments.angle_step_deg > 360.0:
        raise ValueError("--angle-step-deg cannot exceed 360")
    if arguments.samples_per_angle <= 0:
        raise ValueError("--samples-per-angle must be positive")
    if not math.isfinite(arguments.magnetic_north_deg):
        raise ValueError("--magnetic-north-deg must be finite")
    if not math.isfinite(arguments.noise_stddev) or arguments.noise_stddev < 0.0:
        raise ValueError("--noise-stddev must be finite and non-negative")


def generate_angles(step_deg: float) -> Iterable[float]:
    angle = 0.0
    epsilon = 1.0e-10
    while angle < 360.0 - epsilon:
        yield angle
        angle += step_deg


def evaluate_axis(
    center: float,
    relative_angle: float,
    harmonics: list[Harmonic],
    axis: str,
) -> float:
    value = center
    for harmonic in harmonics:
        phase = harmonic.order * relative_angle
        cos_coefficient = getattr(harmonic.cos_coefficient, axis)
        sin_coefficient = getattr(harmonic.sin_coefficient, axis)
        value += cos_coefficient * math.cos(phase)
        value += sin_coefficient * math.sin(phase)
    return value


def generate_measurement(
    robot_angle_deg: float,
    magnetic_north_deg: float,
    center: Vector3,
    harmonics: list[Harmonic],
    noise_stddev: float,
    random_engine: random.Random,
    clamp_output: bool,
    floating_point: bool,
) -> Vector3:
    relative_angle = math.radians(magnetic_north_deg - robot_angle_deg)

    values = Vector3(
        evaluate_axis(center.x, relative_angle, harmonics, "x"),
        evaluate_axis(center.y, relative_angle, harmonics, "y"),
        evaluate_axis(center.z, relative_angle, harmonics, "z"),
    )

    noisy = Vector3(
        values.x + random_engine.gauss(0.0, noise_stddev),
        values.y + random_engine.gauss(0.0, noise_stddev),
        values.z + random_engine.gauss(0.0, noise_stddev),
    )

    if clamp_output:
        noisy = Vector3(
            min(600.0, max(250.0, noisy.x)),
            min(600.0, max(250.0, noisy.y)),
            min(600.0, max(250.0, noisy.z)),
        )

    if not floating_point:
        noisy = Vector3(round(noisy.x), round(noisy.y), round(noisy.z))

    return noisy


def write_csv(arguments: argparse.Namespace) -> int:
    random_engine = random.Random(arguments.seed)

    # A biased, rotated ellipse with cross-axis coupling. These coefficients
    # keep most values in approximately [250, 600].
    center = Vector3(425.0, 420.0, 430.0)
    harmonics = [
        Harmonic(
            order=1,
            cos_coefficient=Vector3(150.0, -22.0, 16.0),
            sin_coefficient=Vector3(35.0, 126.0, -9.0),
        ),
        # Symmetric head/tail distortion around 0 and 180 degrees.
        Harmonic(
            order=2,
            cos_coefficient=Vector3(12.0, -8.0, 4.0),
            sin_coefficient=Vector3(3.0, 5.0, 1.5),
        ),
        # A small third harmonic makes the example less perfectly idealized.
        Harmonic(
            order=3,
            cos_coefficient=Vector3(2.0, -1.5, 0.8),
            sin_coefficient=Vector3(-1.0, 2.0, -0.5),
        ),
    ]

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0

    with arguments.output.open("w", newline="", encoding="utf-8") as output_file:
        output_file.write(
            f"# magnetic_north_deg={arguments.magnetic_north_deg}\n"
            f"# noise_stddev={arguments.noise_stddev}\n"
            f"# seed={arguments.seed}\n"
            "# convention: relative_angle = magnetic_north_angle - robot_angle\n"
        )
        writer = csv.writer(output_file)
        writer.writerow(["angle", "mag_x", "mag_y", "mag_z"])

        for robot_angle_deg in generate_angles(arguments.angle_step_deg):
            for _ in range(arguments.samples_per_angle):
                measurement = generate_measurement(
                    robot_angle_deg=robot_angle_deg,
                    magnetic_north_deg=arguments.magnetic_north_deg,
                    center=center,
                    harmonics=harmonics,
                    noise_stddev=arguments.noise_stddev,
                    random_engine=random_engine,
                    clamp_output=not arguments.no_clamp,
                    floating_point=arguments.floating_point,
                )
                writer.writerow(
                    [
                        f"{robot_angle_deg:.10g}",
                        measurement.x,
                        measurement.y,
                        measurement.z,
                    ]
                )
                row_count += 1

    return row_count


def main() -> int:
    arguments = parse_arguments()
    try:
        validate_arguments(arguments)
        row_count = write_csv(arguments)
    except (OSError, ValueError) as exception:
        print(f"error: {exception}")
        return 1

    print(f"wrote {row_count} rows to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
