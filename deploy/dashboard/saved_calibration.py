"""Portable leader calibration: radians for joints, normalized gripper aperture."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

TAU = 2 * np.pi


def vector(value, size):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"Expected {size} finite values")
    return result


class Unwrapper:
    """Track encoder wrap while sampling continuously; movement must be < pi/sample."""

    def __init__(self):
        self.previous = None
        self.position = None

    def update(self, raw):
        raw = vector(raw, 7)
        if self.previous is None:
            self.position = raw.copy()
        else:
            self.position += (raw - self.previous + np.pi) % TAU - np.pi
        self.previous = raw.copy()
        return self.position.copy()


@dataclass
class Calibration:
    side: str
    port: str
    baudrate: int
    ids: list[int]
    models: list[int]
    home_raw: list[float]
    home_target: list[float]
    signs: list[int]
    raw_min: list[float] | None
    raw_max: list[float] | None
    gripper_closed: float
    gripper_open: float
    version: int = 1

    def __post_init__(self):
        if self.version not in (1, 2) or self.side not in ("left", "right"):
            raise ValueError("Unsupported calibration version or arm side")
        if not self.port or type(self.baudrate) is not int or self.baudrate <= 0:
            raise ValueError("Calibration requires a port and positive baudrate")
        if (
            len(self.ids) != 7
            or len(set(self.ids)) != 7
            or any(type(i) is not int or not 0 <= i <= 252 for i in self.ids)
        ):
            raise ValueError("Calibration requires seven unique motor IDs")
        if len(self.models) != 7 or any(type(m) is not int for m in self.models):
            raise ValueError("Calibration requires seven model numbers")
        home = vector(self.home_raw, 7)
        vector(self.home_target, 6)
        signs = vector(self.signs, 6)
        if not np.isin(signs, [-1, 1]).all():
            raise ValueError("Joint signs must be -1 or +1")
        grip = vector([self.gripper_closed, self.gripper_open], 2)
        if abs(grip[1] - grip[0]) < 0.1:
            raise ValueError("Invalid gripper endpoints")
        if self.version == 2:
            if self.raw_min is not None or self.raw_max is not None:
                raise ValueError("Relative calibration does not use measured joint limits")
            return
        low, high = vector(self.raw_min, 7), vector(self.raw_max, 7)
        if np.any(high - low < 0.1):
            raise ValueError("Every joint/trigger must be swept through at least 0.1 rad")
        # A single encoder revolution cannot identify a multi-turn absolute pose after reboot.
        if np.any(high - low >= TAU - 0.04):
            raise ValueError("Travel must be less than one revolution; restrict the usable range")
        if np.any(home < low) or np.any(home > high):
            raise ValueError("Home pose is outside the measured range")
        grip = vector([self.gripper_closed, self.gripper_open], 2)
        if abs(grip[1] - grip[0]) < 0.1 or np.any(grip < low[6]) or np.any(grip > high[6]):
            raise ValueError("Invalid gripper endpoints")

    def map(self, raw):
        raw = vector(raw, 7)
        if self.version == 2:
            # XL330 torque-off position is a continuous signed encoder count.
            # Preserve full angular displacement; do not scale by measured travel.
            aligned = raw
        else:
            low, high = np.asarray(self.raw_min), np.asarray(self.raw_max)
            midpoint = (low + high) / 2
            # Choose the unique revolution nearest the center of the calibrated usable travel.
            aligned = midpoint + (raw - midpoint + np.pi) % TAU - np.pi
            if np.any(aligned < low - 0.02) or np.any(aligned > high + 0.02):
                raise ValueError(f"{self.side} leader moved outside its calibrated travel")
            aligned = np.clip(aligned, low, high)
        joints = np.asarray(self.home_target) + np.asarray(self.signs) * (
            aligned[:6] - np.asarray(self.home_raw)[:6]
        )
        grip = (aligned[6] - self.gripper_closed) / (self.gripper_open - self.gripper_closed)
        return np.r_[joints, np.clip(grip, 0, 1)]

    def save(self, path):
        # Exclusive creation protects a station's existing measured calibration.
        with Path(path).open("x") as output:
            json.dump(asdict(self), output, indent=2, allow_nan=False)
            output.write("\n")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


class RelativeMapper:
    """Track encoder wrap continuously; start with the leader near its saved home."""

    def __init__(self, calibration):
        self.calibration = calibration
        self.unwrap = Unwrapper()

    def map(self, raw):
        raw = vector(raw, 7)
        if self.calibration.version == 1:
            return self.calibration.map(raw)
        if self.unwrap.previous is None:
            home = np.asarray(self.calibration.home_raw)
            self.unwrap.previous = raw.copy()
            self.unwrap.position = home + (raw - home + np.pi) % TAU - np.pi
        return self.calibration.map(self.unwrap.update(raw))
