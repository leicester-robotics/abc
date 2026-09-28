"""Optional robot-angle safety envelope; never rescales leader motion."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from deploy.gello.control.calibration import vector


def fingerprint(calibration):
    return hashlib.sha256(json.dumps(asdict(calibration), sort_keys=True).encode()).hexdigest()


@dataclass
class JointEnvelope:
    calibration_hashes: list[str]
    low: list[float]
    high: list[float]
    version: int = 1

    def __post_init__(self):
        low, high = vector(self.low, 12), vector(self.high, 12)
        if self.version != 1 or len(self.calibration_hashes) != 2 or np.any(high - low < 0.1):
            raise ValueError("Each joint needs at least 0.1 rad of measured usable range")

    def apply(self, calibrations, model_low, model_high):
        if self.calibration_hashes != [fingerprint(c) for c in calibrations]:
            raise ValueError("Safety envelope belongs to different calibration files")
        indices = np.r_[0:6, 7:13]
        low, high = model_low.copy(), model_high.copy()
        low[indices] = np.maximum(low[indices], self.low)
        high[indices] = np.minimum(high[indices], self.high)
        if np.any(low > high):
            raise ValueError("Safety envelope does not overlap the robot joint limits")
        return low, high

    def save(self, path):
        with Path(path).open("x") as output:
            json.dump(asdict(self), output, indent=2, allow_nan=False)
            output.write("\n")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))
