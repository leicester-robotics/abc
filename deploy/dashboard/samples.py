"""Latest-value buffers retain acquisition time and isolate mutable arrays."""
from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ArmSample:
    acquired_at: float
    position: np.ndarray | None
    velocity: np.ndarray | None = None
    effort: np.ndarray | None = None
    raw: dict = field(default_factory=dict)


@dataclass
class LeaderSample(ArmSample):
    counts: np.ndarray | None = None
    radians: np.ndarray | None = None
    calibrated: bool = False


@dataclass
class CameraSample:
    acquired_at: float
    rgb: np.ndarray
    depth: np.ndarray | None = None
    depth_status: str = 'disabled'


class LatestSample:
    def __init__(self):
        self._lock = threading.Lock()
        self._sample = None
        self._error = ''

    def publish(self, sample):
        with self._lock:
            self._sample = copy.deepcopy(sample)
            self._error = ''

    def read(self):
        with self._lock:
            return copy.deepcopy(self._sample)

    def set_error(self, message: str):
        with self._lock:
            self._error = message

    @property
    def error(self) -> str:
        with self._lock:
            return self._error
