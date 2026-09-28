"""Portable station configuration; runtime control state is never persisted."""
from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ArmConfig:
    side: str
    device: str
    servo_ids: list[int]
    signs: list[int]
    offsets: list[float] | None = None
    mapping_verified: bool = False
    channel: str = ''
    gripper_type: str = 'linear_4310'
    gripper_range: list[float] = field(default_factory=lambda: [-0.6040225, 0.33736414])
    baudrate: int = 4_000_000
    calibration_path: str | None = None

    @property
    def calibrated(self) -> bool:
        return self.offsets is not None or self.calibration_path is not None


@dataclass
class CameraConfig:
    serial: str
    label: str
    depth: bool = False


@dataclass
class DashboardConfig:
    arms: list[ArmConfig] = field(default_factory=list)
    cameras: list[CameraConfig] = field(default_factory=list)
    model_path: str = 'abc_sim/models/yam_bimanual_empty.xml'
    asset_dir: str | None = None
    port: int = 8080
    simulation_home: list[float] | None = None


def validate(cfg: DashboardConfig) -> DashboardConfig:
    def unique(values, name):
        if len(values) != len(set(values)):
            raise ValueError(f'Duplicate {name}')
    unique([a.side for a in cfg.arms], 'arm side')
    unique([a.device for a in cfg.arms], 'serial device')
    unique([a.channel for a in cfg.arms if a.channel], 'CAN channel')
    unique([c.serial for c in cfg.cameras], 'camera serial')
    unique([c.label for c in cfg.cameras], 'camera label')
    if not 1 <= cfg.port <= 65535:
        raise ValueError('Port must be between 1 and 65535')
    if cfg.simulation_home is not None and (len(cfg.simulation_home) != 7 or not all(math.isfinite(v) for v in cfg.simulation_home)):
        raise ValueError('Simulation home must contain seven finite values')
    for arm in cfg.arms:
        if arm.side not in ('left', 'right') or not arm.device:
            raise ValueError('Each arm needs a left/right side and device path')
        if len(arm.servo_ids) != 7 or any(type(i) is not int or not 0 <= i <= 252 for i in arm.servo_ids):
            raise ValueError('Each arm needs seven valid servo IDs')
        unique(arm.servo_ids, 'servo ID')
        if len(arm.signs) != 7 or any(s not in (-1, 1) for s in arm.signs):
            raise ValueError('Joint signs must be seven values of +1 or -1')
        if arm.offsets is not None and (len(arm.offsets) != 7 or not all(math.isfinite(v) for v in arm.offsets)):
            raise ValueError('Offsets must be seven finite radians')
        if len(arm.gripper_range) != 2 or not all(math.isfinite(v) for v in arm.gripper_range) or arm.gripper_range[0] >= arm.gripper_range[1]:
            raise ValueError('Gripper range must be two increasing finite radians')
        if arm.gripper_type not in ('linear_4310', 'crank_4310', 'linear_3507', 'flexible_4310'):
            raise ValueError('Unknown gripper type')
        if arm.baudrate <= 0:
            raise ValueError('Baudrate must be positive')
    return cfg


def load_config(path: Path) -> DashboardConfig:
    try:
        value = json.loads(Path(path).read_text())
        value['arms'] = [ArmConfig(**a) for a in value.get('arms', [])]
        value['cameras'] = [CameraConfig(**c) for c in value.get('cameras', [])]
        return validate(DashboardConfig(**value))
    except (TypeError, KeyError) as error:
        raise ValueError(f'Invalid station configuration: {error}') from error


def save_config(path: Path, cfg: DashboardConfig) -> None:
    validate(cfg)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.')
    try:
        with os.fdopen(fd, 'w') as file:
            json.dump(asdict(cfg), file, indent=2, allow_nan=False)
            file.write('\n')
            file.flush()
            os.fsync(file.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
