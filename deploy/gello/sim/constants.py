"""Shared constants for the ABC Box simulation (matches amazon-far/abc put_bottle.xml)."""

from pathlib import Path

import numpy as np

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
YAM_MJCF = ASSETS_DIR / "yam" / "yam.xml"
ABC_BOX_MJCF = ASSETS_DIR / "abc_box" / "abc_box.xml"

# Arm base poses in the world frame (from ABC put_bottle.xml).
LEFT_BASE_POS = np.array([0.2525, 0.31, 0.76])
RIGHT_BASE_POS = np.array([0.2525, -0.31, 0.76])
TABLE_Z = 0.75  # table contact plane height

# ABC 14-D convention: [left j0..j5, left grip, right j0..j5, right grip].
ABC_JOINT_ORDER = (
    [f"left_joint{i}" for i in range(1, 7)]
    + ["left_gripper"]
    + [f"right_joint{i}" for i in range(1, 7)]
    + ["right_gripper"]
)

# Control runs at 30 Hz; physics substeps at CONTROL_DT / N_SUBSTEPS.
CONTROL_DT = 1.0 / 30.0
N_SUBSTEPS = 16
PHYSICS_DT = CONTROL_DT / N_SUBSTEPS  # ~0.00208 s, close to ABC's tuned 0.002

# Home configuration for one arm (from ABC keyframe), gripper open.
HOME_QPOS_ARM = np.array([0.0, 1.047, 1.047, 0.0, 0.0, 0.0])
