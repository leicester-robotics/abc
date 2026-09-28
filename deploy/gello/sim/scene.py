"""Bimanual ABC Box scene composition via mujoco.MjSpec.

Attaches two ABC-tuned YAM arms (assets/yam/yam.xml) into the task-free cell
(assets/abc_box/abc_box.xml) at the exact base poses from ABC's put_bottle.xml,
with "left_" / "right_" prefixes, then optionally adds task objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from deploy.gello.sim import constants as C


@dataclass
class CubeSpec:
    """A free cube on the table plus a visual target site."""

    half_size: float = 0.0225
    pos: np.ndarray = field(default_factory=lambda: np.array([0.55, -0.25, 0.75 + 0.0225]))
    yaw: float = 0.0
    mass: float = 0.05
    rgba: tuple = (0.85, 0.30, 0.10, 1.0)
    target_pos: np.ndarray = field(default_factory=lambda: np.array([0.55, 0.25]))


def build_spec(cube: CubeSpec | None = None) -> mujoco.MjSpec:
    """Compose the bimanual cell. Returns an uncompiled MjSpec."""
    spec = mujoco.MjSpec.from_file(str(C.ABC_BOX_MJCF))

    for prefix, base_pos in (("left_", C.LEFT_BASE_POS), ("right_", C.RIGHT_BASE_POS)):
        arm = mujoco.MjSpec.from_file(str(C.YAM_MJCF))
        frame = spec.worldbody.add_frame(pos=base_pos.tolist())
        frame.attach_body(arm.body("arm"), prefix, "")

    # Rename wrist cameras to the ABC names ("left", "right").
    for cam in spec.cameras:
        if cam.name == "left_wrist_cam":
            cam.name = "left"
        elif cam.name == "right_wrist_cam":
            cam.name = "right"

    if cube is not None:
        add_cube(spec, cube)

    _add_home_keyframe(spec, cube)
    spec.option.timestep = C.PHYSICS_DT
    return spec


def add_cube(spec: mujoco.MjSpec, cube: CubeSpec) -> None:
    cq = np.array([np.cos(cube.yaw / 2), 0.0, 0.0, np.sin(cube.yaw / 2)])
    body = spec.worldbody.add_body(name="cube", pos=cube.pos.tolist(), quat=cq.tolist())
    body.add_freejoint(name="cube_joint")
    body.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[cube.half_size] * 3,
        rgba=list(cube.rgba),
        mass=cube.mass,
        friction=[1.0, 0.03, 0.003],
        condim=6,
        solref=[0.004, 1],
        solimp=[0.998, 0.998, 0.001, 0.5, 2],
        priority=1,
    )
    # Purely visual target marker on the table.
    spec.worldbody.add_site(
        name="target_site",
        pos=[float(cube.target_pos[0]), float(cube.target_pos[1]), C.TABLE_Z + 0.001],
        size=[0.04, 0.04, 0.001],
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        rgba=[0.1, 0.7, 0.2, 0.35],
    )


def _add_home_keyframe(spec: mujoco.MjSpec, cube: CubeSpec | None) -> None:
    arm_q = C.HOME_QPOS_ARM
    one_arm = np.concatenate([arm_q, [0.0475, -0.0475]])  # two finger joints, open
    qpos = np.concatenate([one_arm, one_arm])
    ctrl = np.concatenate([arm_q, [0.0475], arm_q, [0.0475]])
    if cube is not None:
        cq = np.array([np.cos(cube.yaw / 2), 0.0, 0.0, np.sin(cube.yaw / 2)])
        qpos = np.concatenate([qpos, cube.pos, cq])
    key = spec.add_key(name="home", qpos=qpos.tolist(), ctrl=ctrl.tolist())
    del key


def build_model(cube: CubeSpec | None = None) -> mujoco.MjModel:
    return build_spec(cube).compile()
