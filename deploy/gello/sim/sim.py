"""GELLO preview backed by ABC's existing MuJoCo environment.

Only scene construction and the old teleop convenience API live here. ABC owns
joint indexing, action application, physics stepping, and state projection.
"""

import mujoco
import numpy as np

from abc_sim.config import get_i2rt_sim_config
from abc_sim.env import MuJoCoYAMEnv, project_policy_state
from deploy.gello.sim import constants as C
from deploy.gello.sim.scene import CubeSpec, build_model


class BimanualSim(MuJoCoYAMEnv):
    def __init__(self, cube: CubeSpec | None = None):
        self.cube = cube if cube is not None else CubeSpec()
        config = get_i2rt_sim_config()
        for robot in config.robots.values():
            robot.init_q = [*C.HOME_QPOS_ARM, 1.0]
        super().__init__(
            config=config,
            chunk_dim=1,
            render_cameras=False,
            physics_dt=C.PHYSICS_DT,
            control_decimation=C.N_SUBSTEPS,
        )
        super().reset(randomize=False)
        # Preserve the original preview's open fingers and cube keyframe.
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        mujoco.mj_forward(self.model, self.data)

    def setup_model(self):
        # Bundle the original task-free preview, without downloading task assets.
        self._bind_model(build_model(self.cube))

    def step(self, action):
        return super().step(action, render_obs=False)

    def qpos_abc(self):
        return project_policy_state(
            self.data.qpos, self._qpos_indices, self._gripper_indices, dtype=np.float64
        )

    def arm_qpos(self, side):
        offset = {"left": 0, "right": 7}[side]
        return self.qpos_abc()[offset : offset + 6]

    def home_action(self):
        return self.get_init_q()
