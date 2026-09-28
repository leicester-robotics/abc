"""Interactive joint jogging for the simulated ABC Box only."""

from __future__ import annotations

import argparse
import queue
import time

import mujoco
import numpy as np

from deploy.gello.control.profiles import GELLO_REST_HOME
from deploy.gello.sim import constants as C
from deploy.gello.sim.sim import BimanualSim


class JointTeleop:
    """Keyboard targets with bounded position commands at each physics tick."""

    def __init__(self, sim: BimanualSim, home: np.ndarray | None = None):
        self.sim = sim
        self.side = 0
        self.joint = 0
        limits = []
        for side in ("left", "right"):
            limits.extend(sim.model.joint(f"{side}_joint{i}").range for i in range(1, 7))
            limits.append([0, 1])
        self.low, self.high = np.asarray(limits).T
        self.home = np.asarray(sim.home_action() if home is None else home, dtype=float).copy()
        if (
            self.home.shape != (14,)
            or not np.isfinite(self.home).all()
            or np.any(self.home < self.low)
            or np.any(self.home > self.high)
        ):
            raise ValueError("Reference home must contain 14 finite values within model limits")
        self.command = np.clip(sim.qpos_abc(), self.low, self.high)
        self.target = self.command.copy()

    def key(self, key: int):
        if key == ord("L"):
            self.side = 0
        elif key == ord("R"):
            self.side = 1
        elif ord("1") <= key <= ord("6"):
            self.joint = key - ord("1")
        elif key in (ord("J"), ord("K")):
            self.target[self.side * 7 + self.joint] += 0.05 * (1 if key == ord("K") else -1)
        elif key in (ord("U"), ord("O")):
            self.target[self.side * 7 + 6] += 0.1 * (1 if key == ord("O") else -1)
        elif key == ord("H"):
            self.target = self.home.copy()
        elif key == ord("X"):
            self.command = np.clip(self.sim.qpos_abc(), self.low, self.high)
            self.target = self.command.copy()
        np.clip(self.target, self.low, self.high, out=self.target)

    def advance(self, dt: float):
        # 0.5 rad/s for arm targets; 0.5 normalized aperture/s for grippers.
        step = 0.5 * dt
        self.command += np.clip(self.target - self.command, -step, step)
        return self.command.copy()

    def status(self):
        index = self.side * 7 + self.joint
        return (
            f"{'LEFT' if self.side == 0 else 'RIGHT'} arm | Joint {self.joint + 1}\n"
            f"Measured: {self.sim.qpos_abc()[index]:+.3f} rad | Target: {self.target[index]:+.3f} rad\n"
            "L / R: select arm | 1-6: select joint\n"
            "J / K: jog -/+ 0.05 rad (press repeatedly)\n"
            "U / O: close/open gripper | H: home both arms\n"
            "X: cancel motion and hold | Close window: quit"
        )


def main():
    import mujoco.viewer as mjviewer

    parser = argparse.ArgumentParser(description=__doc__)
    home_args = parser.add_mutually_exclusive_group()
    home_args.add_argument(
        "--home", type=float, nargs=6, help="Reference joint angles for both arms"
    )
    home_args.add_argument(
        "--gello-home",
        action="store_true",
        help="Use the shared GELLO resting pose (joint 4 at its upper stop)",
    )
    args = parser.parse_args()
    sim = BimanualSim()
    if args.gello_home:
        args.home = list(GELLO_REST_HOME)
    home = None if args.home is None else np.r_[args.home, 1.0, args.home, 1.0]
    control = JointTeleop(sim, home=home)
    if home is not None:
        control.key(ord("H"))
    keys = queue.SimpleQueue()
    print(control.status(), flush=True)
    with mjviewer.launch_passive(sim.model, sim.data, key_callback=keys.put) as viewer:
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = sim.model.camera("front").id
        while viewer.is_running():
            started = time.monotonic()
            with viewer.lock():
                while not keys.empty():
                    control.key(keys.get_nowait())
                sim.step(control.advance(C.CONTROL_DT))
                label = control.status()
            viewer.set_texts(
                [(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, label, "")]
            )
            viewer.sync()
            time.sleep(max(0, C.CONTROL_DT - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
