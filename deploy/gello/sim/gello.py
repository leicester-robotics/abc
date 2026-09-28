"""Dual GELLO leaders driving MuJoCo only; no physical follower backend."""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from contextlib import ExitStack, nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np

from deploy.gello.control.calibration import Calibration, RelativeMapper, vector
from deploy.gello.control.dynamixel import DynamixelBus
from deploy.gello.control.joint_limits import JointEnvelope
from deploy.gello.control.state_stream import DEFAULT_SOCKET, StreamBus
from deploy.gello.sim import constants as C
from deploy.gello.sim.sim import BimanualSim
from deploy.gello.sim.teleop import JointTeleop


class LeaderReader:
    """One bounded serial reader per leader; viewer never waits on serial I/O."""

    def __init__(self, calibration, socket_path=None):
        self.socket_path = socket_path
        self.calibration = calibration
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.sample = None
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=2)

    def _run(self):
        if self.socket_path:
            self._run_stream()
            return
        c = self.calibration
        try:
            with DynamixelBus(c.port, c.baudrate) as bus:
                if bus.verify(c.ids) != c.models:
                    raise ValueError("Motor identities differ from calibration")
                # A leader must be passive. Do not change torque state at teleop startup.
                if any(bus.read_torque(i) for i in c.ids):
                    raise ValueError("Leader torque is enabled; relax it through calibration first")
                while not self.stop.is_set():
                    raw, timestamp = bus.read(c.ids)
                    with self.lock:
                        self.sample = (raw, timestamp)
                    self.stop.wait(0.002)
        except Exception as exc:
            with self.lock:
                self.error = str(exc)

    def _run_stream(self):
        c = self.calibration
        while not self.stop.is_set():
            try:
                with StreamBus(self.socket_path, c.side) as bus:
                    if (bus.port_name, bus.baudrate, bus.ids, bus.models) != (
                        c.port,
                        c.baudrate,
                        c.ids,
                        c.models,
                    ):
                        raise ValueError("Monitor identity differs from calibration")
                    while not self.stop.is_set():
                        raw, timestamp = bus.read(c.ids)
                        with self.lock:
                            self.sample, self.error = (raw, timestamp), None
            except (OSError, ValueError) as exc:
                with self.lock:
                    self.sample, self.error = None, str(exc)
                if isinstance(exc, ValueError):
                    return
                self.stop.wait(0.1)

    def latest(self):
        with self.lock:
            if self.error:
                raise OSError(self.error)
            if self.sample is None:
                raise OSError("Waiting for leader")
            return self.sample[0].copy(), self.sample[1]


class GelloController:
    def __init__(self, sim, calibrations, max_age=0.2, envelope=None):
        if [c.side for c in calibrations] != ["left", "right"]:
            raise ValueError("Provide left then right calibration")
        if Path(calibrations[0].port).resolve() == Path(calibrations[1].port).resolve():
            raise ValueError("The two leaders must use distinct serial ports")
        self.calibrations = calibrations
        self.mappers = [RelativeMapper(c) for c in calibrations]
        self.motion = JointTeleop(sim)
        if envelope is not None:
            self.motion.low, self.motion.high = envelope.apply(
                calibrations, self.motion.low, self.motion.high
            )
            if np.any(self.motion.command < self.motion.low) or np.any(
                self.motion.command > self.motion.high
            ):
                raise ValueError("Simulator initial pose is outside the safety envelope")
        self.max_age = max_age
        self.armed = False
        self.status = "DISARMED — Space: enable, X: hold, Q: quit"

    def hold(self, reason):
        self.armed = False
        self.motion.key(ord("X"))
        self.status = reason

    def advance(self, samples, now, *, enable=False):
        try:
            if len(samples) != 2:
                raise ValueError("Need both leaders")
            timestamps = np.asarray([s[1] for s in samples])
            if (
                not np.isfinite(timestamps).all()
                or np.any(now - timestamps > self.max_age)
                or np.any(timestamps > now)
                or np.ptp(timestamps) > self.max_age / 2
            ):
                raise ValueError("Stale or unsynchronized leader samples")
            target = np.concatenate([mapper.map(s[0]) for mapper, s in zip(self.mappers, samples)])
            vector(target, 14)
        except (ValueError, OSError) as exc:
            self.hold(f"FAULT — {exc}; Space to re-arm after correction")
            return self.motion.command.copy(), None
        if enable:
            self.armed = True
        if self.armed:
            self.motion.target = np.clip(target, self.motion.low, self.motion.high)
            self.status = "ENABLED — X: hold, Space: enable, Q: quit"
            return self.motion.advance(C.CONTROL_DT), target
        return self.motion.command.copy(), target


def mock_calibrations():
    result = []
    for side in ("left", "right"):
        result.append(
            Calibration(
                side=side,
                port=f"mock-{side}",
                baudrate=57600,
                ids=list(range(1, 8)),
                models=[1240] * 7,
                home_raw=[0] * 7,
                home_target=C.HOME_QPOS_ARM.tolist(),
                signs=[1] * 6,
                raw_min=[-1] * 7,
                raw_max=[1] * 7,
                gripper_closed=-1,
                gripper_open=1,
            )
        )
    return result


def run(args):
    calibrations = (
        mock_calibrations()
        if args.mock
        else [Calibration.load(args.left), Calibration.load(args.right)]
    )
    sim = BimanualSim()
    if not args.mock:
        import mujoco

        home = np.concatenate([np.r_[c.home_target, 1.0] for c in calibrations])
        JointTeleop(sim, home=home)  # Validate targets against the actual model.
        for c in calibrations:
            for index, angle in enumerate(c.home_target, 1):
                sim.data.qpos[sim.model.joint(f"{c.side}_joint{index}").qposadr[0]] = angle
        mujoco.mj_forward(sim.model, sim.data)
    envelope = JointEnvelope.load(args.limits) if getattr(args, "limits", None) else None
    controller = GelloController(sim, calibrations, envelope=envelope)
    keys = queue.SimpleQueue()
    rows = []
    termination = "completed"
    # Reserve output before opening devices, and never overwrite an existing episode.
    with ExitStack() as stack:
        stack.callback(sim.close)
        output = stack.enter_context(Path(args.record).open("xb")) if args.record else None
        readers = (
            []
            if args.mock
            else [
                stack.enter_context(LeaderReader(c, getattr(args, "socket", None)))
                for c in calibrations
            ]
        )
        if args.headless:
            viewer = None
        else:
            import mujoco
            import mujoco.viewer

            viewer = stack.enter_context(
                mujoco.viewer.launch_passive(
                    sim.model,
                    sim.data,
                    key_callback=keys.put,
                )
            )
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = sim.model.camera("front").id
        started = time.monotonic()
        step = 0
        previous_status = None
        print("MuJoCo followers only. Space: enable; X: hold; Q: quit.", flush=True)
        try:
            while (viewer is None or viewer.is_running()) and (
                args.seconds is None or step * C.CONTROL_DT < args.seconds
            ):
                tick = time.monotonic()
                enable = args.mock and step == 0
                while not keys.empty():
                    key = keys.get_nowait()
                    if key == ord("Q"):
                        termination = "user_quit"
                        return
                    if key == ord("X"):
                        controller.hold("DISARMED — Space to enable")
                        enable = False
                    elif key == ord(" "):
                        enable = True
                try:
                    if args.mock:
                        phase = step * C.CONTROL_DT
                        left, right = np.zeros(7), np.zeros(7)
                        left[0], right[0] = 0.3 * np.sin(phase), -0.3 * np.sin(phase)
                        left[6] = right[6] = np.cos(phase)
                        samples = [(left, tick), (right, tick)]
                    else:
                        samples = [r.latest() for r in readers]
                    with viewer.lock() if viewer else nullcontext():
                        action, target = controller.advance(
                            samples, time.monotonic(), enable=enable
                        )
                except OSError as exc:
                    with viewer.lock() if viewer else nullcontext():
                        controller.hold(f"FAULT — {exc}")
                        action, target = controller.motion.command.copy(), None
                    samples = None
                with viewer.lock() if viewer else nullcontext():
                    state = sim.qpos_abc()
                    sim.step(action)
                    next_state = sim.qpos_abc()
                if not np.isfinite(next_state).all():
                    raise RuntimeError("MuJoCo produced non-finite state")
                if output:
                    rows.append(
                        (
                            tick - started,
                            step * C.CONTROL_DT,
                            state,
                            action,
                            next_state,
                            target if target is not None else np.full(14, np.nan),
                            controller.armed,
                            [s[1] - started for s in samples] if samples else [np.nan] * 2,
                        )
                    )
                if controller.status != previous_status:
                    print(controller.status, flush=True)
                    previous_status = controller.status
                if viewer:
                    viewer.set_texts(
                        [
                            (
                                mujoco.mjtFontScale.mjFONTSCALE_150,
                                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                controller.status,
                                "",
                            )
                        ]
                    )
                    viewer.sync()
                step += 1
                if not args.fast:
                    time.sleep(max(0, C.CONTROL_DT - (time.monotonic() - tick)))
            if viewer is not None and not viewer.is_running():
                termination = "viewer_closed"
        except KeyboardInterrupt:
            termination = "interrupted"
        except Exception:
            termination = "failed"
            raise
        finally:
            if output:
                np.savez_compressed(
                    output,
                    wall_time=np.array([r[0] for r in rows]),
                    sim_time=np.array([r[1] for r in rows]),
                    state=np.array([r[2] for r in rows]).reshape(-1, 14),
                    action=np.array([r[3] for r in rows]).reshape(-1, 14),
                    next_state=np.array([r[4] for r in rows]).reshape(-1, 14),
                    leader_target=np.array([r[5] for r in rows]).reshape(-1, 14),
                    armed=np.array([r[6] for r in rows], dtype=bool),
                    sample_time=np.array([r[7] for r in rows]).reshape(-1, 2),
                    metadata=json.dumps(
                        {
                            "version": 1,
                            "backend": "mujoco",
                            "mock": args.mock,
                            "termination": termination,
                            "control_hz": 30,
                            "joint_order": C.ABC_JOINT_ORDER,
                            "calibrations": [asdict(c) for c in calibrations],
                        }
                    ),
                )
                print(f"Saved {len(rows)} frames to {args.record}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--socket",
        nargs="?",
        const=DEFAULT_SOCKET,
        help="Use running monitor instead of opening USB",
    )
    parser.add_argument("--left", help="Measured left calibration JSON")
    parser.add_argument("--right", help="Measured right calibration JSON")
    parser.add_argument("--limits", help="Optional measured joint safety envelope JSON")
    parser.add_argument(
        "--mock", action="store_true", help="Explicit synthetic leaders, no devices"
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--fast", action="store_true", help="Unpaced headless mock test only")
    parser.add_argument("--record", help="New NPZ diagnostic trajectory (no camera images)")
    args = parser.parse_args()
    if args.mock and (args.left or args.right):
        parser.error("--mock cannot be combined with hardware calibration")
    if not args.mock and not (args.left and args.right):
        parser.error("Supply --left and --right, or explicitly use --mock")
    if args.seconds is not None and (not np.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds must be finite and positive")
    if args.headless and not args.mock:
        parser.error("Hardware leaders require the viewer's explicit Space-to-enable control")
    if args.fast and not (args.mock and args.headless):
        parser.error("--fast requires --mock --headless")
    if (args.record or args.headless) and (args.seconds is None or args.seconds > 600):
        parser.error("Recording/headless runs require --seconds in (0, 600]")
    try:
        run(args)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
