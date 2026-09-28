#!/usr/bin/env python3
"""One vertical out-and-back lift for this station's two YAM arms.

--check validates a path using disabled motor feedback.
--execute enables both arms, calibrates the jaws, and performs the lift.
Uses the separately installed Joint Studio SDK, never the ABC policy stack.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import math
from pathlib import Path
import select
import signal
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
STUDIO = ROOT.parent / "yam/control/single_arm"
SIDES = ("left", "right")
SERIALS = {"left": "205434A258455017", "right": "209037804546500A"}
RATE = 50
MAX_SPEED = np.deg2rad(12.0)
MAX_ACCEL = np.deg2rad(25.0)
# Match get_yam_robot's XML +/- 0.15 rad command-limit buffer. This only
# admits endpoints; path bounds include only the measured start and return pose.
START_LIMIT_BUFFER = 0.15


def rotation_error(actual, target):
    return Rotation.from_matrix(actual @ target.T).as_rotvec()


class Kinematics:
    def __init__(self):
        import mujoco
        from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

        self.mj = mujoco
        path = Path(combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310))
        try:
            self.model = mujoco.MjModel.from_xml_path(str(path))
        finally:
            path.unlink(missing_ok=True)
        self.data = mujoco.MjData(self.model)
        joints = [self.model.joint(f"joint{i}") for i in range(1, 7)]
        self.addresses = np.array([j.qposadr[0] for j in joints])
        self.limits = np.array([j.range.copy() for j in joints])
        self.site = self.model.site("grasp_site").id

    def pose(self, q):
        self.data.qpos[self.addresses] = q
        self.mj.mj_kinematics(self.model, self.data)
        return (
            self.data.site_xpos[self.site].copy(),
            self.data.site_xmat[self.site].reshape(3, 3).copy(),
        )

    def has_self_collision(self, q):
        # Match the SDK GUI's link-collision rule. Check jaw extremes and midpoint
        # because the calibrated opening is not known during read-only preflight.
        self.data.qpos[self.addresses] = q
        for opening in (0.0, 0.5, 1.0):
            for name in ("joint7", "joint8"):
                joint = self.model.joint(name)
                self.data.qpos[joint.qposadr[0]] = (
                    joint.range[0] + opening * np.diff(joint.range)[0]
                )
            self.mj.mj_forward(self.model, self.data)
            for contact in self.data.contact:
                if contact.dist >= -0.001:
                    continue
                g1, g2 = contact.geom1, contact.geom2
                if any(
                    self.model.geom_type[g] == self.mj.mjtGeom.mjGEOM_PLANE
                    for g in (g1, g2)
                ):
                    continue
                b1, b2 = self.model.geom_bodyid[[g1, g2]]
                if (
                    b1 == b2
                    or self.model.body_parentid[b1] == b2
                    or self.model.body_parentid[b2] == b1
                ):
                    continue
                return True
        return False

    def path(self, start, height):
        p0, r0 = self.pose(start)
        # Permit only the measured limit excursion at the start.
        # Do not expand the rest of the XML limits or move farther past a limit.
        lower = np.minimum(self.limits[:, 0], start) - 1e-9
        upper = np.maximum(self.limits[:, 1], start) + 1e-9
        samples = np.linspace(0.0, 1.0, 101)
        result = [start.copy()]
        for fraction in samples[1:]:
            target = p0 + [0.0, 0.0, height * fraction]

            def residual(q):
                p, r = self.pose(q)
                return np.r_[p - target, 0.2 * rotation_error(r, r0)]

            solution = least_squares(
                residual,
                result[-1],
                bounds=(lower, upper),
                max_nfev=100,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
            error = residual(solution.x)
            if np.linalg.norm(error[:3]) > 0.0005 or np.linalg.norm(error[3:]) > 0.0004:
                raise ValueError(
                    f"Vertical lift is unreachable at {height * fraction:.3f} m"
                )
            if np.max(np.abs(solution.x - result[-1])) > 0.04:
                raise ValueError("IK changed joint configuration abruptly")
            result.append(solution.x.copy())
        return CubicSpline(samples, np.array(result), axis=0), p0, r0, lower, upper


def make_plan(kin, starts, *, height=0.05, seconds=5.0, hz=RATE, return_to=None):
    starts = np.asarray(starts, dtype=float)
    if starts.shape != (2, 6) or not np.isfinite(starts).all():
        raise ValueError("Both arms require six finite starting joint positions")
    if not np.isfinite([height, seconds, hz]).all() or not 0 < height <= 0.15:
        raise ValueError("Height must be positive and at most 0.15 m")
    if not 3 <= seconds <= 30 or hz != RATE:
        raise ValueError("Use 3–30 seconds per leg and the fixed 50 Hz rate")
    outside = (starts < kin.limits[:, 0] - START_LIMIT_BUFFER) | (
        starts > kin.limits[:, 1] + START_LIMIT_BUFFER
    )
    if np.any(outside):
        details = []
        for arm, joint in np.argwhere(outside):
            lower, upper = kin.limits[joint]
            details.append(
                f"{SIDES[arm]} joint {joint + 1}: {starts[arm, joint]:.5f} rad; "
                f"model range [{lower:.5f}, {upper:.5f}] rad"
            )
        raise ValueError(
            f"Starting pose exceeds model limits ({START_LIMIT_BUFFER:.2f} rad SDK buffer): "
            + "; ".join(details)
        )
    reference = (
        starts.copy()
        if return_to is None
        else np.array(return_to, dtype=float, copy=True)
    )
    if (
        reference.shape != (2, 6)
        or not np.isfinite(reference).all()
        or np.max(np.abs(reference - starts)) > 0.05
        or np.any(reference < kin.limits[:, 0] - START_LIMIT_BUFFER)
        or np.any(reference > kin.limits[:, 1] + START_LIMIT_BUFFER)
    ):
        raise ValueError("Invalid return reference or startup shift exceeds 0.05 rad")
    paths = [kin.path(q, height) for q in starts]
    # Retiming is shared by both arms, so they arrive and return together.
    duration = seconds
    while True:
        n = math.ceil(duration * hz)
        u = np.linspace(0.0, 1.0, n + 1)
        progress = 10 * u**3 - 15 * u**4 + 6 * u**5
        up = np.stack([path[0](progress) for path in paths], axis=1)
        up[0] = starts
        # Retrace the lift and smoothly blend the small startup displacement
        # back to the saved pre-enable reference. Quintic blending preserves
        # zero endpoint velocity and acceleration, with no final target jump.
        down = up[-2::-1] + progress[1:, None, None] * (reference - starts)
        down[-1] = reference
        q = np.concatenate([up, np.repeat(up[-1:], hz, axis=0), down])
        v = np.diff(q, axis=0) * hz
        a = np.diff(v, axis=0) * hz
        factor = max(
            np.max(np.abs(v)) / MAX_SPEED, np.sqrt(np.max(np.abs(a)) / MAX_ACCEL), 1.0
        )
        if factor <= 1.000001:
            break
        duration *= factor * 1.02
        if duration > 30:
            raise ValueError(
                "Required motion is too slow/ill-conditioned for this demo"
            )
    # Check every dispatched waypoint, including interpolated points.
    for arm, (_, p0, r0, lower, upper) in enumerate(paths):
        for index, waypoint in enumerate(q[:, arm]):
            returning = index > n + hz
            phase_lower = np.minimum(lower, reference[arm]) if returning else lower
            phase_upper = np.maximum(upper, reference[arm]) if returning else upper
            if kin.has_self_collision(waypoint):
                raise ValueError(f"{SIDES[arm]}: model self-collision on the lift path")
            if np.any(waypoint < phase_lower - 1e-7) or np.any(
                waypoint > phase_upper + 1e-7
            ):
                raise ValueError("Interpolated path exceeds joint limits")
            # A shifted original reference requires a small lateral/orientation
            # correction on descent. The ascent remains strictly vertical.
            if returning and np.any(reference[arm] != starts[arm]):
                continue
            p, r = kin.pose(waypoint)
            if (
                np.linalg.norm(p[:2] - p0[:2]) > 0.0005
                or p[2] < p0[2] - 0.0005
                or p[2] > p0[2] + height + 0.0005
                or np.linalg.norm(rotation_error(r, r0)) > 0.002
            ):
                raise ValueError(
                    "Interpolated path fails the vertical/orientation check"
                )
    return {"q": q, "dt": 1.0 / hz, "leg_seconds": n / hz, "height": height}


def check_tracking(measured, commanded, tolerance=0.12):
    measured = np.asarray(measured)
    commanded = np.asarray(commanded)
    if (
        measured.shape != (2, 6)
        or commanded.shape != (2, 6)
        or not np.isfinite(measured).all()
        or not np.isfinite(commanded).all()
    ):
        raise RuntimeError("Invalid joint feedback or target")
    error = np.max(np.abs(measured - commanded))
    if error > tolerance:
        arm, joint = np.unravel_index(
            np.argmax(np.abs(measured - commanded)), measured.shape
        )
        raise RuntimeError(
            f"{SIDES[arm]} joint {joint + 1}: Joint tracking error {error:.3f} rad "
            f"exceeds {tolerance:.3f}; measured={measured[arm, joint]:.4f}, target={commanded[arm, joint]:.4f}"
        )


def check_lift_reached(kin, measured, starts, height):
    for i, side in enumerate(SIDES):
        p, r = kin.pose(measured[i])
        p0, r0 = kin.pose(starts[i])
        if (
            np.linalg.norm(p - (p0 + [0.0, 0.0, height])) > 0.01
            or np.linalg.norm(rotation_error(r, r0)) > 0.05
        ):
            raise RuntimeError(
                f"{side}: measured Cartesian lift missed its 1 cm / 0.05 rad tolerance"
            )


def resolve_channels(devices):
    channels = {}
    for side, serial in SERIALS.items():
        matches = [d for d in devices if d["usb_serial"] == serial]
        if len(matches) != 1:
            raise RuntimeError(f"{side}: expected one adapter with serial {serial}")
        device = matches[0]
        if (
            not device["up"]
            or device["bitrate"] != 1_000_000
            or device["state"] != "ERROR-ACTIVE"
        ):
            raise RuntimeError(
                f"{side}: {device['channel']} must be healthy and UP at 1 Mbit/s"
            )
        channels[side] = device["channel"]
    return channels


def read_disabled_position(bus, motor, feedback_id):
    """Request MIT feedback without enabling torque; PMAX must be 12.5.

    DM's 0x7FF/0xCC status request is used by its refresh_motor_status SDK
    routine. xout register 81 differs from control feedback on this station.
    """
    import can

    deadline = time.monotonic() + 0.1
    while bus.recv(timeout=0) is not None:
        if time.monotonic() > deadline:
            raise RuntimeError("CAN bus is busy before feedback request")
    bus.send(
        can.Message(
            arbitration_id=0x7FF,
            is_extended_id=False,
            data=[motor, 0, 0xCC, 0, 0, 0, 0, 0],
        )
    )
    deadline = time.monotonic() + 0.2
    while time.monotonic() < deadline:
        msg = bus.recv(timeout=0.01)
        if msg is None or msg.arbitration_id != feedback_id:
            continue
        if (
            msg.is_extended_id
            or msg.is_remote_frame
            or msg.is_error_frame
            or len(msg.data) != 8
            or (msg.data[0] & 0x0F) != motor
        ):
            raise RuntimeError(f"Motor {motor}: malformed feedback reply")
        status = msg.data[0] >> 4
        if status != 0:
            raise RuntimeError(
                f"Motor {motor}: preflight requires disabled status, got {status:#x}"
            )
        # Same p16 conversion and +/-12.5 range as the installed SDK.
        return ((msg.data[1] << 8) | msg.data[2]) * 25.0 / 65535 - 12.5
    raise RuntimeError(f"Motor {motor}: feedback request timed out")


def read_start(channels):
    from i2rt.motor_config_tool.utils import RawCanInterface
    from i2rt.motor_config_tool.dm_motor_registers import read_register

    starts = []
    for side in SIDES:
        interface = RawCanInterface(channel=channels[side], bustype="socketcan")
        try:
            q = []
            for motor in range(1, 8):
                expected = {
                    "ESC_ID": motor,
                    "MST_ID": 0x10 + motor,
                    "CTRL_MODE": 1,
                    "PMAX": 12.5,
                    "Gr": 40.0 if motor <= 3 else 10.0,
                    "VMAX": 10.0 if motor <= 3 else 30.0,
                    "TMAX": 28.0 if motor <= 3 else 10.0,
                }
                for reg, value in expected.items():
                    actual = read_register(interface, motor, reg)
                    if not math.isclose(actual, value, rel_tol=1e-5, abs_tol=1e-5):
                        raise RuntimeError(
                            f"{side} motor {motor}: {reg}={actual}, expected {value}"
                        )
                raw = read_disabled_position(interface.bus, motor, 0x10 + motor)
                if not np.isfinite(raw):
                    raise RuntimeError("Nonfinite motor position")
                # Match the SDK's one-revolution startup offset convention.
                if motor <= 6:
                    q.append(
                        raw - 2 * np.pi
                        if raw > np.pi
                        else raw + 2 * np.pi
                        if raw < -np.pi
                        else raw
                    )
            starts.append(q)
        finally:
            interface.close()
    return np.array(starts)


class StartupHold:
    """Hold the measured arm pose while the SDK calibrates only the jaws."""

    def __init__(self):
        from i2rt.robots.utils import (
            ArmType,
            GripperType,
            _load_arm_config,
            combine_arm_and_gripper_xml,
        )
        from i2rt.utils.mujoco_utils import MuJoCoKDL

        self.config = _load_arm_config(ArmType.YAM)
        path = Path(combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310))
        try:
            self.kdl = MuJoCoKDL(str(path))
        finally:
            path.unlink(missing_ok=True)
        self.reference = None

    def capture(self, states):
        q = np.array([s.pos for s in states], dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise RuntimeError("Invalid startup-hold positions")
        self.reference = q

    def check(self, states):
        q = np.array([s.pos for s in states], dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise RuntimeError("Invalid calibration feedback")
        delta = np.abs(q[:6] - self.reference[:6])
        if delta.max() > 0.05:
            raise RuntimeError(
                f"Calibration arm hold: joint {delta.argmax() + 1} shifted "
                f"{delta.max():.3f} rad (maximum 0.050)"
            )

    def command(self, gripper_torque):
        if self.reference is None:
            raise RuntimeError("Startup hold has no captured pose")
        q = self.reference[:6]
        gravity = self.kdl.compute_inverse_dynamics(q, np.zeros(6), np.zeros(6))
        if not np.isfinite(gravity).all() or np.max(np.abs(gravity)) > 25.0:
            raise RuntimeError("Invalid startup gravity compensation")
        # Same pose model, compensation factors, and PD gains as get_yam_robot.
        return dict(
            torques=np.r_[gravity * self.config.gravity_comp_factor, gripper_torque],
            pos=self.reference.copy(),
            vel=np.zeros(7),
            kp=np.r_[self.config.kp, 0.0],
            kd=np.r_[self.config.kd, 0.0],
        )


def calibrate_gripper(
    motor_chain,
    gripper_index=6,
    test_torque=0.2,
    max_duration=2.0,
    position_threshold=0.01,
    check_interval=0.1,
    close_offset=0.05,
):
    """Calibrate the jaws while holding arm joints; clear jaw torque on exit."""
    count = len(motor_chain.motor_list)
    motor_direction = motor_chain.motor_direction[gripper_index]
    hold = getattr(motor_chain, "startup_hold", None)

    def send(gripper_torque):
        if hold is not None:
            motor_chain.set_commands(**hold.command(gripper_torque), get_state=False)
        else:
            torques = np.zeros(count)
            torques[gripper_index] = gripper_torque
            motor_chain.set_commands(torques=torques, get_state=False)

    positions = []
    try:
        positions.append(motor_chain.read_states()[gripper_index].pos)
        for direction in (1, -1):
            start = time.monotonic()
            last_pos = None
            stable_count = 0
            while time.monotonic() - start < max_duration:
                send(direction * test_torque)
                time.sleep(check_interval)
                states = motor_chain.read_states()
                if hold is not None:
                    hold.check(states)
                current = states[gripper_index].pos
                if not np.isfinite(current):
                    raise RuntimeError("Nonfinite gripper feedback during calibration")
                positions.append(current)
                if last_pos is not None:
                    stable_count = (
                        stable_count + 1
                        if abs(current - last_pos) < position_threshold
                        else 0
                    )
                    if stable_count >= 6:
                        break
                last_pos = current
            time.sleep(0.3)
        low, high = min(positions), max(positions)
        if not np.isfinite(positions).all() or high - low <= position_threshold:
            raise RuntimeError("Gripper calibration did not measure a usable stroke")
        limits = [high, low] if motor_direction > 0 else [low, high]
        limits[0] += close_offset * (high - low) * motor_direction
        return limits
    finally:
        send(0.0)


class Session:
    """Keep SDK resources reachable even if a constructor fails partway through."""

    def __init__(self):
        self.robots = []
        self.chains = []
        self.arms = {}

    def connect(self, channels):
        import i2rt.robots.get_robot as sdk
        import i2rt.motor_drivers.dm_driver as driver
        import i2rt.robots.motor_chain_robot as mcr
        from i2rt.robots.utils import ArmType, GripperType

        owner = self
        original_chain, original_robot = sdk.DMChainCanInterface, sdk.MotorChainRobot
        original_checks = driver.run_startup_checks
        original_calibration = mcr.detect_gripper_limits

        class Chain(original_chain):
            def __init__(self, *args, **kwargs):
                owner.chains.append(self)
                self.last_feedback = 0.0
                self.startup_hold = StartupHold()
                super().__init__(*args, **kwargs)

            def _set_commands(self, *args, **kwargs):
                result = super()._set_commands(*args, **kwargs)
                self.last_feedback = time.monotonic()
                return result

            def start_thread(self):
                # The SDK seeds commands with measured efforts. Do not replay
                # those efforts as open-loop torques during startup calibration.
                self.startup_hold.capture(self.read_states())
                self.set_commands(**self.startup_hold.command(0.0), get_state=False)
                super().start_thread()

        class Robot(original_robot):
            def __init__(self, *args, **kwargs):
                owner.robots.append(self)
                self.last_update = 0.0
                self._starting = True
                super().__init__(*args, **kwargs)
                self._starting = False

            def command_joint_pos(self, target):
                # The SDK constructor also sets a target after starting its
                # worker. Both initialization paths must use the same arm pose.
                if self._starting:
                    target = np.array(target, copy=True)
                    target[:6] = self.motor_chain.startup_hold.reference[:6]
                return super().command_joint_pos(target)

            def start_server(self):
                # Install PD before the worker's first update, not after the
                # SDK constructor returns. Keep the calibrated jaw position.
                target = np.array(self.get_joint_pos(), copy=True)
                target[:6] = self.motor_chain.startup_hold.reference[:6]
                self.command_joint_pos(target)
                super().start_server()

            def update(self):
                super().update()
                self.last_update = time.monotonic()

        def checks(*args, **kwargs):
            kwargs["repair"] = False
            return original_checks(*args, **kwargs)

        sdk.DMChainCanInterface, sdk.MotorChainRobot = Chain, Robot
        driver.run_startup_checks = checks
        mcr.detect_gripper_limits = calibrate_gripper
        try:
            for side in SIDES:
                print(
                    f"Connecting {side}: motors enable; gripper calibrates.", flush=True
                )
                self.arms[side] = sdk.get_yam_robot(
                    channel=channels[side],
                    arm_type=ArmType.YAM,
                    gripper_type=GripperType.LINEAR_4310,
                    zero_gravity_mode=False,
                    enable_auto_recovery=False,
                )
        finally:
            sdk.DMChainCanInterface, sdk.MotorChainRobot = (
                original_chain,
                original_robot,
            )
            driver.run_startup_checks = original_checks
            mcr.detect_gripper_limits = original_calibration

    def sample(self):
        positions = []
        for side in SIDES:
            robot = self.arms[side]
            chain = robot.motor_chain
            now = time.monotonic()
            if (
                not chain.running
                or not robot._server_thread.is_alive()
                or not 0 <= now - chain.last_feedback <= 0.15
                or not 0 <= now - robot.last_update <= 0.15
            ):
                raise RuntimeError(f"{side}: stale feedback or stopped SDK worker")
            q = np.asarray(robot.get_joint_pos()).copy()
            if q.shape != (7,) or not np.isfinite(q).all() or not 0 <= q[6] <= 1:
                raise RuntimeError(f"{side}: invalid SDK position")
            positions.append(q)
        return np.array(positions)

    def command(self, arm_q, grips):
        for i, side in enumerate(SIDES):
            self.arms[side].command_joint_pos(np.r_[arm_q[i], grips[i]])

    def hold(self):
        # A constructor interrupted during calibration may own a running chain
        # without a finished Robot. Clear its latched calibration torque first.
        completed = {id(r) for r in self.arms.values()}
        protected_chains = {id(r.motor_chain) for r in self.arms.values()}
        for robot in self.robots:
            if id(robot) not in completed:
                event = getattr(robot, "_stop_event", None)
                if event is not None:
                    event.set()
                thread = getattr(robot, "_server_thread", None)
                if thread is not None and thread.ident is not None:
                    thread.join(timeout=2.0)
                    if thread.is_alive():
                        chain = getattr(robot, "motor_chain", None)
                        protected_chains.add(id(chain))
                        print(
                            "Partial SDK producer did not stop; use hardware power switch."
                        )
        for chain in self.chains:
            if id(chain) not in protected_chains and hasattr(chain, "command_lock"):
                try:
                    startup_hold = getattr(chain, "startup_hold", None)
                    if startup_hold is not None and startup_hold.reference is not None:
                        chain.set_commands(**startup_hold.command(0.0), get_state=False)
                        print(
                            "Cleared jaw calibration torque; retaining startup arm hold.",
                            flush=True,
                        )
                    else:
                        chain.set_commands(
                            torques=np.zeros(len(chain.motor_list)), get_state=False
                        )
                        print("Cleared partial-startup motor commands.", flush=True)
                except Exception as exc:
                    print(
                        f"Partial startup could not be neutralized: {exc}; use hardware power switch."
                    )
        for side, robot in self.arms.items():
            try:
                now = time.monotonic()
                if (
                    robot.motor_chain.running
                    and now - robot.motor_chain.last_feedback < 0.15
                    and now - robot.last_update < 0.15
                ):
                    robot.command_joint_pos(np.array(robot.get_joint_pos(), copy=True))
                else:
                    print(
                        f"{side}: hold cannot be confirmed; support arm / use hardware stop."
                    )
            except Exception as exc:
                print(f"{side}: hold failed: {exc}")

    def close(self):
        # Stop every command producer before disabling either motor chain.
        for robot in self.robots:
            event = getattr(robot, "_stop_event", None)
            if event is not None:
                event.set()
        for robot in self.robots:
            thread = getattr(robot, "_server_thread", None)
            if thread is not None and thread.ident is not None:
                thread.join(timeout=2.0)
                if thread.is_alive():
                    raise RuntimeError(
                        "SDK producer did not stop; support arms and switch off power"
                    )
        errors = []
        for chain in self.chains:
            if getattr(chain, "motor_interface", None) is not None:
                try:
                    chain.close()
                except Exception as exc:
                    errors.append(str(exc))
        if errors:
            raise RuntimeError("Shutdown unconfirmed: " + "; ".join(errors))
        print("All connected motors acknowledged disabled.", flush=True)


def await_disable(session):
    print(
        "Holding. Support both arms, then type disable and Enter to release motor torque.",
        flush=True,
    )
    # Ctrl+C cancels movement, never silently releases an unsupported arm.
    old = {
        s: signal.signal(s, lambda *_: None) for s in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        while True:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                line = sys.stdin.readline()
                if not line:
                    print(
                        "Terminal closed. Support arms and use the hardware power switch.",
                        flush=True,
                    )
                    while True:
                        time.sleep(1.0)
                if line.strip().lower() == "disable":
                    return
            try:
                if len(session.arms) == 2:
                    session.sample()
            except RuntimeError as exc:
                print(
                    f"Hold health check: {exc}. Support arms; hardware stop may be needed.",
                    flush=True,
                )
                time.sleep(1.0)
    finally:
        for s, handler in old.items():
            signal.signal(s, handler)


def perform(session, plan, kin):
    initial = session.sample()
    grips = initial[:, 6].copy()
    check_tracking(initial[:, :6], plan["q"][0], tolerance=0.03)
    deadline = time.monotonic()
    previous = plan["q"][0]
    for i, target in enumerate(plan["q"]):
        time.sleep(max(0.0, deadline - time.monotonic()))
        if time.monotonic() - deadline > 0.1:
            raise RuntimeError("Command schedule missed by over 100 ms")
        measured = session.sample()
        check_tracking(measured[:, :6], previous)
        if np.max(np.abs(measured[:, 6] - grips)) > 0.08:
            raise RuntimeError("Gripper drifted while executing lift")
        # Verify the measured peak after its one-second pause before descending.
        if i == round(plan.get("leg_seconds", 0.0) / plan["dt"]) + RATE:
            check_lift_reached(kin, measured[:, :6], plan["q"][0], plan["height"])
        session.command(target, grips)
        previous = target
        # Delays stretch time; never burst overdue waypoints to catch up.
        deadline = time.monotonic() + plan["dt"]
    # Allow bounded settling, and verify actual return before reporting success.
    end = time.monotonic() + 3.0
    while time.monotonic() < end:
        measured = session.sample()[:, :6]
        check_tracking(measured, previous)
        if np.max(np.abs(measured - previous)) < 0.03:
            errors = [
                np.linalg.norm(kin.pose(measured[i])[0] - kin.pose(previous[i])[0])
                for i in range(2)
            ]
            if max(errors) < 0.01:
                print(
                    "Both arms returned within 1 cm / 0.03 rad of the saved return reference."
                )
                return
        time.sleep(plan["dt"])
    raise RuntimeError("Return did not settle within tolerance; holding")


def finish_session(session):
    # Keep signals masked across the transition from hold to verified disable.
    old = {
        sig: signal.signal(sig, lambda *_: None)
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        await_disable(session)
        session.close()
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


def run_motion(kin, channels, starts, height, seconds):
    """Connect, execute a checked path, and hold until the operator disables."""
    session = Session()
    old_int = signal.getsignal(signal.SIGINT)
    old_term = signal.signal(
        signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    try:
        session.connect(channels)
        ready_deadline = time.monotonic() + 2.0
        while True:
            try:
                session.sample()
                break
            except RuntimeError:
                if time.monotonic() >= ready_deadline:
                    raise
                time.sleep(0.01)
        # Record the held pose after startup; reject a significant startup shift.
        current = session.sample()[:, :6]
        check_tracking(current, starts, tolerance=0.05)
        plan = make_plan(kin, current, height=height, seconds=seconds, return_to=starts)
        perform(session, plan, kin)
    except (Exception, KeyboardInterrupt) as exc:
        signal.signal(signal.SIGINT, lambda *_: None)
        signal.signal(signal.SIGTERM, lambda *_: None)
        print(f"Movement stopped: {type(exc).__name__}: {exc}", flush=True)
        session.hold()
        raise
    finally:
        try:
            if any(
                getattr(c, "motor_interface", None) is not None for c in session.chains
            ):
                finish_session(session)
        finally:
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, old_term)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check", action="store_true", help="Read disabled feedback and check the path"
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Enable, calibrate grippers, lift, return, hold",
    )
    parser.add_argument(
        "--height", type=float, default=0.05, help="Metres; default 0.05"
    )
    parser.add_argument(
        "--seconds", type=float, default=5.0, help="Minimum seconds each way"
    )
    args = parser.parse_args()
    kin = Kinematics()
    if args.execute and not sys.stdin.isatty():
        parser.error(
            "--execute requires an interactive terminal for the final disable command"
        )
    sys.path.insert(0, str(STUDIO))
    from can_identity import acquire_channel, discover_can

    channels = resolve_channels(discover_can())
    with ExitStack() as locks:
        for side in SIDES:
            locks.enter_context(acquire_channel(channels[side], SERIALS[side]))
        starts = read_start(channels)
        plan = make_plan(kin, starts, height=args.height, seconds=args.seconds)
        print(
            f"Path checked: both arms +{args.height * 100:g} cm vertically and back; "
            f"{plan['leg_seconds']:.2f}s each way, 1s pause.",
            flush=True,
        )
        print(
            "Base Z is assumed vertical. Desk objects and inter-arm collisions are not modelled.",
            flush=True,
        )
        if args.check:
            print("Read-only check finished. Motors were not enabled.")
            return
        print(
            "Starting in 3 seconds. Both grippers will calibrate before the arm lift. Ctrl+C cancels.",
            flush=True,
        )
        time.sleep(3.0)
        run_motion(kin, channels, starts, args.height, args.seconds)


if __name__ == "__main__":
    main()
