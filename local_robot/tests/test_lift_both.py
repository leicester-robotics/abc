"""Offline checks: no CAN buses or motor SDK constructors are opened."""

import importlib.util
from pathlib import Path
import numpy as np
import pytest
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[1] / "lift_both.py"


def load():
    assert SCRIPT.exists(), "The standalone lift script has not been implemented"
    spec = importlib.util.spec_from_file_location("lift_both", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Actual register readings from this session, not simulated hardware feedback.
STARTS = np.array(
    [
        [-0.02300978, -0.02147579, -0.01380584, -0.03988361, 0.01073790, -0.28225243],
        [-0.02607763, -0.00306797, 0.0, -0.05522329, -0.06289315, -0.14419420],
    ]
)


def test_both_paths_raise_ten_cm_preserve_orientation_and_return():
    app = load()
    kin = app.Kinematics()
    plan = app.make_plan(kin, STARTS, height=0.10, seconds=5.0, hz=50)
    assert plan["q"].shape[1:] == (2, 6)
    np.testing.assert_allclose(plan["q"][0], STARTS, atol=1e-10)
    np.testing.assert_allclose(plan["q"][-1], STARTS, atol=1e-10)
    for arm in range(2):
        p0, r0 = kin.pose(STARTS[arm])
        positions = []
        for q in plan["q"][::5, arm]:
            p, r = kin.pose(q)
            positions.append(p)
            assert np.linalg.norm(p[:2] - p0[:2]) < 0.001
            assert np.linalg.norm(app.rotation_error(r, r0)) < 0.005
        positions = np.asarray(positions)
        assert abs(positions[:, 2].max() - p0[2] - 0.10) < 0.001
        assert positions[:, 2].min() >= p0[2] - 0.001
    velocity = np.diff(plan["q"], axis=0) * 50
    assert np.max(np.abs(velocity)) <= np.deg2rad(12.0) + 1e-5
    assert np.max(np.abs(velocity[[0, -1]])) < 0.001


@pytest.mark.parametrize("height", [0.0, -0.1, float("nan"), 0.3])
def test_rejects_invalid_lifts(height):
    app = load()
    with pytest.raises(ValueError):
        app.make_plan(app.Kinematics(), STARTS, height=height)


def test_joint_limits_and_nonfinite_start_are_rejected():
    app = load()
    kin = app.Kinematics()
    for value in [float("nan"), 20.0]:
        q = STARTS.copy()
        q[0, 4] = value
        with pytest.raises(ValueError):
            app.make_plan(kin, q)


def test_start_limit_error_identifies_each_joint_before_planning():
    app = load()
    kin = app.Kinematics()
    q = STARTS.copy()
    q[0, 2] = -0.15001
    q[1, 4] = kin.limits[4, 1] + 0.15001
    with pytest.raises(ValueError) as error:
        app.make_plan(kin, q)
    message = str(error.value)
    assert "left joint 3: -0.15001 rad" in message
    assert "right joint 5:" in message
    assert "model range [0.00000, 3.14159] rad" in message
    assert "0.15 rad" in message


def test_reported_base_pose_lifts_without_extending_limit_excursion():
    app = load()
    kin = app.Kinematics()
    starts = np.array(
        [
            [-0.02301, -0.01227, -0.03221, -0.03528, 0.01381, -0.01074],
            [-0.02608, -0.00307, 0.0, -0.02148, -0.04909, -0.14419],
        ]
    )
    plan = app.make_plan(kin, starts)
    lower = np.minimum(kin.limits[:, 0], starts)
    upper = np.maximum(kin.limits[:, 1], starts)
    assert np.all(plan["q"] >= lower - 1e-7)
    assert np.all(plan["q"] <= upper + 1e-7)
    np.testing.assert_allclose(plan["q"][0], starts)
    np.testing.assert_allclose(plan["q"][-1], starts)
    peak = round(plan["leg_seconds"] / plan["dt"])
    for arm in range(2):
        p0, r0 = kin.pose(starts[arm])
        p, r = kin.pose(plan["q"][peak, arm])
        np.testing.assert_allclose(p, p0 + [0, 0, 0.05], atol=0.0005)
        assert np.linalg.norm(app.rotation_error(r, r0)) < 0.002


def test_smooth_return_ends_at_saved_pre_enable_reference():
    app = load()
    kin = app.Kinematics()
    original = np.array(
        [
            [-0.02651, 0.00362, 0.00019, -0.09747, -0.00439, -0.02651],
            [-0.05360, 0.00362, 0.00057, -0.09480, -0.03681, -0.07458],
        ]
    )
    held = original.copy()
    held[:, 1:4] += [0.003, 0.005, 0.015]
    saved = original.copy()
    plan = app.make_plan(kin, held, return_to=original)
    np.testing.assert_allclose(plan["q"][0], held)
    np.testing.assert_allclose(plan["q"][-1], original, atol=1e-12)
    np.testing.assert_array_equal(original, saved)
    velocity = np.diff(plan["q"], axis=0) / plan["dt"]
    acceleration = np.diff(velocity, axis=0) / plan["dt"]
    assert np.max(np.abs(velocity)) <= app.MAX_SPEED + 1e-6
    assert np.max(np.abs(acceleration)) <= app.MAX_ACCEL + 1e-6
    assert np.max(np.abs(velocity[[0, -1]])) < 0.001
    peak = round(plan["leg_seconds"] / plan["dt"])
    for arm in range(2):
        p0, _ = kin.pose(held[arm])
        p1, _ = kin.pose(plan["q"][peak, arm])
        np.testing.assert_allclose(p1, p0 + [0, 0, 0.05], atol=0.0005)


def test_return_reference_rejects_large_startup_shift():
    app = load()
    original = STARTS.copy()
    original[1, 5] += 0.06
    with pytest.raises(ValueError, match="return reference"):
        app.make_plan(app.Kinematics(), STARTS, return_to=original)


def test_tracking_failure_is_detected_before_next_pair_command():
    app = load()
    app.check_tracking(STARTS, STARTS)
    bad = STARTS.copy()
    bad[1, 2] += 0.3
    with pytest.raises(RuntimeError):
        app.check_tracking(bad, STARTS)
    bad[1, 2] = float("nan")
    with pytest.raises(RuntimeError):
        app.check_tracking(bad, STARTS)


def test_adapter_mapping_uses_serials_after_interface_names_swap():
    app = load()
    devices = [
        dict(
            channel="can9",
            usb_serial="205434A258455017",
            up=True,
            bitrate=1000000,
            state="ERROR-ACTIVE",
        ),
        dict(
            channel="can2",
            usb_serial="209037804546500A",
            up=True,
            bitrate=1000000,
            state="ERROR-ACTIVE",
        ),
    ]
    assert app.resolve_channels(devices) == {"left": "can9", "right": "can2"}
    devices[1]["state"] = "ERROR-PASSIVE"
    with pytest.raises(RuntimeError):
        app.resolve_channels(devices)


def test_disabled_position_uses_feedback_query_and_sdk_scale():
    import can

    app = load()
    sent = []
    # Recorded right joint 6 MIT position, rather than its xout=-0.14419.
    encoded = round((-0.07496 + 12.5) * 65535 / 25)
    reply = can.Message(
        arbitration_id=0x16,
        is_extended_id=False,
        data=[6, encoded >> 8, encoded & 255, 0, 0, 0, 25, 25],
    )

    class Bus:
        def send(self, msg):
            sent.append(msg)

        def recv(self, timeout):
            return reply if sent else None

    actual = app.read_disabled_position(Bus(), 6, 0x16)
    assert abs(actual - (-0.07496)) < 0.0002
    assert len(sent) == 1
    assert sent[0].arbitration_id == 0x7FF
    assert list(sent[0].data) == [6, 0, 0xCC, 0, 0, 0, 0, 0]


@pytest.mark.parametrize("data", [[0x16, 0, 0, 0, 0, 0, 0, 0], [6, 0]])
def test_disabled_position_rejects_enabled_or_malformed_reply(data):
    import can

    app = load()

    class Bus:
        sent = False

        def send(self, msg):
            self.sent = True

        def recv(self, timeout):
            if self.sent:
                return can.Message(arbitration_id=0x16, is_extended_id=False, data=data)

    with pytest.raises(RuntimeError):
        app.read_disabled_position(Bus(), 6, 0x16)


def test_preflight_uses_control_feedback_instead_of_xout(monkeypatch):
    from i2rt.motor_config_tool import utils, dm_motor_registers as registers

    app = load()
    closed = []

    class Interface:
        def __init__(self, channel, bustype):
            self.bus = channel

        def close(self):
            closed.append(self.bus)

    def read_register(interface, motor, register):
        assert register != "xout", "xout is not the SDK control position"
        return {
            "ESC_ID": motor,
            "MST_ID": 16 + motor,
            "CTRL_MODE": 1,
            "PMAX": 12.5,
            "Gr": 40 if motor <= 3 else 10,
            "VMAX": 10 if motor <= 3 else 30,
            "TMAX": 28 if motor <= 3 else 10,
        }[register]

    monkeypatch.setattr(utils, "RawCanInterface", Interface)
    monkeypatch.setattr(registers, "read_register", read_register)
    monkeypatch.setattr(
        app, "read_disabled_position", lambda bus, motor, master: -0.07496
    )
    q = app.read_start({"left": "fake-left", "right": "fake-right"})
    np.testing.assert_allclose(q, np.full((2, 6), -0.07496))
    assert closed == ["fake-left", "fake-right"]


def test_stale_chain_feedback_rejected_even_when_sdk_server_is_running():
    app = load()
    now = app.time.monotonic()
    robot = SimpleNamespace(
        motor_chain=SimpleNamespace(running=True, last_feedback=now - 1),
        _server_thread=SimpleNamespace(is_alive=lambda: True),
        last_update=now,
    )
    session = app.Session()
    session.arms = {"left": robot, "right": robot}
    with pytest.raises(RuntimeError, match="stale"):
        session.sample()


def test_no_pair_commands_after_invalid_feedback(monkeypatch):
    app = load()

    class HardwareBoundary:
        count = 0
        commands = []

        def sample(self):
            self.count += 1
            q = np.c_[STARTS, np.ones(2)]
            if self.count > 1:
                q[1, 2] += 0.5
            return q

        def command(self, *args):
            self.commands.append(args)

    session = HardwareBoundary()
    monkeypatch.setattr(app.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="tracking"):
        app.perform(session, {"q": np.array([STARTS]), "dt": 0.02}, app.Kinematics())
    assert session.commands == []


def test_shutdown_attempts_other_chain_when_one_disable_fails():
    app = load()
    closed = []

    def failed():
        closed.append("left")
        raise RuntimeError("missing motor")

    session = app.Session()
    session.chains = [
        SimpleNamespace(motor_interface=True, close=failed),
        SimpleNamespace(motor_interface=True, close=lambda: closed.append("right")),
    ]
    with pytest.raises(RuntimeError, match="Shutdown unconfirmed"):
        session.close()
    assert closed == ["left", "right"]


def test_measured_peak_must_reach_the_requested_height():
    app = load()
    kin = app.Kinematics()
    with pytest.raises(RuntimeError, match="Cartesian"):
        app.check_lift_reached(kin, STARTS, STARTS, 0.1)


def test_self_collision_blocks_plan(monkeypatch):
    app = load()
    kin = app.Kinematics()
    monkeypatch.setattr(kin, "has_self_collision", lambda q: True, raising=False)
    with pytest.raises(ValueError, match="collision"):
        app.make_plan(kin, STARTS)


def test_partial_calibration_commands_are_neutralized_before_waiting():
    app = load()

    class Chain:
        motor_interface = True
        motor_list = list(range(7))
        command_lock = True
        torque = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])

        def set_commands(self, torques, **kwargs):
            self.torque = np.array(torques)

    chain = Chain()
    session = app.Session()
    session.chains = [chain]
    session.hold()
    np.testing.assert_array_equal(chain.torque, np.zeros(7))


def test_late_scheduler_never_bursts_commands(monkeypatch):
    app = load()

    class Clock:
        now = 0.0
        calls = 0

        def sleep(self, seconds):
            self.calls += 1
            self.now += seconds + (0.07 if self.calls == 2 else 0.0)

    clock = Clock()
    monkeypatch.setattr(app.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(app.time, "sleep", clock.sleep)

    class Boundary:
        sent = []

        def sample(self):
            return np.c_[STARTS, np.ones(2)]

        def command(self, *args):
            self.sent.append(clock.now)

    boundary = Boundary()
    plan = {
        "q": np.repeat(STARTS[None], 10, axis=0),
        "dt": 0.02,
        "leg_seconds": 5.0,
        "height": 0.1,
    }
    app.perform(boundary, plan, app.Kinematics())
    assert np.all(np.diff(boundary.sent) >= 0.02 - 1e-8)


def test_interrupt_is_masked_until_shutdown_finishes(monkeypatch):
    app = load()
    handlers = {
        app.signal.SIGINT: app.signal.default_int_handler,
        app.signal.SIGTERM: app.signal.SIG_DFL,
    }
    original = handlers.copy()

    def install(sig, handler):
        previous = handlers[sig]
        handlers[sig] = handler
        return previous

    monkeypatch.setattr(app.signal, "signal", install)
    monkeypatch.setattr(app, "await_disable", lambda session: None)

    class SessionBoundary:
        closed = False

        def close(self):
            handlers[app.signal.SIGINT](app.signal.SIGINT, None)
            self.closed = True

    session = SessionBoundary()
    app.finish_session(session)
    assert session.closed
    assert handlers == original


def test_connect_imports_work_without_abc_or_zmq(monkeypatch):
    # Replace only the hardware-opening factory; execute the real connect path.
    import builtins
    import i2rt.robots.get_robot as sdk
    import i2rt.robots.motor_chain_robot as mcr

    app = load()
    original_import = builtins.__import__
    original_calibration = mcr.detect_gripper_limits

    def guarded_import(name, *args, **kwargs):
        if name == "zmq" or name.startswith("deploy."):
            raise ModuleNotFoundError(f"Unexpected dependency: {name}")
        return original_import(name, *args, **kwargs)

    constructed = []

    def offline_factory(**kwargs):
        constructed.append(kwargs["channel"])
        return object()

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(sdk, "get_yam_robot", offline_factory)
    session = app.Session()
    session.connect({"left": "fake-left", "right": "fake-right"})
    assert constructed == ["fake-left", "fake-right"]
    assert set(session.arms) == {"left", "right"}
    assert mcr.detect_gripper_limits is original_calibration


def test_sdk_worker_first_dispatch_has_original_arm_hold(monkeypatch):
    import i2rt.robots.get_robot as sdk

    app = load()
    reference = np.r_[STARTS[0], 2.0]
    measured = np.r_[STARTS[0] + 0.01, 0.5]
    dispatched = []

    class FakeRobot:
        def __init__(self, **kwargs):
            self.motor_chain = SimpleNamespace(
                startup_hold=SimpleNamespace(reference=reference)
            )
            self.target = None
            # Deterministically exercise the SDK thread's earliest possible run.
            self.start_server()
            self.command_joint_pos(measured)
            np.testing.assert_allclose(self.target[:6], reference[:6])

        def get_joint_pos(self):
            return measured.copy()

        def command_joint_pos(self, target):
            self.target = target.copy()

        def start_server(self):
            assert self.target is not None, "Worker dispatched before hold was set"
            np.testing.assert_allclose(self.target[:6], reference[:6])
            assert self.target[6] == measured[6]
            dispatched.append(self.target.copy())

    monkeypatch.setattr(sdk, "MotorChainRobot", FakeRobot)
    monkeypatch.setattr(sdk, "get_yam_robot", lambda **kw: sdk.MotorChainRobot())
    session = app.Session()
    session.connect({"left": "fake-left", "right": "fake-right"})
    assert len(dispatched) == 2
    session.arms["left"].command_joint_pos(measured)
    np.testing.assert_allclose(session.arms["left"].target, measured)


@pytest.mark.parametrize("interrupt", [False, True])
def test_calibration_only_torques_gripper_and_clears_on_exit(monkeypatch, interrupt):
    app = load()

    class Clock:
        now = 0.0

        def sleep(self, seconds):
            self.now += seconds
            if interrupt:
                raise KeyboardInterrupt

    clock = Clock()
    monkeypatch.setattr(app.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(app.time, "sleep", clock.sleep)

    class Chain:
        motor_list = list(range(7))
        motor_direction = np.ones(7)
        commands = []
        position = 0.0

        def read_states(self):
            return [SimpleNamespace(pos=self.position, eff=9.0) for _ in range(7)]

        def set_commands(self, torques, **kwargs):
            self.commands.append(np.array(torques, copy=True))
            self.position = float(np.sign(torques[6]))

    chain = Chain()
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            app.calibrate_gripper(chain)
    else:
        limits = app.calibrate_gripper(chain)
        assert limits[0] > limits[1]
    assert chain.commands
    assert all(np.all(q[:6] == 0) for q in chain.commands)
    np.testing.assert_array_equal(chain.commands[-1], np.zeros(7))


def test_startup_hold_keeps_arm_targets_and_sdk_gains_during_jaw_torque():
    app = load()
    hold = app.StartupHold()
    start = np.r_[STARTS[0], 1.25]
    states = [SimpleNamespace(pos=float(q)) for q in start]
    hold.capture(states)
    command = hold.command(0.5)
    np.testing.assert_array_equal(command["pos"], start)
    np.testing.assert_array_equal(
        command["kp"], [80.0, 80.0, 80.0, 10.0, 10.0, 10.0, 0.0]
    )
    np.testing.assert_array_equal(command["kd"], [5.0, 5.0, 5.0, 1.5, 1.5, 1.5, 0.0])
    assert np.isfinite(command["torques"]).all()
    assert command["torques"][6] == 0.5
    stopped = hold.command(0.0)
    np.testing.assert_array_equal(stopped["kp"][:6], command["kp"][:6])
    assert stopped["torques"][6] == 0.0
    states[2].pos += 0.072
    with pytest.raises(RuntimeError, match="joint 3"):
        hold.check(states)


def test_tracking_error_identifies_arm_and_joint():
    app = load()
    bad = STARTS.copy()
    bad[1, 4] += 0.072
    with pytest.raises(RuntimeError, match="right joint 5"):
        app.check_tracking(bad, STARTS, tolerance=0.05)


@pytest.mark.parametrize("interrupt", [False, True])
def test_calibration_preserves_arm_hold_and_clears_jaw_torque(monkeypatch, interrupt):
    app = load()
    now = [0.0]

    def sleep(seconds):
        now[0] += seconds
        if interrupt:
            raise KeyboardInterrupt

    monkeypatch.setattr(app.time, "sleep", sleep)
    monkeypatch.setattr(app.time, "monotonic", lambda: now[0])

    class Chain:
        motor_list = list(range(7))
        motor_direction = np.ones(7)

        def __init__(self):
            self.position = np.r_[STARTS[0], 0.0]
            self.commands = []
            self.startup_hold = app.StartupHold()
            self.startup_hold.capture(self.read_states())

        def read_states(self):
            return [SimpleNamespace(pos=float(q)) for q in self.position]

        def set_commands(self, **kwargs):
            self.commands.append(kwargs)
            self.position[6] = float(np.sign(kwargs["torques"][6]))

    chain = Chain()
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            app.calibrate_gripper(chain)
    else:
        app.calibrate_gripper(chain)
    for command in chain.commands:
        np.testing.assert_array_equal(command["pos"][:6], STARTS[0])
        np.testing.assert_array_equal(
            command["kp"], [80.0, 80.0, 80.0, 10.0, 10.0, 10.0, 0.0]
        )
    assert chain.commands[-1]["torques"][6] == 0.0
    chain.command_lock = True
    session = app.Session()
    session.chains = [chain]
    session.hold()
    np.testing.assert_array_equal(
        chain.commands[-1]["kp"], [80.0, 80.0, 80.0, 10.0, 10.0, 10.0, 0.0]
    )
    assert chain.commands[-1]["torques"][6] == 0.0
