import json
from dataclasses import asdict
from unittest.mock import Mock

import numpy as np
import pytest

from deploy.gello.control.calibration import Calibration, Unwrapper
from deploy.gello.control.dynamixel import DynamixelBus


def calibration(**changes):
    values = dict(
        side="left",
        port="/dev/test-left",
        baudrate=57600,
        ids=list(range(1, 8)),
        models=[1240] * 7,
        home_raw=[6.1] * 7,
        home_target=[0] * 6,
        signs=[1, -1, 1, 1, 1, 1],
        raw_min=[5.1] * 7,
        raw_max=[7.1] * 7,
        gripper_closed=7.1,
        gripper_open=5.1,
    )
    return Calibration(**(values | changes))


def test_wrap_signs_and_reversed_gripper():
    c = calibration()
    result = c.map(np.full(7, 6.5) % (2 * np.pi))
    np.testing.assert_allclose(result, [0.4, -0.4, 0.4, 0.4, 0.4, 0.4, 0.3])
    # Power-cycle position counters can move by full turns without changing physical pose.
    np.testing.assert_allclose(c.map(np.full(7, 6.5 - 4 * np.pi)), result)


def test_outside_travel_and_nan_rejected():
    for raw in (np.full(7, 8), np.full(7, np.nan), [0] * 6):
        with pytest.raises(ValueError):
            calibration().map(raw)


@pytest.mark.parametrize(
    "change",
    [
        {"ids": [1] * 7},
        {"raw_max": [5.15] * 7},
        {"raw_max": [12] * 7},
        {"signs": [0] * 6},
        {"home_raw": [4] * 7},
        {"gripper_open": 7.1},
        {"home_target": [float("nan")] * 6},
        {"version": 2},
    ],
)
def test_invalid_calibration(change):
    with pytest.raises(ValueError):
        calibration(**change)


def test_calibration_roundtrip_protects_existing(tmp_path):
    path = tmp_path / "calibration.json"
    original = calibration()
    original.save(path)
    assert asdict(Calibration.load(path)) == asdict(original)
    with pytest.raises(FileExistsError):
        original.save(path)
    assert json.loads(path.read_text())["side"] == "left"


def test_continuous_unwrap_crosses_zero_both_ways():
    unwrap = Unwrapper()
    for angle in [6.1, 6.2, 6.4, 6.5, 6.3, 6.0]:
        np.testing.assert_allclose(unwrap.update(np.full(7, angle % (2 * np.pi))), angle)


def fake_bus():
    bus = object.__new__(DynamixelBus)
    bus.port_name = "test"
    bus.port = Mock()
    bus.packet = Mock()
    bus.scan = Mock(return_value={1: 1240})
    bus.ping = Mock(return_value=1240)
    bus.read_torque = Mock(return_value=0)
    bus.packet.write1ByteTxRx.return_value = (0, 0)
    return bus


@pytest.mark.parametrize("mode", ["not_isolated", "wrong_model", "extra_motor", "torque"])
def test_id_assignment_refuses_unsafe_preconditions(mode):
    bus = fake_bus()
    if mode == "extra_motor":
        bus.scan.return_value[2] = 1240
    if mode == "torque":
        bus.read_torque.return_value = 1
    with pytest.raises(ValueError):
        bus.assign_id(1, 7, 999 if mode == "wrong_model" else 1240, isolated=mode != "not_isolated")
    bus.packet.write1ByteTxRx.assert_not_called()


def test_id_assignment_only_writes_id_and_verifies():
    bus = fake_bus()
    bus.scan.side_effect = [{1: 1240}, {7: 1240}]
    bus.assign_id(1, 7, 1240, isolated=True)
    bus.packet.write1ByteTxRx.assert_called_once_with(bus.port, 1, 7, 7)
    assert bus.ping.call_args.args == (7,)


def test_read_signed_ticks_and_packet_errors():
    bus = fake_bus()
    bus.packet.getProtocolVersion.return_value = 2.0
    bus.packet.syncReadTx.return_value = 0
    bus.packet.readRx.return_value = ([0, 252, 255, 255], 0, 0)  # -1024 ticks
    values, timestamp = bus.read([1, 2])
    np.testing.assert_allclose(values, [-np.pi / 2] * 2)
    assert timestamp > 0
    bus.packet.readRx.return_value = ([0] * 4, 0, 128)
    with pytest.raises(OSError):
        bus.read([1, 2])
    bus.packet.readRx.side_effect = [([0] * 4, 0, 0), ([], -3001, 0)]
    with pytest.raises(OSError):
        bus.read([1, 2])


def test_cleanup_preserves_original_open_error(monkeypatch):
    import dynamixel_sdk

    port = Mock(ser=None)
    port.openPort.side_effect = PermissionError("serial access denied")
    monkeypatch.setattr(dynamixel_sdk, "PortHandler", lambda _: port)
    with pytest.raises(PermissionError, match="serial access denied"):
        DynamixelBus("test")
    port.closePort.assert_not_called()


def test_capture_retries_short_loss_and_journals_pose(tmp_path, monkeypatch):
    import time

    from deploy.gello.control import gello_cli

    bus = Mock(port_name="test", baudrate=57600)
    bus.read.side_effect = [OSError("dropped packet"), (np.zeros(7), time.monotonic())]
    monkeypatch.setattr(gello_cli.sys, "stdin", Mock(isatty=lambda: True, readline=lambda: "\n"))
    monkeypatch.setattr(gello_cli.select, "select", lambda *args: ([True], [], []))
    journal = tmp_path / "progress.jsonl"
    capture = gello_cli.Capture(bus, list(range(1, 8)), journal=journal)
    np.testing.assert_array_equal(capture.until_enter("neutral"), np.zeros(7))
    record = json.loads(journal.read_text())
    assert record["prompt"] == "neutral"
    assert record["unwrapped"] == [0] * 7
    assert record["ids"] == list(range(1, 8))
    assert bus.read.call_count == 2


def test_capture_persistent_loss_does_not_save_a_pose(tmp_path, monkeypatch):
    from deploy.gello.control import gello_cli

    bus = Mock()
    bus.read.side_effect = OSError("disconnected")
    monkeypatch.setattr(gello_cli.sys, "stdin", Mock(isatty=lambda: True))
    journal = tmp_path / "progress.jsonl"
    with pytest.raises(OSError, match="disconnected"):
        gello_cli.Capture(bus, list(range(1, 8)), journal=journal).until_enter("neutral")
    assert bus.read.call_count == 3
    assert not journal.exists()


def test_raw_pose_capture_metadata_and_no_motor_writes():
    import time

    from deploy.gello.control.poses import capture_pose

    bus = Mock(port_name="test", baudrate=57600)
    bus.read.return_value = (np.full(7, np.pi), time.monotonic())
    record = capture_pose(bus, list(range(1, 8)), [1200] * 6 + [1190])
    np.testing.assert_allclose(record["position_deg"], [180] * 7)
    assert record["baudrate"] == 57600
    assert [call[0] for call in bus.mock_calls] == ["read"]


def test_diagnostic_monitor_keeps_missing_motor_invalid():
    from deploy.gello.control.monitor import read_motor

    bus = fake_bus()
    bus.packet.ping.return_value = (0, -3001, 0)
    row = read_motor(bus, 3)
    assert row["id"] == 3
    assert row["position_rad"] is None
    bus.packet.read4ByteTxRx.assert_not_called()


def test_diagnostic_monitor_reports_voltage_alert_with_position():
    from deploy.gello.control.monitor import read_motor

    bus = fake_bus()
    bus.packet.ping.return_value = (1200, 0, 128)
    bus.packet.read4ByteTxRx.return_value = (1024, 0, 128)
    bus.packet.read2ByteTxRx.return_value = (31, 0, 128)
    bus.packet.read1ByteTxRx.return_value = (1, 0, 128)
    row = read_motor(bus, 1)
    assert row["position_deg"] == 90
    assert row["registers"]["voltage"]["value"] == 31
    assert row["error"]
    bus.packet.write1ByteTxRx.assert_not_called()


def test_monitor_latest_file_is_complete_json(tmp_path):
    from deploy.gello.control.monitor import write_latest

    p = tmp_path / "live.json"
    write_latest(p, {"running": True, "arms": {"left": {"motors": []}}})
    write_latest(p, {"running": False, "arms": {}})
    assert json.loads(p.read_text()) == {"running": False, "arms": {}}
    assert not p.with_name(p.name + ".tmp").exists()


def test_relative_mapping_is_one_to_one_without_joint_sweeps(tmp_path):
    c = calibration(version=2, raw_min=None, raw_max=None)
    raw = np.asarray(c.home_raw).copy()
    raw[:6] += np.deg2rad([1, 1, 200, -15, 2, 10])
    result = c.map(raw)
    np.testing.assert_allclose(result[:6], np.deg2rad([1, -1, 200, -15, 2, 10]))
    path = tmp_path / "relative.json"
    c.save(path)
    np.testing.assert_allclose(Calibration.load(path).map(raw), result)


def test_relative_mapper_wraps_initial_pose_and_tracks_beyond_half_turn():
    from deploy.gello.control.calibration import RelativeMapper

    c = calibration(version=2, raw_min=None, raw_max=None, home_raw=[0] * 7)
    mapper = RelativeMapper(c)
    raw = np.zeros(7)
    raw[4] = np.deg2rad(355)
    assert np.isclose(mapper.map(raw)[4], np.deg2rad(-5))
    for degrees in range(0, 211, 10):
        raw[4] = np.deg2rad(degrees)
        target = mapper.map(raw)
    assert np.isclose(target[4], np.deg2rad(210))
