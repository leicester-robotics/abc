"""Calibration recovery regressions using fake streams only."""

import json
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from deploy.gello.control import calibrate_stream, gello_cli


def arguments(tmp_path):
    return SimpleNamespace(
        side="left",
        socket="unused",
        port="left-adapter",
        baudrate=2000000,
        ids=list(range(1, 8)),
        home=[0] * 6,
        negative_joints=[],
        output=tmp_path / "left.json",
        resume=False,
    )


class Bus:
    port_name = "left-adapter"
    baudrate = 2000000
    ids = list(range(1, 8))
    models = [1200] * 6 + [1190]
    paced = True

    def __init__(self):
        self.identity = (self.port_name, self.baudrate, self.ids, self.models)
        self.read = Mock()
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True


def test_reconnect_preserves_accepted_steps_and_discards_provisional_limits(tmp_path):
    args = arguments(tmp_path)
    capture = gello_cli.make_capture(args, Bus(), Bus.models)
    capture.accepted = {
        "home_raw": [1] * 7,
        "signs": [1, -1],
        "joint1_limits": [0.5, 1.5],
        "gripper_closed": 0.7,
        "direction_baseline": {"joint": 3, "raw": [1] * 7},
    }
    capture.low, capture.high = np.full(7, -10.0), np.full(7, 10.0)
    capture.last_good = capture.last_checkpoint = time.monotonic()
    capture.latest = (np.zeros(7), np.zeros(7))
    replacement = Bus()
    capture.reconnect(replacement)
    assert capture.bus is replacement
    assert capture.accepted["signs"] == [1, -1]
    assert "direction_baseline" not in capture.accepted
    np.testing.assert_allclose(capture.low, [0.5, 1, 1, 1, 1, 1, 0.7])
    np.testing.assert_allclose(capture.high, [1.5, 1, 1, 1, 1, 1, 1])
    assert capture.latest is None
    assert capture.last_good is None
    assert capture.unwrap.previous is None


def test_restore_progress_uses_accepted_home_and_realigns_encoder_revolution(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    bus = Bus()
    capture = gello_cli.make_capture(args, bus, bus.models)
    record = {
        "configuration": capture.metadata,
        "port": args.port,
        "baudrate": args.baudrate,
        "ids": args.ids,
        "accepted": {"home_raw": [6.2] * 7, "signs": [1, -1]},
        "raw_min": [-30] * 7,
        "raw_max": [30] * 7,
    }
    capture.journal.write_text(json.dumps(record) + '\n{"interrupted":')
    calibrate_stream.restore_progress(capture, args)
    assert capture.accepted["signs"] == [1, -1]
    np.testing.assert_allclose(capture.low, [6.2] * 7)
    np.testing.assert_allclose(capture.high, [6.2] * 7)
    bus.read.side_effect = lambda ids: (np.full(7, 6.2 - 2 * np.pi), time.monotonic())
    monkeypatch.setattr(gello_cli.sys, "stdin", Mock(isatty=lambda: True, readline=lambda: "\n"))
    monkeypatch.setattr(gello_cli.select, "select", lambda *args: ([True], [], []))
    restored = gello_cli.return_to_neutral(capture, np.full(7, 6.2), "confirm neutral")
    np.testing.assert_allclose(restored, [6.2] * 7)


def test_reconnect_far_from_home_rebases_without_contaminating_limits(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    bus = Bus()
    capture = gello_cli.make_capture(args, bus, bus.models)
    home = np.full(7, 6.2)
    capture.accepted = {
        "home_raw": home.tolist(),
        "signs": [1, -1],
        "joint2_limits": [5.5, 6.5],
    }
    capture.reconnect(bus)
    samples = []
    for delta in [4.0, 3.0, 2.0, 1.0, 0.0]:
        raw = home.copy()
        raw[0] = (home[0] + delta) % (2 * np.pi)
        samples.append(raw)
    bus.read.side_effect = lambda ids: (samples.pop(0), time.monotonic())
    monkeypatch.setattr(gello_cli.sys, "stdin", Mock(isatty=lambda: True, readline=lambda: "\n"))
    monkeypatch.setattr(
        gello_cli.select, "select", lambda *args: ([] if samples else [True], [], [])
    )
    restored = gello_cli.return_to_neutral(capture, home, "resume neutral")
    np.testing.assert_allclose(restored, home)
    np.testing.assert_allclose(capture.low, [6.2, 5.5, 6.2, 6.2, 6.2, 6.2, 6.2])
    np.testing.assert_allclose(capture.high, [6.2, 6.5, 6.2, 6.2, 6.2, 6.2, 6.2])
    np.testing.assert_allclose(capture.unwrap.update(home + 0.02), home + 0.02)
    assert capture.recovering is False
    assert capture.accepted["signs"] == [1, -1]


def test_resumed_calibration_skips_accepted_direction_checks(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    bus = Bus()
    capture = gello_cli.make_capture(args, bus, bus.models)
    capture.accepted = {"home_raw": [0] * 7, "signs": [1, -1]}
    prompts = []

    def until_enter(prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return np.zeros(7)
        raise gello_cli.CalibrationStreamError("disconnect")

    monkeypatch.setattr(capture, "until_enter", until_enter)
    with pytest.raises(gello_cli.CalibrationStreamError):
        gello_cli.capture_calibration(args, bus, bus.models, capture=capture)
    assert "CONFIRM NEUTRAL" in prompts[0]
    assert "JOINT 3 DIRECTION" in prompts[1]
    assert capture.accepted["signs"] == [1, -1]


def test_run_pauses_reconnects_and_retains_same_capture(tmp_path, monkeypatch, capsys):
    args = arguments(tmp_path)
    first, second = Bus(), Bus()
    factory = Mock(side_effect=[first, OSError("monitor unavailable"), second])
    monkeypatch.setattr(calibrate_stream, "StreamBus", factory)
    monkeypatch.setattr(calibrate_stream.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(calibrate_stream.sys, "stdin", Mock(isatty=lambda: False))
    captures = []

    def run(args, bus, models, capture):
        captures.append(capture)
        if len(captures) == 1:
            capture.accepted = {"home_raw": [0] * 7, "signs": [-1, 1]}
            capture.low, capture.high = np.full(7, -2.0), np.full(7, 2.0)
            raise gello_cli.CalibrationStreamError("missing status packet")
        assert capture.accepted["signs"] == [-1, 1]
        np.testing.assert_array_equal(capture.low, np.zeros(7))
        np.testing.assert_array_equal(capture.high, np.zeros(7))

    monkeypatch.setattr(calibrate_stream, "capture_calibration", run)
    calibrate_stream.run_calibration(args)
    assert captures[0] is captures[1]
    assert first.closed and second.closed
    assert factory.call_count == 3
    assert "PAUSED" in capsys.readouterr().out


def test_changed_identity_is_rejected_after_stream_failure(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    first, second = Bus(), Bus()
    second.identity = ("other-arm", second.baudrate, second.ids, second.models)
    monkeypatch.setattr(calibrate_stream, "StreamBus", Mock(side_effect=[first, second]))
    monkeypatch.setattr(calibrate_stream.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(calibrate_stream.sys, "stdin", Mock(isatty=lambda: False))
    procedure = Mock(side_effect=gello_cli.CalibrationStreamError("missing packet"))
    monkeypatch.setattr(calibrate_stream, "capture_calibration", procedure)
    with pytest.raises(ValueError, match="identity changed"):
        calibrate_stream.run_calibration(args)
    procedure.assert_called_once()
    assert first.closed and second.closed


def test_file_error_stops_instead_of_reconnecting(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    factory = Mock(return_value=Bus())
    monkeypatch.setattr(calibrate_stream, "StreamBus", factory)
    monkeypatch.setattr(calibrate_stream.sys, "stdin", Mock(isatty=lambda: False))
    monkeypatch.setattr(
        calibrate_stream, "capture_calibration", Mock(side_effect=OSError("disk full"))
    )
    with pytest.raises(OSError, match="disk full") as error:
        calibrate_stream.run_calibration(args)
    assert not isinstance(error.value, gello_cli.CalibrationStreamError)
    factory.assert_called_once()


def test_checkpoint_error_is_not_wrapped_as_stream_error(tmp_path, monkeypatch):
    bus = Bus()
    bus.read.side_effect = lambda ids: (np.zeros(7), time.monotonic())
    capture = gello_cli.Capture(bus, bus.ids)
    capture.journal = Mock()
    capture.journal.open.side_effect = OSError("disk full")
    monkeypatch.setattr(gello_cli.sys, "stdin", Mock(isatty=lambda: True, readline=lambda: "\n"))
    monkeypatch.setattr(gello_cli.select, "select", lambda *args: ([True], [], []))
    with pytest.raises(OSError, match="disk full") as error:
        capture.until_enter("home")
    assert not isinstance(error.value, gello_cli.CalibrationStreamError)
