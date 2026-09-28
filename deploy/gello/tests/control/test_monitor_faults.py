"""Dropout diagnostics remain read-only and outside normal sampling."""

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np

from deploy.gello.control import monitor


class Stop:
    def __init__(self):
        self.stopped = False

    def is_set(self):
        return self.stopped

    def set(self):
        self.stopped = True

    def wait(self, seconds):
        return self.stopped


def fake_bus():
    bus = MagicMock()
    bus.__enter__.return_value = bus
    bus.port = SimpleNamespace(readPort=lambda length: b"")
    bus.verify.return_value = [1200] * 6 + [1190]
    return bus


def test_healthy_sampling_has_no_diagnostics_or_log(tmp_path, monkeypatch):
    stop, bus = Stop(), fake_bus()

    def read(ids):
        stop.set()
        return np.zeros(7), time.monotonic()

    bus.read.side_effect = read
    monkeypatch.setattr(monitor, "DynamixelBus", Mock(return_value=bus))
    probe = Mock(side_effect=AssertionError("diagnostics on healthy acquisition"))
    monkeypatch.setattr(monitor, "read_motor", probe)
    path = tmp_path / "left.errors.jsonl"
    reader = monitor.ArmReader("test", 2000000, stop, side="left", error_log=path)
    reader.run()
    assert reader.snapshot()["error"] is None
    assert reader.snapshot()["last_error_timestamp_utc"] is None
    probe.assert_not_called()
    assert not path.exists()
    bus.relax.assert_not_called()
    bus.assign_id.assert_not_called()


def test_original_error_persisted_before_probes_and_retained_after_recovery(tmp_path, monkeypatch):
    stop = Stop()
    first, second = fake_bus(), fake_bus()
    first.read.side_effect = OSError("missing status packet")

    def recovered(ids):
        stop.set()
        return np.ones(7), time.monotonic()

    second.read.side_effect = recovered
    factory = Mock(side_effect=[first, second])
    monkeypatch.setattr(monitor, "DynamixelBus", factory)
    path = tmp_path / "left.errors.jsonl"

    def probe(bus, motor_id):
        original = json.loads(path.read_text().splitlines()[0])
        assert original["event"] == "acquisition_error"
        assert original["error"] == "missing status packet"
        assert not bus.__exit__.called
        if motor_id == 2:
            raise OSError("probe timeout")
        return {"id": motor_id, "model": 1200, "registers": {"voltage": {"value": 45}}}

    diagnostics = Mock(side_effect=probe)
    monkeypatch.setattr(monitor, "read_motor", diagnostics)
    reader = monitor.ArmReader("test", 2000000, stop, side="left", error_log=path)
    reader.run()
    state = reader.snapshot()
    assert state["error"] is None
    assert state["last_error"] == "missing status packet"
    assert state["last_error_timestamp_utc"]
    assert state["read_errors"] == 1
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event"] for row in rows] == [
        "acquisition_error",
        "motor_diagnostics",
        "motor_diagnostics",
    ]
    assert [row.get("phase") for row in rows[1:]] == ["fault", "reconnected"]
    assert rows[1]["motors"][1] == {"id": 2, "error": "probe timeout"}
    assert len(rows[1]["motors"]) == len(rows[2]["motors"]) == 7
    assert diagnostics.call_count == 14
    assert factory.call_count == 2
    for bus in (first, second):
        bus.relax.assert_not_called()
        bus.assign_id.assert_not_called()


def test_log_failure_stops_monitor_and_exposes_original_error(tmp_path, monkeypatch):
    stop, bus = Stop(), fake_bus()
    bus.read.side_effect = OSError("frame timeout")
    monkeypatch.setattr(monitor, "DynamixelBus", Mock(return_value=bus))
    probe = Mock()
    monkeypatch.setattr(monitor, "read_motor", probe)
    reader = monitor.ArmReader(
        "test", 2000000, stop, side="left", error_log=tmp_path / "missing" / "errors.jsonl"
    )
    reader.run()
    assert stop.is_set()
    assert isinstance(reader.fatal_error, OSError)
    state = reader.snapshot()
    assert state["last_error"] == "frame timeout"
    assert state["diagnostics_log_error"]
    assert state["valid"] is False
    probe.assert_not_called()


def test_connection_open_error_is_logged_without_probe(tmp_path, monkeypatch):
    stop = Stop()

    def unavailable(*args):
        stop.set()
        raise OSError("adapter unavailable")

    monkeypatch.setattr(monitor, "DynamixelBus", unavailable)
    probe = Mock()
    monkeypatch.setattr(monitor, "read_motor", probe)
    path = tmp_path / "right.errors.jsonl"
    reader = monitor.ArmReader("right", 2000000, stop, side="right", error_log=path)
    reader.run()
    assert json.loads(path.read_text())["error"] == "adapter unavailable"
    assert reader.snapshot()["last_error"] == "adapter unavailable"
    probe.assert_not_called()
