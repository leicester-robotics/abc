"""Monitor behavior without serial devices or wall-clock sleeps."""

from unittest.mock import MagicMock, call

import numpy as np
import pytest

from deploy.gello.control import monitor
from deploy.gello.control.dynamixel import DynamixelBus


class Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now


class Stop:
    def __init__(self, clock):
        self.clock = clock
        self.stopped = False
        self.on_wait = lambda: None

    def is_set(self):
        return self.stopped

    def wait(self, seconds):
        self.on_wait()
        self.clock.now += seconds
        return self.stopped


def fake_bus():
    bus = MagicMock(spec=DynamixelBus)
    bus.port = MagicMock()
    bus.__enter__.return_value = bus
    bus.verify.return_value = [1200] * 6 + [1190]
    return bus


def test_terminal_order_and_invalid_arm_values():
    arms = {
        "left": {"valid": True, "position_deg": list(range(7))},
        "right": {"valid": False, "position_deg": [99] * 7},
    }
    assert monitor.format_line(arms) == (
        "Right: [NA,NA,NA,NA,NA,NA,NA] Left: [0.00,1.00,2.00,3.00,4.00,5.00,6.00]"
    )


def test_fresh_pose_becomes_stale_without_worker_updates(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(monitor.time, "monotonic", clock.monotonic)
    reader = monitor.ArmReader("test", 2000000, Stop(clock))
    assert reader.snapshot()["valid"] is False
    reader.publish(error=None, position_deg=[12.0] * 7, sample_monotonic=clock.now)
    assert reader.snapshot()["valid"] is True
    clock.now += 0.201
    stale = reader.snapshot()
    assert stale["age_seconds"] == pytest.approx(0.201)
    assert stale["valid"] is False
    assert monitor.format_line({"right": stale, "left": stale}) == (
        "Right: [NA,NA,NA,NA,NA,NA,NA] Left: [NA,NA,NA,NA,NA,NA,NA]"
    )


@pytest.mark.parametrize(
    ("previous", "period", "now", "expected"),
    [(0.0, 0.01, 0.005, 0.01), (0.0, 0.01, 0.025, 0.03), (1.0, 0.2, 2.05, 2.2)],
)
def test_deadline_skips_missed_slots(previous, period, now, expected):
    deadline = monitor.next_deadline(previous, period, now)
    assert deadline == pytest.approx(expected)
    assert deadline > now


def test_worker_reads_without_motor_writes_and_preserves_sample_time(monkeypatch):
    clock = Clock()
    stop = Stop(clock)
    bus = fake_bus()

    def read(ids):
        started = clock.now
        clock.now += 0.003
        stop.stopped = True
        return np.arange(7) * np.pi / 2, started

    bus.read.side_effect = read
    factory = MagicMock(return_value=bus)
    monkeypatch.setattr(monitor, "DynamixelBus", factory)
    monkeypatch.setattr(monitor.time, "monotonic", clock.monotonic)
    reader = monitor.ArmReader("test-left", 2000000, stop)
    reader.run()
    state = reader.snapshot()
    assert state["valid"] is True
    assert state["sequence"] == 1
    assert state["sample_monotonic"] == 100.0
    assert state["read_duration_ms"] == pytest.approx(3)
    np.testing.assert_allclose(state["position_deg"], np.arange(7) * 90)
    factory.assert_called_once_with("test-left", 2000000)
    assert bus.method_calls == [call.verify(list(range(1, 8))), call.read(list(range(1, 8)))]
    bus.__exit__.assert_called_once()


def test_worker_error_invalidates_pose_and_reconnects(monkeypatch):
    clock = Clock()
    stop = Stop(clock)
    first, second = fake_bus(), fake_bus()
    first.read.side_effect = [
        (np.zeros(7), clock.now),
        OSError("motor disconnected"),
    ]

    def recovered_read(ids):
        stop.stopped = True
        return np.ones(7), clock.now

    second.read.side_effect = recovered_read
    factory = MagicMock(side_effect=[first, second])
    monkeypatch.setattr(monitor, "DynamixelBus", factory)
    monkeypatch.setattr(monitor.time, "monotonic", clock.monotonic)
    reader = monitor.ArmReader("test-right", 57600, stop)
    observations = []
    stop.on_wait = lambda: observations.append(reader.snapshot())
    reader.run()
    failed = next(state for state in observations if state["error"] is not None)
    assert failed["error"] == "motor disconnected"
    assert failed["valid"] is False
    assert failed["position_deg"] is None
    assert failed["position_rad"] is None
    assert failed["read_hz"] == 0.0
    recovered = reader.snapshot()
    assert recovered["valid"] is True
    assert recovered["error"] is None
    assert recovered["sequence"] == 2
    np.testing.assert_allclose(recovered["position_rad"], np.ones(7))
    assert factory.call_count == 2
    ids = list(range(1, 8))
    assert first.method_calls == [call.verify(ids), call.read(ids), call.read(ids)]
    assert second.method_calls == [call.verify(ids), call.read(ids)]
    first.__exit__.assert_called_once()
    second.__exit__.assert_called_once()


def test_arm_state_is_independent(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(monitor.time, "monotonic", clock.monotonic)
    stop = Stop(clock)
    left = monitor.ArmReader("left", 2000000, stop)
    right = monitor.ArmReader("right", 57600, stop)
    left.publish(error=None, position_deg=[1] * 7, sample_monotonic=clock.now)
    right.publish(error="connection failed", position_deg=None)
    assert left.snapshot()["valid"] is True
    assert right.snapshot()["valid"] is False
    copied = left.snapshot()
    copied["error"] = "external change"
    assert left.snapshot()["error"] is None


def test_timing_reports_latency_violation_without_hiding_good_rate():
    from deploy.gello.control.monitor import Timing

    timing = Timing(100)
    timing.record(1.0, 1.002)
    timing.record(1.01, 1.0105)
    result = timing.snapshot()
    assert abs(result["average_hz"] - 100) < 1e-6
    assert abs(result["max_operation_duration_ms"] - 2) < 1e-6
    assert result["operations_over_1ms"] == 1


def test_deadline_is_future_at_float_boundary():
    from deploy.gello.control.monitor import next_deadline

    assert next_deadline(0, 0.01, 0.29) > 0.29


def test_usb_call_timing_is_separate_from_frame_timing(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(monitor.time, "monotonic", clock.monotonic)

    def read(length):
        clock.now += 0.0002
        return b"x" * length

    timed = monitor.UsbReadTiming(read)
    assert timed(4) == b"xxxx"
    assert timed.snapshot()["max_call_ms"] == pytest.approx(0.2)
    assert timed.snapshot()["calls_over_1ms"] == 0
