"""Local calibration transport tests; no USB devices are opened."""

import copy
import io
import json
import socket
import threading
import time
from unittest.mock import Mock

import numpy as np
import pytest

from deploy.gello.control import dynamixel, state_stream


def snapshot(sequence=1, **arm_changes):
    arm = {
        "sequence": sequence,
        "error": None,
        "valid": True,
        "sample_monotonic": time.monotonic(),
        "port": "test-left",
        "baudrate": 2000000,
        "ids": list(range(1, 8)),
        "models": [1200] * 6 + [1190],
        "position_rad": list(range(7)),
    }
    arm.update(arm_changes)
    return {"running": True, "arms": {"left": arm}}


@pytest.fixture(autouse=True)
def forbid_usb(monkeypatch):
    factory = Mock(side_effect=AssertionError("Stream must not open USB"))
    monkeypatch.setattr(dynamixel, "DynamixelBus", factory)
    yield
    factory.assert_not_called()


def stream_from_frames(monkeypatch, frames):
    source = io.BytesIO(b"".join((json.dumps(frame) + "\n").encode() for frame in frames))
    sock = Mock()
    sock.makefile.return_value = source
    monkeypatch.setattr(state_stream.socket, "socket", Mock(return_value=sock))
    return state_stream.StreamBus("unused-local-socket", "left"), sock


def test_stream_reads_fresh_values_and_skips_duplicate_sequences(monkeypatch):
    initial = snapshot()
    updated = snapshot(2, position_rad=[0.5] * 7)
    bus, sock = stream_from_frames(monkeypatch, [initial, initial, initial, updated])
    with bus:
        values, timestamp = bus.read(list(range(1, 8)))
        np.testing.assert_array_equal(values, np.full(7, 0.5))
        assert timestamp == updated["arms"]["left"]["sample_monotonic"]
        assert bus.port_name == "test-left"
        assert bus.baudrate == 2000000
        assert bus.sequence == 2
        assert not hasattr(bus, "relax")
        assert not hasattr(bus, "assign_id")
    sock.close.assert_called_once()
    assert bus.file.closed


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"sample_monotonic": -1}, "stale"),
        ({"sample_monotonic": float("inf")}, "stale"),
        ({"error": "motor disconnected"}, "unavailable"),
        ({"valid": False}, "unavailable"),
        ({"sequence": 0}, "restarted"),
        ({"port": "different-adapter"}, "identity changed"),
        ({"baudrate": 57600}, "identity changed"),
        ({"ids": list(range(7, 0, -1))}, "identity changed"),
        ({"models": [1200] * 7}, "identity changed"),
    ],
)
def test_stream_rejects_unusable_samples(monkeypatch, change, message):
    updated = snapshot(2)
    updated["arms"]["left"].update(change)
    bus, _ = stream_from_frames(monkeypatch, [snapshot(), updated])
    with bus, pytest.raises(OSError, match=message):
        bus.read(list(range(1, 8)))


def test_stream_rejects_stopped_monitor(monkeypatch):
    stopped = snapshot(2)
    stopped["running"] = False
    bus, _ = stream_from_frames(monkeypatch, [snapshot(), stopped])
    with bus, pytest.raises(OSError, match="unavailable"):
        bus.read(list(range(1, 8)))


@pytest.mark.parametrize("positions", [[0] * 6, [float("nan")] * 7])
def test_stream_rejects_invalid_positions(monkeypatch, positions):
    bus, _ = stream_from_frames(monkeypatch, [snapshot(), snapshot(2, position_rad=positions)])
    with bus, pytest.raises(OSError, match="Invalid encoder positions"):
        bus.read(list(range(1, 8)))


def test_stream_rejects_wrong_ids_and_disconnection(monkeypatch):
    bus, _ = stream_from_frames(monkeypatch, [snapshot()])
    with bus:
        with pytest.raises(ValueError, match="Requested IDs"):
            bus.read([7, 6, 5, 4, 3, 2, 1])
        with pytest.raises(OSError, match="disconnected"):
            bus.read(list(range(1, 8)))


def test_constructor_closes_socket_when_initial_state_is_unavailable(monkeypatch):
    source = io.BytesIO((json.dumps(snapshot(error="Connecting")) + "\n").encode())
    sock = Mock()
    sock.makefile.return_value = source
    monkeypatch.setattr(state_stream.socket, "socket", Mock(return_value=sock))
    with pytest.raises(OSError, match="unavailable"):
        state_stream.StreamBus("unused-local-socket", "left")
    assert source.closed
    sock.close.assert_called_once()


def test_local_server_streams_updates_without_duplicate_frames(tmp_path):
    path = tmp_path / "state.sock"
    lock = threading.Lock()
    state = snapshot()

    def current():
        with lock:
            return copy.deepcopy(state)

    server = state_stream.StateServer(path, current)
    try:
        assert path.stat().st_mode & 0o777 == 0o600
        with socket.socket(socket.AF_UNIX) as client:
            client.settimeout(0.2)
            client.connect(str(path))
            first = json.loads(client.recv(65536))
            assert first["arms"]["left"]["sequence"] == 1
            client.settimeout(0.025)
            with pytest.raises(TimeoutError):
                client.recv(65536)
            with lock:
                state = snapshot(2, position_rad=[0.25] * 7)
            client.settimeout(0.2)
            second = json.loads(client.recv(65536))
            assert second["arms"]["left"]["position_rad"] == [0.25] * 7
        with state_stream.StreamBus(path, "left") as bus:
            with lock:
                state = snapshot(3, position_rad=[0.75] * 7)
            positions, timestamp = bus.read(list(range(1, 8)))
            np.testing.assert_array_equal(positions, np.full(7, 0.75))
            assert timestamp == state["arms"]["left"]["sample_monotonic"]
    finally:
        server.close()
    assert not path.exists()
    assert not server.thread.is_alive()


def test_server_refuses_to_replace_regular_file(tmp_path):
    path = tmp_path / "state.sock"
    path.write_text("keep this file")
    with pytest.raises(OSError, match="Refusing to replace"):
        state_stream.StateServer(path, snapshot)
    assert path.read_text() == "keep this file"
