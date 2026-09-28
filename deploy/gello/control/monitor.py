"""Read-only encoder monitor: 100 Hz sampling/display and 5 Hz JSON snapshots."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from deploy.gello.control.dynamixel import MODELS, DynamixelBus
from deploy.gello.control.state_stream import DEFAULT_SOCKET, StateServer


def read_motor(bus, motor_id):
    row = {"id": motor_id, "position_rad": None, "position_deg": None}
    model, result, error = bus.packet.ping(bus.port, motor_id)
    row.update(model=model if result == 0 else None, ping_result=result, status_error=error)
    if result != 0:
        row["error"] = bus.packet.getTxRxResult(result)
        return row
    if model not in MODELS:
        row["error"] = f"Unsupported model {model}; no control-table reads attempted"
        return row
    row["model_name"] = MODELS[model]
    values = {}
    for key, address, size in (
        ("position_ticks", 132, 4),
        ("voltage", 144, 2),
        ("hardware_error", 70, 1),
    ):
        value, result, error = getattr(bus.packet, f"read{size}ByteTxRx")(
            bus.port, motor_id, address
        )
        values[key] = {
            "value": value if result == 0 else None,
            "comm_result": result,
            "status_error": error,
        }
    row["registers"] = values
    ticks = values["position_ticks"]["value"]
    if ticks is not None:
        ticks = ticks if ticks < 2**31 else ticks - 2**32
        row["position_rad"] = ticks * 2 * np.pi / 4096
        row["position_deg"] = ticks * 360 / 4096
        row["position_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    if any(v["comm_result"] != 0 for v in values.values()):
        row["error"] = "One or more register reads failed"
    elif (
        row["status_error"]
        or any(v["status_error"] for v in values.values())
        or values["hardware_error"]["value"]
    ):
        row["error"] = "Motor reports a hardware/status error; see registers"
    return row


def write_latest(path, snapshot):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


class Timing:
    """Measured host timings; not a guarantee about device sampling or screen refresh."""

    def __init__(self, hz):
        self.period = 1 / hz
        self.previous = None
        self.first = None
        self.count = 0
        self.max_jitter_ms = 0.0
        self.max_duration_ms = 0.0
        self.over_1ms_count = 0
        self.lock = threading.Lock()

    def record(self, started, completed):
        with self.lock:
            if self.previous is not None:
                self.max_jitter_ms = max(
                    self.max_jitter_ms, abs(started - self.previous - self.period) * 1000
                )
            if self.first is None:
                self.first = started
            self.previous = started
            self.count += 1
            duration_ms = (completed - started) * 1000
            self.max_duration_ms = max(self.max_duration_ms, duration_ms)
            self.over_1ms_count += duration_ms > 1

    def snapshot(self):
        with self.lock:
            duration = 0 if self.first is None else self.previous - self.first
            return {
                "count": self.count,
                "average_hz": (self.count - 1) / duration if duration > 0 else None,
                "max_interval_jitter_ms": self.max_jitter_ms,
                "max_operation_duration_ms": self.max_duration_ms,
                "operations_over_1ms": self.over_1ms_count,
            }


class UsbReadTiming:
    """Time nonblocking host read calls, separately from complete frame acquisition."""

    def __init__(self, read):
        self.read = read
        self.calls = 0
        self.max_ms = 0.0
        self.over_1ms = 0

    def __call__(self, length):
        started = time.monotonic()
        try:
            return self.read(length)
        finally:
            duration_ms = (time.monotonic() - started) * 1000
            self.calls += 1
            self.max_ms = max(self.max_ms, duration_ms)
            self.over_1ms += duration_ms > 1

    def snapshot(self):
        return {
            "calls": self.calls,
            "max_call_ms": self.max_ms,
            "calls_over_1ms": self.over_1ms,
            "scope": "nonblocking host read calls since connection; not frame acquisition",
        }


class ArmReader(threading.Thread):
    """Own one serial port; slow/disconnected arms cannot stall the other arm."""

    def __init__(self, port, baud, stop, hz=100, *, side=None, error_log=None):
        super().__init__(daemon=True)
        self.port, self.baud, self.stop, self.hz = port, baud, stop, hz
        self.side, self.error_log = side, error_log
        self.fatal_error = None
        self.lock = threading.Lock()
        self.timing = Timing(hz)
        self.state = {
            "port": port,
            "baudrate": baud,
            "sequence": 0,
            "read_errors": 0,
            "last_error": None,
            "last_error_timestamp_utc": None,
            "position_deg": None,
            "error": "Connecting",
            "read_hz": 0.0,
        }

    def log_fault_event(self, event):
        if self.error_log is None:
            return
        try:
            with Path(self.error_log).open("a") as output:
                output.write(json.dumps(event, allow_nan=False) + "\n")
        except (OSError, ValueError) as exc:
            self.fatal_error = exc
            self.publish(diagnostics_log_error=str(exc))
            self.stop.set()
            raise MonitorLogError(f"Cannot append diagnostics to {self.error_log}: {exc}") from exc

    def record_fault(self, exc):
        timestamp = datetime.now(timezone.utc).isoformat()
        errors = self.state["read_errors"] + 1
        self.publish(
            error=str(exc),
            read_errors=errors,
            last_error=str(exc),
            last_error_timestamp_utc=timestamp,
            position_deg=None,
            position_rad=None,
            read_hz=0.0,
        )
        # Persist the original exception before probing a possibly broken connection.
        self.log_fault_event(
            {
                "event": "acquisition_error",
                "timestamp_utc": timestamp,
                "side": self.side,
                "port": self.port,
                "baudrate": self.baud,
                "error_index": errors,
                "error": str(exc),
            }
        )

    def diagnose(self, bus, phase):
        motors = []
        for motor_id in range(1, 8):
            if self.stop.is_set():
                break
            try:
                motors.append(read_motor(bus, motor_id))
            except Exception as exc:
                motors.append({"id": motor_id, "error": str(exc)})
        event = {
            "event": "motor_diagnostics",
            "phase": phase,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "side": self.side,
            "port": self.port,
            "baudrate": self.baud,
            "error_index": self.state["read_errors"],
            "motors": motors,
        }
        self.log_fault_event(event)
        self.publish(last_diagnostics=event)

    def publish(self, **values):
        with self.lock:
            self.state = {**self.state, **values}

    def snapshot(self):
        with self.lock:
            state = dict(self.state)
        timestamp = state.get("sample_monotonic")
        state["age_seconds"] = None if timestamp is None else time.monotonic() - timestamp
        state["timing"] = self.timing.snapshot()
        state["read_latency_within_1ms"] = (
            state.get("read_duration_ms") is not None
            and state["read_duration_ms"] <= 1
            and state["error"] is None
        )
        state["valid"] = (
            state["error"] is None and timestamp is not None and state["age_seconds"] < 0.2
        )
        return state

    def run(self):
        ids = list(range(1, 8))
        sequence = 0
        while not self.stop.is_set():
            try:
                with DynamixelBus(self.port, self.baud) as bus:
                    try:
                        models = bus.verify(ids)
                        if self.state["read_errors"]:
                            self.diagnose(bus, "reconnected")
                        usb_timing = UsbReadTiming(bus.port.readPort)
                        bus.port.readPort = usb_timing
                        window, count, rate = None, 0, 0.0
                        deadline = time.monotonic()
                        while not self.stop.is_set():
                            positions, timestamp = bus.read(ids)
                            now = time.monotonic()
                            self.timing.record(timestamp, now)
                            sequence += 1
                            if window is None:
                                window = now
                            else:
                                count += 1
                            if now - window >= 1:
                                rate = count / (now - window)
                                window, count = now, 0
                            self.publish(
                                sequence=sequence,
                                usb_reads=usb_timing.snapshot(),
                                ids=ids,
                                models=models,
                                position_rad=positions.tolist(),
                                position_deg=np.rad2deg(positions).tolist(),
                                sample_monotonic=timestamp,
                                read_duration_ms=(now - timestamp) * 1000,
                                read_hz=rate,
                                error=None,
                            )
                            deadline = next_deadline(deadline, 1 / self.hz, now)
                            self.stop.wait(max(0, deadline - time.monotonic()))
                    except (OSError, ValueError) as exc:
                        self.record_fault(exc)
                        self.diagnose(bus, "fault")
            except MonitorLogError:
                return
            except (OSError, ValueError) as exc:
                try:
                    self.record_fault(exc)
                except MonitorLogError:
                    return
            if not self.stop.is_set():
                self.stop.wait(0.5)


class MonitorLogError(RuntimeError):
    """A diagnostics write failed; stop explicitly instead of losing fault evidence."""


def next_deadline(previous, period, now):
    # Skip missed slots rather than emitting bursts to catch up.
    deadline = previous + max(1, int((now - previous) / period) + 1) * period
    return deadline + period if deadline <= now else deadline


def format_line(arms):
    def values(side):
        arm = arms[side]
        if not arm["valid"]:
            return ",".join(["NA"] * 7)
        return ",".join(f"{value:.2f}" for value in arm["position_deg"])

    return f"Right: [{values('right')}] Left: [{values('left')}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-port", required=True)
    parser.add_argument("--left-baud", type=int, default=2000000)
    parser.add_argument("--right-port", required=True)
    parser.add_argument("--right-baud", type=int, default=2000000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--socket", default=DEFAULT_SOCKET, help="Local calibration state stream")
    parser.add_argument("--read-hz", type=float, default=100)
    parser.add_argument("--print-hz", type=float, default=100)
    parser.add_argument("--json-hz", type=float, default=5)
    args = parser.parse_args()
    if any(not np.isfinite(v) or v <= 0 for v in (args.read_hz, args.print_hz, args.json_hz)):
        parser.error("Rates must be finite and positive")
    if Path(args.left_port).resolve() == Path(args.right_port).resolve():
        parser.error("Left and right ports must be distinct")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    readers = {
        side: ArmReader(
            port,
            baud,
            stop,
            args.read_hz,
            side=side,
            error_log=args.output.with_name(args.output.name + f".{side}.errors.jsonl"),
        )
        for side, port, baud in [
            ("left", args.left_port, args.left_baud),
            ("right", args.right_port, args.right_baud),
        ]
    }

    print_timing, json_timing = Timing(args.print_hz), Timing(args.json_hz)

    def snapshot(running=True):
        return {
            "running": running,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "requested_read_hz": args.read_hz,
            "timing": {"terminal": print_timing.snapshot(), "json": json_timing.snapshot()},
            "print_hz": args.print_hz,
            "json_hz": args.json_hz,
            "arms": {side: reader.snapshot() for side, reader in readers.items()},
        }

    writer_errors = []

    def save_loop():
        deadline = time.monotonic()
        try:
            while not stop.is_set():
                started = time.monotonic()
                write_latest(args.output, snapshot())
                json_timing.record(started, time.monotonic())
                deadline = next_deadline(deadline, 1 / args.json_hz, time.monotonic())
                stop.wait(max(0, deadline - time.monotonic()))
        except OSError as exc:
            writer_errors.append(exc)
            stop.set()

    server = StateServer(args.socket, snapshot)
    writer = threading.Thread(target=save_loop, daemon=True)
    for reader in readers.values():
        reader.start()
    writer.start()
    try:
        deadline = time.monotonic()
        while not stop.is_set():
            started = time.monotonic()
            print(format_line(snapshot()["arms"]), flush=True)
            print_timing.record(started, time.monotonic())
            deadline = next_deadline(deadline, 1 / args.print_hz, time.monotonic())
            stop.wait(max(0, deadline - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.close()
        for reader in readers.values():
            reader.join(timeout=2)
        writer.join()
        write_latest(args.output, snapshot(running=False))
    if writer_errors:
        raise writer_errors[0]
    for reader in readers.values():
        if reader.fatal_error is not None:
            raise reader.fatal_error


if __name__ == "__main__":
    main()
