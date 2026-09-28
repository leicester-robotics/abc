"""Discover leaders, assign isolated motor IDs, and capture manual calibration."""

from __future__ import annotations

import argparse
import glob
import json
import select
import sys
import time
from pathlib import Path

import numpy as np

from deploy.gello.control.calibration import Calibration, Unwrapper
from deploy.gello.control.dynamixel import MODELS, DynamixelBus


class CalibrationStreamError(OSError):
    """Acquisition interruption: retry the stage only after reconnecting and rehoming."""


class Capture:
    def __init__(self, bus, ids, journal=None, metadata=None):
        self.bus, self.ids = bus, ids
        self.unwrap = Unwrapper()
        self.low = np.full(7, np.inf)
        self.high = np.full(7, -np.inf)
        self.journal = journal
        self.metadata = metadata or {}
        self.accepted = {}
        self.session = time.time_ns()
        self.last_good = None
        self.last_checkpoint = None
        self.latest = None
        self.recovering = False

    def reconnect(self, bus):
        self.bus = bus
        self.unwrap = Unwrapper()
        self.last_good = self.last_checkpoint = self.latest = None
        self.accepted.pop("direction_baseline", None)
        self.recovering = "home_raw" in self.accepted
        self.restore_accepted_travel()

    def restore_accepted_travel(self):
        if "home_raw" in self.accepted:
            home = np.asarray(self.accepted["home_raw"], dtype=float)
            self.low, self.high = home.copy(), home.copy()
            # Discard provisional travel from the interrupted stage.
            for index in range(6):
                limits = self.accepted.get(f"joint{index + 1}_limits")
                if limits is not None:
                    self.low[index], self.high[index] = limits
            if "trigger_limits" in self.accepted:
                self.low[6], self.high[6] = self.accepted["trigger_limits"]
            for key in ("gripper_closed", "gripper_open"):
                if key in self.accepted:
                    self.low[6] = min(self.low[6], self.accepted[key])
                    self.high[6] = max(self.high[6], self.accepted[key])
        else:
            self.low, self.high = np.full(7, np.inf), np.full(7, -np.inf)

    def confirm_rehome(self, pose):
        home = np.asarray(self.accepted["home_raw"], dtype=float)
        aligned = home + (pose - home + np.pi) % (2 * np.pi) - np.pi
        self.unwrap.position = aligned.copy()
        self.latest = (self.latest[0].copy(), aligned.copy())
        self.restore_accepted_travel()
        self.recovering = False
        return aligned

    def checkpoint(self, prompt, event):
        if self.journal is None or self.latest is None:
            return
        raw, position = self.latest
        record = {
            "session": self.session,
            "prompt": prompt,
            "event": event,
            "port": self.bus.port_name,
            "baudrate": self.bus.baudrate,
            "ids": self.ids,
            "timestamp": time.time(),
            "configuration": self.metadata,
            "accepted": self.accepted,
            "raw": raw.tolist(),
            "unwrapped": position.tolist(),
            "raw_min": self.low.tolist(),
            "raw_max": self.high.tolist(),
        }
        with self.journal.open("a") as output:
            output.write(json.dumps(record, allow_nan=False) + "\n")
        self.last_checkpoint = time.monotonic()

    def capture_limits(self, indices, prompt):
        """Keep observed extrema across outages; Enter accepts the observed range."""
        print(prompt + " Press Enter when the recorded limits look right.", flush=True)
        if not sys.stdin.isatty():
            raise ValueError("Calibration needs an interactive terminal")
        samples, last_print, disconnected = 0, 0.0, False
        home = np.asarray(self.accepted["home_raw"], dtype=float)
        indices = np.asarray(indices, dtype=int)
        self.low[indices] = self.high[indices] = home[indices]
        try:
            while True:
                if select.select([sys.stdin], [], [], 0)[0]:
                    if not sys.stdin.readline():
                        raise EOFError("Calibration input closed")
                    if samples and np.all(self.high[indices] - self.low[indices] >= 0.1):
                        self.checkpoint(prompt, "capture")
                        self.last_good = time.monotonic()
                        return self.latest[1].copy()
                    missing = indices[self.high[indices] - self.low[indices] < 0.1] + 1
                    print(
                        f"Move joints {missing.tolist()} through at least 5.8 degrees; "
                        "existing limits are retained. Then press Enter again.",
                        flush=True,
                    )
                try:
                    if disconnected and hasattr(self.bus, "reopen"):
                        self.bus.reopen()
                    raw, started = self.bus.read(self.ids)
                    if not 0 <= time.monotonic() - started < 0.2:
                        raise OSError("Stale encoder sample")
                except OSError as exc:
                    if not disconnected:
                        print(
                            f"\nRead interrupted: {exc}\nKeeping recorded limits. "
                            "Reconnects automatically; Enter accepts the range already seen.",
                            flush=True,
                        )
                        self.checkpoint(prompt, "paused_limits_retained")
                    disconnected = True
                    time.sleep(0.1)
                    continue
                if disconnected:
                    print("Readings restored; continuing the SAME limit sweep.", flush=True)
                    # Torque-off XL330 positions are signed continuous tick counts.
                    # Do not infer missing movement by wrapping a large raw delta.
                    if self.unwrap.previous is not None:
                        self.unwrap.position += raw - self.unwrap.previous
                        self.unwrap.previous = raw.copy()
                disconnected = False
                position = self.unwrap.update(raw)
                self.low[indices] = np.minimum(self.low[indices], position[indices])
                self.high[indices] = np.maximum(self.high[indices], position[indices])
                self.latest = (raw.copy(), position.copy())
                self.last_good = time.monotonic()
                samples += 1
                if self.last_good - last_print >= 0.2:
                    lows = ",".join(f"{v:.1f}" for v in np.rad2deg(self.low))
                    highs = ",".join(f"{v:.1f}" for v in np.rad2deg(self.high))
                    print(f"Min: [{lows}] Max: [{highs}] degrees | Enter to accept", flush=True)
                    last_print = self.last_good
                if self.last_checkpoint is None or self.last_good - self.last_checkpoint >= 1:
                    self.checkpoint(prompt, "provisional")
                if not getattr(self.bus, "paced", False):
                    time.sleep(0.01)
        except (OSError, ValueError, EOFError, KeyboardInterrupt):
            self.checkpoint(prompt, "interrupted")
            raise

    def until_enter(self, prompt):
        """Continuously sample; checkpoint provisional travel separately from captures."""
        print(prompt + " Press Enter when ready.", flush=True)
        if not sys.stdin.isatty():
            raise ValueError("Calibration needs an interactive terminal")
        if self.last_good is None:
            self.last_good = time.monotonic()
        failures = 0
        try:
            while True:
                try:
                    raw, started = self.bus.read(self.ids)
                except OSError as exc:
                    failures += 1
                    if failures >= 3 or time.monotonic() - self.last_good > 0.2:
                        raise CalibrationStreamError(str(exc)) from exc
                    time.sleep(0.01)
                    continue
                if time.monotonic() - started > 0.2:
                    raise CalibrationStreamError(
                        "Calibration read exceeded 200 ms; check serial latency/connection"
                    )
                if time.monotonic() - self.last_good > 0.2:
                    raise CalibrationStreamError(
                        "Calibration sampling gap exceeded 200 ms; progress journal retained"
                    )
                self.last_good = time.monotonic()
                failures = 0
                position = self.unwrap.update(raw)
                self.low = np.minimum(self.low, position)
                self.high = np.maximum(self.high, position)
                self.latest = (raw.copy(), position.copy())
                if select.select([sys.stdin], [], [], 0)[0]:
                    if not sys.stdin.readline():
                        raise EOFError("Calibration input closed")
                    self.checkpoint(prompt, "capture")
                    return position
                if self.last_checkpoint is None or time.monotonic() - self.last_checkpoint >= 1:
                    self.checkpoint(prompt, "provisional")
                if not getattr(self.bus, "paced", False):
                    time.sleep(0.01)
        except (OSError, EOFError, KeyboardInterrupt):
            self.checkpoint(prompt, "interrupted")
            raise


def calibrate(args):
    if Path(args.output).exists():
        raise ValueError("Output already exists; choose a new calibration filename")
    with DynamixelBus(args.port, args.baudrate) as bus:
        models = bus.verify(args.ids)
        print("Support the leader. Calibration disables torque; move it by hand only.")
        input("Press Enter to relax this leader and begin: ")
        bus.relax(args.ids)
        capture_calibration(args, bus, models)


def return_to_neutral(capture, home, prompt):
    """Separate return motion from the next measurement and rebase after settling."""
    while True:
        pose = capture.until_enter(
            prompt + "\nReturn ALL joints to the original neutral/home pose and fully OPEN "
            "the trigger. Support the arm, let it settle, then hold still."
        )
        difference = pose - home
        if getattr(capture, "recovering", False) is True:
            difference = (difference + np.pi) % (2 * np.pi) - np.pi
        delta = np.rad2deg(difference)
        outside = np.flatnonzero(np.abs(delta) > 5)
        if not len(outside):
            if getattr(capture, "recovering", False) is True:
                pose = capture.confirm_rehome(pose)
            capture.checkpoint("neutral", "accepted")
            return pose
        detail = ", ".join(f"J{i + 1}: {delta[i]:+.1f} degrees" for i in outside)
        print(
            f"Still away from neutral ({detail}). Adjust and press Enter again; "
            "within 5 degrees is enough.",
            flush=True,
        )


def make_capture(args, bus, models):
    negative_joints = set(getattr(args, "negative_joints", []) or [])
    journal = Path(str(args.output) + ".progress.jsonl")
    print(f"Each captured step is saved to {journal} (not a finished calibration).", flush=True)
    return Capture(
        bus,
        args.ids,
        journal=journal,
        metadata={
            "side": args.side,
            "models": models,
            "home_target": args.home,
            "negative_joints": sorted(negative_joints),
        },
    )


def capture_calibration(args, bus, models, capture=None):
    negative_joints = set(getattr(args, "negative_joints", []) or [])
    capture = capture if capture is not None else make_capture(args, bus, models)
    if "home_raw" in capture.accepted:
        home = np.asarray(capture.accepted["home_raw"], dtype=float)
        if getattr(capture, "limits_only", False):
            baseline = home.copy()
        else:
            baseline = return_to_neutral(
                capture, home, "\nRESUME — CONFIRM NEUTRAL BEFORE CONTINUING"
            )
    else:
        home = capture.until_enter(
            f"\nStep 1/13 — HOME ({args.side.upper()}).\n"
            "Place the arm in the comfortable reference pose from your photo. "
            "Keep the base facing forward and the trigger fully OPEN. "
            f"This maps to simulator joints {args.home}."
        )
        capture.accepted["home_raw"] = home.tolist()
        capture.checkpoint("home", "accepted")
        baseline = home.copy()
    signs = capture.accepted.setdefault("signs", [])
    for index in range(len(signs), 6):
        requested_direction = -1 if index + 1 in negative_joints else 1
        direction_name = "NEGATIVE" if requested_direction == -1 else "POSITIVE"
        direction_key = "J" if requested_direction == -1 else "K"
        while True:
            capture.accepted["direction_baseline"] = {"joint": index + 1, "raw": baseline.tolist()}
            capture.checkpoint(f"joint{index + 1}.baseline", "accepted")
            moved = capture.until_enter(
                f"\nStep {2 + index * 2}/13 — JOINT {index + 1} DIRECTION.\n"
                f"Starting from neutral, move ONLY motor ID {index + 1} "
                "(count from base 1 toward handle 6). "
                f"In the MuJoCo viewer press H to return to home, {args.side[0].upper()}, "
                f"then {index + 1}, then {direction_key} several times to see the {direction_name} direction. "
                f"Move physical joint {index + 1} in that direction by 10–20 degrees, away from any home stop. "
                "Support the other joints at neutral and hold this pose until you press Enter. "
                + (
                    "The next prompt will ask you to return to neutral."
                    if index < 5
                    else "Next, calibrate the trigger; no return to neutral is needed."
                )
            )
            delta = moved - baseline
            print(
                "Measured changes, joints 1–6 (degrees): "
                + str(np.round(np.rad2deg(delta[:6]), 1).tolist()),
                flush=True,
            )
            if np.any(np.abs(np.delete(delta[:6], index)) > 0.15):
                print("Other joints moved; this movement was not accepted.", flush=True)
            elif abs(delta[index]) < 0.1:
                print(
                    "Not enough movement; use at least 5.8 degrees on the next attempt.", flush=True
                )
            elif abs(delta[index]) >= np.pi:
                print("Movement exceeded half a turn; retry with 10–20 degrees.", flush=True)
            else:
                break
            baseline = return_to_neutral(capture, home, f"Retry joint {index + 1} — RESET")
        signs.append(requested_direction * (1 if delta[index] > 0 else -1))
        capture.accepted.pop("direction_baseline", None)
        capture.checkpoint(f"joint{index + 1}.direction", "accepted")
        if index < 5:
            baseline = return_to_neutral(
                capture, home, f"\nStep {3 + index * 2}/13 — RETURN TO NEUTRAL"
            )
    if "gripper_open" not in capture.accepted:
        capture.capture_limits(
            [6],
            "\nStep 13/13 — TRIGGER ENDPOINTS.\n"
            "No joint limit sweep is needed. Keep the arm wherever comfortable. "
            "Fully close and fully open the trigger, then press Enter. "
            "No return to neutral is required.",
        )
        capture.accepted["trigger_limits"] = [float(capture.low[6]), float(capture.high[6])]
        capture.checkpoint("trigger_limits", "accepted")
    # Home was explicitly recorded with the trigger fully open. The nearer
    # swept endpoint is open; the opposite endpoint is closed. No extra pose needed.
    if "gripper_open" not in capture.accepted:
        endpoints = [float(capture.low[6]), float(capture.high[6])]
        open_index = int(np.argmin(np.abs(np.asarray(endpoints) - home[6])))
        if abs(endpoints[open_index] - home[6]) > np.deg2rad(5):
            raise ValueError("Trigger home was not near an open endpoint; verify the home capture")
        capture.accepted["gripper_open"] = endpoints[open_index]
        capture.accepted["gripper_closed"] = endpoints[1 - open_index]
        capture.checkpoint("trigger_endpoints_from_sweep", "accepted")
    closed, opened = capture.accepted["gripper_closed"], capture.accepted["gripper_open"]
    calibration = Calibration(
        side=args.side,
        port=args.port,
        baudrate=args.baudrate,
        ids=args.ids,
        models=models,
        home_raw=home.tolist(),
        home_target=args.home,
        signs=signs,
        raw_min=None,
        raw_max=None,
        version=2,
        gripper_closed=float(closed),
        gripper_open=float(opened),
    )
    calibration.save(args.output)
    print(f"Saved {args.output}. Joint signs: {signs}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan", help="Read-only protocol-2 discovery; no motor writes")
    scan.add_argument(
        "--port", action="append", help="Repeat for multiple ports; default by-id ports"
    )
    scan.add_argument("--baudrates", nargs="+", type=int, default=[57600, 1000000, 2000000, 115200])
    assign = sub.add_parser("set-id", help="Assign ONE physically isolated motor's EEPROM ID")
    assign.add_argument("--old-id", type=int, required=True)
    assign.add_argument("--new-id", type=int, required=True)
    assign.add_argument("--model", type=int, required=True)
    cal = sub.add_parser(
        "calibrate", help="Interactive manual limits, signs, home and trigger capture"
    )
    cal.add_argument("--side", choices=["left", "right"], required=True)
    cal.add_argument("--ids", nargs=7, type=int, default=list(range(1, 8)))
    cal.add_argument("--home", nargs=6, type=float, default=[0, 1.047, 1.047, 0, 0, 0])
    cal.add_argument(
        "--negative-joints",
        nargs="*",
        type=int,
        choices=range(1, 7),
        default=[],
        help="Joint IDs to test in the negative simulator direction",
    )
    cal.add_argument("--output", required=True)
    for command in (assign, cal):
        command.add_argument("--port", required=True)
        command.add_argument("--baudrate", type=int, default=57600)
    args = parser.parse_args()
    try:
        if args.command == "scan":
            ports = args.port or sorted(glob.glob("/dev/serial/by-id/*"))
            if not ports:
                raise OSError("No serial ports found; specify --port")
            failed = False
            for port in ports:
                for baud in args.baudrates:
                    try:
                        with DynamixelBus(port, baud) as bus:
                            found = bus.scan()
                        print(
                            json.dumps(
                                {
                                    "port": port,
                                    "baudrate": baud,
                                    "motors": {
                                        i: {"model": m, "name": MODELS.get(m, "unsupported")}
                                        for i, m in sorted(found.items())
                                    },
                                }
                            ),
                            flush=True,
                        )
                    except OSError as exc:
                        failed = True
                        print(f"{port} at {baud}: {exc}", file=sys.stderr)
            if failed:
                return 1
        elif args.command == "set-id":
            print(f"Port: {args.port}; model {args.model}; ID {args.old_id} -> {args.new_id}.")
            print(
                "Power off before rewiring. Connect ONLY this one motor, then power it on. "
                "A scan cannot distinguish several motors sharing an ID."
            )
            if (
                input("Type 'one motor connected' to perform this ID write: ")
                != "one motor connected"
            ):
                raise ValueError("ID write cancelled")
            with DynamixelBus(args.port, args.baudrate) as bus:
                bus.assign_id(args.old_id, args.new_id, args.model, isolated=True)
            print("ID verified. Power off before reconnecting the chain.")
        else:
            calibrate(args)
    except (OSError, ValueError, EOFError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Stopped; no motion commands were sent.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
