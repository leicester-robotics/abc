"""Interactive calibration from the running encoder monitor, without USB access."""

import argparse
import json
import sys
import termios
import time
from pathlib import Path

import numpy as np

from deploy.gello.control.gello_cli import (
    CalibrationStreamError,
    capture_calibration,
    make_capture,
)
from deploy.gello.control.profiles import GELLO_REST_HOME, GELLO_REST_NEGATIVE_JOINTS
from deploy.gello.control.state_stream import DEFAULT_SOCKET, StreamBus


def restore_progress(capture, args):
    records = []
    for line in capture.journal.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # A final interrupted append may be incomplete.
    if not records:
        raise ValueError("No readable progress checkpoint")
    record = records[-1]
    if (
        record["configuration"] != capture.metadata
        or record["port"] != args.port
        or record["baudrate"] != args.baudrate
        or record["ids"] != args.ids
    ):
        raise ValueError("Saved calibration configuration differs from this arm or command")
    accepted = record["accepted"]
    if "home_raw" in accepted:
        home = np.asarray(accepted["home_raw"], dtype=float)
        if home.shape != (7,) or not np.isfinite(home).all():
            raise ValueError("Invalid saved home")
        signs = accepted.get("signs", [])
        if len(signs) > 6 or any(sign not in (-1, 1) for sign in signs):
            raise ValueError("Invalid saved joint signs")
        for i in range(6):
            limits = accepted.get(f"joint{i + 1}_limits")
            if limits is not None:
                limits = np.asarray(limits, dtype=float)
                if (
                    limits.shape != (2,)
                    or not np.isfinite(limits).all()
                    or not limits[0] <= home[i] <= limits[1]
                    or not 0.1 <= limits[1] - limits[0] < 2 * np.pi - 0.04
                ):
                    raise ValueError("Invalid saved joint limits")
    capture.accepted = accepted
    capture.reconnect(capture.bus)
    print(
        f"Restored {len(accepted.get('signs', []))} completed direction checks; "
        "neutral confirmation is required.",
        flush=True,
    )


def run_calibration(args):
    capture, identity = None, None
    announced = False
    while True:
        try:
            bus = StreamBus(args.socket, args.side)
        except OSError as exc:
            if not announced:
                print(f"PAUSED: waiting for encoder monitor ({exc}). Ctrl-C stops.", flush=True)
                announced = True
            time.sleep(0.5)
            continue
        announced = False
        with bus:
            if identity is not None and bus.identity != identity:
                raise ValueError("Encoder identity changed; refusing to resume another arm")
            identity = bus.identity
            args.port, args.baudrate, args.ids = bus.port_name, bus.baudrate, bus.ids
            if capture is None:
                capture = make_capture(args, bus, bus.models)
                if args.resume:
                    restore_progress(capture, args)
            else:
                capture.reconnect(bus)
                print(
                    "Connection restored. Completed steps retained; interrupted travel discarded.",
                    flush=True,
                )
            if sys.stdin.isatty():
                termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
            try:
                capture_calibration(args, bus, bus.models, capture=capture)
                return
            except CalibrationStreamError as exc:
                print(
                    f"\nPAUSED: {exc}\nStop moving the arm. Waiting to reconnect; "
                    "then return to neutral before retrying the interrupted step.",
                    flush=True,
                )
        time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default=DEFAULT_SOCKET)
    parser.add_argument("--side", choices=["left", "right"], required=True)
    parser.add_argument("--home", nargs=6, type=float, default=list(GELLO_REST_HOME))
    parser.add_argument(
        "--negative-joints",
        nargs="*",
        type=int,
        choices=range(1, 7),
        default=list(GELLO_REST_NEGATIVE_JOINTS),
        help="Joint IDs whose direction check moves negative, away from a home stop",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Restore matching saved checkpoints, then confirm neutral",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new calibration filename")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_calibration(args)
    except (OSError, ValueError, EOFError) as exc:
        print(f"Calibration stopped: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Calibration stopped; progress journal retained.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
