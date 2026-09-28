"""One-pass joint safety-envelope capture from the running monitor."""

import argparse
import select
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from deploy.gello.control.calibration import Calibration, RelativeMapper
from deploy.gello.control.joint_limits import JointEnvelope, fingerprint
from deploy.gello.control.monitor import write_latest
from deploy.gello.control.state_stream import DEFAULT_SOCKET, StreamBus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--socket", default=DEFAULT_SOCKET)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new filename")
    if not sys.stdin.isatty():
        parser.error("Capture requires an interactive terminal")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    calibrations = [Calibration.load(args.left), Calibration.load(args.right)]
    if [c.side for c in calibrations] != ["left", "right"]:
        parser.error("Supply left then right calibration")
    mappers = [RelativeMapper(c) for c in calibrations]
    low = np.concatenate([c.home_target for c in calibrations])
    high = low.copy()
    latest_write, latest_print = 0, 0
    progress = Path(str(args.output) + ".progress.json")
    print(
        "Move both leaders through the joint ranges you want to allow.\n"
        "All 12 joint bounds update together. 1:1 mapping stays unchanged.\n"
        "Enter saves; no neutral return needed. Dropouts retain observed bounds.\n"
        "Start near the saved home poses so encoder turns can be aligned.",
        flush=True,
    )
    errors = [None, None]
    with ExitStack() as stack:
        buses = [stack.enter_context(StreamBus(args.socket, c.side)) for c in calibrations]
        for c, bus in zip(calibrations, buses):
            if bus.identity != (c.port, c.baudrate, c.ids, c.models):
                raise ValueError("Monitor identity differs from calibration")
        while True:
            if select.select([sys.stdin], [], [], 0)[0]:
                if not sys.stdin.readline():
                    raise EOFError("Input closed")
                try:
                    envelope = JointEnvelope(
                        [fingerprint(c) for c in calibrations], low.tolist(), high.tolist()
                    )
                except ValueError as exc:
                    print(f"{exc}; keep sweeping, then press Enter.", flush=True)
                else:
                    envelope.save(args.output)
                    print(
                        f"Saved {args.output}. Motion remains 1:1 inside these bounds.", flush=True
                    )
                    return 0
            for index, (bus, mapper, c) in enumerate(zip(buses, mappers, calibrations)):
                try:
                    if errors[index]:
                        bus.reopen()
                    raw, _ = bus.read(c.ids)
                    q = mapper.map(raw)[:6]
                    section = slice(index * 6, index * 6 + 6)
                    low[section] = np.minimum(low[section], q)
                    high[section] = np.maximum(high[section], q)
                    errors[index] = None
                except OSError as exc:
                    if errors[index] is None:
                        print(f"{c.side}: {exc}; retaining min/max.", flush=True)
                    errors[index] = str(exc)
            now = time.monotonic()
            if now - latest_print >= 0.2:
                for index, side in enumerate(["Left", "Right"]):
                    section = slice(index * 6, index * 6 + 6)
                    print(
                        f"{side} min: {np.rad2deg(low[section]).round(1).tolist()} "
                        f"max: {np.rad2deg(high[section]).round(1).tolist()}",
                        flush=True,
                    )
                latest_print = now
            if now - latest_write >= 0.2:
                write_latest(
                    progress,
                    {
                        "low": low.tolist(),
                        "high": high.tolist(),
                        "errors": errors,
                        "timestamp": time.time(),
                    },
                )
                latest_write = now
            if any(errors):
                time.sleep(0.05)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, EOFError) as exc:
        raise SystemExit(str(exc))
    except KeyboardInterrupt:
        raise SystemExit("Stopped; observed limits remain in the progress file.")
