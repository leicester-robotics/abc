"""Read raw GELLO encoder poses without changing any motor settings."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from deploy.gello.control.dynamixel import DynamixelBus


def capture_pose(bus, ids, models):
    angles, started = bus.read(ids)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sample_monotonic": started,
        "read_duration_ms": (time.monotonic() - started) * 1000,
        "port": bus.port_name,
        "baudrate": bus.baudrate,
        "ids": ids,
        "models": models,
        "position_rad": angles.tolist(),
        "position_deg": np.rad2deg(angles).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--baudrate", type=int, required=True)
    parser.add_argument("--ids", nargs=7, type=int, default=list(range(1, 8)))
    parser.add_argument("--count", type=int, default=1, help="Number of poses to read")
    parser.add_argument("--interval", type=float, default=0.1, help="Seconds between reads")
    parser.add_argument("--output", help="Save to a NEW JSONL file; refuses overwrites")
    args = parser.parse_args()
    if args.count < 1 or not np.isfinite(args.interval) or args.interval < 0:
        parser.error("--count must be positive and --interval finite and nonnegative")
    try:
        with Path(args.output).open("x") if args.output else nullcontext() as output:
            with DynamixelBus(args.port, args.baudrate) as bus:
                models = bus.verify(args.ids)
                for index in range(args.count):
                    line = json.dumps(capture_pose(bus, args.ids, models), allow_nan=False)
                    print(line, flush=True)
                    if output:
                        output.write(line + "\n")
                        output.flush()
                    if index + 1 < args.count:
                        time.sleep(args.interval)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Stopped. Previously captured poses remain saved.\n")


if __name__ == "__main__":
    main()
