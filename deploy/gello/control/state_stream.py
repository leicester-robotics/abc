"""Local read-only state stream; the monitor remains the sole USB owner."""

import json
import os
import socket
import socketserver
import stat
import threading
import time
from pathlib import Path

import numpy as np

DEFAULT_SOCKET = f"/tmp/gontrol-encoders-{os.getuid()}.sock"


class StateServer:
    def __init__(self, path, snapshot):
        self.path = Path(path)
        self.stop = threading.Event()
        if self.path.exists():
            info = self.path.lstat()
            if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
                raise OSError(f"Refusing to replace {self.path}")
            with socket.socket(socket.AF_UNIX) as probe:
                try:
                    probe.connect(str(self.path))
                except ConnectionRefusedError:
                    self.path.unlink()
                else:
                    raise OSError(f"Another monitor is serving {self.path}")
        owner = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                self.request.settimeout(0.2)
                previous = None
                try:
                    while not owner.stop.is_set():
                        state = snapshot()
                        token = tuple((a["sequence"], a["error"]) for a in state["arms"].values())
                        if token != previous:
                            self.request.sendall(
                                (json.dumps(state, allow_nan=False) + "\n").encode()
                            )
                            previous = token
                        owner.stop.wait(0.001)
                except OSError:
                    pass  # A slow or closed subscriber never blocks acquisition.

        class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
            daemon_threads = True
            block_on_close = False

        self.server = Server(str(self.path), Handler)
        self.path.chmod(0o600)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.path.unlink(missing_ok=True)


class StreamBus:
    """Calibration input only: intentionally has no motor-write API."""

    paced = True

    def __init__(self, path, side):
        self.path = path
        self.side = side
        self.sequence = -1
        self.sock = socket.socket(socket.AF_UNIX)
        self.sock.settimeout(0.2)
        try:
            self.sock.connect(str(path))
            self.file = self.sock.makefile("rb")
            self.identity = None
            arm = self._next()
            self.port_name, self.baudrate = arm["port"], arm["baudrate"]
            self.ids, self.models = arm["ids"], arm["models"]
            self.identity = (self.port_name, self.baudrate, self.ids, self.models)
        except BaseException:
            self.close()
            raise

    def _next(self):
        while True:
            line = self.file.readline(1024 * 1024)
            if not line:
                raise OSError("Encoder monitor disconnected; calibration journal retained")
            try:
                state = json.loads(line)
                arm = state["arms"][self.side]
                if not state["running"] or not arm["valid"] or arm["error"]:
                    raise OSError(f"Monitor {self.side} unavailable: {arm.get('error')}")
                age = time.monotonic() - arm["sample_monotonic"]
                if not 0 <= age < 0.2:
                    raise OSError("Encoder stream stale; calibration journal retained")
                if self.identity is not None and self.identity != (
                    arm["port"],
                    arm["baudrate"],
                    arm["ids"],
                    arm["models"],
                ):
                    raise OSError("Encoder identity changed during calibration")
                if arm["sequence"] < self.sequence:
                    raise OSError("Encoder stream restarted during calibration")
                if arm["sequence"] == self.sequence:
                    continue
                self.sequence = arm["sequence"]
                return arm
            except (KeyError, TypeError, ValueError) as exc:
                raise OSError(f"Invalid encoder stream: {exc}") from exc

    def read(self, ids):
        if ids != self.ids:
            raise ValueError("Requested IDs differ from the monitor IDs")
        arm = self._next()
        values = np.asarray(arm["position_rad"], dtype=float)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise OSError("Invalid encoder positions")
        return values, arm["sample_monotonic"]

    def reopen(self):
        """Replace a failed buffered socket; never silently switch to another arm."""
        replacement = StreamBus(self.path, self.side)
        if replacement.identity != self.identity:
            replacement.close()
            raise ValueError("Encoder identity changed; refusing to continue limit capture")
        self.close()
        self.sock, self.file = replacement.sock, replacement.file
        self.sequence = replacement.sequence

    def close(self):
        if hasattr(self, "file"):
            self.file.close()
        self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
