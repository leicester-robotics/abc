"""Independent device workers with bounded waits and reconnect status."""
import threading
import time


class DeviceWorker:
    rate = 30.

    def __init__(self, buffer):
        self.buffer = buffer
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name=type(self).__name__)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                self.open()
                while not self._stop.is_set():
                    start = time.monotonic()
                    try:
                        self.read_once()
                    except TimeoutError as error:
                        self.buffer.set_error(str(error), transient=getattr(error, "transient_read", False))
                    self._stop.wait(max(0., 1 / self.rate - (time.monotonic() - start)))
            except Exception as error:
                self.buffer.set_error(f'{type(error).__name__}: {error}')
            finally:
                try:
                    self._close_device()
                except Exception as error:
                    self.buffer.set_error(f'{self.buffer.error}; cleanup: {error}')
            self._stop.wait(1.)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.)
            if self._thread.is_alive():
                self.buffer.set_error('Device worker did not stop within 3 seconds')
                return
        self._close_device()
