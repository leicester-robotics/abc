"""RealSense acquisition in separate processes, isolated from motor threads."""
import multiprocessing as mp
import queue
import time

import numpy as np


def camera_worker(name, config, output, stop):
    import pyrealsense2 as rs
    pipeline, rcfg = rs.pipeline(), rs.config()
    rcfg.enable_device(config.serial)
    rcfg.enable_stream(rs.stream.color, config.width, config.height, rs.format.rgb8, 30)
    started = False
    try:
        pipeline.start(rcfg)
        started = True
        while not stop.is_set():
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            frame = frames.get_color_frame()
            if not frame:
                raise RuntimeError('no color frame')
            item = ('rgb', name, np.asanyarray(frame.get_data()).copy(), time.perf_counter_ns())
            # Keep recent frames while the parent initializes the motors.
            # Sequence/timestamp gaps during recording are checked by validation.
            try:
                output.put_nowait(item)
            except queue.Full:
                try:
                    output.get_nowait()
                except queue.Empty:
                    pass
                try:
                    output.put_nowait(item)
                except queue.Full:
                    pass
    except BaseException as exc:
        try:
            output.put(('error', name, repr(exc), time.perf_counter_ns()), timeout=.2)
        except queue.Full:
            pass
    finally:
        if started:
            pipeline.stop()
        output.cancel_join_thread()


class CameraStreams:
    def __init__(self, configs, *, worker=camera_worker):
        self.configs = configs
        self.worker = worker
        self.context = mp.get_context('spawn')
        self.stop = self.context.Event()
        self.queues = {name: self.context.Queue(maxsize=4) for name in configs}
        self.processes = {}

    def start(self):
        for name, config in self.configs.items():
            process = self.context.Process(target=self.worker,
                                           args=(name, config, self.queues[name], self.stop), daemon=True)
            self.processes[name] = process
            process.start()
        for name in self.configs:
            self.read(name, timeout=30)

    def read(self, name, *, timeout=.3):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                item = self.queues[name].get(timeout=min(.1, max(.001, deadline-time.monotonic())))
            except queue.Empty:
                if not self.processes[name].is_alive():
                    raise RuntimeError(f'{name}: camera process stopped')
                continue
            if item[0] == 'error':
                raise RuntimeError(f'{name}: {item[2]}')
            if time.perf_counter() - item[3]/1e9 <= .3:
                return item
        raise RuntimeError(f'{name}: stale camera frames')

    def close(self):
        self.stop.set()
        for process in self.processes.values():
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()  # camera worker only; never a motor producer
                process.join(timeout=2)
        for output in self.queues.values():
            output.close()
