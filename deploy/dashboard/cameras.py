"""RealSense capture independent of control and browser update loops."""
import time
import numpy as np
from .samples import CameraSample
from .workers import DeviceWorker


class RealSenseSource:
    def __init__(self, config):
        self.config = config
        self.pipeline = None
        self.started = False
        self.depth_status = 'disabled'

    def start(self):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(self.config.serial)
        # These RGB settings are supported by D405; depth is optional.
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        if self.config.depth:
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        try:
            self.pipeline.start(cfg)
            self.started = True
            self.depth_status = 'enabled' if self.config.depth else 'disabled'
        except RuntimeError:
            if not self.config.depth:
                raise
            self.pipeline = rs.pipeline()
            cfg.disable_stream(rs.stream.depth)
            self.pipeline.start(cfg)
            self.started = True
            self.depth_status = 'unavailable; RGB only'
        self.colorizer = rs.colorizer()

    def read(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=500)
        except RuntimeError as error:
            if 'Frame didn' in str(error):
                raise TimeoutError('Waiting for camera frames') from error
            raise
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError('No RGB frame')
        depth = frames.get_depth_frame()
        preview = np.asanyarray(self.colorizer.colorize(depth).get_data()).copy() if depth else None
        return np.asanyarray(color.get_data()).copy(), preview, self.depth_status

    def close(self):
        if self.started:
            try:
                self.pipeline.stop()
            finally:
                self.started = False


class CameraWorker(DeviceWorker):
    rate = 30.

    def __init__(self, config, buffer, source_factory=None):
        super().__init__(buffer)
        self.source_factory = source_factory or (lambda: RealSenseSource(config))
        self.source = None

    def open(self):
        self.source = self.source_factory()
        try:
            self.source.start()
        except Exception:
            self._close_device()
            raise

    def read_once(self):
        rgb, depth, status = self.source.read()
        self.buffer.publish(CameraSample(time.monotonic(), rgb, depth, status))

    def _close_device(self):
        if self.source is not None:
            source, self.source = self.source, None
            source.close()
