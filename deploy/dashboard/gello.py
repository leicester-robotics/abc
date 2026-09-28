"""Read-only GELLO acquisition: no mode, torque, or current writes."""
from __future__ import annotations
import glob
import time
import numpy as np
from .samples import LeaderSample
from .workers import DeviceWorker


def encoder_radians(counts):
    return (np.asarray(counts, dtype=float) / 2048. - 1.) * np.pi


def map_encoder(counts, config):
    if not config.calibrated:
        return None
    position = np.asarray(config.signs) * (encoder_radians(counts) - config.offsets)
    position[-1] = np.interp(position[-1], config.gripper_range, [0., 1.])
    return position


def calibration_offsets(counts, signs):
    """Nearest pi/4 offsets at the repository's documented zero pose."""
    raw = encoder_radians(counts)
    reference = np.array([0., 0., 0., 0., 0., 0., .357])
    return (np.round((raw - reference / np.asarray(signs)) / (np.pi / 4)) * (np.pi / 4)).tolist()


class GelloReader(DeviceWorker):
    rate = 100.

    def __init__(self, config, buffer, connection_factory=None, sync_factory=None):
        super().__init__(buffer)
        self.config = config
        self.connection_factory = connection_factory
        self.sync_factory = sync_factory
        self.connection = None
        self.sync = None

    def open(self):
        if self.connection_factory:
            self.connection = self.connection_factory()
        else:
            from deploy.robot.leaders.dynamixel import Dynamixel
            self.connection = Dynamixel.Config(baudrate=self.config.baudrate, device_name=self.config.device).instantiate()
        try:
            if self.sync_factory:
                self.sync = self.sync_factory(self.connection, self.config.servo_ids)
            else:
                from dynamixel_sdk import GroupSyncRead
                self.sync = GroupSyncRead(self.connection.portHandler, self.connection.packetHandler, 132, 4)
            for motor_id in self.config.servo_ids:
                if not self.sync.addParam(motor_id):
                    raise RuntimeError(f'Cannot register servo {motor_id}')
        except Exception:
            self._close_device()
            raise

    def read_once(self):
        if self.sync.txRxPacket() != 0:
            raise ConnectionError('GELLO read failed; check power, baudrate, and servo IDs')
        counts = []
        for motor_id in self.config.servo_ids:
            if not self.sync.isAvailable(motor_id, 132, 4):
                raise ConnectionError(f'No fresh position from servo {motor_id}')
            value = self.sync.getData(motor_id, 132, 4)
            counts.append(value - 2**32 if value >= 2**31 else value)
        counts = np.array(counts)
        self.buffer.publish(LeaderSample(
            acquired_at=time.monotonic(), position=map_encoder(counts, self.config),
            counts=counts, radians=encoder_radians(counts), calibrated=self.config.calibrated,
            raw={'servo_ids': self.config.servo_ids, 'device': self.config.device},
        ))

    def _close_device(self):
        if self.connection is not None:
            self.connection.disconnect()
            self.connection = None


def discover_devices():
    import pyrealsense2 as rs
    return {
        'serial_adapters': sorted(glob.glob('/dev/serial/by-id/*')),
        'cameras': [{'serial': d.get_info(rs.camera_info.serial_number),
                     'name': d.get_info(rs.camera_info.name)} for d in rs.context().query_devices()],
    }
