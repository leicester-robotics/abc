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
    rate = 400.

    def __init__(self, config, buffer, connection_factory=None, sync_factory=None):
        super().__init__(buffer)
        self.config = config
        self.connection_factory = connection_factory
        self.sync_factory = sync_factory
        self.connection = None
        self.sync = None
        self.mapper = None
        self._individual_reads = False

    def open(self):
        self._individual_reads = False
        if self.config.calibration_path:
            from pathlib import Path
            from .saved_calibration import Calibration, RelativeMapper
            cal = Calibration.load(Path(self.config.calibration_path))
            if (cal.port, cal.ids, cal.baudrate) != (self.config.device, self.config.servo_ids, self.config.baudrate):
                raise ValueError('Saved calibration does not match arm device, IDs and baudrate')
            self.mapper = RelativeMapper(cal)
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
        started = time.monotonic()
        if not self._individual_reads:
            result = self.sync.txRxPacket()
            if result != 0:
                if not hasattr(self.connection, 'packetHandler'):
                    raise ConnectionError('GELLO read failed; check power, baudrate, and servo IDs')
                self._individual_reads = True
        counts = []
        for motor_id in self.config.servo_ids:
            if self._individual_reads:
                value, result, error = self.connection.packetHandler.read4ByteTxRx(
                    self.connection.portHandler, motor_id, 132)
                if result != 0 or error != 0:
                    raise ConnectionError(f'Invalid encoder reply from servo {motor_id}: {result}/{error}')
            else:
                if not self.sync.isAvailable(motor_id, 132, 4):
                    raise ConnectionError(f'No fresh position from servo {motor_id}')
                value = self.sync.getData(motor_id, 132, 4)
            counts.append(value - 2**32 if value >= 2**31 else value)
        counts = np.array(counts)
        self.buffer.publish(LeaderSample(
            acquired_at=started, position=self.mapper.map(counts * (np.pi / 2048.)) if self.mapper else map_encoder(counts, self.config),
            counts=counts, radians=encoder_radians(counts), calibrated=self.config.calibrated,
            raw={'servo_ids': self.config.servo_ids, 'device': self.config.device,
                 'read_mode': 'individual' if self._individual_reads else 'sync',
                 'read_ms': round((time.monotonic()-started)*1000, 2)},
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
