"""GELLO acquisition with optional bounded haptics owned by the same serial thread."""
from __future__ import annotations
import glob
import time
import numpy as np
from .samples import LeaderSample
from .workers import DeviceWorker
from .haptics import HapticOutput


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

    def __init__(self, config, buffer, connection_factory=None, sync_factory=None, haptics=None):
        super().__init__(buffer)
        self.haptics = haptics
        self.haptic_output = None
        self._haptic_previous = 0.
        self._haptic_filtered = np.zeros(7)
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
        if self.haptic_output is not None:
            self.haptic_output.connection = self.connection
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
        self._update_haptics()
        self.buffer.publish(LeaderSample(
            acquired_at=started, position=self.mapper.map(counts * (np.pi / 2048.)) if self.mapper else map_encoder(counts, self.config),
            counts=counts, radians=encoder_radians(counts), calibrated=self.config.calibrated,
            raw={'servo_ids': self.config.servo_ids, 'device': self.config.device,
                 'read_mode': 'individual' if self._individual_reads else 'sync',
                 'read_ms': round((time.monotonic()-started)*1000, 2),
                 'haptics_active': bool(self.haptic_output and self.haptic_output.active),
                 'haptic_current_ma': self.haptic_output.current.tolist() if self.haptic_output else [0]*7},
        ))

    def _update_haptics(self):
        now=time.monotonic()
        command=self.haptics.read() if self.haptics is not None else None
        valid=command is not None and command.enabled and 0<=now-command.acquired_at<=.1
        if not valid:
            if self.haptic_output is not None and (self.haptic_output.active or self.haptic_output.original):
                self.haptic_output.close()
            self._haptic_filtered[:]=0
            self._haptic_previous=now
            return
        if self.haptic_output is None:self.haptic_output=HapticOutput(self.connection,self.config.servo_ids)
        if not self.haptic_output.active:
            if self.haptic_output.original:self.haptic_output.close()
            self.haptic_output.enable()
            self._haptic_previous=time.monotonic()
            return
        dt=now-self._haptic_previous
        if dt<.03:return
        self._haptic_filtered+=np.clip(command.current_ma-self._haptic_filtered,-25*min(dt,.1),25*min(dt,.1))
        self.haptic_output.write(self._haptic_filtered)
        self._haptic_previous=now

    def _close_device(self):
        try:
            if self.haptic_output is not None:
                self.haptic_output.close()
                self.haptic_output = None
        finally:
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
