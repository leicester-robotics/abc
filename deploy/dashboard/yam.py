"""Explicit-connect YAM adapter and non-actuating observation subscription."""
from __future__ import annotations
import time
import numpy as np
from .samples import ArmSample, LatestSample
from .workers import DeviceWorker


def decode_observation(values, extras):
    values = np.asarray(values)
    timestamp = extras.get('timestamp')
    if values.shape != (28,) or not np.isfinite(values).all() or timestamp is None or not np.isfinite(timestamp):
        raise ValueError('Expected 28 finite observation values and acquisition timestamp')
    return ArmSample(float(timestamp), values[:7].copy(), values[7:14].copy(), values[14:21].copy(),
                     {'command': values[21:28].tolist(), 'source': 'existing ABC follower'})


class FeedbackCapture(LatestSample):
    def publish(self, acquired_at, position, velocity, effort, raw):
        super().publish(ArmSample(acquired_at, position, velocity, effort, raw))


class PassiveYamWorker(DeviceWorker):
    def __init__(self, side, buffer):
        super().__init__(buffer)
        self.side = side
        self.context = None
        self.socket = None

    def open(self):
        import zmq
        from deploy.robot.communication import create_subscriber
        self.context = zmq.Context()
        self.socket = create_subscriber(self.context, f'follower_{self.side}_obs', conflate=1)

    def read_once(self):
        from deploy.robot.communication import subscribe
        values, extras = subscribe(self.socket, timeout_ms=200)
        self.buffer.publish(decode_observation(values, extras))

    def _close_device(self):
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        if self.context is not None:
            self.context.term()
            self.context = None


class YamAdapter:
    """Capture timestamps where the motor thread publishes a complete feedback batch.

    The installed i2rt observation API returns cached data without timestamps.
    This instance-local hook records a sample only after all motor replies arrive
    and absolute positions update, under the driver's existing state lock.
    """
    def __init__(self, config, factory=None):
        self.config = config
        self.factory = factory
        self.robot = None
        self.capture = FeedbackCapture()
        self._original_update = None

    def connect(self):
        if not self.config.mapping_verified or not self.config.calibrated:
            raise RuntimeError('Verify station mapping and calibration before connecting')
        if self.factory is None:
            from deploy.robot.followers.yam_follower import _patch_gripper_calibration
            from i2rt.robots.get_robot import get_yam_robot
            from i2rt.robots.utils import GripperType
            _patch_gripper_calibration()
            factory = lambda: get_yam_robot(channel=self.config.channel,
                gripper_type=GripperType[self.config.gripper_type.upper()], zero_gravity_mode=False)
        else:
            factory = self.factory
        self.robot = factory()
        try:
            chain = self.robot.motor_chain
            self._original_update = chain._update_absolute_positions
            original = self._original_update

            def updated(feedback):
                original(feedback)
                now = time.monotonic()
                positions = np.array([chain._joint_position_real_to_sim_idx(chain.absolute_positions[i], i)
                                      for i in range(len(feedback))])
                velocity = np.array([f.velocity*chain.motor_direction[i] for i,f in enumerate(feedback)])
                effort = np.array([f.torque*chain.motor_direction[i] for i,f in enumerate(feedback)])
                remapper = self.robot.remapper
                self.capture.publish(now, remapper.to_command_joint_pos_space(positions),
                    remapper.to_command_joint_vel_space(velocity), effort,
                    {'motor_position_rad': positions.tolist(),
                     'motor_errors': [f.error_code for f in feedback],
                     'temperature_rotor': [f.temperature_rotor for f in feedback],
                     'temperature_mos': [f.temperature_mos for f in feedback]})
            chain._update_absolute_positions = updated
            deadline = time.monotonic()+1.
            while self.capture.read() is None and time.monotonic() < deadline:
                time.sleep(.005)
            if self.capture.read() is None:
                raise RuntimeError('Driver did not publish a fresh feedback batch; teleop disabled')
        except Exception:
            self.close()
            raise

    def observe(self):
        if self.robot is None:
            raise RuntimeError('YAM is not connected')
        if not self.robot.motor_chain.running or not self.robot._server_thread.is_alive():
            raise RuntimeError('YAM control thread stopped; physical state unconfirmed')
        sample = self.capture.read()
        if sample is None:
            raise RuntimeError('No acquisition-stamped YAM feedback')
        return sample

    def command(self, target):
        self.robot.command_joint_pos(np.asarray(target).copy())

    def hold(self, target):
        self.command(target)

    def close(self):
        if self.robot is not None:
            # i2rt.close stops threads and releases torque; never calls move_joints.
            self.robot.close()
            if self._original_update is not None:
                self.robot.motor_chain._update_absolute_positions = self._original_update
            self.robot = None
