"""Physical startup using the isolated, integrity-checked station SDK."""
import importlib.util
import os
from pathlib import Path
import time
import numpy as np
from .driver_setup import verify_sdk


def backend_blocker():
    directory=os.environ.get('DASHBOARD_SDK_DIR')
    if not directory:
        return 'Physical connection locked: prepare and launch with the verified dashboard SDK for confirmed torque release and partial-startup cleanup.'
    try:
        root=verify_sdk(directory)
        spec=importlib.util.find_spec('i2rt')
        if spec is None or Path(spec.origin).resolve()!=root/'i2rt/__init__.py':
            raise RuntimeError('The prepared SDK must be first on PYTHONPATH')
    except Exception as error:return str(error)
    return ''


def close_owned(channel):
    from i2rt.motor_drivers.dm_driver import DMChainCanInterface
    from i2rt.robots.motor_chain_robot import MotorChainRobot
    robot=MotorChainRobot.active_for_channel(channel)
    chain=DMChainCanInterface.active_for_channel(channel)
    if robot is not None:robot.close()
    elif chain is not None:chain.close()


def validate_gripper_travel(span, bounds):
    if bounds is None or not bounds[0]<span<bounds[1]:
        raise ValueError(f'Measured gripper travel {span:.4f} rad outside station range {bounds}')


def calibrate_gripper(motor_chain,gripper_index=6,test_torque=.2,max_duration=2.,
                      position_threshold=.01,check_interval=.1,close_offset=0.,travel_range=None):
    """Sweep only the gripper while the six arm joints hold their measured pose."""
    if gripper_index!=6 or len(motor_chain.motor_list)!=7:
        raise ValueError('Expected six YAM joints and one gripper')
    initial=np.array([s.pos for s in motor_chain.read_states()])
    if not np.isfinite(initial).all():raise ValueError('Invalid initial motor position')
    kp=np.array([80.,80.,80.,10.,10.,10.,0.])
    kd=np.array([5.,5.,5.,1.5,1.5,1.5,0.])
    positions=[initial[-1]]
    try:
        for direction in (1,-1):
            torques=np.zeros(7);torques[-1]=direction*min(abs(test_torque),.2)
            started=time.monotonic();last=None;stable=0
            while time.monotonic()-started<max_duration:
                motor_chain.set_commands(torques=torques,pos=initial.copy(),vel=np.zeros(7),kp=kp,kd=kd)
                time.sleep(check_interval)
                states=motor_chain.read_states()
                q=np.array([s.pos for s in states])
                if not np.isfinite(q).all() or np.any(np.abs(q[:6]-initial[:6])>.15):
                    raise RuntimeError('Arm moved during gripper calibration; connection aborted')
                positions.append(q[-1])
                stable=stable+1 if last is not None and abs(q[-1]-last)<position_threshold else 0
                last=q[-1]
                if stable>=6:break
            if stable<6:raise RuntimeError('Gripper did not reach a confirmed end stop before timeout')
    finally:
        motor_chain.set_commands(torques=np.zeros(7),pos=initial.copy(),vel=np.zeros(7),kp=kp,kd=kd)
    low,high=min(positions),max(positions)
    validate_gripper_travel(high-low,travel_range)
    direction=motor_chain.motor_direction[-1]
    limits=[high,low] if direction>0 else [low,high]
    # Dashboard coordinates use measured stops; a closed-end overtravel offset
    # would make zero unreachable and keep alignment waiting at a closed gripper.
    limits[0]+=close_offset*(high-low)*direction
    return limits


class PhysicalConnection:
    def __init__(self,config):self.config=config;self.robot=None

    def connect(self):
        blocker=backend_blocker()
        if blocker:raise RuntimeError(blocker)
        if self.config.gripper_type != 'linear_4310':
            raise RuntimeError('Physical commissioning currently supports the station LINEAR_4310 grippers only')
        if self.config.robot_gripper_travel_range is None:
            raise RuntimeError('Station robot gripper travel range is required')
        from functools import partial
        from i2rt.robots import utils
        from i2rt.robots.get_robot import get_yam_robot
        # This is the symbol imported locally by MotorChainRobot.__init__.
        utils.detect_gripper_limits=partial(calibrate_gripper,travel_range=self.config.robot_gripper_travel_range)
        self.robot=get_yam_robot(channel=self.config.channel,
            gripper_type=utils.GripperType(self.config.gripper_type),zero_gravity_mode=False)
        return self.robot

    def close(self):
        # Registries retain ownership even when a constructor failed before returning.
        close_owned(self.config.channel)
        self.robot=None
