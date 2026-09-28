import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

class PhysicalTests(unittest.TestCase):
    def test_gripper_calibration_holds_arm_and_clears_gripper_effort(self):
        from deploy.dashboard.physical import calibrate_gripper
        class Chain:
            motor_list=list(range(7));motor_direction=np.ones(7)
            def __init__(self):self.commands=[];self.q=np.array([.1,.5,1.,-.5,.1,.2,.3])
            def read_states(self):return [SimpleNamespace(pos=q) for q in self.q]
            def set_commands(self,torques,**kwargs):
                self.commands.append((np.array(torques),kwargs))
                self.q[6]=3.285 if torques[6]>0 else -3.285
        chain=Chain();clock=[0.]
        def sleep(dt):clock[0]+=dt
        with patch('deploy.dashboard.physical.time.monotonic',side_effect=lambda:clock[0]),patch('deploy.dashboard.physical.time.sleep',side_effect=sleep):
            limits=calibrate_gripper(chain,max_duration=1.,check_interval=.1,travel_range=(6.,7.))
        self.assertGreater(abs(limits[1]-limits[0]),.9)
        for torque,cmd in chain.commands:
            np.testing.assert_allclose(cmd['pos'][:6],[.1,.5,1.,-.5,.1,.2])
            self.assertTrue(np.all(cmd['kp'][:6]>0))
            np.testing.assert_equal(torque[:6],0.)
        np.testing.assert_equal(chain.commands[-1][0],0.)

    def test_failed_initialization_retains_cleanup_owner(self):
        from deploy.dashboard.physical import close_owned
        class Chain:
            def __init__(self):self.closed=False
            def close(self):self.closed=True
        chain=Chain()
        with patch('i2rt.motor_drivers.dm_driver.DMChainCanInterface.active_for_channel',return_value=chain,create=True),patch('i2rt.robots.motor_chain_robot.MotorChainRobot.active_for_channel',return_value=None,create=True):
            close_owned('fake')
        self.assertTrue(chain.closed)

    def test_gripper_sweep_timeout_does_not_become_calibration(self):
        from deploy.dashboard.physical import calibrate_gripper
        class Chain:
            motor_list=list(range(7));motor_direction=np.ones(7)
            def __init__(self):self.q=np.array([0.,.5,1.,0.,0.,0.,0.]);self.effort=None
            def read_states(self):return [SimpleNamespace(pos=q) for q in self.q]
            def set_commands(self,torques,**kwargs):
                self.effort=np.array(torques);self.q[-1]+=.3*np.sign(torques[-1])
        clock=[0.];chain=Chain()
        def sleep(dt):clock[0]+=dt
        with patch('deploy.dashboard.physical.time.monotonic',side_effect=lambda:clock[0]),patch('deploy.dashboard.physical.time.sleep',side_effect=sleep):
            with self.assertRaisesRegex(RuntimeError,'end stop'):
                calibrate_gripper(chain,max_duration=.4,check_interval=.1)
        np.testing.assert_equal(chain.effort,0.)

    def test_station_gripper_span_matches_recorded_calibration(self):
        from deploy.dashboard.physical import validate_gripper_travel
        # Sept 20 recorded limits include a 5% closed-end offset.
        span=(6.693252446875931-1.2742206318152736)/1.05
        validate_gripper_travel(span,(4.5,6.))
        with self.assertRaises(ValueError):validate_gripper_travel(.3,(4.5,6.))
