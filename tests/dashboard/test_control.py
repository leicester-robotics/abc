import unittest
import numpy as np
from deploy.dashboard.config import ArmConfig
from deploy.dashboard.samples import ArmSample, LeaderSample, LatestSample
from deploy.dashboard.control import ControlSupervisor

class Adapter:
    def __init__(self, clock):
        self.clock=clock; self.q=np.array([0.,.5,1.,0.,0.,0.,.5]); self.commands=[]; self.closed=False; self.stale=False
    def connect(self):pass
    def observe(self):return ArmSample(self.clock()-(1 if self.stale else 0),self.q.copy())
    def command(self,q):self.commands.append(q.copy())
    def hold(self,q):self.command(q)
    def close(self):self.closed=True

class ControlTests(unittest.TestCase):
    def setUp(self):
        self.now=10.; self.clock=lambda:self.now
        self.cfg=ArmConfig('left','test',list(range(7)),[1]*7,[0.]*7,True,'can0')
        self.buf=LatestSample(); self.adapter=Adapter(self.clock); self.created=[]
        def factory(cfg):self.created.append(cfg.side);return self.adapter
        self.supervisor=ControlSupervisor([self.cfg],{'left':self.buf},factory,clock=self.clock)
        self.publish(); self.supervisor.heartbeat('a',self.now)
    def publish(self,q=None,age=0):
        self.buf.publish(LeaderSample(self.now-age,self.adapter.q.copy() if q is None else q,calibrated=True))
    def request(self,action,client='a'):
        self.supervisor.request(action,client);self.supervisor.tick(self.now)
    def enable(self):self.request('connect');self.request('enable');self.assertEqual(self.supervisor.mode,'teleop')
    def test_startup_never_creates_robot(self):
        self.supervisor.tick(self.now);self.assertEqual(self.created,[])
    def test_alignment_limits(self):
        self.request('connect')
        for index,delta in [(0,.151),(6,.101)]:
            q=self.adapter.q.copy();q[index]+=delta;self.publish(q)
            self.request('enable');self.assertNotEqual(self.supervisor.mode,'teleop')
    def test_reject_stale_nonfinite_out_of_range_and_calibration(self):
        self.request('connect')
        for bad in [np.full(7,np.nan),np.full(7,np.inf),np.full(7,100.)]:
            self.publish(bad);self.request('enable');self.assertNotEqual(self.supervisor.mode,'teleop')
        self.publish(age=.201);self.request('enable');self.assertNotEqual(self.supervisor.mode,'teleop')
        self.publish();self.cfg.mapping_verified=False;self.request('enable');self.assertNotEqual(self.supervisor.mode,'teleop')
    def test_slew_limit(self):
        self.enable();self.now+=.1
        q=self.adapter.q.copy();q[0]+=.4;q[-1]+=.4;self.publish(q)
        self.supervisor.heartbeat('a',self.now);self.supervisor.tick(self.now)
        self.assertLessEqual(np.max(np.abs(self.adapter.commands[-1]-self.adapter.q)),.050001)
    def test_nonowner_cannot_renew_or_enable(self):
        self.enable();self.now+=.501;self.publish()
        self.supervisor.heartbeat('b',self.now);self.request('enable','b')
        self.assertNotEqual(self.supervisor.mode,'teleop')
        self.assertIsNone(self.supervisor.owner)
    def test_stale_tracking_faults_and_reconnect_does_not_enable(self):
        self.enable();self.now+=.201;self.supervisor.heartbeat('a',self.now)
        self.supervisor.tick(self.now);self.assertEqual(self.supervisor.mode,'fault')
        self.publish();self.supervisor.heartbeat('a',self.now);self.supervisor.tick(self.now)
        self.assertEqual(self.supervisor.mode,'fault')
    def test_return_to_sim_holds_measured_position(self):
        self.enable();self.request('simulation')
        self.assertEqual(self.supervisor.mode,'holding')
        np.testing.assert_equal(self.adapter.commands[-1],self.adapter.q)
    def test_disconnect_revokes_and_cleanup_does_not_home(self):
        self.enable();self.supervisor.disconnect('a');self.supervisor.tick(self.now)
        self.assertNotEqual(self.supervisor.mode,'teleop')
        self.supervisor.close();self.assertTrue(self.adapter.closed)
        self.assertTrue(all(np.array_equal(q,self.adapter.q) for q in self.adapter.commands))
    def test_feedback_loss_reports_fault(self):
        self.enable();self.adapter.stale=True;self.supervisor.tick(self.now)
        self.assertEqual(self.supervisor.mode,'fault')

    def test_stop_is_latched_when_request_queue_is_full(self):
        self.enable()
        for _ in range(32):self.supervisor.request('enable','b')
        self.supervisor.request('simulation','a')
        self.supervisor.heartbeat('a',self.now)
        self.supervisor.tick(self.now)
        self.assertNotEqual(self.supervisor.mode,'teleop')

    def test_failed_release_is_sticky_and_retains_adapter(self):
        self.enable()
        def failed_close():raise RuntimeError('release unconfirmed')
        self.adapter.close=failed_close
        self.request('release')
        self.assertEqual(self.supervisor.mode,'fault')
        self.assertIn('left',self.supervisor.adapters)
        self.request('simulation');self.request('recover')
        self.assertEqual(self.supervisor.mode,'fault')
        self.request('connect');self.assertEqual(len(self.created),1)
        self.adapter.close=lambda:None
        self.request('release')
        self.assertEqual(self.supervisor.mode,'simulation')
        self.assertEqual(self.supervisor.adapters,{})

    def test_encoder_noise_at_rest_home_is_clipped_to_command_limits(self):
        from deploy.dashboard.control import checked,HIGH,LOW
        q=np.array([0.,-.007,0.,1.576,0.,0.,.5])
        result=checked(ArmSample(self.now,q),self.now)
        self.assertEqual(result[1],LOW[1])
        self.assertEqual(result[3],HIGH[3])
        q[1]=-.02
        with self.assertRaises(ValueError):checked(ArmSample(self.now,q),self.now)

    def test_physical_connect_requires_fresh_support_confirmation(self):
        self.supervisor.require_support=True
        self.request('connect')
        self.assertEqual(self.created,[])
        self.request('confirm_support')
        self.request('connect')
        self.assertEqual(self.supervisor.mode,'holding')
        self.request('release')
        self.request('connect')
        self.assertEqual(len(self.created),1)

    def test_shutdown_cause_survives_failed_observation(self):
        self.enable()
        self.adapter.close=lambda:(_ for _ in ()).throw(RuntimeError('motor 7 lost ACK'))
        self.request('release')
        self.adapter.observe=lambda:(_ for _ in ()).throw(RuntimeError('not connected'))
        self.supervisor.tick(self.now)
        self.assertIn('motor 7 lost ACK',self.supervisor.error)

    def test_enable_from_different_pose_aligns_gradually_without_jump(self):
        self.request('connect')
        goal=self.adapter.q.copy();goal[0]+=.8;self.publish(goal)
        before=self.adapter.q.copy()
        self.request('enable')
        self.assertEqual(self.supervisor.mode,'aligning')
        np.testing.assert_allclose(self.adapter.commands[-1],before)
        self.now+=.1;self.publish(goal);self.supervisor.heartbeat('a',self.now)
        self.supervisor.tick(self.now)
        self.assertGreater(self.adapter.commands[-1][0],before[0])
        self.assertLessEqual(self.adapter.commands[-1][0]-before[0],.025001)
        self.now+=.501;self.publish(goal);self.supervisor.tick(self.now)
        self.assertEqual(self.supervisor.mode,'holding')

    def test_alignment_enters_teleop_only_after_robot_reaches_leader(self):
        self.request('connect')
        goal=self.adapter.q.copy();goal[0]+=.5;self.publish(goal);self.request('enable')
        self.assertEqual(self.supervisor.mode,'aligning')
        for _ in range(30):
            self.now+=.1;self.adapter.q=self.adapter.commands[-1].copy()
            self.publish(goal);self.supervisor.heartbeat('a',self.now);self.supervisor.tick(self.now)
        self.assertEqual(self.supervisor.mode,'teleop')
        np.testing.assert_allclose(self.adapter.commands[-1],goal)

    def test_moving_leader_during_alignment_stops_motion(self):
        self.request('connect');goal=self.adapter.q.copy();goal[0]+=.8
        self.publish(goal);self.request('enable');self.now+=.1
        goal[0]+=.1;self.publish(goal);self.supervisor.heartbeat('a',self.now)
        self.supervisor.tick(self.now)
        self.assertEqual(self.supervisor.mode,'holding')
        self.assertIn('Keep GELLO still',self.supervisor.error)

    def test_connect_stays_holding_if_browser_not_ready_for_home_move(self):
        self.supervisor.home_on_connect=True
        self.supervisor._heartbeats.clear()
        self.request('connect')
        self.assertEqual(self.supervisor.mode,'holding')
        self.assertFalse(self.adapter.closed)

    def test_force_feedback_only_has_live_targets_during_teleop(self):
        commands=LatestSample()
        self.supervisor.haptics={'left':commands}
        self.adapter.observe=lambda:ArmSample(self.now,self.adapter.q.copy(),effort=np.ones(7),raw={'gripper_direction':-1})
        self.request('connect');self.assertFalse(commands.read().enabled)
        self.request('enable');self.assertTrue(commands.read().enabled)
        self.request('simulation');self.assertFalse(commands.read().enabled)

    def test_connect_uses_saved_home_not_current_leader_pose(self):
        self.supervisor.home_on_connect=True
        home=np.array([0.,0.,0.,1.5708,0.,0.,.5])
        self.supervisor.home_pose=home
        self.request('connect')
        np.testing.assert_allclose(self.supervisor._alignment_goals['left'],np.clip(home,[-2.617,0,0,-1.57,-1.57,-2.09,0],[3.13,3.65,3.13,1.57,1.57,2.09,1]))
        self.assertEqual(self.supervisor._alignment_then,'holding')

    def test_haptic_publication_preserves_feedback_age(self):
        commands=LatestSample();self.supervisor.haptics={'left':commands}
        self.supervisor.mode='teleop'
        self.supervisor.observations={'left':ArmSample(self.now-.09,self.adapter.q.copy(),effort=np.ones(7),raw={'gripper_direction':-1})}
        self.supervisor._publish()
        self.assertTrue(commands.read().enabled)
        self.assertAlmostEqual(commands.read().acquired_at,self.now-.09)
        self.assertGreater(self.now+.02-commands.read().acquired_at,.1)
