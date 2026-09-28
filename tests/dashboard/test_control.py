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
