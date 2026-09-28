import unittest
import numpy as np
from deploy.dashboard.leader_hold import CompliantHold

class HoldTests(unittest.TestCase):
    def test_captures_without_jump_resists_and_yields(self):
        hold=CompliantHold();q=np.zeros(7)
        np.testing.assert_equal(hold.current(q,1.,100,6),np.zeros(7))
        q[1]=np.deg2rad(2)
        current=hold.current(q,1.03,100,6)
        self.assertLess(current[1],0);self.assertLessEqual(abs(current[1]),75)
        q[1]=np.deg2rad(30);q[-1]=1
        current=hold.current(q,1.06,100,6)
        self.assertAlmostEqual(hold.anchor[1],np.deg2rad(24))
        self.assertEqual(current[-1],0)
        self.assertTrue(np.all(np.abs(current)<=75))

    def test_wrap_disable_and_stale_recapture(self):
        hold=CompliantHold();q=np.full(7,2*np.pi-.01)
        hold.current(q,1.,100,6)
        q[:]=.01
        current=hold.current(q,1.03,100,6)
        self.assertTrue(np.all(np.abs(current[:6])<25))
        np.testing.assert_equal(hold.current(q,2.,100,6),np.zeros(7))
        np.testing.assert_equal(hold.current(q,2.03,0,6),np.zeros(7))
        self.assertIsNone(hold.anchor)
