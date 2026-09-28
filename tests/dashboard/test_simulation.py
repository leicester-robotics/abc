import unittest
import os
from pathlib import Path
import numpy as np
from deploy.dashboard.simulation import Simulation
from deploy.dashboard.samples import LeaderSample

MODEL=Path('abc_sim/models/yam_bimanual_empty.xml')
ASSETS=Path(os.environ.get('DASHBOARD_TEST_ASSETS', '/home/tarik/code/abc/abc_sim/models/assets/i2rt_yam/assets'))

class SimulationTests(unittest.TestCase):
    def setUp(self): self.sim=Simulation(MODEL,asset_dir=ASSETS)
    def test_physics_moves_toward_target(self):
        start=self.sim.snapshot('left').position
        target=start.copy(); target[0]+=.2
        self.sim.set_target('left',target)
        for _ in range(50):self.sim.step(.01)
        state=self.sim.snapshot('left')
        self.assertGreater(state.position[0],start[0]+.03)
        self.assertTrue(np.isfinite(state.position).all())
        self.assertLess(abs(target[0]-state.position[0]),.2)
    def test_gripper_range_and_right_arm_independent(self):
        right=self.sim.targets['right'].copy()
        target=self.sim.snapshot('left').position; target[-1]=1.
        self.sim.set_target('left',target)
        idx=self.sim.actuator_ids['left'][-1]
        self.assertAlmostEqual(self.sim.data.ctrl[idx],self.sim.model.actuator_ctrlrange[idx,1])
        np.testing.assert_equal(self.sim.targets['right'],right)
    def test_stale_leader_does_not_change_target(self):
        initial=self.sim.targets['left'].copy()
        self.sim.follow('left',LeaderSample(10.,np.ones(7),calibrated=True),now=11.)
        np.testing.assert_equal(self.sim.targets['left'],initial)
    def test_bounded_catchup(self):
        self.sim.step(100.)
        self.assertLessEqual(self.sim.data.time,.051)
    def test_missing_mesh_error(self):
        with self.assertRaisesRegex(FileNotFoundError,'asset'):
            Simulation(MODEL,asset_dir=Path('/does/not/exist'))
