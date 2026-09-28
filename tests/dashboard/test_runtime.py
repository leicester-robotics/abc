import queue
import time
import unittest
from pathlib import Path
import numpy as np
from deploy.dashboard.runtime import SimulationRunner
from deploy.dashboard.simulation import Simulation
from deploy.dashboard.config import DashboardConfig,ArmConfig
from deploy.dashboard.samples import LatestSample,LeaderSample
from tests.dashboard.test_simulation import MODEL,ASSETS

class RuntimeTests(unittest.TestCase):
    def test_physics_continues_without_browser_updates(self):
        sim=Simulation(MODEL,ASSETS)
        runner=SimulationRunner(sim,DashboardConfig(),{},queue.Queue(),lambda:{'mode':'simulation'},demo=True)
        runner.start()
        try:
            before=runner.snapshot()['simulation']['left'].raw['simulation_time']
            time.sleep(.15)  # Browser/camera presentation does no work during this wait.
            after=runner.snapshot()['simulation']['left'].raw['simulation_time']
            self.assertGreater(after-before,.08)
        finally:runner.close()
    def test_relative_preview_and_stale_input(self):
        arm=ArmConfig('left','test',list(range(7)),[1]*7)
        buf=LatestSample();actions=queue.Queue();now=time.monotonic()
        buf.publish(LeaderSample(now,None,counts=np.full(7,2048),radians=np.zeros(7)))
        sim=Simulation(MODEL,ASSETS)
        runner=SimulationRunner(sim,DashboardConfig(arms=[arm]),{'left':buf},actions,lambda:{'mode':'simulation'})
        actions.put(('preview',None));runner.tick(now)
        initial=runner.snapshot()['targets']['left'].copy()
        angles=np.zeros(7);angles[0]=.2
        buf.publish(LeaderSample(now+.01,None,counts=np.full(7,2048),radians=angles))
        runner.tick(now+.01)
        self.assertAlmostEqual(runner.snapshot()['targets']['left'][0],initial[0]+.2)
        angles[0]=.4
        buf.publish(LeaderSample(now-1.,None,counts=np.full(7,2048),radians=angles))
        runner.tick(now+.02)
        self.assertAlmostEqual(runner.snapshot()['targets']['left'][0],initial[0]+.2)
