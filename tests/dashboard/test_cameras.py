import unittest
import numpy as np
from deploy.dashboard.config import CameraConfig
from deploy.dashboard.samples import LatestSample
from deploy.dashboard.cameras import CameraWorker

class Source:
    def __init__(self,fail=False):self.closed=False; self.fail=fail
    def start(self):
        if self.fail:raise RuntimeError('unplugged')
    def read(self):return np.zeros((20,30,3),dtype=np.uint8),None,'unavailable'
    def close(self):self.closed=True

class CameraTests(unittest.TestCase):
    def test_partial_start_closes_and_other_camera_continues(self):
        bad=Source(True); good=Source(); a=LatestSample(); b=LatestSample()
        first=CameraWorker(CameraConfig('a','top'),a,source_factory=lambda:bad)
        second=CameraWorker(CameraConfig('b','wrist'),b,source_factory=lambda:good)
        with self.assertRaises(RuntimeError):first.open()
        self.assertTrue(bad.closed)
        second.open(); second.read_once(); second.close()
        self.assertEqual(b.read().rgb.shape,(20,30,3)); self.assertTrue(good.closed)
    def test_depth_unavailable_keeps_rgb(self):
        buf=LatestSample(); worker=CameraWorker(CameraConfig('a','top',True),buf,source_factory=Source)
        worker.open(); worker.read_once(); worker.close()
        self.assertIsNone(buf.read().depth)
        self.assertEqual(buf.read().depth_status,'unavailable')

    def test_transient_frame_timeout_keeps_pipeline_open(self):
        import time
        class WarmingSource(Source):
            starts=0;reads=0
            def start(self):self.starts+=1
            def read(self):
                self.reads+=1
                if self.reads==1:raise TimeoutError('warming up')
                return super().read()
        source=WarmingSource();buf=LatestSample()
        worker=CameraWorker(CameraConfig('a','top'),buf,source_factory=lambda:source)
        worker.start()
        deadline=time.monotonic()+.5
        while buf.read() is None and time.monotonic()<deadline:time.sleep(.01)
        worker.close()
        self.assertIsNotNone(buf.read())
        self.assertEqual(source.starts,1)
