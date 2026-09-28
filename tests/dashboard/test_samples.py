import unittest
import numpy as np
from deploy.dashboard.samples import ArmSample, LatestSample

class SampleTests(unittest.TestCase):
    def test_snapshot_isolation_and_timestamp(self):
        data = np.zeros(7)
        buffer = LatestSample()
        buffer.publish(ArmSample(10., data))
        data[0] = 99
        sample = buffer.read()
        self.assertEqual(sample.acquired_at, 10.)
        self.assertEqual(sample.position[0], 0.)
        sample.position[0] = 42
        self.assertEqual(buffer.read().position[0], 0.)
        buffer.set_error('lost connection')
        self.assertEqual(buffer.read().acquired_at, 10.)
        self.assertEqual(buffer.error, 'lost connection')
