import unittest
import numpy as np
from deploy.dashboard.yam import decode_observation, FeedbackCapture

class YamTests(unittest.TestCase):
    def test_passive_observation_uses_source_timestamp(self):
        sample=decode_observation(np.arange(28,dtype=float),{'timestamp':12.})
        self.assertEqual(sample.acquired_at,12.)
        np.testing.assert_equal(sample.position,np.arange(7))
        np.testing.assert_equal(sample.effort,np.arange(14,21))
    def test_invalid_passive_observations(self):
        for array,extras in [(np.zeros(27),{'timestamp':1.}), (np.zeros(28),{}), (np.full(28,np.nan),{'timestamp':1.})]:
            with self.assertRaises(ValueError):decode_observation(array,extras)
    def test_capture_does_not_refresh_cached_feedback(self):
        cap=FeedbackCapture()
        cap.publish(5.,np.zeros(7),np.zeros(7),np.zeros(7),{})
        self.assertEqual(cap.read().acquired_at,5.)
        self.assertEqual(cap.read().acquired_at,5.)
