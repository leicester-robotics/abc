import unittest
from deploy.dashboard.ui import heartbeat_html, format_arm
from deploy.dashboard.samples import ArmSample
import numpy as np

class UITests(unittest.TestCase):
    def test_heartbeat_is_browser_originated_and_stops_when_hidden(self):
        html=heartbeat_html('button-id')
        self.assertIn('setInterval',html)
        self.assertIn('100',html)
        self.assertIn('visibilityState',html)
        self.assertIn('button-id',html)
        self.assertIn('click()',html)
    def test_telemetry_labels_source_units_and_staleness(self):
        text=format_arm('Physical YAM',ArmSample(1.,np.zeros(7)),now=2.)
        self.assertIn('Physical YAM',text)
        self.assertIn('rad',text)
        self.assertIn('STALE',text)
        self.assertIn('normalized',text)
