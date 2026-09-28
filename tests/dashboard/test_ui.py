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

    def test_sim_target_and_error_are_visible(self):
        from deploy.dashboard.ui import format_simulation
        text=format_simulation(ArmSample(1.,np.zeros(7)),np.full(7,.1),now=1.)
        self.assertIn('Target',text)
        self.assertIn('Error',text)
        self.assertIn('0.1000',text)

    def test_offsets_and_measured_rates_are_visible(self):
        from deploy.dashboard.ui import format_diagnostics
        from deploy.dashboard.config import ArmConfig
        arm=ArmConfig('left','test',list(range(7)),[1]*7,[.125]*7)
        text=format_diagnostics(arm,50.,[.05]*7)
        self.assertIn('0.125',text)
        self.assertIn('50.0 Hz',text)
        self.assertIn('Alignment',text)

    def test_fault_panel_explains_current_write_timeout_and_keeps_original(self):
        from deploy.dashboard.ui import format_fault
        text=format_fault({'mode':'fault','error':'Confirm arms supported',
            'last_fault':{'reason':'right: GELLO 5 write 102: -3001/0','time':'2026-09-28T12:00:00','stage':'teleop'}})
        for expected in ['right','GELLO 5','acknowledgment','Current message','2026-09-28','Clear fault']:
            self.assertIn(expected,text)
