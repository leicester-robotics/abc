import json
import tempfile
import unittest
from pathlib import Path
from deploy.dashboard.config import load_config, save_config


def station():
    return {'arms': [{'side': 'left', 'device': '/dev/test', 'servo_ids': list(range(7)),
                      'signs': [1]*7}], 'cameras': []}


class ConfigTests(unittest.TestCase):
    def load(self, value):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'config.json'
            path.write_text(json.dumps(value))
            return load_config(path)

    def test_missing_offsets_is_uncalibrated(self):
        cfg = self.load(station())
        self.assertFalse(cfg.arms[0].calibrated)
        self.assertEqual(cfg.port, 8080)

    def test_reject_invalid_calibration(self):
        for field, value in [('signs', [0]*7), ('offsets', [float('nan')]*7),
                             ('offsets', [0]*6), ('servo_ids', [1]*7),
                             ('gripper_type', 'unknown')]:
            with self.subTest(field=field):
                data = station(); data['arms'][0][field] = value
                with self.assertRaises(ValueError): self.load(data)

    def test_duplicate_side_or_device_rejected(self):
        for field in ['side', 'device']:
            data = station()
            other = dict(data['arms'][0], side='right', device='/dev/other')
            other[field] = data['arms'][0][field]
            data['arms'].append(other)
            with self.assertRaises(ValueError): self.load(data)

    def test_runtime_mode_not_persisted(self):
        cfg = self.load(station())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'station.json'
            save_config(path, cfg)
            self.assertNotIn('mode', json.loads(path.read_text()))
            self.assertEqual(load_config(path), cfg)

    def test_duplicate_camera_serial_rejected(self):
        data = station(); data['cameras'] = [{'serial':'a','label':'top'}, {'serial':'a','label':'left'}]
        with self.assertRaises(ValueError): self.load(data)
