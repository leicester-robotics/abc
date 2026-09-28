import tempfile
import unittest
from pathlib import Path

import cv2
import h5py
import numpy as np

from local_robot.scripted_collection import EpisodeWriter, validate_episode
from local_robot.scripted_export import export_session, synthetic_recording


class ExportTests(unittest.TestCase):
    def test_synthetic_recording_exports_three_views_and_14d_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(10):
                synthetic_recording(root / 'recordings' / f'episode_{i:03d}.h5')
            summary = export_session(root)
            self.assertEqual(len(summary['exported_episodes']), 10)
            self.assertEqual(len(list((root/'export/train').iterdir())), 8)
            self.assertEqual(len(list((root/'export/val').iterdir())), 2)
            ep = next((root/'export/train').iterdir())
            rows = np.fromfile(ep/'states_actions.bin', dtype=np.float64).reshape(-1, 28)
            self.assertTrue(np.isfinite(rows).all())
            self.assertGreater(len(rows), 30)
            video = cv2.VideoCapture(str(ep/'combined_camera-images-rgb.mp4'))
            ok, frame = video.read()
            video.release()
            self.assertTrue(ok)
            self.assertEqual(frame.shape, (504, 224, 3))

    def test_corrupt_or_incomplete_recording_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'episode.h5'
            synthetic_recording(path)
            with h5py.File(path, 'r+') as f:
                f['timestamps/q_left'][40] = f['timestamps/q_left'][39]
            with self.assertRaises(ValueError):
                validate_episode(path)

    def test_camera_missing_motion_coverage_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'episode.h5'
            synthetic_recording(path)
            with h5py.File(path, 'r+') as f:
                f['data/right'].resize(30, axis=0)
                f['timestamps/right'].resize(30, axis=0)
            with self.assertRaisesRegex(ValueError, 'coverage'):
                validate_episode(path)

    def test_unusable_input_fails_before_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'recordings/episode.h5'
            synthetic_recording(path)
            with h5py.File(path, 'r+') as f:
                f.attrs['usable'] = False
            with self.assertRaisesRegex(ValueError, 'unusable'):
                export_session(root, expected_episodes=1)
            self.assertFalse((root/'export').exists())

    def test_shifted_camera_timestamps_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'episode.h5'
            synthetic_recording(path)
            with h5py.File(path, 'r+') as f:
                f['timestamps/left'][:] += 1_000_000_000
            with self.assertRaisesRegex(ValueError, 'coverage'):
                validate_episode(path)


if __name__ == '__main__':
    unittest.main()
