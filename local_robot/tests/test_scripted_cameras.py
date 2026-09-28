"""Camera process tests use generated frames; no camera SDK or devices."""
import time
import unittest
from types import SimpleNamespace
import numpy as np

from local_robot.scripted_cameras import CameraStreams


def generated_camera(name, config, output, stop):
    while not stop.is_set():
        try:
            output.put(('rgb', name, np.full((4, 4, 3), 123, np.uint8), time.perf_counter_ns()), timeout=.1)
        except Exception:
            pass
        stop.wait(.02)


def failed_camera(name, config, output, stop):
    output.put(('error', name, 'camera disconnected', time.perf_counter_ns()))


class CameraTests(unittest.TestCase):
    def test_receives_frames_from_separate_processes_and_joins(self):
        streams = CameraStreams({n: SimpleNamespace() for n in ('top', 'left', 'right')}, worker=generated_camera)
        try:
            streams.start()
            for name in ('top', 'left', 'right'):
                item = streams.read(name)
                self.assertEqual(item[:2], ('rgb', name))
                self.assertTrue((item[2] == 123).all())
                self.assertLess(time.perf_counter() - item[3]/1e9, .3)
            self.assertTrue(all(p.pid is not None for p in streams.processes.values()))
        finally:
            streams.close()
        self.assertTrue(all(not p.is_alive() for p in streams.processes.values()))

    def test_worker_failure_propagates_before_robot_connection(self):
        streams = CameraStreams({'top': SimpleNamespace()}, worker=failed_camera)
        try:
            with self.assertRaisesRegex(RuntimeError, 'camera disconnected'):
                streams.start()
        finally:
            streams.close()


if __name__ == '__main__':
    unittest.main()
