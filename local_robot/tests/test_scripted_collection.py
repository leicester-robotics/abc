import tempfile
import unittest
import signal
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import numpy as np

from local_robot import scripted_collection as sc


class Clock:
    def __init__(self):
        self.t = 0.

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += dt + .01  # deliberately delayed; must never catch up


class Rig:
    def __init__(self, clock, failure=None):
        self.clock = clock
        self.q = np.zeros((2, 7))
        self.q[:, 6] = [.3, .7]
        self.commands = []
        self.failure = failure
        self.held = False

    def health(self):
        if self.failure and len(self.commands) == 35:
            raise self.failure

    def sample(self):
        return self.q.copy()

    def command(self, q):
        self.q = q.copy()
        self.commands.append((self.clock.now(), q.copy()))

    def hold(self):
        self.held = True


class Writer:
    def __init__(self, failure=False):
        self.failure = failure
        self.complete = False
        self.aborted = False

    def check(self):
        if self.failure:
            raise OSError('writer failed')

    def finish(self, usable):
        self.complete = usable
        self.aborted = not usable


class ScriptedTests(unittest.TestCase):
    def test_smooth_legs_have_gentle_departure_arrival_and_bounded_speed(self):
        start = np.zeros((2, 7))
        target = start.copy()
        target[:, :6] = [.1, .5, 1., -1., .2, .3]
        up = sc.smooth_leg(start, target)
        down = sc.smooth_leg(target, start)
        for leg, a, b in ((up, start, target), (down, target, start)):
            np.testing.assert_allclose(leg[0], a)
            np.testing.assert_allclose(leg[-1], b)
            velocity = np.diff(leg[:, :, :6], axis=0) * 30
            self.assertLessEqual(np.abs(velocity).max(), .25 + 1e-9)
            self.assertLess(np.abs(velocity[[0, -1]]).max(), .001)
            acceleration = np.diff(velocity, axis=0) * 30
            self.assertLess(np.abs(acceleration).max(), .5)

    def test_recovery_masks_repeated_interrupts(self):
        previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            sc.mask_recovery_signals()
            for s in previous:
                self.assertEqual(signal.getsignal(s), signal.SIG_IGN)
        finally:
            for s, handler in previous.items():
                signal.signal(s, handler)

    def test_fault_hold_waits_for_inflight_command(self):
        events = []
        class Session:
            def sample(self): return np.zeros((2, 7))
            def hold(self): events.append('hold')
        rig = sc.LiveRig(Session(), None)
        def fail(): raise RuntimeError('camera failed')
        with rig.lock:
            worker = threading.Thread(target=rig._guard, args=(fail,))
            worker.start()
            time.sleep(.05)
            self.assertEqual(events, [])
        worker.join()
        self.assertEqual(events, ['hold'])
        with self.assertRaises(RuntimeError):
            rig.command(np.zeros((2, 7)))

    def test_abort_revokes_previously_promoted_file(self):
        import h5py
        with tempfile.TemporaryDirectory() as tmp:
            writer = sc.EpisodeWriter(Path(tmp)/'episode.h5', {})
            writer.finish(True)
            with h5py.File(writer.path, 'r+') as f:
                f.attrs['usable'] = True
            writer.finish(False)
            with h5py.File(writer.path) as f:
                self.assertFalse(f.attrs['usable'])

    def test_sensor_workers_stop_before_supported_shutdown(self):
        events = []
        rig = SimpleNamespace(health=lambda: None, sample=lambda: np.zeros((2, 7)),
                              close=lambda: events.append('sensors'))
        session = SimpleNamespace(close=lambda: events.append('motors'))
        with patch.object(sc, 'mask_recovery_signals'):
            sc.shutdown_supported(rig, session, np.zeros((2, 6)))
        self.assertEqual(events, ['sensors', 'motors'])

    def test_cycles_endpoints_grippers_speed_and_delays(self):
        clock = Clock()
        rig = Rig(clock)
        start = rig.sample()
        target = start.copy()
        target[:, :6] = [.2, -.1, .15, -.2, 0, .1]
        writers = []
        def create(i):
            w = Writer()
            writers.append(w)
            return w
        done = sc.collect_cycles(rig, start, target, create, lambda w: None,
                                 episodes=10, clock=clock.now, sleep=clock.sleep)
        self.assertEqual(done, 10)
        self.assertTrue(all(w.complete for w in writers))
        np.testing.assert_allclose(rig.q, start)
        commands = np.array([q for _, q in rig.commands])
        times = np.array([t for t, _ in rig.commands])
        self.assertGreaterEqual(np.diff(times).min(), 1 / 30 - 1e-8)
        self.assertLessEqual((np.abs(np.diff(commands[:, :, :6], axis=0)) /
                             np.diff(times)[:, None, None]).max(), .25 + 1e-8)
        np.testing.assert_allclose(commands[:, :, 6], np.tile([.3, .7], (len(commands), 1)))
        self.assertTrue(any(np.allclose(q, target) for q in commands))

    def test_interrupt_and_camera_or_feedback_failure_abort_and_hold(self):
        for failure in (KeyboardInterrupt(), RuntimeError('camera lost'), RuntimeError('stale feedback')):
            clock = Clock()
            rig = Rig(clock, failure)
            writer = Writer()
            with self.assertRaises(type(failure)):
                sc.collect_cycles(rig, rig.q.copy(), rig.q.copy() + np.array([.2]*6+[0]),
                                  lambda i: writer, lambda w: None, clock=clock.now, sleep=clock.sleep)
            self.assertTrue(rig.held)
            self.assertTrue(writer.aborted)
            self.assertFalse(writer.complete)

    def test_writer_failure_stops_before_motion(self):
        clock = Clock()
        rig = Rig(clock)
        writer = Writer(True)
        with self.assertRaises(OSError):
            sc.collect_cycles(rig, rig.q, rig.q, lambda i: writer, lambda w: None,
                              clock=clock.now, sleep=clock.sleep)
        self.assertEqual(rig.commands, [])
        self.assertTrue(rig.held)
        self.assertTrue(writer.aborted)

    def test_measured_return_error_rejects_episode_and_holds(self):
        clock = Clock()
        class DriftingRig(Rig):
            def sample(self):
                q = super().sample()
                # Two 2-second legs and three 1-second holds: 210 commands.
                if len(self.commands) >= 210:
                    q[:, 0] += .04
                return q
        rig = DriftingRig(clock)
        writer = Writer()
        target = rig.q.copy()
        target[:, 0] = .1
        with self.assertRaisesRegex(RuntimeError, 'return endpoint'):
            sc.collect_cycles(rig, rig.q, target, lambda i: writer, lambda w: None,
                              clock=clock.now, sleep=clock.sleep)
        self.assertFalse(writer.complete)
        self.assertTrue(rig.held)

    def test_real_writer_failure_propagates_and_remains_unusable(self):
        with tempfile.TemporaryDirectory() as temp:
            writer = sc.EpisodeWriter(Path(temp)/'episode.h5', {})
            writer.put(('invalid', 'left', np.zeros(1), 1))
            with self.assertRaises(RuntimeError):
                writer.finish(True)
            import h5py
            with h5py.File(writer.path) as f:
                self.assertFalse(f.attrs['usable'])


if __name__ == '__main__':
    unittest.main()
