"""Standalone, rate-limited supported-start -> profile pose -> start collection.

Uses the existing local SDK session (no leader, policy, or blocking reset).
Failed recordings remain unusable; faults hold until an operator types disable.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import signal
import threading
import time

import cv2
import h5py
import numpy as np

from deploy.robot.recorders.base import RecorderBase

SIDES = ('left', 'right')
CAMERAS = ('top', 'left', 'right')
PERIOD = 1 / 30
SPEED = .25
PROMPT = 'move both arms to the initial pose and return to the supported start'


def mask_recovery_signals():
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal.SIG_IGN)


def shutdown_supported(rig, session, starts):
    rig.health()
    if np.max(np.abs(rig.sample()[:, :6] - starts)) > .03:
        raise RuntimeError('supported-start shutdown check failed')
    mask_recovery_signals()
    rig.close()
    session.close()


def smooth_leg(start, target):
    """Straight joint-space path with zero endpoint speed and acceleration.

    Quintic smoothstep has peak derivative 1.875 and peak second derivative
    10/sqrt(3). Size duration against both speed and 0.5 rad/s² acceleration.
    Callers send each point at least PERIOD apart, never catching up.
    """
    start, target = np.asarray(start), np.asarray(target)
    delta = float(np.max(np.abs(target[:, :6] - start[:, :6])))
    duration = max(2., 1.875 * delta / SPEED, np.sqrt((10 / np.sqrt(3)) * delta / .5))
    steps = int(np.ceil(duration / PERIOD))
    t = np.linspace(0, 1, steps + 1)
    alpha = 10*t**3 - 15*t**4 + 6*t**5
    points = start + alpha[:, None, None] * (target - start)
    points[0], points[-1] = start, target
    return points


class EpisodeWriter:
    """Bounded asynchronous writer; errors are observed by the motion controller."""
    def __init__(self, path, metadata):
        self.path = Path(path)
        self.metadata = metadata
        self.queue = queue.Queue(maxsize=512)
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def check(self):
        if self.error is not None:
            raise RuntimeError('HDF5 writer failed') from self.error

    def put(self, item):
        self.check()
        self.queue.put_nowait(item)

    def _run(self):
        try:
            with h5py.File(self.path, 'x') as f:
                f.attrs.update(recording_type='teleop', source='scripted_local_yam',
                               task_name=PROMPT, collection_name='scripted_cycles', usable=False)
                for key, value in self.metadata.items():
                    f.attrs[key] = value
                data, timestamps = f.create_group('data'), f.create_group('timestamps')
                while True:
                    item = self.queue.get()
                    if item is None:
                        break
                    kind, name, value, stamp = item
                    if kind == 'rgb':
                        ok, jpeg = cv2.imencode('.jpg', value[:, :, ::-1],
                                               [cv2.IMWRITE_JPEG_QUALITY, 90])
                        if not ok:
                            raise RuntimeError('JPEG encoding failed')
                        RecorderBase._append_stream(data, timestamps, name, jpeg, stamp,
                                                    h5py.vlen_dtype(np.uint8))
                    elif kind == 'follower' and value.shape == (28,) and np.isfinite(value).all():
                        for key, values in ((f'q_{name}', value[:6]),
                                            (f'q_gripper_{name}', value[6:7]),
                                            (f'q_vel_{name}', value[7:14]),
                                            (f'q_eff_{name}', value[14:21]),
                                            (f'q_des_{name}', value[21:28])):
                            RecorderBase._append_stream(data, timestamps, key, values, stamp,
                                                        'float32', len(values))
                    else:
                        raise ValueError('invalid recorder item')
                f.flush()
        except BaseException as exc:
            self.error = exc

    def finish(self, usable):
        if self.thread.is_alive():
            try:
                self.queue.put(None, timeout=2)
            except queue.Full:
                raise RuntimeError('writer queue stalled')
            self.thread.join(timeout=15)
        if self.thread.is_alive():
            raise RuntimeError('writer did not finish')
        self.check()
        # Validation promotes this later; even a complete cycle starts unusable.
        with h5py.File(self.path, 'r+') as f:
            f.attrs['cycle_complete'] = bool(usable)
            if not usable:
                f.attrs['usable'] = False


def collect_cycles(rig, start, target, create_writer, validate, *, episodes=10,
                   clock=time.monotonic, sleep=time.sleep):
    start, target = np.array(start, copy=True), np.array(target, copy=True)
    if start.shape != (2, 7) or target.shape != (2, 7) or not np.isfinite([start, target]).all():
        raise ValueError('expected finite two-arm positions')
    if not np.array_equal(start[:, 6], target[:, 6]) or episodes < 1:
        raise ValueError('grippers must be constant and episodes positive')
    up, down = smooth_leg(start, target), smooth_leg(target, start)
    last_sent = clock()
    writer = None

    def send(q):
        nonlocal last_sent
        # Each increment is <= speed * period; delays stretch the trajectory.
        sleep(max(0., PERIOD - (clock() - last_sent)))
        rig.health()
        writer.check()
        measured = rig.sample()
        if np.max(np.abs(measured[:, :6] - q[:, :6])) > .15:
            raise RuntimeError('arm tracking error exceeds 0.15 rad')
        rig.command(q)
        last_sent = clock()

    def hold(q):
        for _ in range(30):
            send(q)

    try:
        for episode in range(episodes):
            rig.health()
            writer = create_writer(episode)
            hold(start)
            for q in up[1:]:
                send(q)
            hold(target)
            if np.max(np.abs(rig.sample()[:, :6] - target[:, :6])) > .03:
                raise RuntimeError('top endpoint outside 0.03 rad')
            for q in down[1:]:
                send(q)
            hold(start)
            if np.max(np.abs(rig.sample()[:, :6] - start[:, :6])) > .03:
                raise RuntimeError('return endpoint outside 0.03 rad')
            if hasattr(rig, 'set_writer'):
                rig.set_writer(None)
            writer.finish(True)
            validate(writer)
            print(f'Validated episode {episode + 1}/{episodes}', flush=True)
            writer = None
        return episodes
    except BaseException:
        if hasattr(rig, 'enter_recovery'):
            rig.enter_recovery()
        rig.hold()
        if hasattr(rig, 'set_writer'):
            rig.set_writer(None)
        if writer is not None:
            try:
                writer.finish(False)
            except BaseException:
                pass  # file defaults to unusable, including writer failures
        raise


class LiveRig:
    """Continuous follower telemetry and three independent camera streams."""
    def __init__(self, session, config, cameras=None):
        self.session, self.config = session, config
        self.cameras = cameras
        self.lock = threading.RLock()
        self.writer = None
        self.error = None
        self.seen = {}
        self.stop = threading.Event()
        self.threads = []
        self.last_command = session.sample()

    def _emit(self, item):
        with self.lock:
            self.seen[(item[0], item[1])] = item[3] / 1e9
            if self.writer:
                self.writer.put(item)

    def _guard(self, fn, *args):
        try:
            fn(*args)
        except BaseException as exc:
            with self.lock:
                self.error = exc
                self.session.hold()

    def start(self):
        for fn, args in [(self._telemetry, ())] + [(self._camera, (c,)) for c in CAMERAS]:
            thread = threading.Thread(target=self._guard, args=(fn, *args), daemon=True)
            self.threads.append(thread)
            thread.start()
        deadline = time.monotonic() + 30
        while len(self.seen) != 5:
            if self.error:
                raise RuntimeError('sensor startup failed') from self.error
            if time.monotonic() > deadline:
                raise RuntimeError('sensor startup timeout')
            time.sleep(.05)
        self.health()

    def _telemetry(self):
        from deploy.robot.followers.yam_follower import YAMFollowerNode
        from types import SimpleNamespace
        while not self.stop.is_set():
            self.session.sample()  # checks CAN and SDK worker freshness
            with self.lock:
                for i, side in enumerate(SIDES):
                    obs = YAMFollowerNode.get_robot_obs(SimpleNamespace(robot=self.session.arms[side]))
                    self._emit(('follower', side, np.r_[obs, self.last_command[i]], time.perf_counter_ns()))
            self.stop.wait(PERIOD)

    def _camera(self, name):
        while not self.stop.is_set():
            self._emit(self.cameras.read(name))

    def set_writer(self, writer):
        with self.lock:
            self.writer = writer

    def health(self):
        if self.error:
            raise RuntimeError('sensor/recorder worker failed') from self.error
        now = time.perf_counter()
        with self.lock:
            for kind, names in (('follower', SIDES), ('rgb', CAMERAS)):
                for name in names:
                    if now - self.seen.get((kind, name), -np.inf) > .3:
                        raise RuntimeError(f'stale {kind}: {name}')
        self.session.sample()

    def sample(self):
        return self.session.sample()

    def command(self, q):
        with self.lock:
            if self.error:
                raise RuntimeError('sensor failed before command') from self.error
            self.session.command(q[:, :6], q[:, 6])
            self.last_command = q.copy()

    def hold(self):
        with self.lock:
            self.session.hold()

    def enter_recovery(self):
        mask_recovery_signals()

    def close(self):
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=3)
            if thread.is_alive():
                raise RuntimeError('sensor worker did not stop')


def validate_episode(path, *, promote=False):
    """Decode every camera frame, check telemetry and both measured endpoints."""
    from deploy.recording import io as rio
    with h5py.File(path, 'r') as f:
        if not f.attrs.get('cycle_complete', False):
            raise ValueError('cycle did not complete')
        start, target = f.attrs['start_q'], f.attrs['target_q']
        report = {'path': str(path), 'streams': {}}
        ranges = {}
        for name in (*CAMERAS, *[f'{p}_{s}' for s in SIDES for p in ('q', 'q_gripper', 'q_vel', 'q_eff', 'q_des')]):
            data = f['data'][name]
            ts = f['timestamps'][name][:].astype(np.int64)
            if len(ts) < 30 or len(ts) != len(data) or np.any(np.diff(ts) <= 0):
                raise ValueError(f'{name}: missing or non-increasing timestamps')
            gap = np.diff(ts).max() * 1e-9
            if gap > .3:
                raise ValueError(f'{name}: telemetry/frame gap {gap:.3f}s')
            if name in CAMERAS:
                for i in range(len(data)):
                    image = rio.decode_frame(f, name, i)
                    if image is None or image.size == 0:
                        raise ValueError(f'{name}: corrupt frame {i}')
            elif not np.isfinite(data[:]).all():
                raise ValueError(f'{name}: nonfinite telemetry')
            report['streams'][name] = {'samples': len(ts), 'max_gap_s': float(gap)}
            ranges[name] = (int(ts[0]), int(ts[-1]))
        motion_start = min(ranges[f'q_{side}'][0] for side in SIDES)
        motion_end = max(ranges[f'q_{side}'][1] for side in SIDES)
        for name, (begin, end) in ranges.items():
            if abs(begin - motion_start) > 300_000_000 or abs(end - motion_end) > 300_000_000:
                raise ValueError(f'{name}: stream coverage does not match full motion')
        synced = rio.sync_streams(f, list(ranges))
        for name in ranges:
            ts = f['timestamps'][name][:]
            age_ns = synced.base_ts_ns.astype(np.int64) - ts[synced.indices[name]].astype(np.int64)
            if np.any(age_ns < 0) or np.any(age_ns > 300_000_000):
                raise ValueError(f'{name}: stale synchronized samples')
        report['endpoints'] = {}
        for i, side in enumerate(SIDES):
            measured = f['data'][f'q_{side}'][:]
            action = f['data'][f'q_des_{side}'][:]
            bottom_error = float(np.max(np.abs(measured[-1] - start[i, :6])))
            top_rows = np.max(np.abs(action[:, :6] - target[i, :6]), axis=1) < 1e-5
            top_error = float(np.min(np.max(np.abs(measured[top_rows] - target[i, :6]), axis=1)))
            if bottom_error > .03 or top_error > .03 or np.max(np.abs(measured[0] - start[i, :6])) > .03:
                raise ValueError(f'{side}: endpoint disagreement')
            if np.max(np.abs(action[:, 6] - start[i, 6])) > 1e-6:
                raise ValueError(f'{side}: gripper command changed')
            # Verify telemetry is present throughout each leg, not only holds.
            changes = np.max(np.abs(np.diff(action[:, :6], axis=0)), axis=1)
            if np.count_nonzero(changes > 1e-6) < 4:
                raise ValueError(f'{side}: motion telemetry missing')
            report['endpoints'][side] = {'return_error_rad': bottom_error, 'top_error_rad': top_error}
    Path(path).with_suffix('.validation.json').write_text(json.dumps(report, indent=2) + '\n')
    if promote:
        with h5py.File(path, 'r+') as f:
            f.attrs['usable'] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=10)
    args = parser.parse_args()
    if os.environ.get('ROBOT_PROFILE') != 'local_yam_config':
        raise ValueError('ROBOT_PROFILE must be local_yam_config')
    if not 1 <= args.episodes <= 10:
        raise ValueError('episodes must be between 1 and 10')
    from deploy.robot.config import get_i2rt_config
    from local_robot.lift_both import Session, Kinematics, read_start, await_disable
    from local_robot.scripted_cameras import CameraStreams
    cfg = get_i2rt_config()
    folder = args.session / 'recordings'
    folder.mkdir(parents=True, exist_ok=False)
    channels = {s: cfg.robots[s].follower.channel for s in SIDES}
    starts = read_start(channels)
    targets = np.array([cfg.robots[s].init_q[:6] for s in SIDES])
    kin = Kinematics()
    for start, target in zip(starts, targets):
        if np.any(start < kin.limits[:, 0] - .15) or np.any(start > kin.limits[:, 1] + .15):
            raise ValueError('start outside SDK limits')
        if np.any(target < kin.limits[:, 0]) or np.any(target > kin.limits[:, 1]):
            raise ValueError('target outside joint limits')
        for alpha in np.linspace(0, 1, 150):
            if kin.has_self_collision(start + alpha * (target - start)):
                raise ValueError('self collision on scripted path')
    print('Measured supported start:', starts.tolist(), flush=True)
    print('Profile target:', targets.tolist(), flush=True)
    session, rig = Session(), None
    cameras = CameraStreams(cfg.cameras)
    def interrupt(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupt)
    try:
        # RealSense readiness and native initialization precede motor enable.
        cameras.start()
        session.connect(channels)
        deadline = time.monotonic() + 2
        while True:
            try:
                start = session.sample()
                break
            except RuntimeError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(.01)
        if np.max(np.abs(start[:, :6] - starts)) > .03:
            raise RuntimeError('initialization shifted the supported start')
        start[:, :6] = starts
        target = start.copy()
        target[:, :6] = targets
        rig = LiveRig(session, cfg, cameras)
        rig.start()
        metadata = {'start_q': start, 'target_q': target, 'speed_rad_s': SPEED,
                    'control_hz': 30, 'robot_profile': 'local_yam_config',
                    'trajectory': 'quintic_smoothstep', 'acceleration_rad_s2': .5}
        (args.session / 'poses.json').write_text(json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in metadata.items()}, indent=2))
        def create(i):
            writer = EpisodeWriter(folder / f'episode_{i:03d}.h5', {**metadata, 'episode_index': i})
            rig.set_writer(writer)
            return writer
        collect_cycles(rig, start, target, create,
                       lambda w: validate_episode(w.path, promote=True), episodes=args.episodes)
        shutdown_supported(rig, session, starts)
    except BaseException as exc:
        mask_recovery_signals()
        session.hold()
        print(f'COLLECTION STOPPED: {exc!r}. No further progression.', flush=True)
        if session.chains:
            await_disable(session)
            if rig:
                rig.close()
            session.close()
        raise
    finally:
        if rig:
            rig.close()
        cameras.close()


if __name__ == '__main__':
    main()
