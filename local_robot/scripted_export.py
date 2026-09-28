"""Validate scripted HDF5 episodes and reuse the upstream dataset exporter."""
import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np

from local_robot.scripted_collection import EpisodeWriter, validate_episode, CAMERAS, SIDES


def synthetic_recording(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    start = np.zeros((2, 7))
    start[:, 6] = [.3, .7]
    target = start.copy()
    target[:, :6] = [.2, .1, .15, -.2, 0, .1]
    writer = EpisodeWriter(path, {'start_q': start, 'target_q': target, 'synthetic': True})
    poses = ([start] * 30 + [start + (target-start)*a for a in np.linspace(0, 1, 31)] +
             [target] * 30 + [target + (start-target)*a for a in np.linspace(0, 1, 31)] + [start] * 30)
    for i, q in enumerate(poses):
        for cam_index, cam in enumerate(CAMERAS):
            image = np.zeros((48, 64, 3), dtype=np.uint8)
            image[:, :, cam_index] = 100 + i % 100
            # Sensor streams precede the base top timeline by <= 2 ns.
            stamp = 1_000_000_000 + round(i * 1e9 / 30)
            while writer.queue.qsize() > 480:
                writer.check()
                time.sleep(.01)
            writer.put(('rgb', cam, image, stamp + 2))
        for j, side in enumerate(SIDES):
            writer.put(('follower', side, np.r_[q[j], np.zeros(14), q[j]], stamp))
    writer.finish(True)
    validate_episode(path, promote=True)


def export_session(session, *, expected_episodes=10):
    from deploy.recording.export import ExportConfig, export_recordings, RunningStats
    session = Path(session).resolve()
    paths = sorted((session/'recordings').glob('*.h5'))
    if len(paths) != expected_episodes:
        raise ValueError(f'expected {expected_episodes} episodes, found {len(paths)}')
    for path in paths:
        from deploy.recording.io import is_unusable
        with h5py.File(path, 'r') as f:
            if is_unusable(f) or not f.attrs.get('usable', False):
                raise ValueError(f'{path}: recording is unusable')
        validate_episode(path)
    summary = export_recordings(ExportConfig(
        input_pattern=str(session/'recordings/*.h5'), output_dir=str(session/'export'),
        mode='whole_episodes', output_fps=30, val_mode='split', val_ratio=.2, seed=123,
        compute_norm_stats=True, concat_review_mp4=True))
    n_val = max(1, round(expected_episodes * .2))
    train_count = len(list((session/'export/train').glob('*/episode_metadata.json')))
    val_count = len(list((session/'export/val').glob('*/episode_metadata.json')))
    if (len(summary['exported_episodes']) != expected_episodes or summary['skipped_files'] or
            summary['skipped_segments'] or train_count != expected_episodes - n_val or val_count != n_val):
        raise RuntimeError('export did not produce the complete expected train/validation split')
    # Upstream computes stats before the split. Replace with training-only stats
    # to keep the two held-out episodes out of normalization.
    state, action = RunningStats(), RunningStats()
    for path in sorted((session/'export/train').glob('*/states_actions.bin')):
        rows = np.fromfile(path, dtype=np.float64).reshape(-1, 28)
        state.update(rows[:, :14])
        action.update(rows[:, 14:])
    (session/'export/norm_stats.json').write_text(json.dumps({'norm_stats': {
        'state': state.statistics(), 'actions': action.statistics()}}, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, type=Path)
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()
    if args.synthetic:
        for i in range(10):
            synthetic_recording(args.session/'recordings'/f'episode_{i:03d}.h5')
    export_session(args.session)


if __name__ == '__main__':
    main()
