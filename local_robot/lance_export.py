"""Offline scripted HDF5 -> Gontrol-compatible Lance training rows.

Keeps raw files; publishes a complete immutable export directory only on success.
Uses the existing camera timeline and causal synchronization, without retiming.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import cv2
import h5py
import numpy as np

from deploy.recording import io as rio
from deploy.recording.export import RunningStats
from local_robot.scripted_collection import validate_episode

CAMERA_COLUMNS = dict(zip(rio.CAMERA_KEYS, ('image_top', 'image_left_wrist', 'image_right_wrist')))
STREAMS = (*rio.CAMERA_KEYS, *rio.STATE_KEYS, *rio.ACTION_KEYS)


def fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def schema():
    import pyarrow as pa
    return pa.schema([
        ('episode_id', pa.string()), ('task', pa.string()), ('frame_index', pa.int32()),
        ('timestamp_ns', pa.int64()),
        *[(name, pa.large_binary()) for name in CAMERA_COLUMNS.values()],
        ('state', pa.list_(pa.float32(), 14)), ('action', pa.list_(pa.float32(), 14)),
        ('split', pa.string()), ('usable', pa.bool_()),
        ('source_timestamps_ns', pa.struct([(key, pa.int64()) for key in STREAMS])),
    ])


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return value


def episode_table(path, split, image_size=None):
    import pyarrow as pa
    with h5py.File(path, 'r') as f:
        if rio.is_unusable(f) or not f.attrs.get('usable', False):
            raise ValueError(f'{path}: recording is unusable')
    # Existing strict checks include all decoded cameras, measured endpoints,
    # monotonically increasing timestamps and complete fresh stream coverage.
    validation = validate_episode(path)
    with h5py.File(path, 'r') as f:
        synced = rio.sync_streams(f, list(STREAMS))
        states = np.concatenate([synced.gather(f, k) for k in rio.STATE_KEYS], axis=1)
        actions = np.concatenate([synced.gather(f, k) for k in rio.ACTION_KEYS], axis=1)
        n = len(states)
        if n < 30 or states.shape != (n, 14) or actions.shape != (n, 14):
            raise ValueError(f'{path}: expected at least 30 frames with 14-dimensional state/actions')
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f'{path}: nonfinite state/action')
        task = rio.decode_attr(f.attrs.get('task_name', '')).strip()
        if not task:
            raise ValueError(f'{path}: missing task prompt')
        images = {}
        for cam, column in CAMERA_COLUMNS.items():
            frames = []
            # Copy JPEGs byte-for-byte unless an explicit resize is requested.
            for index in synced.indices[cam]:
                jpeg = f['data'][cam][int(index)].tobytes()
                if image_size is not None:
                    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                    if (decoded.shape[1], decoded.shape[0]) != tuple(image_size):
                        decoded = cv2.resize(decoded, tuple(image_size), interpolation=cv2.INTER_AREA)
                        ok, encoded = cv2.imencode('.jpg', decoded, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        if not ok:
                            raise RuntimeError(f'{path}: JPEG encode failed')
                        jpeg = encoded.tobytes()
                frames.append(jpeg)
            images[column] = frames
        source_times = {key: f['timestamps'][key][:][synced.indices[key]].astype(np.int64) for key in STREAMS}
        table = pa.Table.from_pydict({
            'episode_id': [path.stem]*n, 'task': [task]*n,
            'frame_index': np.arange(n, dtype=np.int32),
            'timestamp_ns': synced.base_ts_ns.astype(np.int64), **images,
            'state': states.astype(np.float32).tolist(), 'action': actions.astype(np.float32).tolist(),
            'split': [split]*n, 'usable': [True]*n,
            'source_timestamps_ns': [{key: int(source_times[key][i]) for key in STREAMS} for i in range(n)],
        }, schema=schema())
        metadata = {'episode_id': path.stem, 'task': task, 'split': split, 'num_frames': n,
                    'source_h5': str(path), 'source_attrs': {k: _jsonable(v) for k,v in f.attrs.items()},
                    'validation': validation, 'avg_hz': synced.avg_hz}
    return table, metadata, states, actions


def export_lance(session, output=None, *, seed=123, image_size=None):
    import lance
    import pyarrow
    session = Path(session).resolve()
    output = Path(output).resolve() if output else session/'lance_export'
    if output.exists():
        raise FileExistsError(f'{output} already exists; use a fresh output directory')
    if image_size is not None and (len(image_size) != 2 or min(image_size) < 2):
        raise ValueError('image_size must be (width, height), both >= 2')
    paths = sorted((session/'recordings').glob('*.h5'))
    if len(paths) < 2:
        raise ValueError('need at least two episodes for independent train/validation splits')
    n_val = max(1, round(len(paths)*.2))
    val_ids = set(np.random.default_rng(seed).choice([p.stem for p in paths], n_val, replace=False))
    output.parent.mkdir(parents=True, exist_ok=True)
    # A failure removes only this operation's temporary directory, never inputs.
    with tempfile.TemporaryDirectory(prefix='.lance-export-', dir=output.parent) as temp:
        stage = Path(temp)/'export'
        stage.mkdir()
        state_stats, action_stats = RunningStats(), RunningStats()
        episodes, offset = [], 0
        dataset = None
        for path in paths:
            before_hash = file_sha256(path)
            split = 'val' if path.stem in val_ids else 'train'
            table, meta, states, actions = episode_table(path, split, image_size)
            if file_sha256(path) != before_hash:
                raise RuntimeError(f'{path}: recording changed during export')
            meta.update(row_start=offset, source_sha256=before_hash)
            offset += table.num_rows
            dataset = lance.write_dataset(table, str(stage/'dataset.lance'),
                                          mode='create' if dataset is None else 'append',
                                          data_storage_version='2.2')
            episodes.append(meta)
            ep_dir = stage/split/path.stem
            ep_dir.mkdir(parents=True)
            (ep_dir/'episode_metadata.json').write_text(json.dumps({
                'task_name': meta['task'], 'cameras': list(rio.CAMERA_KEYS), 'usable': True,
                'num_frames': table.num_rows, 'source_h5': str(path)}, indent=2)+'\n')
            if split == 'train':
                state_stats.update(states)
                action_stats.update(actions)
            print(f'[lance] {path.stem}: {table.num_rows} frames ({split})', flush=True)
        stats = {'norm_stats': {'state': state_stats.statistics(), 'actions': action_stats.statistics()}}
        (stage/'norm_stats.json').write_text(json.dumps(stats, indent=2)+'\n')
        manifest = {
            'schema_version': 1, 'lance_version': dataset.version, 'num_rows': offset,
            'seed': seed, 'camera_keys': list(rio.CAMERA_KEYS), 'state_dim': 14, 'action_dim': 14,
            'action_convention': 'absolute_joint_position', 'joint_units': 'radians',
            'gripper_units': 'normalized_aperture_0_closed_1_open',
            'timeline': 'top_camera_timestamps_causal_hold_no_retiming',
            'max_sample_age_ns': 300_000_000,
            'image_size': list(image_size) if image_size else None,
            'packages': {'pylance': lance.__version__, 'pyarrow': pyarrow.__version__},
            'normalization_split': 'train', 'norm_stats_sha256': file_sha256(stage/'norm_stats.json'),
            'episodes': episodes,
        }
        if dataset.count_rows() != offset:
            raise RuntimeError('Lance row count disagrees with manifest')
        manifest['fingerprint'] = fingerprint(manifest)
        (stage/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
        stage.rename(output)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--image-size', type=int, nargs=2, metavar=('WIDTH','HEIGHT'))
    args = p.parse_args()
    export_lance(args.session, args.output, image_size=args.image_size)


if __name__ == '__main__':
    main()
