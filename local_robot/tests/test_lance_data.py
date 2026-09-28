"""Synthetic fixtures only: no robot or camera access."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from local_robot.scripted_export import synthetic_recording
from local_robot.lance_export import export_lance


@pytest.fixture
def recording_root(tmp_path):
    root = tmp_path/'session'
    for i in range(2):
        synthetic_recording(root/'recordings'/f'episode_{i:03d}.h5')
    return root


def test_export_preserves_jpeg_bytes_and_causal_state(recording_root):
    import lance
    out = recording_root/'lance'
    manifest = export_lance(recording_root, out)
    ds = lance.dataset(str(out/'dataset.lance'), version=manifest['lance_version'])
    row = ds.take([40]).to_pylist()[0]
    with h5py.File(recording_root/'recordings/episode_000.h5') as f:
        assert row['image_top'] == f['data/top'][40].tobytes()
        expected = np.r_[f['data/q_left'][40], f['data/q_gripper_left'][40],
                         f['data/q_right'][40], f['data/q_gripper_right'][40]]
        np.testing.assert_array_equal(row['state'], expected)
        assert row['timestamp_ns'] == f['timestamps/top'][40]
        assert row['source_timestamps_ns']['q_left'] <= row['timestamp_ns']
    assert row['episode_id'] == 'episode_000'
    assert row['frame_index'] == 40
    assert ds.count_rows() == 304
    assert {e['split'] for e in manifest['episodes']} == {'train', 'val'}
    assert all(len(e['source_sha256']) == 64 for e in manifest['episodes'])
    assert manifest['action_convention'] == 'absolute_joint_position'


def test_export_rejects_bad_recordings_without_publishing(recording_root):
    path = recording_root/'recordings/episode_001.h5'
    with h5py.File(path, 'r+') as f:
        f['data/q_left'][40, 0] = np.nan
    with pytest.raises(ValueError):
        export_lance(recording_root, recording_root/'lance')
    assert not (recording_root/'lance').exists()


def test_export_rejects_unusable_and_preserves_existing_destination(recording_root):
    with h5py.File(recording_root/'recordings/episode_001.h5', 'r+') as f:
        f.attrs['usable'] = False
    with pytest.raises(ValueError, match='unusable'):
        export_lance(recording_root, recording_root/'lance')
    out = recording_root/'existing'
    out.mkdir()
    (out/'keep').write_text('original')
    with pytest.raises(FileExistsError):
        export_lance(recording_root, out)
    assert (out/'keep').read_text() == 'original'


def test_normalization_uses_only_training_episodes(recording_root):
    # Fixed seed 123 picks episode_000 for validation for this two-episode fixture.
    with h5py.File(recording_root/'recordings/episode_000.h5', 'r+') as f:
        for key in ('q_gripper_left', 'q_gripper_right'):
            f['data'][key][:] = .99
    out = recording_root/'lance'
    manifest = export_lance(recording_root, out)
    assert [e['episode_id'] for e in manifest['episodes'] if e['split']=='val'] == ['episode_000']
    stats = json.loads((out/'norm_stats.json').read_text())['norm_stats']
    np.testing.assert_allclose(np.array(stats['state']['mean'])[[6,13]], [.3,.7], atol=1e-6)
    with pytest.raises(FileExistsError):
        export_lance(recording_root, out)


def make_dataset(root, split='train'):
    from abc_minimal.config import DiTConfig
    from abc_minimal.preprocess import load_norm_stats
    from local_robot.lance_dataset import LanceEpisodeDataset
    return LanceEpisodeDataset(root/split, load_norm_stats(root/'norm_stats.json'),
                               train=False, default_task_name='', mask_state_ratio=0,
                               model_config=DiTConfig(hidden_size=256, depth=4, num_heads=8))


def test_loader_chunk_boundary_images_and_prompt(recording_root):
    import torch
    root = recording_root/'lance'
    export_lance(recording_root, root)
    ds = make_dataset(root)
    assert len(ds) == 123  # 152 frames, 30-step chunks
    sample = ds[122]  # last valid chunk reaches last row of episode_001
    assert sample['actions'].shape == (30, 14)
    assert sample['state'].shape == (14,)
    assert set(sample['images']) == {'top', 'left', 'right'}
    assert all(v.shape == (3,224,224) and torch.isfinite(v).all() for v in sample['images'].values())
    assert sample['prompt'] == 'move both arms to the initial pose and return to the supported start'
    assert torch.isfinite(sample['actions']).all()
    with pytest.raises(IndexError):
        ds[123]
    with pytest.raises(IndexError):
        ds[-1]


def test_loader_pins_snapshot_after_dataset_append(recording_root):
    import lance
    root = recording_root/'lance'
    manifest = export_lance(recording_root, root)
    ds = make_dataset(root)
    current = lance.dataset(str(root/'dataset.lance'))
    lance.write_dataset(current.take([0]), str(root/'dataset.lance'), mode='append')
    assert lance.dataset(str(root/'dataset.lance')).count_rows() == 305
    assert ds.snapshot.version == manifest['lance_version']
    assert ds.snapshot.count_rows() == 304
    assert len(ds) == 123


@pytest.mark.parametrize('problem', ['manifest', 'stats'])
def test_loader_rejects_changed_manifest_or_stats(recording_root, problem):
    root = recording_root/'lance'
    export_lance(recording_root, root)
    if problem == 'stats':
        (root/'norm_stats.json').write_text('{}')
    else:
        data = json.loads((root/'manifest.json').read_text())
        data['episodes'][0]['split'] = 'train'
        (root/'manifest.json').write_text(json.dumps(data))
    from local_robot.lance_dataset import load_manifest
    with pytest.raises(ValueError, match='normalization|fingerprint'):
        load_manifest(root)


def test_loader_survives_spawned_worker(recording_root):
    from torch.utils.data import DataLoader
    root = recording_root/'lance'
    export_lance(recording_root, root)
    ds = make_dataset(root)
    _ = ds[0]  # opens parent handle; the worker must open its own snapshot
    loader = DataLoader(ds, batch_size=1, num_workers=1, multiprocessing_context='spawn', timeout=30)
    batch = next(iter(loader))
    assert batch['actions'].shape == (1,30,14)


def test_chunks_stop_at_each_episode_and_read_only_one_image_row(recording_root):
    import os
    synthetic_recording(recording_root/'recordings/episode_002.h5')
    root = recording_root/'lance'
    export_lance(recording_root, root)
    ds = make_dataset(root)
    assert len(ds) == 246
    snapshot = ds.snapshot
    calls = []
    class Reads:
        def take(self, indices, columns):
            calls.append((list(indices), list(columns)))
            return snapshot.take(indices, columns=columns)
    ds._snapshot, ds._pid = Reads(), os.getpid()
    last = ds.read_raw(122)
    first = ds.read_raw(123)
    assert last['episode_id'] != first['episode_id']
    assert last['frame_index'] == 122
    assert first['frame_index'] == 0
    assert [len(indices) for indices, _ in calls] == [30,1,30,1]
    assert all(not any(c.startswith('image_') for c in cols) for _, cols in calls[::2])


@pytest.mark.parametrize('problem', ['missing', 'stale', 'broken_jpeg'])
def test_bad_sensor_data_cannot_be_exported(recording_root, problem):
    path = recording_root/'recordings/episode_001.h5'
    with h5py.File(path, 'r+') as f:
        if problem == 'missing':
            del f['data/q_des_right']
        elif problem == 'stale':
            f['timestamps/right'][:] += 1_000_000_000
        else:
            f['data/top'][10] = np.array([1,2,3], dtype=np.uint8)
    with pytest.raises((ValueError, KeyError)):
        export_lance(recording_root, recording_root/'lance')
    assert not (recording_root/'lance').exists()
