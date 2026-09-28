"""Lance adapter for the unchanged ABC sampler, preprocessing and trainer."""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from abc_minimal.config import PromptConfig
from abc_minimal.dataloader import EpisodeDataset, _prompt_timeline
from abc_minimal.preprocess import augment_and_normalize, normalize, preset_for_backbone
from local_robot.lance_export import CAMERA_COLUMNS, file_sha256, fingerprint


def load_manifest(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    payload = {k: v for k, v in manifest.items() if k != 'fingerprint'}
    if manifest.get('fingerprint') != fingerprint(payload):
        raise ValueError('Lance manifest fingerprint mismatch')
    if manifest.get('schema_version') != 1 or manifest.get('action_convention') != 'absolute_joint_position':
        raise ValueError('unsupported Lance dataset schema/action convention')
    if manifest.get('camera_keys') != ['top', 'left', 'right'] or (manifest['state_dim'], manifest['action_dim']) != (14, 14):
        raise ValueError('unexpected Lance cameras or dimensions')
    if file_sha256(root/'norm_stats.json') != manifest['norm_stats_sha256']:
        raise ValueError('Lance normalization statistics changed')
    offset, ids = 0, set()
    for ep in manifest['episodes']:
        if (ep['row_start'] != offset or ep['num_frames'] < 30 or ep['episode_id'] in ids or
                Path(ep['episode_id']).name != ep['episode_id'] or ep['split'] not in ('train', 'val')):
            raise ValueError('invalid episode manifest')
        ids.add(ep['episode_id'])
        offset += ep['num_frames']
    if offset != manifest['num_rows']:
        raise ValueError('manifest row count mismatch')
    return manifest


class LanceEpisodeDataset(EpisodeDataset):
    """Read images at t and only scalar columns for the future action chunk.

    Public constructor matches EpisodeDataset so the existing loader builders
    and deterministic sampler remain unchanged. Handles are local to each worker.
    """
    def __init__(self, data_dir, norm_stats, train, default_task_name, mask_state_ratio,
                 model_config, prompt_config=None, operator_label_maps=None, norm_preset='auto'):
        data_dir = Path(data_dir).resolve()
        self.root = data_dir.parent
        self.manifest = load_manifest(self.root)
        if data_dir.name not in ('train', 'val'):
            raise ValueError('Lance data_dir must select train or val')
        self.model_config = model_config
        self.camera_keys = tuple(model_config.camera_keys)
        if self.camera_keys != ('top', 'left', 'right') or (model_config.state_dim, model_config.action_dim) != (14,14):
            raise ValueError('Lance adapter requires three standard cameras and 14-dimensional state/actions')
        self.norm_stats = norm_stats
        self.train = train
        self.mask_state_ratio = mask_state_ratio
        self.image_size = getattr(getattr(model_config, 'backbone', None), 'image_size', 224)
        self.norm_preset = preset_for_backbone(model_config.vision_backbone) if norm_preset == 'auto' else norm_preset
        self.prompt_config = prompt_config or PromptConfig()
        self.operator_label_maps = operator_label_maps or {}
        self._subtask_cache, self._operator_cache = {}, {}
        self._snapshot, self._pid = None, None
        self.records, self.episodes = [], []
        for ep in self.manifest['episodes']:
            if ep['split'] != data_dir.name:
                continue
            usable = ep['num_frames'] - model_config.chunk_length + 1
            if usable <= 0:
                continue
            ep_dir = data_dir/ep['episode_id']
            # The manifest pins the task text; mutable sidecars cannot replace it.
            task = ep['task'] or default_task_name
            timeline = _prompt_timeline(ep_dir, {'task_name': task}, task)
            self.records.append(ep)
            self.episodes.append((ep_dir, ep['num_frames'], usable, self.camera_keys, task, timeline))
        if not self.episodes:
            raise ValueError(f'no trainable Lance episodes in {data_dir}')
        self.cum = np.cumsum([e[2] for e in self.episodes])

    @property
    def snapshot(self):
        if self._snapshot is None or self._pid != os.getpid():
            import lance
            self._snapshot = lance.dataset(str(self.root/'dataset.lance'), version=self.manifest['lance_version'])
            if self._snapshot.count_rows() != self.manifest['num_rows']:
                raise ValueError('pinned Lance snapshot does not match manifest')
            self._pid = os.getpid()
        return self._snapshot

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_snapshot'], state['_pid'] = None, None
        return state

    def locate(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        ep_index = int(np.searchsorted(self.cum, index, side='right'))
        frame = int(index - (self.cum[ep_index-1] if ep_index else 0))
        return ep_index, frame

    def read_raw(self, index):
        ep_index, frame = self.locate(index)
        ep = self.records[ep_index]
        row = ep['row_start'] + frame
        length = self.model_config.chunk_length
        scalars = self.snapshot.take(list(range(row, row+length)),
                                     columns=['episode_id', 'frame_index', 'state', 'action', 'timestamp_ns'])
        if (set(scalars['episode_id'].to_pylist()) != {ep['episode_id']} or
                scalars['frame_index'].to_pylist() != list(range(frame, frame+length))):
            raise ValueError('Lance chunk crosses an episode boundary or contains unordered frames')
        states = np.asarray(scalars['state'].to_pylist(), dtype=np.float32)
        actions = np.asarray(scalars['action'].to_pylist(), dtype=np.float32)
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError('Lance chunk contains nonfinite values')
        encoded = self.snapshot.take([row], columns=list(CAMERA_COLUMNS.values())).to_pylist()[0]
        images = {}
        for cam, column in CAMERA_COLUMNS.items():
            bgr = cv2.imdecode(np.frombuffer(encoded[column], np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f'corrupt JPEG in {ep["episode_id"]}/{frame}/{cam}')
            images[cam] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return {'state': states[0], 'actions': actions, 'images': images,
                'timestamp_ns': scalars['timestamp_ns'][0].as_py(),
                'episode_id': ep['episode_id'], 'frame_index': frame}

    def __getitem__(self, index):
        raw = self.read_raw(index)
        ep_index, frame = self.locate(index)
        ep_dir, _, _, _, task, timeline = self.episodes[ep_index]
        state = normalize(raw['state'], self.norm_stats['state']).astype(np.float32)
        actions = normalize(raw['actions'], self.norm_stats['actions']).astype(np.float32)
        masked = bool(self.train and torch.rand(1).item() < self.mask_state_ratio)
        if masked:
            state = np.zeros_like(state)
        images = augment_and_normalize(
            {cam: torch.from_numpy(image).permute(2,0,1).float()/255 for cam, image in raw['images'].items()},
            self.train, norm_preset=self.norm_preset, image_size=self.image_size)
        return {'state': torch.from_numpy(state), 'actions': torch.from_numpy(actions),
                'images': images, 'state_is_masked': masked,
                'prompt': self.build_prompt(ep_dir, frame, task, timeline)}
