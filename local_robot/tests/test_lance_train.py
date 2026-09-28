import json
from unittest.mock import patch

import pytest

from local_robot.lance_train import training_context, bind_run


def test_training_context_selects_lance_and_restores_originals():
    from abc_minimal import dataloader
    from local_robot.lance_dataset import LanceEpisodeDataset
    original = dataloader.EpisodeDataset
    original_loader = dataloader.DataLoader
    with training_context():
        assert dataloader.EpisodeDataset is LanceEpisodeDataset
        # Spawn is required: Rust's thread pool must not be inherited via fork.
        loader = dataloader.DataLoader([1,2], num_workers=1)
        assert loader.multiprocessing_context.get_start_method() == 'spawn'
    assert dataloader.EpisodeDataset is original
    assert dataloader.DataLoader is original_loader


def test_resume_rejects_different_dataset(tmp_path):
    binding = {'fingerprint': 'a'*64, 'lance_version': 2}
    bind_run(tmp_path/'run', binding, resume=False)
    bind_run(tmp_path/'run', binding, resume=True)
    with pytest.raises(ValueError, match='different'):
        bind_run(tmp_path/'run', {**binding,'fingerprint':'b'*64}, resume=True)
    with pytest.raises(FileExistsError):
        bind_run(tmp_path/'run', binding, resume=False)
