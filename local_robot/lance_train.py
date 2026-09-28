"""Train the small local DiT against a pinned Lance export; no hardware access."""
import argparse
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch


def bind_run(run_dir, manifest, *, resume):
    run_dir = Path(run_dir)
    binding = {'backend': 'lance', 'fingerprint': manifest['fingerprint'],
               'lance_version': manifest['lance_version']}
    path = run_dir/'dataset_binding.json'
    if resume:
        if not path.exists() or json.loads(path.read_text()) != binding:
            raise ValueError('resume references a different or unbound dataset')
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        path.write_text(json.dumps(binding, indent=2)+'\n')
    return binding


@contextmanager
def training_context():
    from abc_minimal import dataloader
    from local_robot.lance_dataset import LanceEpisodeDataset
    original_loader = dataloader.DataLoader
    def loader(*args, **kwargs):
        if kwargs.get('num_workers', 0) > 0:
            kwargs['multiprocessing_context'] = 'spawn'
        return original_loader(*args, **kwargs)
    with patch.object(dataloader, 'EpisodeDataset', LanceEpisodeDataset), \
         patch.object(dataloader, 'DataLoader', loader):
        yield


def run(dataset_root, run_dir, *, resume=False, steps=200):
    import torch
    from abc_minimal import train_loop
    from local_robot import scripted_train
    from local_robot.lance_dataset import load_manifest
    dataset_root, run_dir = Path(dataset_root).resolve(), Path(run_dir).resolve()
    manifest = load_manifest(dataset_root)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the local training test')
    binding = bind_run(run_dir, manifest, resume=resume)
    if resume:
        ckpt = torch.load(run_dir/'checkpoints/200.pt', map_location='cpu', mmap=True, weights_only=False)
        if ckpt.get('train_config', {}).get('dataset_binding') != binding:
            raise ValueError('checkpoint references a different dataset snapshot')
        del ckpt
    build_config = scripted_train.build_config
    save_checkpoint = train_loop.save_checkpoint
    def config(*args, **kwargs):
        return replace(build_config(*args, **kwargs), cache_root=str(dataset_root))
    def save(*args, **kwargs):
        kwargs['train_config'] = {**kwargs['train_config'], 'dataset_binding': binding}
        return save_checkpoint(*args, **kwargs)
    with training_context(), patch.object(scripted_train, 'build_config', config), \
         patch.object(train_loop, 'save_checkpoint', save):
        scripted_train.run(run_dir, resume=resume, steps=steps)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, type=Path)
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    run(args.dataset, args.run_dir, resume=args.resume, steps=args.steps)


if __name__ == '__main__':
    main()
