"""CPU loader comparison using the same episode/frame samples and resolution."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from abc_minimal.config import DiTConfig
from abc_minimal.dataloader import EpisodeDataset
from abc_minimal.preprocess import load_norm_stats
from local_robot.lance_dataset import LanceEpisodeDataset


def summarize(durations, *, cpu_seconds):
    elapsed = sum(durations)
    return {'samples': len(durations), 'wall_seconds': elapsed, 'cpu_seconds': cpu_seconds,
            'samples_per_second': len(durations)/elapsed,
            'median_ms': float(np.median(durations)*1000),
            'p95_ms': float(np.percentile(durations, 95)*1000),
            'cpu_percent_one_core': 100*cpu_seconds/elapsed}


def measure_loaders(backends, indices, rounds, *, clock=time.perf_counter, cpu_clock=time.process_time):
    # Warm every sampled index in both backends, not just the parity-check subset.
    for ds in backends.values():
        for index in indices:
            _ = ds[index]
    durations = {name: [] for name in backends}
    cpu = {name: 0. for name in backends}
    for round_index in range(rounds):
        order = list(backends.items())
        if round_index % 2:
            order.reverse()
        for name, ds in order:
            cpu_start = cpu_clock()
            for index in indices:
                begin = clock()
                _ = ds[index]
                durations[name].append(clock()-begin)
            cpu[name] += cpu_clock()-cpu_start
    return durations, cpu


def benchmark(mp4_root, lance_root, output, *, samples=128, rounds=3):
    if samples < 1 or rounds < 1:
        raise ValueError('samples and rounds must be positive')
    torch.set_num_threads(4)
    mp4_root, lance_root = Path(mp4_root).resolve(), Path(lance_root).resolve()
    model = DiTConfig(hidden_size=256, depth=4, num_heads=8)
    def dataset(cls, root):
        return cls(root/'train', load_norm_stats(root/'norm_stats.json'), train=False,
                   default_task_name='', mask_state_ratio=0, model_config=model)
    old, new = dataset(EpisodeDataset, mp4_root), dataset(LanceEpisodeDataset, lance_root)
    if [(e[0].name, e[1]) for e in old.episodes] != [(e[0].name, e[1]) for e in new.episodes]:
        raise ValueError('benchmark exports must contain identical episodes and frame counts')
    first_raw = new.read_raw(0)
    video = cv2.VideoCapture(str(old.episodes[0][0]/'combined_camera-images-rgb.mp4'))
    try:
        size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))/3)
    finally:
        video.release()
    if any((v.shape[1], v.shape[0]) != size for v in first_raw['images'].values()):
        raise ValueError('benchmark source image resolutions differ')
    rng = np.random.default_rng(123)
    indices = rng.integers(0, len(old), samples).tolist()
    image_errors = []
    for index in indices[:16]:
        a, b = old[index], new[index]
        torch.testing.assert_close(a['state'], b['state'], rtol=0, atol=1e-5)
        torch.testing.assert_close(a['actions'], b['actions'], rtol=0, atol=1e-5)
        if a['prompt'] != b['prompt']:
            raise ValueError('benchmark prompt mismatch')
        image_errors.append(float(torch.stack([(a['images'][c]-b['images'][c]).abs().mean() for c in model.camera_keys]).mean()))
    durations, cpu = measure_loaders({'mp4': old, 'lance': new}, indices, rounds)
    mp4_files = [p for split in ('train','val') for p in (mp4_root/split).rglob('*') if p.is_file()]
    lance_files = [p for p in lance_root.rglob('*') if p.is_file()]
    report = {
        'fixture': 'same episodes, indices, 30-action chunks, resolution, CPU preprocessing; warm caches',
        'source_image_size': list(size), 'seed': 123, 'torch_threads': 4, 'rounds': rounds,
        'manifest_fingerprint': new.manifest['fingerprint'],
        'scalar_prompt_parity_samples': min(16, samples),
        'mean_normalized_image_absolute_difference': float(np.mean(image_errors)),
        'mp4': summarize(durations['mp4'], cpu_seconds=cpu['mp4']),
        'lance': summarize(durations['lance'], cpu_seconds=cpu['lance']),
        'mp4_training_bytes': sum(p.stat().st_size for p in mp4_files),
        'lance_export_bytes': sum(p.stat().st_size for p in lance_files),
        'limits': 'Synthetic flat-color images only. Excludes raw HDF5, review reel, checkpoints and dependencies. '
                  'JPEG and MP4 encoding differ. No cold-cache, remote storage or multiworker throughput claim.',
    }
    Path(output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mp4-root', type=Path, required=True)
    p.add_argument('--lance-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=128)
    args = p.parse_args()
    benchmark(args.mp4_root, args.lance_root, args.output, samples=args.samples)


if __name__ == '__main__':
    main()
