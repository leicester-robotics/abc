"""Latency, memory and accuracy check for serve_local.py policies (no robot).

Feeds real frames from a cached episode through the same policy the server
builds (RTC prefix included), prints per-chunk latency, peak VRAM, pinned and
anonymous host memory and, for the VLA, a prefix/Gemma/head breakdown plus the
compute-only time with host copies skipped. --save/--compare store and diff
the actions so int8 runs can be checked against a bf16 run.

  uv run python -m local_robot.bench_serve_local \
      --policy.checkpoint-path=cache/vla_abc130k_200000_v2.pt --policy.diffusion-steps=5 \
      --save=/tmp/vla_bf16.npz
  uv run python -m local_robot.bench_serve_local ... --offload.quantize=all \
      --compare=/tmp/vla_bf16.npz
"""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from abc_minimal.dataloader import decode_frame
from abc_minimal.episode_io import discover_episodes, load_episode
from local_robot.serve_local import LocalArgs, create_policy
from local_robot.vla_offload import GiB


@dataclass
class BenchArgs(LocalArgs):
    episodes: str = "cache/val_real"
    frames: int = 6
    save: str = ""
    compare: str = ""
    alt_prompt: str = "pick up the cube"


def real_observations(args: BenchArgs, camera_keys, count):
    episode = discover_episodes(Path(args.episodes), limit=1)[0]
    metadata, states, _ = load_episode(episode)
    cameras = metadata.get("cameras", list(camera_keys))
    for idx in np.linspace(0, len(states) - 1, count).astype(int):
        images = decode_frame(episode, int(idx), len(states), cameras, camera_keys)
        yield {
            "state": states[idx].astype(np.float32),
            "images": {k: (v * 255).round().to(torch.uint8).numpy() for k, v in images.items()},
        }


def rss_anon_gb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("RssAnon:"):
            return int(line.split()[1]) / 2**20
    return float("nan")


class Timer:
    """CUDA-event time spent inside wrapped callables, per call."""

    def __init__(self):
        self.pending: dict[str, list] = {}

    def wrap(self, name, fn):
        def timed(*a, **kw):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn(*a, **kw)
            end.record()
            self.pending.setdefault(name, []).append((start, end))
            return out

        return timed

    def collect(self):
        torch.cuda.synchronize()
        out = {k: sum(s.elapsed_time(e) for s, e in v) for k, v in self.pending.items()}
        self.pending.clear()
        return out


def run(policy, observations, prefix_length, action_dim, timer=None):
    prefix = np.zeros((prefix_length, action_dim), dtype=np.float32)
    actions, latencies, parts = [], [], []
    for obs in observations:
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = policy.infer(dict(obs), action_prefix=prefix, prefix_length=prefix_length)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)
        actions.append(out["actions"])
        prefix = out["actions"][-prefix_length:]
        if timer is not None:
            parts.append(timer.collect())
    return np.stack(actions), latencies, parts


def main(args: BenchArgs) -> None:
    start = time.monotonic()
    kind, policy = create_policy(args)
    cfg = policy.config
    engine = policy._policy
    prefix_length = cfg.rtc_prefix_length or 4
    observations = list(real_observations(args, cfg.model.camera_keys, args.frames))
    run(policy, observations[:2], prefix_length, cfg.model.action_dim)
    print(
        f"[bench] startup {time.monotonic() - start:.0f} s, peak GPU "
        f"{torch.cuda.max_memory_allocated() / GiB:.2f} GiB, reserved "
        f"{torch.cuda.memory_reserved() / GiB:.2f} GiB"
    )
    torch.cuda.reset_peak_memory_stats()

    timer = None
    if kind == "vla":
        timer = Timer()
        vla = engine.model.vla
        vla.prepare = timer.wrap("prefix", vla.prepare)
        gemma = vla.gemma_model.model
        gemma.forward = timer.wrap("gemma", gemma.forward)
        head = engine.model.diffusion_head
        head.sample = timer.wrap("head", head.sample)
    action_dim = cfg.model.action_dim
    actions, latencies, parts = run(policy, observations, prefix_length, action_dim, timer)
    peak = torch.cuda.max_memory_allocated() / GiB
    policy.live_prompt = args.alt_prompt
    alt, _, _ = run(policy, observations[:1], prefix_length, action_dim, timer)

    ms = np.array(latencies[1:])
    print(f"[bench] {kind}: {ms.mean():.0f} ms/chunk (min {ms.min():.0f}, max {ms.max():.0f})")
    if parts:
        mean = {k: np.mean([p[k] for p in parts[1:]]) for k in parts[0]}
        print("[bench] breakdown ms: " + ", ".join(f"{k} {v:.0f}" for k, v in mean.items()))
    print(
        f"[bench] peak GPU {peak:.2f} GiB, pinned host "
        f"{getattr(engine, 'pinned_bytes', 0) / GiB:.2f} GiB, RssAnon {rss_anon_gb():.2f} GiB"
    )
    print(
        f"[bench] finite={np.isfinite(actions).all()} shape={actions.shape[1:]} "
        f"prompt change max|d|={np.abs(alt[0] - actions[0]).max():.4f}"
    )

    streamer = getattr(engine, "streamer", None)
    if streamer is not None:
        copy_ms = []
        for _ in range(5):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(streamer.stream)
            streamer.gpu[0].copy_(streamer.host[0], non_blocking=True)
            end.record(streamer.stream)
            end.synchronize()
            copy_ms.append(start.elapsed_time(end))
        per_layer = float(np.median(copy_ms))
        streamer._fetch = lambda k: streamer.ready[k % len(streamer.gpu)].record(streamer.stream)
        policy.live_prompt = cfg.prompt
        _, compute, _ = run(policy, observations, prefix_length, action_dim, timer)
        print(
            f"[bench] H2D {per_layer:.1f} ms/layer x {streamer.count} streamed "
            f"({streamer.nbytes * streamer.count / GiB:.2f} GiB/chunk, "
            f"{streamer.nbytes / per_layer / 1e6:.1f} GB/s); compute-only "
            f"{np.mean(compute[1:]):.0f} ms/chunk"
        )

    if args.save:
        np.savez(args.save, actions=actions, alt=alt)
    if args.compare:
        ref = np.load(args.compare)
        print(
            f"[bench] vs {args.compare}: max|d| {np.abs(actions - ref['actions']).max():.4f}, "
            f"mean|d| {np.abs(actions - ref['actions']).mean():.4f}, "
            f"action std {ref['actions'].std():.3f}"
        )


if __name__ == "__main__":
    main(tyro.cli(BenchArgs))
