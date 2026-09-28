"""Serve ABC-DiT or ABC-VLA on this station's 8 GB GPU, with live text prompts.

The policy type is sniffed from the checkpoint like deploy/serve_policy.py
(--policy-type overrides it). DiT runs in bf16 as in serve_bf16.py. The VLA
keeps as many Gemma layers on the GPU as --offload.gpu-budget-gb allows and
streams the rest from pinned host memory (see vla_offload.py).

Run in its own terminal, then start deploy/deploy_policy.py with
--remote-host=localhost. Type a prompt here and press Enter to change the
task; it applies from the next action chunk.

  uv run python -m local_robot.serve_local \
      --policy.checkpoint-path=cache/bottles_75k.pt --policy.diffusion-steps=5
  uv run python -m local_robot.serve_local \
      --policy.checkpoint-path=cache/vla_abc130k_200000_v2.pt --policy.diffusion-steps=5
"""

import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import tyro

from abc_minimal.policy import shared_inference_fields
from deploy.policy.selector import sniff_policy_kind
from deploy.serve_policy_config import Args
from deploy.websocket_server import WebsocketPolicyServer
from local_robot.serve_bf16 import LivePromptPolicy, read_prompts
from local_robot.vla_offload import (
    GiB,
    OffloadConfig,
    OffloadVLAInferencePolicy,
    OffloadVLAPolicyConfig,
)


@dataclass
class LocalArgs(Args):
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    """VLA only: VRAM budget and host streaming of Gemma layers."""


class LiveVLAPolicy(LivePromptPolicy):
    engine_cls = OffloadVLAInferencePolicy


def create_policy(args: LocalArgs) -> tuple[str, LivePromptPolicy]:
    if not args.policy.checkpoint_path:
        raise ValueError("A checkpoint is required")
    kind = args.policy_type
    if kind == "auto":
        kind = sniff_policy_kind(args.policy.checkpoint_path)
    if kind == "vla":
        config = OffloadVLAPolicyConfig(
            **shared_inference_fields(args.policy),
            model=args.vla_model,
            offload=args.offload,
        )
        return kind, LiveVLAPolicy(config)
    return kind, LivePromptPolicy(args.dit_config())


def warmup(policy: LivePromptPolicy) -> None:
    cfg = policy.config
    h, w = cfg.camera_height, cfg.camera_width
    obs = {
        "state": np.zeros(cfg.model.state_dim, dtype=np.float32),
        "images": {
            cam: np.zeros((3, h, w), dtype=np.uint8) for cam in cfg.model.camera_keys
        },
    }
    prefix_length = cfg.rtc_prefix_length or 4
    prefix = np.zeros((prefix_length, cfg.model.action_dim), dtype=np.float32)
    for _ in range(2):
        policy.infer(dict(obs))
    for _ in range(3):
        policy.infer(dict(obs), action_prefix=prefix, prefix_length=prefix_length)
    torch.cuda.synchronize()
    pinned = getattr(policy._policy, "pinned_bytes", 0)
    print(
        f"[serve_local] warmup: {policy.last_latency_s * 1000:.0f} ms/chunk, "
        f"{torch.cuda.max_memory_allocated() / GiB:.2f} GiB peak GPU "
        f"({torch.cuda.memory_reserved() / GiB:.2f} GiB reserved), "
        f"{pinned / GiB:.2f} GiB pinned host",
        flush=True,
    )


def main(args: LocalArgs) -> None:
    start = time.monotonic()
    kind, policy = create_policy(args)
    warmup(policy)
    print(
        f"[serve_local] {kind}, steps={args.policy.diffusion_steps}, "
        f"rtc prefix={args.policy.rtc_prefix_length}, "
        f"fast_inference={args.policy.fast_inference}, "
        f"startup {time.monotonic() - start:.0f} s"
    )
    print(f"[serve_local] prompt: {policy.live_prompt!r} (type a new one + Enter)")
    print(f"[serve_local] serving on 0.0.0.0:{args.port}", flush=True)
    threading.Thread(target=read_prompts, args=(policy,), daemon=True).start()
    WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=args.port).serve_forever()


if __name__ == "__main__":
    main(tyro.cli(LocalArgs))
