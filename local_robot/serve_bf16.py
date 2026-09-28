"""Serve ABC-DiT in bf16 with live text prompts, for this station's 8 GB GPU.

The upstream server builds the 2.0B-param DiT on the GPU in fp32 (~8.1 GB),
which does not fit. This loads the checkpoint on the CPU, casts to bf16
(~4 GB), then moves it to the GPU. Upstream code is left untouched.

Run in its own terminal, then start deploy/deploy_policy.py with
--remote-host=localhost. Type a prompt here and press Enter to change the
task; it applies from the next action chunk. Recordings keep the --prompt
given to deploy_policy.py.

  uv run python -m local_robot.serve_bf16 \
      --policy.checkpoint-path=cache/bottles_75k.pt --policy.diffusion-steps=5
"""

import threading
import time

import numpy as np
import torch
import tyro

from abc_minimal import policy as abc_policy
from abc_minimal.dit import DiTPolicy, load_pretrained
from deploy.policy.base import DeployPolicy
from deploy.serve_policy_config import Args
from deploy.websocket_server import WebsocketPolicyServer


class BF16DiTInferencePolicy(abc_policy.DiTInferencePolicy):
    def __init__(self, checkpoint, config, device, model_config=None):
        self.config = config
        self.model_config = model_config if model_config is not None else config.model
        self.device = torch.device(device)
        self.diffusion_steps = config.diffusion_steps
        self.chunk_length = self.model_config.chunk_length
        self.action_dim = self.model_config.action_dim
        self.model = DiTPolicy(self.model_config)
        ckpt = load_pretrained(self.model, checkpoint)
        self.model.to(dtype=torch.bfloat16).to(self.device)
        self.model.img_backbone.set_bfloat16(True)
        self.model.eval()
        self.norm_preset = abc_policy.preset_for_backbone(self.model_config.vision_backbone)
        self.norm_stats = abc_policy.resolve_norm_stats(ckpt, config.norm_stats_path)
        self.trained_max_prefix = abc_policy.resolve_trained_max_prefix(ckpt)
        del ckpt
        self.embedder = abc_policy.CLIPTextEmbedder(config.clip, device=self.device)
        self._prompt = config.prompt
        self.task_vec = self.embedder.encode([self._prompt]).to(
            device=self.device, dtype=torch.bfloat16
        )
        self._fast_inference_enabled = False


class LivePromptPolicy(DeployPolicy):
    engine_cls = BF16DiTInferencePolicy

    def __init__(self, config):
        super().__init__(config)
        self.live_prompt = config.prompt

    def infer(self, obs, **kwargs):
        obs["prompt"] = self.live_prompt
        start = time.perf_counter()
        out = super().infer(obs, **kwargs)
        self.last_latency_s = time.perf_counter() - start
        return out


def read_prompts(policy: LivePromptPolicy) -> None:
    while True:
        try:
            line = input().strip()
        except EOFError:
            return
        if line:
            policy.live_prompt = line
            print(f"[prompt] now: {line!r}", flush=True)


def warmup(policy: LivePromptPolicy, args: Args) -> None:
    cfg = policy.config
    h, w = cfg.camera_height, cfg.camera_width
    obs = {
        "state": np.zeros(cfg.model.state_dim, dtype=np.float32),
        "images": {
            cam: np.zeros((3, h, w), dtype=np.uint8) for cam in cfg.model.camera_keys
        },
    }
    for _ in range(3):
        policy.infer(dict(obs))
    torch.cuda.synchronize()
    print(
        f"[serve_bf16] warmup: {policy.last_latency_s * 1000:.0f} ms/chunk, "
        f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB peak GPU",
        flush=True,
    )


def main(args: Args) -> None:
    policy = LivePromptPolicy(args.dit_config())
    warmup(policy, args)
    print(f"[serve_bf16] prompt: {policy.live_prompt!r} (type a new one + Enter)")
    print(f"[serve_bf16] serving on 0.0.0.0:{args.port}", flush=True)
    threading.Thread(target=read_prompts, args=(policy,), daemon=True).start()
    WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=args.port).serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
