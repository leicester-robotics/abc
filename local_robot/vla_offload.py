"""ABC-VLA inference for a GPU smaller than the model.

The Gemma prefix runs once per observation and the action head runs once per
diffusion step, so SigLIP, the projections and the head stay on the GPU. As
many Gemma decoder layers as fit a VRAM budget stay resident; the rest live in
pinned host memory and rotate through a few preallocated GPU slots, copied on
a side stream ahead of compute. Optionally Gemma layers are stored as int8
(weight-only, per-row scales), halving their VRAM and PCIe cost.

The 1.3 GB token table stays on the CPU: token ids depend only on the prompt,
so the text embeddings, ids and attention masks are cached until it changes.
"""

from __future__ import annotations

import copy
import itertools
import warnings
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from abc_minimal import policy as abc_policy
from abc_minimal.checkpointing import model_state_dict
from abc_minimal.config import validate_vla_checkpoint_config
from abc_minimal.gemma import model as gemma_model
from abc_minimal.gemma.preprocessor import tokenize_raw_input
from abc_minimal.vla import VLAPolicy, inference_model_config

GiB = 2**30
_ALIGN = 512


@dataclass
class OffloadConfig:
    gpu_budget_gb: float | None = None
    """VRAM for all VLA weights; default: free VRAM at startup minus reserve_gb."""
    reserve_gb: float = 0.5
    """VRAM kept free for activations, cuBLAS workspace and the display."""
    quantize: Literal["none", "streamed", "all"] = "none"
    """int8 weight-only Gemma layers: none, only the host-streamed ones, or all."""
    num_buffers: int = 2
    """GPU slots the streamed layers rotate through."""
    trim_padding: bool = False
    """Run Gemma on the real tokens only, not all fixed_seq_len. Padding sits
    after every real token under a causal mask, so only rounding changes."""


@dataclass
class OffloadVLAPolicyConfig(abc_policy.VLAPolicyConfig):
    offload: OffloadConfig = field(default_factory=OffloadConfig)


class Int8Linear(nn.Module):
    """Weight-only int8 linear with one scale per output row."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        w = weight.detach().float()
        scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127
        self.register_buffer("weight", torch.round(w / scale).to(torch.int8))
        self.register_buffer("scale", scale.squeeze(1).to(weight.dtype))

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype)) * self.scale


def quantize_linears(module: nn.Module, device: torch.device) -> nn.Module:
    """Quantize on ``device`` one weight at a time, then move the rest there."""
    for name, child in list(module.named_modules()):
        if isinstance(child, gemma_model.Linear):
            parent_name, _, attr = name.rpartition(".")
            setattr(
                module.get_submodule(parent_name), attr, Int8Linear(child.weight.to(device))
            )
    return module.to(device)


def int8_nbytes(module: nn.Module) -> int:
    total = 0
    for child in module.modules():
        if isinstance(child, gemma_model.Linear):
            total += child.weight.numel() + child.weight.shape[0] * child.weight.element_size()
        else:
            total += sum(t.nbytes for t in child.parameters(recurse=False))
    return total


def _tensors(module: nn.Module) -> list[tuple[str, torch.Tensor]]:
    return list(itertools.chain(module.named_parameters(), module.named_buffers()))


def _layout(module: nn.Module) -> tuple[list[tuple[str, int, torch.Size, torch.dtype]], int]:
    specs, offset = [], 0
    for name, tensor in _tensors(module):
        specs.append((name, offset, tensor.shape, tensor.dtype))
        offset += -(-tensor.nbytes // _ALIGN) * _ALIGN
    return specs, offset


def _view(flat: torch.Tensor, offset: int, shape: torch.Size, dtype: torch.dtype):
    nbytes = shape.numel() * dtype.itemsize
    return flat[offset : offset + nbytes].view(dtype).view(shape)


def _bind(module: nn.Module, flat: torch.Tensor, specs) -> None:
    for name, offset, shape, dtype in specs:
        owner_name, _, attr = name.rpartition(".")
        owner = module.get_submodule(owner_name)
        view = _view(flat, offset, shape, dtype)
        if attr in owner._parameters:
            owner._parameters[attr] = nn.Parameter(view, requires_grad=False)
        else:
            owner._buffers[attr] = view


class LayerStreamer:
    """Rotate host-resident decoder layers through a ring of GPU slots.

    Each copy starts as soon as its slot's previous layer finishes, and the ring
    wraps across calls, so the next observation's first streamed layers load
    while the robot is executing actions.
    """

    def __init__(self, first: nn.Module, count: int, device: torch.device, num_buffers: int):
        self.specs, self.nbytes = _layout(first)
        self.count = count
        # cudaHostRegister pins exactly this size; the caching host allocator
        # would round each allocation up to a power of two.
        self.host = torch.empty((count, self.nbytes), dtype=torch.uint8)
        cudart = torch.cuda.cudart()
        torch.cuda.check_error(cudart.cudaHostRegister(self.host.data_ptr(), self.host.nbytes, 0))
        weakref.finalize(self, cudart.cudaHostUnregister, self.host.data_ptr())
        self.store(0, first)
        self.gpu = [
            torch.empty(self.nbytes, dtype=torch.uint8, device=device)
            for _ in range(min(num_buffers, count))
        ]
        self.slots = []
        for flat in self.gpu:
            slot = copy.deepcopy(first)
            _bind(slot, flat, self.specs)
            self.slots.append(slot)
        self.stream = torch.cuda.Stream(device)
        self.ready = [torch.cuda.Event() for _ in self.gpu]
        self.freed = [torch.cuda.Event() for _ in self.gpu]
        self.step = 0

    def store(self, index: int, layer: nn.Module) -> None:
        """Copy a layer into its host row and free the original."""
        row = self.host[index]
        for (_, tensor), (_, offset, shape, dtype) in zip(_tensors(layer), self.specs):
            _view(row, offset, shape, dtype).copy_(tensor)
        layer.to("meta")

    def start(self) -> None:
        for k in range(len(self.gpu)):
            self._fetch(k)

    def _fetch(self, k: int) -> None:
        b = k % len(self.gpu)
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(self.freed[b])
            self.gpu[b].copy_(self.host[k % self.count], non_blocking=True)
            self.ready[b].record(self.stream)

    def _resync(self) -> None:
        torch.cuda.synchronize()
        self.step = 0
        for k in range(len(self.gpu)):
            self._fetch(k)

    def run(self, index: int, attn_type, *args):
        if index == 0 and self.step % self.count:
            self._resync()
        b = self.step % len(self.gpu)
        torch.cuda.current_stream().wait_event(self.ready[b])
        slot = self.slots[b]
        slot.attn_type = slot.self_attn.attn_type = attn_type
        out = slot(*args)
        self.freed[b].record()
        self._fetch(self.step + len(self.gpu))
        self.step += 1
        return out


class StreamedLayer(nn.Module):
    """Stand-in for a host-resident Gemma3DecoderLayer inside GemmaModel.layers."""

    def __init__(self, streamer: LayerStreamer, index: int, attn_type):
        super().__init__()
        self.streamer = streamer
        self.index = index
        self.attn_type = attn_type

    def forward(self, hidden_states, freqs_cis, mask, local_mask):
        return self.streamer.run(
            self.index, self.attn_type, hidden_states, freqs_cis, mask, local_mask
        )


class OffloadVLAInferencePolicy(abc_policy.VLAInferencePolicy):
    """VLAInferencePolicy that fits a small GPU; ``infer`` is upstream's."""

    def __init__(self, checkpoint: Path, config: Any, device: str, model_config: Any = None):
        self.config = config
        self.model_config = model_config if model_config is not None else config.model
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("the offloading VLA policy needs a CUDA device")
        self.diffusion_steps = config.diffusion_steps
        self.camera_keys = tuple(self.model_config.camera_keys)
        self.chunk_length = self.model_config.chunk_length
        self.action_dim = self.model_config.action_dim
        offload = getattr(config, "offload", OffloadConfig())

        checkpoint = Path(checkpoint).expanduser().resolve()
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        if not validate_vla_checkpoint_config(self.model_config, ckpt, source=str(checkpoint)):
            warnings.warn(
                "VLA checkpoint has no architecture metadata; only tensor keys "
                "and shapes can be validated",
                RuntimeWarning,
                stacklevel=2,
            )
        with torch.device("meta"):
            self.model = VLAPolicy(
                inference_model_config(self.model_config),
                backbone_dtype=torch.bfloat16,
                backbone_autocast=False,
            )
        self.model.load_state_dict(model_state_dict(ckpt), strict=True, assign=True)
        self.norm_stats = abc_policy.resolve_norm_stats(ckpt, config.norm_stats_path)
        self.trained_max_prefix = abc_policy.resolve_trained_max_prefix(ckpt)
        del ckpt
        self._restore_nonpersistent_buffers()
        self.model.eval()
        self._place(offload)
        self.trim_padding = offload.trim_padding
        self._prefix_key = None
        self.model.vla.prepare = self._prepare
        self._fast_inference_enabled = False

    def _restore_nonpersistent_buffers(self) -> None:
        gemma = self.model.vla.gemma_model
        for name in ("local_freqs_cis", "global_freqs_cis"):
            for suffix in ("_cos", "_sin"):
                del gemma._buffers[name + suffix]
            gemma_model.refresh_rotary_freqs_buffers(gemma, name)
        siglip = gemma.siglip_vision_model
        siglip.register_buffer(
            "position_ids",
            torch.arange(siglip.num_positions).expand((1, -1)),
            persistent=False,
        )

    def _place(self, offload: OffloadConfig) -> None:
        gemma = self.model.vla.gemma_model
        layers = list(gemma.model.layers)[: self.model.feature_layer + 1]
        embedder = gemma.text_token_embedder
        gemma.model.layers = nn.ModuleList()
        gemma.text_token_embedder = None
        self.model.to(self.device)
        gemma.text_token_embedder = embedder

        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info(self.device)
        used = torch.cuda.memory_allocated(self.device)
        available = used + free - int(offload.reserve_gb * GiB)
        if offload.gpu_budget_gb is None:
            budget = available
        else:
            budget = int(offload.gpu_budget_gb * GiB)
            if budget > available:
                warnings.warn(
                    f"--offload.gpu-budget-gb={offload.gpu_budget_gb} exceeds the "
                    f"{available / GiB:.2f} GiB available; using that",
                    RuntimeWarning,
                    stacklevel=2,
                )
                budget = available
        layer_budget = budget - used

        bf16_bytes = _layout(layers[0])[1]
        int8_bytes = int8_nbytes(layers[0])
        resident_bytes = int8_bytes if offload.quantize == "all" else bf16_bytes
        stream_bytes = bf16_bytes if offload.quantize == "none" else int8_bytes
        n = len(layers)
        if n * resident_bytes <= layer_budget:
            num_resident = n
        else:
            slots = offload.num_buffers * stream_bytes
            num_resident = int((layer_budget - slots) // resident_bytes)
            if num_resident < 0:
                raise RuntimeError(
                    f"{layer_budget / GiB:.2f} GiB of VRAM left for Gemma layers cannot "
                    f"hold {offload.num_buffers} streaming slots; free VRAM or lower "
                    "--offload.num-buffers / use --offload.quantize"
                )
        num_streamed = n - num_resident
        streamed = {int((j + 0.5) * n / num_streamed) for j in range(num_streamed)}

        self.streamer = None
        stand_ins = {}
        for j, i in enumerate(sorted(streamed)):
            layer = layers[i]
            if offload.quantize != "none":
                layer = quantize_linears(layer, self.device)
            if self.streamer is None:
                self.streamer = LayerStreamer(
                    layer, num_streamed, self.device, offload.num_buffers
                )
            else:
                self.streamer.store(j, layer)
            stand_ins[i] = StreamedLayer(self.streamer, j, layer.attn_type)
        placed = []
        for i, layer in enumerate(layers):
            if i in stand_ins:
                placed.append(stand_ins[i])
            elif offload.quantize == "all":
                placed.append(quantize_linears(layer, self.device))
            else:
                placed.append(layer.to(self.device))
        gemma.model.layers = nn.ModuleList(placed)
        if self.streamer is not None:
            self.streamer.start()
        torch.cuda.empty_cache()

        self.pinned_bytes = self.streamer.host.nbytes if self.streamer else 0
        print(
            f"[vla_offload] budget {budget / GiB:.2f} GiB: {num_resident}/{n} Gemma layers "
            f"resident ({'int8' if offload.quantize == 'all' else 'bf16'}), "
            f"{num_streamed} streamed "
            f"({'bf16' if offload.quantize == 'none' else 'int8'}) from "
            f"{self.pinned_bytes / GiB:.2f} GiB pinned host memory",
            flush=True,
        )

    def _build_prefix(self, prompts: tuple[str, ...], device: torch.device):
        vla = self.model.vla
        placeholders = torch.empty(len(prompts), len(self.camera_keys), 0)
        processed = tokenize_raw_input(
            vla.tokenizer,
            vla._raw_input(placeholders, list(prompts)),
            vla.model_config,
            torch.device("cpu"),
        )
        context_ids = processed["user_input_token_ids"]
        context_lengths = processed["prompt_lengths"]
        if max(context_lengths) > vla.fixed_seq_len:
            raise ValueError(
                f"multimodal sequence needs {max(context_lengths)} tokens but "
                f"fixed_seq_len={vla.fixed_seq_len}"
            )
        text_embeds = vla._embed_text(context_ids).to(device)
        context_ids = context_ids.to(device)
        seq_len = context_ids.shape[1] if self.trim_padding else vla.fixed_seq_len
        input_ids = torch.full(
            (len(prompts), seq_len),
            vla.tokenizer.pad_id,
            dtype=context_ids.dtype,
            device=device,
        )
        input_ids[:, : context_ids.shape[1]] = context_ids
        positions = torch.arange(seq_len, device=device)
        lengths = torch.tensor(context_lengths, device=device)
        valid = positions[None, :] < lengths[:, None]
        attention, local_attention = vla._attention_masks(input_ids, text_embeds.dtype)
        return input_ids, context_ids, text_embeds, attention, local_attention, positions, valid

    def _prepare(self, batch):
        """GemmaVLABackbone.prepare with the prompt-only parts cached."""
        vla = self.model.vla
        images = batch["images"]
        key = tuple(batch["prompt"])
        if key != self._prefix_key:
            self._prefix = self._build_prefix(key, images.device)
            self._prefix_key = key
        input_ids, context_ids, text_embeds, attention, local_attention, positions, valid = (
            self._prefix
        )
        normalized = vla._normalize_images(images)
        context_embeds = vla._inject_state(text_embeds, context_ids, batch["state"])
        context_embeds = vla._encode_images(
            context_embeds, normalized.to(text_embeds.dtype), context_ids
        )
        input_embeds = torch.zeros(
            images.shape[0],
            input_ids.shape[1],
            vla.model_config.hidden_size,
            dtype=text_embeds.dtype,
            device=images.device,
        )
        context_length = context_ids.shape[1]
        input_embeds[:, :context_length] = torch.where(
            valid[:, :context_length, None], context_embeds, 0.0
        )
        return input_ids, input_embeds, attention, local_attention, positions, ~valid

    def _compile_for_fast_inference(self, compile_mode: str) -> None:
        """Compile the resident hot paths: SigLIP and the head's velocity network."""
        kwargs: dict[str, Any] = {"dynamic": False, "mode": compile_mode or None}
        gemma = self.model.vla.gemma_model
        gemma.siglip_vision_model.forward = torch.compile(  # type: ignore[method-assign]
            gemma.siglip_vision_model.forward, **kwargs
        )
        head = self.model.diffusion_head
        head.predict_velocity = torch.compile(head.predict_velocity, **kwargs)  # type: ignore[method-assign]
