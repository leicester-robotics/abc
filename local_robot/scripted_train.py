"""Small local DiT pipeline test using the unchanged ABC trainer.

Run once for 200 steps, then --resume for checkpoint reload + one more step.
The isolated export cache intentionally contains no pretrained DINO weights.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import patch

from abc_minimal.config import TrainConfig, DiTConfig, MixtureComponent, OptimConfig


def build_config(session, *, resume=False, steps=200):
    session = Path(session).resolve()
    return TrainConfig(
        cache_root=str(session/'export'), output_dir=str(session/'checkpoints'),
        seed=123, batch_size=1, num_workers=0, train_steps=201 if resume else steps,
        mixture=[MixtureComponent('train', 'val', 1., '')],
        load_pretrained=False, compile=False, dino_bf16=True,
        log_every=1 if resume else 10, val_every=100, val_batches=4,
        ckpt_every=1 if resume else 100,
        resume_from=str(session/'checkpoints/200.pt') if resume else None,
        optim=OptimConfig(lr_warmup_steps=20),
        model=DiTConfig(hidden_size=256, depth=4, num_heads=8, chunk_length=30))


class TrainingAudit:
    def __init__(self):
        self.rows = []
        self.loss = None
        self.grad_norm = None
        self.shapes = None

    def check_gradients(self, parameters):
        import torch
        grads = [p.grad for p in parameters if p.grad is not None]
        if not grads or not bool(torch.stack([g.detach().isfinite().all() for g in grads]).all()):
            raise FloatingPointError('missing or nonfinite gradients')

    def forward(self, model, args, loss):
        import torch
        batch = args[0]
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('nonfinite loss')
        if set(batch['images']) != {'top', 'left', 'right'}:
            raise ValueError('training must load all three camera views')
        if batch['actions'].shape[1:] != (30, 14) or batch['state'].shape[-1] != 14:
            raise ValueError('unexpected training dimensions')
        if batch['task_vec_clip'].shape[-1] != 512 or not bool(batch['task_vec_clip'].isfinite().all()):
            raise ValueError('invalid CLIP text conditioning')
        self.shapes = {k: list(v.shape) for k, v in batch['images'].items()}
        if model.training:
            self.loss = float(loss.detach())


def run(session, *, resume=False, steps=200):
    import torch
    from abc_minimal import train_loop
    cfg = build_config(session, resume=resume, steps=steps)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this local GPU pipeline test')
    if (Path(cfg.cache_root)/'dinov3_vitb16_pretrain_lvd1689m.pth').exists():
        raise ValueError('isolated test cache must not contain pretrained DINO weights')
    torch.set_num_threads(4)
    audit = TrainingAudit()
    build_model = train_loop._build_dit_model
    build_optim = train_loop._build_optimizer
    clip_grad = torch.nn.utils.clip_grad_norm_
    model_ref = []
    initial = {}

    def audited_model(*args, **kwargs):
        model, stats = build_model(*args, **kwargs)
        model.register_forward_hook(audit.forward)
        model_ref.append(model)
        initial['head'] = model.final_layer.linear.weight.detach().cpu().clone()
        initial['vision'] = model.img_backbone.dinov3_model.patch_embed.proj.weight.detach().cpu().clone()
        return model, stats

    def audited_optimizer(model, *args, **kwargs):
        optim = build_optim(model, *args, **kwargs)
        previous = []
        def before(*_):
            previous[:] = [model.final_layer.linear.weight.detach().clone()]
        def after(*_):
            delta = float((model.final_layer.linear.weight.detach() - previous[0]).abs().max())
            if not delta > 0:
                raise RuntimeError('optimizer did not update head parameters')
            audit.rows.append({'step': (200 if resume else 0) + len(audit.rows) + 1,
                               'loss': audit.loss, 'grad_norm': audit.grad_norm,
                               'head_max_update': delta})
        optim.register_step_pre_hook(before)
        optim.register_step_post_hook(after)
        return optim

    def audited_clip(parameters, max_norm, *args, **kwargs):
        parameters = list(parameters)
        audit.check_gradients(parameters)
        kwargs['error_if_nonfinite'] = True
        norm = clip_grad(parameters, max_norm, *args, **kwargs)
        audit.grad_norm = float(norm)
        return norm

    session = Path(session)
    session.mkdir(parents=True, exist_ok=True)
    label = 'resume' if resume else 'train'
    (session/f'{label}_config.json').write_text(json.dumps(asdict(cfg), indent=2) + '\n')
    try:
        with patch.object(train_loop, '_build_dit_model', audited_model), \
             patch.object(train_loop, '_build_optimizer', audited_optimizer), \
             patch.object(torch.nn.utils, 'clip_grad_norm_', audited_clip):
            train_loop.main(cfg)
        expected = 1 if resume else steps
        if len(audit.rows) != expected:
            raise RuntimeError(f'expected {expected} optimizer steps, got {len(audit.rows)}')
        model = model_ref[0]
        vision_delta = float((model.img_backbone.dinov3_model.patch_embed.proj.weight.detach().cpu() - initial['vision']).abs().max())
        if not vision_delta > 0:
            raise RuntimeError('DINO parameters did not update')
        print(f'Verified {expected} finite optimizer steps; DINO max update {vision_delta:.6g}', flush=True)
    finally:
        report = {'camera_shapes': audit.shapes, 'steps': audit.rows,
                  'vision_max_update': locals().get('vision_delta'),
                  'peak_gpu_gib': torch.cuda.max_memory_allocated()/2**30}
        (session/f'{label}_audit.json').write_text(json.dumps(report, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--steps', type=int, default=200)
    args = parser.parse_args()
    run(args.session, resume=args.resume, steps=args.steps)


if __name__ == '__main__':
    main()
