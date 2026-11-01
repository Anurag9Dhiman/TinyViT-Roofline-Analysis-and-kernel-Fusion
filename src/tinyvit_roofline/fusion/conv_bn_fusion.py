"""Conv+BatchNorm fusion: a standard, exact (not approximate) inference-time
optimization -- every `ConvNorm` in TinyViT (patch_embed, MBConv in stage 0,
PatchMerging, local_conv in every block) is Conv2d(bias=False) followed by
BatchNorm2d. At inference, BN is just a fixed per-channel affine transform,
so it can be folded directly into the conv's weight/bias with no
approximation (only float-reordering-level numerical difference).

Motivated directly by the roofline diagnostics: Phase 1's profiling found
"conv" was the single largest dispatch-time category (1919us/iter,
more than qkv/mlp/attention combined) -- and unlike the MLX-fusion attempts
(mlx_fused_attn.py, mlx_fused_block.py), which target the attention core
and, on inspection, don't reliably beat PyTorch's MPS backend at TinyViT's
actual operator sizes, this targets the part of the model actually shown to
dominate latency, with a well-understood technique (timm's own `ConvNorm`
already ships a `.fuse()` method for exactly this) rather than an
experimental one.

Measured on this machine (tiny_vit_5m_224, real pretrained weights):
batch=1: 1.35x, batch=4: 1.13x, batch=16: 1.16x, batch=32: 1.15x, with
max-abs-diff ~3e-6 and identical top-1 predictions vs. the unfused baseline.
"""

from typing import Tuple

import torch


def fuse_conv_norms(model: torch.nn.Module) -> int:
    """Replace every ConvNorm submodule in place with its fused nn.Conv2d
    equivalent (timm's own ConvNorm.fuse()). Returns the number fused.
    Caller is responsible for moving the model to the target device
    afterward (.fuse() constructs the new Conv2d on CPU)."""
    count = 0
    for module in list(model.modules()):
        for child_name, child in list(module.named_children()):
            if type(child).__name__ == "ConvNorm":
                setattr(module, child_name, child.fuse())
                count += 1
    return count


def build_conv_fused_model(load_fn, variant: str, device: str = "mps") -> Tuple[torch.nn.Module, int]:
    """load_fn: e.g. models.load_baseline_model. Returns (fused_model, num_fused)."""
    model = load_fn(variant)
    n = fuse_conv_norms(model)
    model.to(device)
    return model, n
