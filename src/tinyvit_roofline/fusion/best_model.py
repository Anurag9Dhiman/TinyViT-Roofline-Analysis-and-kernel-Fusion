"""The single reusable entry point for this project's actual finding: the
one combination of optimizations that was validated, end-to-end, to beat
the standard baseline on real ImageNet-1k images with full or near-full
accuracy preserved (section 7/8/10/13 of report.md) -- conv+BN fusion
(exact) + channels_last (exact) + optional fp16 (not exact, small real
accuracy cost, see section 7).

Every other lever tried in this project (MLX attention fusion, torch.compile,
MLX int8 quantization, bf16, native kernel fusion) either failed outright or
doesn't generalize past a narrow synthetic config -- this is the one thing
actually worth taking out of the research and using. See README.md's
headline results for the measured speedups (1.08-1.27x exact, up to ~1.49x
with fp16, varying by model scale).
"""

import torch

from ..models import load_baseline_model
from .conv_bn_fusion import fuse_conv_norms


def build_best_model(variant: str, fp16: bool = False, device: str = "mps") -> torch.nn.Module:
    """Loads the standard pretrained TinyViT checkpoint and applies the
    validated optimization stack: conv+BN fusion, then channels_last
    memory format, then (optionally) fp16.

    fp16=False (default) is numerically exact (max-abs-diff ~1e-6 vs. the
    unmodified baseline). fp16=True is the fastest arm in the whole project
    but is NOT exact -- expect a small, real accuracy cost (~0.1pp top-1 on
    ImageNet-1k, shrinking with model scale; section 7 of report.md).

    Caller is responsible for converting input tensors to match: always
    `.to(memory_format=torch.channels_last)`, and additionally `.half()`
    if fp16=True.
    """
    model = load_baseline_model(variant)
    fuse_conv_norms(model)
    model.to(device)
    model = model.to(memory_format=torch.channels_last)
    if fp16:
        model = model.half()
    return model
