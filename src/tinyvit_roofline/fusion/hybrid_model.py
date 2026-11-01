"""Wrap a real, pretrained TinyViT model's attention submodules with the
MLX-fused implementation, in place, so the full model can run real inference
(and be scored for real accuracy) with window attention actually routed
through MLX -- not just the isolated-operator comparison in ablation/.

Important, deliberately NOT hidden: PyTorch-MPS tensors and MLX arrays do not
share memory even though both run on the same physical unified memory --
there is no zero-copy bridge between the two frameworks' own Metal buffer
allocators. Every wrapped attention call pays a real GPU->CPU->GPU round
trip (`.to('cpu').numpy()` then `mx.array(...)`, and back). This overhead is
real and reported, not glossed over: it may offset or exceed the fused
kernel's own speed advantage once integrated into a real multi-block model,
which is itself a finding worth having, not just a footnote.
"""

import types
from typing import List

import numpy as np
import torch

from .mlx_fused_attn import build_mlx_attention


def _make_mlx_routed_forward(compiled_mlx_fn):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import mlx.core as mx

        orig_device, orig_dtype = x.device, x.dtype
        x_np = x.detach().to("cpu", torch.float32).numpy()
        x_mlx = mx.array(x_np)
        out_mlx = compiled_mlx_fn(x_mlx)
        mx.eval(out_mlx)
        out_np = np.array(out_mlx)
        return torch.from_numpy(out_np).to(orig_device, orig_dtype)

    return forward


def wrap_blocks_with_mlx_attention(model) -> List[str]:
    """Monkeypatch every TinyVitBlock's `.attn.forward` to route through the
    MLX-fused implementation (weights ported from that exact block's real
    trained attn submodule). Returns the list of wrapped module names."""
    wrapped = []
    for name, module in model.named_modules():
        if type(module).__name__ == "TinyVitBlock":
            mlx_attn = build_mlx_attention(module.attn)
            compiled_fn = mlx_attn.compiled()
            module.attn.forward = types.MethodType(_make_mlx_routed_forward(compiled_fn), module.attn)
            wrapped.append(name)
    return wrapped
