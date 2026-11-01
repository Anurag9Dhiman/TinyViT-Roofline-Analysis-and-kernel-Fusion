"""PyTorch-level optimization arm of the ablation: same weights, same op
graph as the baseline `Attention` module (timm's own qkv-linear -> SDPA ->
proj-linear path) -- no manual algorithmic rewrite -- wrapped in
`torch.compile` so Inductor can fuse the reshape/permute/elementwise chain
and reduce dispatch count on its own.

Why not hand-rewrite the attention math? Phase 1/3 established that the
baseline already uses timm's fused-attention path (F.scaled_dot_product_
attention) with a single fused QKV linear -- there isn't a naive unfused
reference to "optimize away" here; the remaining overhead is dispatch count
and small-op efficiency, which is exactly what a compiler pass targets
without changing the algorithm. This keeps the ablation honest: this arm
tests what a compiler can do to the standard implementation, not a
hand-tuned rewrite competing unfairly with the "standard baseline" premise
of the whole project.
"""

import torch

from .pytorch_baseline_attn import Attention, build_attention


class OptimizedAttention(torch.nn.Module):
    def __init__(self, baseline_attn: Attention, mode: str = "default"):
        super().__init__()
        self.attn = baseline_attn
        self._compiled = None
        self._compile_mode = mode
        self._compile_error = None

    def _ensure_compiled(self):
        if self._compiled is None and self._compile_error is None:
            try:
                self._compiled = torch.compile(self.attn, mode=self._compile_mode)
            except Exception as e:  # torch.compile/Inductor support for MPS can be partial
                self._compile_error = repr(e)
        return self._compiled

    def forward(self, x):
        compiled = self._ensure_compiled()
        if compiled is not None:
            try:
                return compiled(x)
            except Exception as e:
                self._compile_error = repr(e)
                self._compiled = None
        return self.attn(x)  # fall back to the identical uncompiled baseline module

    @property
    def compiled_successfully(self) -> bool:
        return self._compiled is not None and self._compile_error is None

    @property
    def compile_error(self):
        return self._compile_error


def build_optimized(dim: int, window_size: int, num_heads=None, mode: str = "default") -> OptimizedAttention:
    baseline = build_attention(dim, window_size, num_heads)
    return OptimizedAttention(baseline, mode=mode)
