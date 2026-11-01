"""MLX-fused window attention: same math as timm's Attention module (norm ->
qkv-linear -> split/permute -> SDPA -> merge -> proj-linear), real weights
ported from the PyTorch checkpoint (weight_port.py), but the attention core
runs through MLX's `mx.fast.scaled_dot_product_attention` -- a single fused
Metal kernel for QK^T + softmax + AV, as opposed to PyTorch's MPS backend,
which (per Phase 1/3's findings) falls back to
`aten::_scaled_dot_product_attention_math_for_mps`, an opaque-to-the-profiler
but NOT actually a single fused Metal kernel in the same sense.

This is the "kernel fusion via MLX/Metal" arm the project targets: fewer
intermediate tensors materialized in unified memory, fewer kernel
dispatches, for the same numerical operation.
"""

from typing import Dict

import mlx.core as mx


class MLXWindowAttention:
    def __init__(self, weights: Dict[str, mx.array], num_heads: int, key_dim: int, val_dim: int):
        self.w = weights
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.val_dim = val_dim
        self.out_dim = val_dim * num_heads
        self.scale = key_dim ** -0.5

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        w = self.w

        # Fused LayerNorm (mx.fast.layer_norm), not a hand-rolled mean/var/
        # normalize chain -- same "use the fused primitive" principle as SDPA.
        x_norm = mx.fast.layer_norm(x, w["norm_weight"], w["norm_bias"], eps=1e-5)

        # Fused QKV projection (single linear, same as baseline)
        qkv = x_norm @ w["qkv_weight"].T + w["qkv_bias"]
        qkv = qkv.reshape(B, N, self.num_heads, -1)
        q, k, v = mx.split(qkv, [self.key_dim, 2 * self.key_dim], axis=3)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Fused attention core: single Metal kernel for QK^T + softmax + AV
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=w["attention_bias"])

        out = out.transpose(0, 2, 1, 3).reshape(B, N, self.out_dim)
        out = out @ w["proj_weight"].T + w["proj_bias"]
        return out

    def compiled(self):
        """A `mx.compile`-wrapped callable. MLX is a lazy-graph framework:
        an uncompiled call rebuilds and re-traces the whole op graph in
        Python on every invocation (measured ~1.3x slower on this machine
        for this op at TinyViT's real shapes -- see scripts/15_mlx_
        primitive_microbench.py's mlx_attn_compile_benefit; smaller than
        this docstring once claimed, before that script existed to check
        it), whereas mx.compile caches the compiled graph across calls
        with the same input shapes -- the fair, idiomatic-MLX comparison
        point against PyTorch's compiled `torch.compile` arm, and against
        PyTorch's own persistent (non-retraced) op dispatch."""
        if not hasattr(self, "_compiled_fn"):
            self._compiled_fn = mx.compile(lambda inp: self(inp))
        return self._compiled_fn


def build_mlx_attention(torch_attn) -> MLXWindowAttention:
    from .weight_port import port_attention_weights

    weights = port_attention_weights(torch_attn)
    return MLXWindowAttention(weights, torch_attn.num_heads, torch_attn.key_dim, torch_attn.val_dim)
