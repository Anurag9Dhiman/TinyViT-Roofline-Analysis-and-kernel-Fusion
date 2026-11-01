"""Closed-form FLOPs / bytes-moved estimators for each TinyViT window-attention
sub-stage, parametrized by real shapes (so they track whatever variant/config
is actually profiled, not a single hand-derived case).

Convention: FLOPs counts multiply-adds as 2 FLOPs (standard matmul counting).
Bytes assume fp32 (itemsize=4) unless dtype is passed. "Layout" ops (window
partition/reverse) are pure data movement: 0 FLOPs, but a full read+write of
the feature map each, because TinyViT's window partition uses
`.transpose(2,3).reshape(...)`, and transpose-then-reshape on a
non-contiguous tensor forces a real copy on the MPS backend -- confirmed by
the "layout" self-time we measured on the uninstrumented baseline
(profiling/torch_profile.py) is nonzero, not a free view.
"""

from dataclasses import dataclass
from typing import Dict

ITEMSIZE_FP32 = 4


@dataclass
class StageAI:
    name: str
    flops: float
    bytes_moved: float

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.bytes_moved if self.bytes_moved else float("inf")

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "flops": self.flops,
            "bytes_moved": self.bytes_moved,
            "arithmetic_intensity_flops_per_byte": self.arithmetic_intensity,
        }


def qkv_proj(batch_windows: int, tokens_per_window: int, dim: int, itemsize: int = ITEMSIZE_FP32) -> StageAI:
    """Linear(dim -> 3*dim) applied to (batch_windows, tokens_per_window, dim)."""
    m = batch_windows * tokens_per_window
    flops = 2.0 * m * dim * (3 * dim)
    bytes_moved = itemsize * (m * dim + dim * (3 * dim) + m * (3 * dim))  # read x, read W, write y
    return StageAI("qkv_proj", flops, bytes_moved)


def attn_score_exact(batch_windows: int, tokens_per_window: int, dim: int, num_heads: int, itemsize: int = ITEMSIZE_FP32) -> StageAI:
    """Q @ K^T across all heads combined: num_heads*head_dim == dim."""
    n = tokens_per_window
    flops = 2.0 * batch_windows * n * n * dim
    read_bytes = itemsize * 2 * batch_windows * n * dim  # Q, K
    write_bytes = itemsize * batch_windows * num_heads * n * n  # attn scores
    return StageAI("attn_score", flops, read_bytes + write_bytes)


def softmax(batch_windows: int, tokens_per_window: int, num_heads: int, itemsize: int = ITEMSIZE_FP32) -> StageAI:
    """~5 FLOPs/element heuristic (exp, running sum, divide) over the (n x n) attn matrix, per head."""
    n = tokens_per_window
    elems = batch_windows * num_heads * n * n
    flops = 5.0 * elems
    bytes_moved = itemsize * 2 * elems  # read logits, write probs (in place in practice, but count both passes)
    return StageAI("softmax", flops, bytes_moved)


def attn_value(batch_windows: int, tokens_per_window: int, dim: int, num_heads: int, itemsize: int = ITEMSIZE_FP32) -> StageAI:
    n = tokens_per_window
    flops = 2.0 * batch_windows * n * n * dim
    read_bytes = itemsize * (batch_windows * num_heads * n * n + batch_windows * n * dim)  # attn probs + V
    write_bytes = itemsize * batch_windows * n * dim
    return StageAI("attn_value", flops, read_bytes + write_bytes)


def sdpa_fused(batch_windows: int, tokens_per_window: int, dim: int, num_heads: int, itemsize: int = ITEMSIZE_FP32) -> StageAI:
    """Combined QK^T + softmax + AV, matching the single opaque
    aten::_scaled_dot_product_attention_math_for_mps call we can actually time
    on this backend (see torch_profile.py's note on why this can't be split)."""
    qk = attn_score_exact(batch_windows, tokens_per_window, dim, num_heads, itemsize)
    sm = softmax(batch_windows, tokens_per_window, num_heads, itemsize)
    av = attn_value(batch_windows, tokens_per_window, dim, num_heads, itemsize)
    return StageAI("sdpa_qk_softmax_av", qk.flops + sm.flops + av.flops, qk.bytes_moved + sm.bytes_moved + av.bytes_moved)


def out_proj(batch_windows: int, tokens_per_window: int, dim: int, itemsize: int = ITEMSIZE_FP32) -> StageAI:
    m = batch_windows * tokens_per_window
    flops = 2.0 * m * dim * dim
    bytes_moved = itemsize * (m * dim + dim * dim + m * dim)
    return StageAI("out_proj", flops, bytes_moved)


def window_layout_op(batch: int, height: int, width: int, dim: int, itemsize: int = ITEMSIZE_FP32) -> StageAI:
    """Window partition or window reverse: 0 FLOPs, one full read + write of
    the feature map (transpose(2,3).reshape(...) forces a real copy)."""
    elems = batch * height * width * dim
    return StageAI("window_layout", 0.0, itemsize * 2 * elems)


def local_conv_depthwise(batch: int, height: int, width: int, dim: int, kernel_size: int = 3, itemsize: int = ITEMSIZE_FP32) -> StageAI:
    flops = 2.0 * batch * height * width * dim * kernel_size * kernel_size
    bytes_moved = itemsize * (batch * height * width * dim * 2 + dim * kernel_size * kernel_size)  # read+write features, read weights
    return StageAI("local_conv", flops, bytes_moved)


def mlp(batch: int, tokens: int, dim: int, mlp_ratio: float = 4.0, itemsize: int = ITEMSIZE_FP32) -> StageAI:
    hidden = int(dim * mlp_ratio)
    m = batch * tokens
    flops = 2.0 * m * dim * hidden + 2.0 * m * hidden * dim
    bytes_moved = itemsize * (m * dim + dim * hidden + m * hidden) + itemsize * (m * hidden + hidden * dim + m * dim)
    return StageAI("mlp", flops, bytes_moved)


def block_ai_breakdown(
    batch: int,
    resolution: int,
    window_size: int,
    dim: int,
    num_heads: int,
    mlp_ratio: float = 4.0,
    itemsize: int = ITEMSIZE_FP32,
) -> Dict[str, StageAI]:
    """All stages for one TinyVitBlock forward, given the block's real config."""
    h = w = resolution
    is_full_window = (h == window_size and w == window_size)
    n_win_h = 1 if is_full_window else -(-h // window_size)
    n_win_w = 1 if is_full_window else -(-w // window_size)
    batch_windows = batch * n_win_h * n_win_w
    tokens_per_window = window_size * window_size if not is_full_window else h * w

    out = {
        "qkv_proj": qkv_proj(batch_windows, tokens_per_window, dim, itemsize),
        "sdpa_qk_softmax_av": sdpa_fused(batch_windows, tokens_per_window, dim, num_heads, itemsize),
        "out_proj": out_proj(batch_windows, tokens_per_window, dim, itemsize),
        "local_conv": local_conv_depthwise(batch, h, w, dim, 3, itemsize),
        "mlp": mlp(batch, h * w, dim, mlp_ratio, itemsize),
    }
    if not is_full_window:
        out["window_partition"] = window_layout_op(batch, h, w, dim, itemsize)
        out["window_reverse"] = window_layout_op(batch, h, w, dim, itemsize)
    return out


def conv2d_ai(
    in_shape,
    out_shape,
    kernel_size,
    groups: int = 1,
    itemsize: int = ITEMSIZE_FP32,
) -> StageAI:
    """Generic Conv2d FLOPs/bytes from actual (captured) input/output shapes,
    used for patch_embed / MBConv / PatchMerging -- covers the "convolution
    blocks" stage the roofline is also supposed to characterize, without
    hand-deriving each stage's resolution."""
    n, c_in, h_in, w_in = in_shape
    n2, c_out, h_out, w_out = out_shape
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    flops = 2.0 * n * c_out * h_out * w_out * (c_in // groups) * kh * kw
    bytes_moved = itemsize * (
        n * c_in * h_in * w_in  # read input
        + c_out * (c_in // groups) * kh * kw  # read weights
        + n * c_out * h_out * w_out  # write output
    )
    return StageAI("conv2d", flops, bytes_moved)


def profile_conv_layers(model, x) -> Dict[str, StageAI]:
    """Forward-hook every nn.Conv2d, capture its real input/output shape on
    one forward pass, and return per-module AI. Generic and automatic --
    covers patch_embed, every MBConv conv, and PatchMerging without manually
    tracking resolutions through the stem."""
    import torch.nn as nn

    shapes = {}
    handles = []

    def make_hook(name, module):
        def hook(mod, inp, out):
            shapes[name] = (tuple(inp[0].shape), tuple(out.shape), mod.kernel_size, mod.groups)
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(make_hook(name, module)))

    try:
        import torch
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()

    out = {}
    for name, (in_shape, out_shape, ks, groups) in shapes.items():
        out[name] = conv2d_ai(in_shape, out_shape, ks, groups)
    return out
