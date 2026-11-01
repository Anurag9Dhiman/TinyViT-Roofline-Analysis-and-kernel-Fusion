"""Controlled window-attention microbenchmark: sweep window size, embedding
dim, window count, and batch size; decompose
T_attention = T_qkv + T_reshape + T_attn(sdpa) + T_qk + T_softmax + T_av + T_proj
using the same isolated-microbenchmark methodology as roofline/op_bench.py
(warmup + many iterations + one final sync, to avoid torch.mps.synchronize()'s
fixed-cost problem documented there).

Uses timm's actual Attention module (timm.models.tiny_vit.Attention) so this
is the real op, not a re-implementation, just driven directly instead of
through a full TinyViT forward pass.
"""

import time
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from timm.models.tiny_vit import Attention


def _time_call(fn, warmup: int = 15, iters: int = 100) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters


def heads_for_dim(dim: int, head_dim: int = 32) -> int:
    return max(1, dim // head_dim)


def build_attention(dim: int, window_size: int, num_heads: Optional[int] = None, device: str = "mps") -> Attention:
    num_heads = num_heads or heads_for_dim(dim)
    while dim % num_heads != 0:
        num_heads -= 1
    key_dim = dim // num_heads
    attn = Attention(dim, key_dim, num_heads, attn_ratio=1, resolution=(window_size, window_size), device=device)
    attn.eval()
    return attn


def run_config(window_size: int, dim: int, num_windows: int, batch_size: int, warmup: int = 15, iters: int = 100) -> Dict:
    attn = build_attention(dim, window_size)
    batch_windows = batch_size * num_windows
    tokens = window_size * window_size
    x = torch.randn(batch_windows, tokens, dim, device="mps")

    t_total = _time_call(lambda: attn(x), warmup, iters)

    with torch.no_grad():
        attn_bias = attn.get_attention_biases(x.device)
        x_norm = attn.norm(x)
    t_norm = _time_call(lambda: attn.norm(x), warmup, iters)
    t_qkv = _time_call(lambda: attn.qkv(x_norm), warmup, iters)

    with torch.no_grad():
        qkv = attn.qkv(x_norm)
        B, N, _ = x_norm.shape
        q, k, v = qkv.view(B, N, attn.num_heads, -1).split([attn.key_dim, attn.key_dim, attn.val_dim], dim=3)

    def _split_permute():
        qq, kk, vv = qkv.view(B, N, attn.num_heads, -1).split([attn.key_dim, attn.key_dim, attn.val_dim], dim=3)
        return qq.permute(0, 2, 1, 3), kk.permute(0, 2, 1, 3), vv.permute(0, 2, 1, 3)

    t_reshape = _time_call(_split_permute, warmup, iters)

    with torch.no_grad():
        qp = q.permute(0, 2, 1, 3).contiguous()
        kp = k.permute(0, 2, 1, 3).contiguous()
        vp = v.permute(0, 2, 1, 3).contiguous()

    t_attn_fused = _time_call(lambda: F.scaled_dot_product_attention(qp, kp, vp, attn_mask=attn_bias), warmup, iters)

    with torch.no_grad():
        scale = attn.scale
        qk = (qp * scale) @ kp.transpose(-2, -1) + attn_bias
    t_qk = _time_call(lambda: (qp * scale) @ kp.transpose(-2, -1) + attn_bias, warmup, iters)

    with torch.no_grad():
        sm = qk.softmax(dim=-1)
    t_softmax = _time_call(lambda: qk.softmax(dim=-1), warmup, iters)
    t_av = _time_call(lambda: sm @ vp, warmup, iters)

    with torch.no_grad():
        av_out = sm @ vp
        merged = av_out.transpose(1, 2).reshape(B, N, attn.out_dim)
    t_proj = _time_call(lambda: attn.proj(merged), warmup, iters)

    manual_sum = t_norm + t_qkv + t_reshape + t_qk + t_softmax + t_av + t_proj
    t_dispatch_estimate = max(0.0, t_total - (t_norm + t_qkv + t_reshape + t_attn_fused + t_proj))

    return {
        "window_size": window_size,
        "dim": dim,
        "num_heads": attn.num_heads,
        "num_windows": num_windows,
        "batch_size": batch_size,
        "batch_windows": batch_windows,
        "tokens_per_window": tokens,
        "t_total_us": t_total * 1e6,
        "t_norm_us": t_norm * 1e6,
        "t_qkv_us": t_qkv * 1e6,
        "t_reshape_us": t_reshape * 1e6,
        "t_attention_fused_sdpa_us": t_attn_fused * 1e6,
        "t_qk_score_us": t_qk * 1e6,
        "t_softmax_us": t_softmax * 1e6,
        "t_attention_value_us": t_av * 1e6,
        "t_proj_us": t_proj * 1e6,
        "t_manual_unfused_sum_us": manual_sum * 1e6,
        "t_dispatch_overhead_estimate_us": t_dispatch_estimate * 1e6,
        "dispatch_overhead_fraction_of_total": t_dispatch_estimate / t_total if t_total else 0.0,
    }
