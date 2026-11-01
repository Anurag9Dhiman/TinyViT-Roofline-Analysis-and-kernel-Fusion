"""Full MLX-fused TinyVitBlock: window partition -> attention -> window
reverse -> local_conv -> mlp, entirely in MLX (real weights ported from the
matching PyTorch block).

Why go beyond just the attention operator (mlx_fused_attn.py)? The roofline
diagnostics (Phase 3) found the attention core was already reasonably
efficient in the standard PyTorch baseline (single fused QKV linear,
F.scaled_dot_product_attention) -- fusing *only* that part left the actually
worst-utilized ops untouched: local_conv (1-13% of its roofline bound) and
the window partition/reverse reshapes. Wrapping just the attention call also
pays a PyTorch<->MLX round trip while leaving local_conv/mlp/reshapes still
running in PyTorch eager, unfused, exactly as in the baseline -- so it could
never touch the ablation's actual bottleneck. This module fuses the whole
block (window partition through mlp) as one MLX-compiled region: one round
trip per block instead of one per attention call, and mx.compile gets a
chance to fuse the reshape/conv/mlp chain the way it fused nothing before.
"""

from typing import Dict

import mlx.core as mx
import mlx.nn as nn


class MLXTinyVitBlock:
    def __init__(self, weights: Dict):
        self.w = weights
        aw = weights["attn"]
        self.attn_w = aw
        self.num_heads = weights["num_heads"]
        self.key_dim = weights["key_dim"]
        self.val_dim = weights["val_dim"]
        self.out_dim = self.val_dim * self.num_heads
        self.scale = self.key_dim ** -0.5
        self.window_size = weights["window_size"]
        self.lc = weights["local_conv"]
        self.mlp_w = weights["mlp"]

    def _attn(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        w = self.attn_w
        x_norm = mx.fast.layer_norm(x, w["norm_weight"], w["norm_bias"], eps=1e-5)
        qkv = x_norm @ w["qkv_weight"].T + w["qkv_bias"]
        qkv = qkv.reshape(B, N, self.num_heads, -1)
        q, k, v = mx.split(qkv, [self.key_dim, 2 * self.key_dim], axis=3)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=w["attention_bias"])
        out = out.transpose(0, 2, 1, 3).reshape(B, N, self.out_dim)
        return out @ w["proj_weight"].T + w["proj_bias"]

    def _local_conv(self, x: mx.array) -> mx.array:
        # x: (B, H, W, C) -- MLX conv2d is natively NHWC, unlike PyTorch's
        # NCHW, so no permute dance is needed here (the baseline permutes to
        # BCHW and back; that's real, measured "layout" overhead this
        # implementation simply doesn't pay).
        out = mx.conv2d(x, self.lc["weight"], stride=self.lc["stride"],
                         padding=self.lc["padding"], groups=self.lc["groups"])
        return out + self.lc["bias"]

    def _mlp(self, x: mx.array) -> mx.array:
        w = self.mlp_w
        x_norm = mx.fast.layer_norm(x, w["norm_weight"], w["norm_bias"], eps=1e-5)
        h = x_norm @ w["fc1_weight"].T + w["fc1_bias"]
        h = nn.gelu(h)
        return h @ w["fc2_weight"].T + w["fc2_bias"]

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, C = x.shape
        ws = self.window_size
        shortcut = x

        if H == ws and W == ws:
            x = x.reshape(B, H * W, C)
            x = self._attn(x)
            x = x.reshape(B, H, W, C)
        else:
            pad_b = (ws - H % ws) % ws
            pad_r = (ws - W % ws) % ws
            xp = mx.pad(x, [(0, 0), (0, pad_b), (0, pad_r), (0, 0)]) if (pad_b or pad_r) else x
            pH, pW = H + pad_b, W + pad_r
            nH, nW = pH // ws, pW // ws
            windows = xp.reshape(B, nH, ws, nW, ws, C).transpose(0, 1, 3, 2, 4, 5).reshape(B * nH * nW, ws * ws, C)
            windows = self._attn(windows)
            xp = windows.reshape(B, nH, nW, ws, ws, C).transpose(0, 1, 3, 2, 4, 5).reshape(B, pH, pW, C)
            x = xp[:, :H, :W] if (pad_b or pad_r) else xp

        x = shortcut + x  # residual 1 (drop_path is identity at inference)
        x = self._local_conv(x)  # replaces x (not an extra residual -- matches TinyVitBlock.forward)
        x_flat = x.reshape(B, H * W, C)
        x_flat = x_flat + self._mlp(x_flat)  # residual 2
        return x_flat.reshape(B, H, W, C)

    def compiled(self):
        if not hasattr(self, "_compiled_fn"):
            self._compiled_fn = mx.compile(lambda inp: self(inp))
        return self._compiled_fn


def build_mlx_block(torch_block) -> MLXTinyVitBlock:
    from .weight_port import port_block_weights

    return MLXTinyVitBlock(port_block_weights(torch_block))
