"""Port a timm `Attention` module's real trained weights into MLX arrays, so
the MLX-fused implementation is numerically comparable to the PyTorch
baseline on the same checkpoint -- not just shape-compatible with random
weights.
"""

from typing import Dict

import mlx.core as mx
import torch


def torch_to_mx(t: torch.Tensor) -> mx.array:
    return mx.array(t.detach().to("cpu", torch.float32).numpy())


def port_attention_weights(attn) -> Dict[str, mx.array]:
    """attn: an instance of timm.models.tiny_vit.Attention (or the identical
    module accessed via block.attn on a full TinyViT)."""
    attn_bias_full = attn.attention_biases[:, attn.attention_bias_idxs]  # (num_heads, N, N), already gathered
    return {
        "norm_weight": torch_to_mx(attn.norm.weight),
        "norm_bias": torch_to_mx(attn.norm.bias),
        "qkv_weight": torch_to_mx(attn.qkv.weight),  # (out, in) -- PyTorch Linear convention
        "qkv_bias": torch_to_mx(attn.qkv.bias),
        "proj_weight": torch_to_mx(attn.proj.weight),
        "proj_bias": torch_to_mx(attn.proj.bias),
        "attention_bias": torch_to_mx(attn_bias_full),  # pre-gathered additive bias, (num_heads, N, N)
    }


def port_conv_norm_fused(conv_norm) -> Dict[str, mx.array]:
    """conv_norm: a timm.models.tiny_vit.ConvNorm (Sequential of Conv2d(bias=False)
    + BatchNorm2d). Fuses BN into the conv at inference time (same math as
    ConvNorm.fuse() in timm's own source) and converts to MLX's NHWC conv2d
    weight layout (C_out, KH, KW, C_in/groups), instead of PyTorch's
    (C_out, C_in/groups, KH, KW)."""
    conv, bn = conv_norm.conv, conv_norm.bn
    with torch.no_grad():
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        fused_weight = conv.weight * scale[:, None, None, None]
        fused_bias = bn.bias - bn.running_mean * scale
    return {
        "weight": torch_to_mx(fused_weight.permute(0, 2, 3, 1)),  # -> (C_out, KH, KW, C_in/groups)
        "bias": torch_to_mx(fused_bias),
        "groups": conv.groups,
        "stride": conv.stride[0],
        "padding": conv.padding[0],
    }


def port_linear_weights(linear) -> Dict[str, mx.array]:
    """linear: a torch.nn.Linear. Weight shape (out_features, in_features) --
    the same convention mx.quantized_matmul/matmul expect for x @ w.T."""
    return {
        "weight": torch_to_mx(linear.weight),
        "bias": torch_to_mx(linear.bias) if linear.bias is not None else None,
    }


def port_mlp_weights(mlp) -> Dict[str, mx.array]:
    """mlp: a timm.models.tiny_vit.NormMlp (norm -> fc1 -> act -> fc2)."""
    return {
        "norm_weight": torch_to_mx(mlp.norm.weight),
        "norm_bias": torch_to_mx(mlp.norm.bias),
        "fc1_weight": torch_to_mx(mlp.fc1.weight),
        "fc1_bias": torch_to_mx(mlp.fc1.bias),
        "fc2_weight": torch_to_mx(mlp.fc2.weight),
        "fc2_bias": torch_to_mx(mlp.fc2.bias),
    }


def port_block_weights(block) -> Dict:
    """block: a timm.models.tiny_vit.TinyVitBlock. Full weight set for
    reimplementing the whole block (window partition -> attention -> window
    reverse -> local_conv -> mlp) in MLX, not just the attention operator."""
    return {
        "attn": port_attention_weights(block.attn),
        "local_conv": port_conv_norm_fused(block.local_conv),
        "mlp": port_mlp_weights(block.mlp),
        "window_size": block.window_size,
        "num_heads": block.attn.num_heads,
        "key_dim": block.attn.key_dim,
        "val_dim": block.attn.val_dim,
    }
