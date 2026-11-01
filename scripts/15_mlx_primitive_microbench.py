#!/usr/bin/env python3
"""Phase 15: isolated PyTorch-MPS vs MLX microbenchmark for LayerNorm and
depthwise conv2d, at TinyViT's real operator sizes.

Why this script exists: the project's README/report both state, as a load-
bearing explanation for why full-block MLX fusion (fusion/mlx_fused_block.py)
doesn't help, that "mx.fast.layer_norm measured ~10x slower than PyTorch's
LayerNorm at these shapes, and MLX's depthwise conv2d ~3x slower." That claim
was never backed by a committed script or a results/raw/*.json -- every
other number in this project is reproducible by construction (that's the
whole methodology), so an unverified claim sitting in the "key findings"
list is itself a bug. This script measures it properly, the same isolated-
microbenchmark way as roofline/op_bench.py and ablation/run_ablation.py: a
real captured input tensor (correct shape/dtype), real ported checkpoint
weights, warmup, many iterations, one amortized sync (torch.mps.synchronize
for PyTorch, mx.eval per call for MLX -- same pattern already used and
validated in ablation/run_ablation.py's MLX timing).

Tests one representative TinyVitBlock per stage (stage1/dim128/ws7,
stage2/dim160/ws14, stage3/dim320/ws7 on tiny_vit_5m_224) -- the same three
configs enumerate_attention_blocks finds for this variant's block_idx=0.

LayerNorm: compared on the real windowed-attention input shape
(B*num_windows, window_size^2, dim) -- identical layout for both frameworks,
no permute involved either side.

Depthwise conv2d (local_conv, BatchNorm fused into the conv exactly as
conv_bn_fusion.py does -- not an approximate comparison): PyTorch gets its
real captured BCHW input (local_conv's actual received tensor, post the
block's own `x.permute(0,3,1,2)`); MLX gets the inverse-permuted BHWC
tensor, since MLX's conv2d is natively NHWC and never needs that permute.
Reported two ways -- "kernel only" (isolates the claim being tested) and
"kernel + required PyTorch layout permute" (the real-world cost the
baseline actually pays, same "layout" category op_classify.py already
tracks separately from "conv").
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import torch.nn.functional as F
import mlx.core as mx

from tinyvit_roofline.config import MODEL_REGISTRY, RAW_DIR
from tinyvit_roofline.models import load_baseline_model, enumerate_attention_blocks, get_module_by_name
from tinyvit_roofline.fusion.weight_port import port_attention_weights, port_conv_norm_fused, torch_to_mx
from tinyvit_roofline.fusion.mlx_fused_attn import build_mlx_attention

WARMUP = 20
ITERS = 300


def _time_torch(fn, warmup: int = WARMUP, iters: int = ITERS) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters


def _time_mlx(fn, warmup: int = WARMUP, iters: int = ITERS) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    return (time.perf_counter() - t0) / iters


def _capture_input(model, x_full, target_module) -> torch.Tensor:
    captured = {}

    def hook(mod, inp, out):
        captured["x"] = inp[0].detach().clone()

    h = target_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(x_full)
    finally:
        h.remove()
    return captured["x"]


def bench_block(model, bi, x_full) -> dict:
    block = get_module_by_name(model, bi.module_name)
    attn = block.attn

    # ---- LayerNorm (attn.norm): real windowed-token input, shape (B*nW, ws^2, dim) ----
    x_norm_in = _capture_input(model, x_full, attn.norm)
    w_attn = port_attention_weights(attn)
    x_norm_mx = torch_to_mx(x_norm_in)

    t_ln_torch = _time_torch(lambda: F.layer_norm(x_norm_in, (bi.dim,), attn.norm.weight, attn.norm.bias, eps=1e-5))

    ln_mlx_fn = mx.compile(lambda inp: mx.fast.layer_norm(inp, w_attn["norm_weight"], w_attn["norm_bias"], eps=1e-5))
    t_ln_mlx = _time_mlx(lambda: ln_mlx_fn(x_norm_mx))

    with torch.no_grad():
        ln_out_torch = F.layer_norm(x_norm_in, (bi.dim,), attn.norm.weight, attn.norm.bias, eps=1e-5)
    ln_out_mlx = mx.fast.layer_norm(x_norm_mx, w_attn["norm_weight"], w_attn["norm_bias"], eps=1e-5)
    ln_diff = float((torch.from_numpy(np.array(ln_out_mlx)).to("mps") - ln_out_torch).abs().max())

    # ---- mx.compile vs. uncompiled re-tracing, on the real attention module (same windowed input) ----
    # Verifies another unbacked claim (mlx_fused_attn.py's docstring / Key Finding 5): an uncompiled MLX
    # call "measured ~4x slower" than the mx.compile-wrapped one, with no script or results/raw/*.json
    # behind that number either, same bug class as the layer_norm/conv2d claim this script already fixes.
    mlx_attn = build_mlx_attention(attn)
    t_attn_uncompiled = _time_mlx(lambda: mlx_attn(x_norm_mx))
    mlx_attn_compiled = mlx_attn.compiled()
    t_attn_compiled = _time_mlx(lambda: mlx_attn_compiled(x_norm_mx))

    # ---- Depthwise conv2d (local_conv, BN fused): real BCHW input as the block actually sees it ----
    x_conv_bchw = _capture_input(model, x_full, block.local_conv)  # local_conv's real post-permute input
    conv_fused_torch = block.local_conv.fuse().to("mps")  # pure function: does NOT mutate block.local_conv
    w_conv = port_conv_norm_fused(block.local_conv)
    x_conv_bhwc = x_conv_bchw.permute(0, 2, 3, 1).contiguous()  # what MLX's native-NHWC conv2d would receive
    x_conv_mx = torch_to_mx(x_conv_bhwc)

    t_conv_torch_kernel = _time_torch(lambda: conv_fused_torch(x_conv_bchw))
    # Real-world sequence: the block holds BHWC, permutes to BCHW, then convs -- exactly what
    # x.permute(0, 3, 1, 2) does in timm's TinyVitBlock.forward right before calling local_conv.
    t_conv_torch_with_permute = _time_torch(
        lambda: conv_fused_torch(x_conv_bhwc.permute(0, 3, 1, 2).contiguous())
    )

    def _mlx_conv(inp):
        out = mx.conv2d(inp, w_conv["weight"], stride=w_conv["stride"], padding=w_conv["padding"], groups=w_conv["groups"])
        return out + w_conv["bias"]

    conv_mlx_fn = mx.compile(_mlx_conv)
    t_conv_mlx = _time_mlx(lambda: conv_mlx_fn(x_conv_mx))

    with torch.no_grad():
        conv_out_torch = conv_fused_torch(x_conv_bchw).permute(0, 2, 3, 1).contiguous()
    conv_out_mlx = _mlx_conv(x_conv_mx)
    conv_diff = float((torch.from_numpy(np.array(conv_out_mlx)).to("mps") - conv_out_torch).abs().max())

    return {
        "stage_idx": bi.stage_idx,
        "dim": bi.dim,
        "window_size": bi.window_size,
        "layer_norm": {
            "input_shape": list(x_norm_in.shape),
            "pytorch_us": t_ln_torch * 1e6,
            "mlx_us": t_ln_mlx * 1e6,
            "mlx_vs_pytorch_speedup": t_ln_torch / t_ln_mlx,
            "max_abs_diff": ln_diff,
        },
        "mlx_attn_compile_benefit": {
            "input_shape": list(x_norm_in.shape),
            "uncompiled_us": t_attn_uncompiled * 1e6,
            "compiled_us": t_attn_compiled * 1e6,
            "compiled_speedup": t_attn_uncompiled / t_attn_compiled,
        },
        "depthwise_conv2d": {
            "input_shape_bchw": list(x_conv_bchw.shape),
            "pytorch_kernel_only_us": t_conv_torch_kernel * 1e6,
            "pytorch_kernel_plus_layout_roundtrip_us": t_conv_torch_with_permute * 1e6,
            "mlx_us": t_conv_mlx * 1e6,
            "mlx_vs_pytorch_kernel_only_speedup": t_conv_torch_kernel / t_conv_mlx,
            "mlx_vs_pytorch_kernel_plus_layout_speedup": t_conv_torch_with_permute / t_conv_mlx,
            "max_abs_diff": conv_diff,
        },
    }


def run(variant: str) -> dict:
    print(f"[15] loading {variant} ...")
    model = load_baseline_model(variant)
    x_full = torch.randn(1, 3, 224, 224, device="mps")
    infos = [bi for bi in enumerate_attention_blocks(model, variant) if bi.block_idx == 0]

    blocks = []
    for bi in infos:
        print(f"[15] stage {bi.stage_idx} (dim={bi.dim}, window_size={bi.window_size}) ...")
        r = bench_block(model, bi, x_full)
        ln, cv = r["layer_norm"], r["depthwise_conv2d"]
        print(f"     layer_norm: pytorch={ln['pytorch_us']:.2f}us  mlx={ln['mlx_us']:.2f}us  "
              f"(mlx is {1/ln['mlx_vs_pytorch_speedup']:.1f}x slower)  max_abs_diff={ln['max_abs_diff']:.2e}")
        print(f"     conv2d:     pytorch(kernel)={cv['pytorch_kernel_only_us']:.2f}us  "
              f"pytorch(+layout)={cv['pytorch_kernel_plus_layout_roundtrip_us']:.2f}us  mlx={cv['mlx_us']:.2f}us  "
              f"(mlx is {1/cv['mlx_vs_pytorch_kernel_only_speedup']:.1f}x slower vs. kernel-only)  "
              f"max_abs_diff={cv['max_abs_diff']:.2e}")
        mc = r["mlx_attn_compile_benefit"]
        print(f"     mlx_compile: uncompiled={mc['uncompiled_us']:.2f}us  compiled={mc['compiled_us']:.2f}us  "
              f"(compiling is {mc['compiled_speedup']:.1f}x faster than re-tracing)")
        blocks.append(r)

    mean_ln_slowdown = sum(1 / b["layer_norm"]["mlx_vs_pytorch_speedup"] for b in blocks) / len(blocks)
    mean_conv_slowdown = sum(1 / b["depthwise_conv2d"]["mlx_vs_pytorch_kernel_only_speedup"] for b in blocks) / len(blocks)
    mean_compile_speedup = sum(b["mlx_attn_compile_benefit"]["compiled_speedup"] for b in blocks) / len(blocks)

    return {
        "phase": "15_mlx_primitive_microbench",
        "variant": variant,
        "methodology_note": (
            "Verifies two claims in README.md/report.md that had no backing results/raw/*.json before "
            "this script existed: (1) full-block MLX fusion doesn't pay off because mx.fast.layer_norm "
            "and MLX's depthwise conv2d are slower than PyTorch-MPS at these shapes, and (2) an "
            "uncompiled MLX call re-traces the op graph and is markedly slower than the mx.compile-"
            "wrapped version."
        ),
        "warmup": WARMUP,
        "iters": ITERS,
        "blocks": blocks,
        "mean_layer_norm_mlx_slowdown_x": mean_ln_slowdown,
        "mean_conv2d_mlx_slowdown_x_kernel_only": mean_conv_slowdown,
        "mean_mlx_compile_speedup_x": mean_compile_speedup,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="tiny_vit_5m_224", choices=list(MODEL_REGISTRY.keys()))
    args = ap.parse_args()

    result = run(args.variant)
    out_path = RAW_DIR / f"15_mlx_primitive_microbench_{args.variant}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[15] mean layer_norm slowdown (MLX vs PyTorch): {result['mean_layer_norm_mlx_slowdown_x']:.1f}x")
    print(f"[15] mean conv2d slowdown (MLX vs PyTorch, kernel only): {result['mean_conv2d_mlx_slowdown_x_kernel_only']:.1f}x")
    print(f"[15] mean mx.compile speedup over uncompiled re-tracing: {result['mean_mlx_compile_speedup_x']:.1f}x")
    print(f"[15] wrote {out_path}")
