#!/usr/bin/env python3
"""Phase 16: does MLX's primitive slowdown (Phase 15) change at fp16?

Two established findings motivate this: Phase 14 found fp16 is the biggest
single speedup lever in this project for PyTorch-MPS (~1.3-1.5x on top of
the best combination). Phase 15 found MLX's layer_norm/conv2d are ~12-17x
slower than PyTorch-MPS's at fp32, at TinyViT's real operator sizes. Those
two findings were never tested together: if fp16 helps MLX's Metal kernels
more than it helps PyTorch's (plausible -- Metal's fp16 throughput is often
disproportionately higher than fp32's on Apple GPUs), the ~12-17x gap found
at fp32 might narrow substantially at fp16. If it doesn't, that's useful
too -- it rules out "just use fp16" as a fix for MLX's primitive weakness
on this hardware.

Same isolated-microbenchmark methodology as Phase 15 (real captured shapes,
real ported weights, warmup + many iterations + one amortized sync), same
three representative blocks (one per stage). The only difference: every
tensor and weight is cast to float16 (torch) / mx.float16 (MLX) before
timing. Correctness is checked two ways -- fp16-PyTorch vs. fp16-MLX (same
as Phase 15's gate), and fp16 vs. the fp32 reference (characterizes fp16's
own numerical cost here, same as Phase 14's project-wide fp16 correctness
tracking).
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

    # ---- LayerNorm, fp16 on both sides ----
    x_norm_in = _capture_input(model, x_full, attn.norm)
    w_attn = port_attention_weights(attn)

    x_norm_in_h = x_norm_in.half()
    w_h, b_h = attn.norm.weight.half(), attn.norm.bias.half()
    t_ln_torch = _time_torch(lambda: F.layer_norm(x_norm_in_h, (bi.dim,), w_h, b_h, eps=1e-5))

    x_norm_mx_h = torch_to_mx(x_norm_in).astype(mx.float16)
    norm_w_h, norm_b_h = w_attn["norm_weight"].astype(mx.float16), w_attn["norm_bias"].astype(mx.float16)
    ln_mlx_fn = mx.compile(lambda inp: mx.fast.layer_norm(inp, norm_w_h, norm_b_h, eps=1e-5))
    t_ln_mlx = _time_mlx(lambda: ln_mlx_fn(x_norm_mx_h))

    with torch.no_grad():
        ln_out_torch_fp32 = F.layer_norm(x_norm_in, (bi.dim,), attn.norm.weight, attn.norm.bias, eps=1e-5)
        ln_out_torch_h = F.layer_norm(x_norm_in_h, (bi.dim,), w_h, b_h, eps=1e-5)
    ln_out_mlx_h = mx.fast.layer_norm(x_norm_mx_h, norm_w_h, norm_b_h, eps=1e-5)
    ln_mlx_vs_torch_diff = float((torch.from_numpy(np.array(ln_out_mlx_h, dtype=np.float32)).to("mps") - ln_out_torch_h.float()).abs().max())
    ln_fp16_vs_fp32_diff = float((ln_out_torch_h.float() - ln_out_torch_fp32).abs().max())

    # ---- Depthwise conv2d (local_conv, BN fused), fp16 on both sides ----
    x_conv_bchw = _capture_input(model, x_full, block.local_conv)
    conv_fused_torch = block.local_conv.fuse().to("mps")
    conv_fused_torch_h = block.local_conv.fuse().to("mps").half()
    w_conv = port_conv_norm_fused(block.local_conv)
    x_conv_bhwc = x_conv_bchw.permute(0, 2, 3, 1).contiguous()

    x_conv_bchw_h = x_conv_bchw.half()
    t_conv_torch = _time_torch(lambda: conv_fused_torch_h(x_conv_bchw_h))

    x_conv_mx_h = torch_to_mx(x_conv_bhwc).astype(mx.float16)
    conv_w_h, conv_b_h = w_conv["weight"].astype(mx.float16), w_conv["bias"].astype(mx.float16)

    def _mlx_conv_h(inp):
        out = mx.conv2d(inp, conv_w_h, stride=w_conv["stride"], padding=w_conv["padding"], groups=w_conv["groups"])
        return out + conv_b_h

    conv_mlx_fn_h = mx.compile(_mlx_conv_h)
    t_conv_mlx = _time_mlx(lambda: conv_mlx_fn_h(x_conv_mx_h))

    with torch.no_grad():
        conv_out_torch_fp32 = conv_fused_torch(x_conv_bchw).permute(0, 2, 3, 1).contiguous()
        conv_out_torch_h = conv_fused_torch_h(x_conv_bchw_h).permute(0, 2, 3, 1).contiguous()
    conv_out_mlx_h = _mlx_conv_h(x_conv_mx_h)
    conv_mlx_vs_torch_diff = float((torch.from_numpy(np.array(conv_out_mlx_h, dtype=np.float32)).to("mps") - conv_out_torch_h.float()).abs().max())
    conv_fp16_vs_fp32_diff = float((conv_out_torch_h.float() - conv_out_torch_fp32).abs().max())

    return {
        "stage_idx": bi.stage_idx,
        "dim": bi.dim,
        "window_size": bi.window_size,
        "layer_norm": {
            "pytorch_fp16_us": t_ln_torch * 1e6,
            "mlx_fp16_us": t_ln_mlx * 1e6,
            "mlx_vs_pytorch_fp16_speedup": t_ln_torch / t_ln_mlx,
            "mlx_vs_pytorch_fp16_max_abs_diff": ln_mlx_vs_torch_diff,
            "fp16_vs_fp32_max_abs_diff": ln_fp16_vs_fp32_diff,
        },
        "depthwise_conv2d": {
            "pytorch_fp16_kernel_us": t_conv_torch * 1e6,
            "mlx_fp16_us": t_conv_mlx * 1e6,
            "mlx_vs_pytorch_fp16_speedup": t_conv_torch / t_conv_mlx,
            "mlx_vs_pytorch_fp16_max_abs_diff": conv_mlx_vs_torch_diff,
            "fp16_vs_fp32_max_abs_diff": conv_fp16_vs_fp32_diff,
        },
    }


def run(variant: str) -> dict:
    print(f"[16] loading {variant} ...")
    model = load_baseline_model(variant)
    x_full = torch.randn(1, 3, 224, 224, device="mps")
    infos = [bi for bi in enumerate_attention_blocks(model, variant) if bi.block_idx == 0]

    blocks = []
    for bi in infos:
        print(f"[16] stage {bi.stage_idx} (dim={bi.dim}, window_size={bi.window_size}) ...")
        r = bench_block(model, bi, x_full)
        ln, cv = r["layer_norm"], r["depthwise_conv2d"]
        print(f"     layer_norm fp16: pytorch={ln['pytorch_fp16_us']:.2f}us  mlx={ln['mlx_fp16_us']:.2f}us  "
              f"(mlx is {1/ln['mlx_vs_pytorch_fp16_speedup']:.1f}x slower)  "
              f"mlx_vs_torch_diff={ln['mlx_vs_pytorch_fp16_max_abs_diff']:.2e}  "
              f"fp16_vs_fp32_diff={ln['fp16_vs_fp32_max_abs_diff']:.2e}")
        print(f"     conv2d fp16:     pytorch={cv['pytorch_fp16_kernel_us']:.2f}us  mlx={cv['mlx_fp16_us']:.2f}us  "
              f"(mlx is {1/cv['mlx_vs_pytorch_fp16_speedup']:.1f}x slower)  "
              f"mlx_vs_torch_diff={cv['mlx_vs_pytorch_fp16_max_abs_diff']:.2e}  "
              f"fp16_vs_fp32_diff={cv['fp16_vs_fp32_max_abs_diff']:.2e}")
        blocks.append(r)

    mean_ln_slowdown = sum(1 / b["layer_norm"]["mlx_vs_pytorch_fp16_speedup"] for b in blocks) / len(blocks)
    mean_conv_slowdown = sum(1 / b["depthwise_conv2d"]["mlx_vs_pytorch_fp16_speedup"] for b in blocks) / len(blocks)

    # Compare against Phase 15's fp32 numbers for the same variant, if present.
    fp32_path = RAW_DIR / f"15_mlx_primitive_microbench_{variant}.json"
    comparison = None
    if fp32_path.exists():
        fp32 = json.loads(fp32_path.read_text())
        comparison = {
            "layer_norm_mlx_slowdown_fp32": fp32["mean_layer_norm_mlx_slowdown_x"],
            "layer_norm_mlx_slowdown_fp16": mean_ln_slowdown,
            "conv2d_mlx_slowdown_fp32": fp32["mean_conv2d_mlx_slowdown_x_kernel_only"],
            "conv2d_mlx_slowdown_fp16": mean_conv_slowdown,
        }

    return {
        "phase": "16_mlx_fp16_primitive_microbench",
        "variant": variant,
        "methodology_note": (
            "Tests whether Phase 15's MLX-vs-PyTorch primitive slowdown (layer_norm ~14x, conv2d ~12x at "
            "fp32) changes at fp16 -- the precision Phase 14 found to be this project's single biggest "
            "PyTorch-MPS speedup lever. Same isolated microbenchmark methodology as Phase 15."
        ),
        "warmup": WARMUP,
        "iters": ITERS,
        "blocks": blocks,
        "mean_layer_norm_mlx_slowdown_fp16_x": mean_ln_slowdown,
        "mean_conv2d_mlx_slowdown_fp16_x": mean_conv_slowdown,
        "comparison_vs_phase15_fp32": comparison,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="tiny_vit_5m_224", choices=list(MODEL_REGISTRY.keys()))
    args = ap.parse_args()

    result = run(args.variant)
    out_path = RAW_DIR / f"16_mlx_fp16_primitive_microbench_{args.variant}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[16] mean layer_norm slowdown at fp16 (MLX vs PyTorch): {result['mean_layer_norm_mlx_slowdown_fp16_x']:.1f}x")
    print(f"[16] mean conv2d slowdown at fp16 (MLX vs PyTorch): {result['mean_conv2d_mlx_slowdown_fp16_x']:.1f}x")
    if result["comparison_vs_phase15_fp32"]:
        c = result["comparison_vs_phase15_fp32"]
        print(f"[16] layer_norm: fp32={c['layer_norm_mlx_slowdown_fp32']:.1f}x -> fp16={c['layer_norm_mlx_slowdown_fp16']:.1f}x")
        print(f"[16] conv2d:     fp32={c['conv2d_mlx_slowdown_fp32']:.1f}x -> fp16={c['conv2d_mlx_slowdown_fp16']:.1f}x")
    print(f"[16] wrote {out_path}")
