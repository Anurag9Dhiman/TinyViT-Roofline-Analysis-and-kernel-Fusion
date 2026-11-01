#!/usr/bin/env python3
"""Phase 17: does MLX's int8 quantized matmul beat PyTorch-MPS fp16 on the
QKV/MLP linears -- the bottleneck Phase 13 found remains after conv+BN
fusion, and the one Phase 14's fp16 already targets?

Motivation: PyTorch's native quantization ops are not implemented for the
MPS backend at all (`aten::quantize_per_tensor` raises `NotImplementedError`
-- confirmed directly, not assumed), so int8 was never on the table as a
PyTorch-MPS lever in this project. MLX, however, ships `mx.quantize` /
`mx.quantized_matmul` (affine int8/int4 group quantization) that runs on
this same GPU. That's a real asymmetry worth testing: on exactly the op
category (`linear_qkv_proj_or_mlp`) where Phase 15 found MLX's other
primitives (layer_norm, conv2d) are ~12-17x slower than PyTorch's, does
MLX's int8 quantized matmul -- a capability PyTorch-MPS entirely lacks --
close that gap, or even win outright against this project's best existing
lever for this bottleneck (fp16, Phase 14)?

Tests two representative Linears per block (qkv_proj and mlp.fc1 -- the
two largest matmuls in a TinyVitBlock) across the same three stages Phase
15/16 used. Four arms per linear: pytorch_fp32 (baseline), pytorch_fp16
(Phase 14's lever), mlx_fp32 (mx.compile'd matmul, the fair MLX comparison
point per Phase 15's methodology), mlx_int8 (mx.quantize with group_size=32,
bits=8 -- 32 chosen because it's the only group size that evenly divides
every embed_dim across all three model variants; 64 fails on dim=160/96).
Correctness is checked against the pytorch_fp32 reference for all three
non-reference arms, same framework Phase 14/16 already use for fp16's
numerical cost.
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
from tinyvit_roofline.fusion.weight_port import port_linear_weights, torch_to_mx

WARMUP = 20
ITERS = 300
GROUP_SIZE = 32
BITS = 8


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


def bench_linear(linear: torch.nn.Linear, x_torch: torch.Tensor) -> dict:
    with torch.no_grad():
        ref = linear(x_torch)

    t_fp32 = _time_torch(lambda: linear(x_torch))

    x_h, w_h, b_h = x_torch.half(), linear.weight.half(), linear.bias.half()
    with torch.no_grad():
        out_fp16 = F.linear(x_h, w_h, b_h)
    t_fp16 = _time_torch(lambda: F.linear(x_h, w_h, b_h))
    diff_fp16 = float((out_fp16.float() - ref).abs().max())

    w = port_linear_weights(linear)
    x_mx = torch_to_mx(x_torch)
    mlx_fp32_fn = mx.compile(lambda inp: inp @ w["weight"].T + w["bias"])
    t_mlx_fp32 = _time_mlx(lambda: mlx_fp32_fn(x_mx))
    out_mlx_fp32 = mlx_fp32_fn(x_mx)
    diff_mlx_fp32 = float((torch.from_numpy(np.array(out_mlx_fp32, dtype=np.float32)).to("mps") - ref).abs().max())

    wq, scales, biases = mx.quantize(w["weight"], group_size=GROUP_SIZE, bits=BITS)

    def _mlx_int8(inp):
        return mx.quantized_matmul(inp, wq, scales=scales, biases=biases, transpose=True,
                                    group_size=GROUP_SIZE, bits=BITS) + w["bias"]

    mlx_int8_fn = mx.compile(_mlx_int8)
    t_mlx_int8 = _time_mlx(lambda: mlx_int8_fn(x_mx))
    out_mlx_int8 = mlx_int8_fn(x_mx)
    diff_mlx_int8 = float((torch.from_numpy(np.array(out_mlx_int8, dtype=np.float32)).to("mps") - ref).abs().max())

    return {
        "input_shape": list(x_torch.shape),
        "out_features": linear.out_features,
        "in_features": linear.in_features,
        "pytorch_fp32_us": t_fp32 * 1e6,
        "pytorch_fp16_us": t_fp16 * 1e6,
        "mlx_fp32_us": t_mlx_fp32 * 1e6,
        "mlx_int8_us": t_mlx_int8 * 1e6,
        "speedup_vs_fp32": {
            "pytorch_fp16": t_fp32 / t_fp16,
            "mlx_fp32": t_fp32 / t_mlx_fp32,
            "mlx_int8": t_fp32 / t_mlx_int8,
        },
        "max_abs_diff_vs_pytorch_fp32": {
            "pytorch_fp16": diff_fp16,
            "mlx_fp32": diff_mlx_fp32,
            "mlx_int8": diff_mlx_int8,
        },
    }


def bench_block(model, bi) -> dict:
    block = get_module_by_name(model, bi.module_name)
    x_full = torch.randn(1, 3, 224, 224, device="mps")

    qkv_in = _capture_input(model, x_full, block.attn.qkv)
    fc1_in = _capture_input(model, x_full, block.mlp.fc1)

    return {
        "stage_idx": bi.stage_idx,
        "dim": bi.dim,
        "qkv_proj": bench_linear(block.attn.qkv, qkv_in),
        "mlp_fc1": bench_linear(block.mlp.fc1, fc1_in),
    }


def _print_linear(name: str, r: dict) -> None:
    su = r["speedup_vs_fp32"]
    diff = r["max_abs_diff_vs_pytorch_fp32"]
    print(f"     {name}: pytorch_fp32={r['pytorch_fp32_us']:.2f}us  pytorch_fp16={r['pytorch_fp16_us']:.2f}us "
          f"({su['pytorch_fp16']:.2f}x)  mlx_fp32={r['mlx_fp32_us']:.2f}us ({su['mlx_fp32']:.2f}x)  "
          f"mlx_int8={r['mlx_int8_us']:.2f}us ({su['mlx_int8']:.2f}x)")
    print(f"       max_abs_diff vs pytorch_fp32: fp16={diff['pytorch_fp16']:.2e}  "
          f"mlx_fp32={diff['mlx_fp32']:.2e}  mlx_int8={diff['mlx_int8']:.2e}")


def run(variant: str) -> dict:
    print(f"[17] loading {variant} ...")
    model = load_baseline_model(variant)
    infos = [bi for bi in enumerate_attention_blocks(model, variant) if bi.block_idx == 0]

    blocks = []
    for bi in infos:
        print(f"[17] stage {bi.stage_idx} (dim={bi.dim}) ...")
        r = bench_block(model, bi)
        _print_linear("qkv_proj", r["qkv_proj"])
        _print_linear("mlp_fc1 ", r["mlp_fc1"])
        blocks.append(r)

    def _mean(key, arm):
        vals = [b[key]["speedup_vs_fp32"][arm] for b in blocks]
        return sum(vals) / len(vals)

    summary = {}
    for key in ("qkv_proj", "mlp_fc1"):
        summary[key] = {arm: _mean(key, arm) for arm in ("pytorch_fp16", "mlx_fp32", "mlx_int8")}

    return {
        "phase": "17_mlx_quantized_linear_microbench",
        "variant": variant,
        "methodology_note": (
            "PyTorch's native quantization ops (aten::quantize_per_tensor) are not implemented for the "
            "MPS backend (confirmed directly) -- so int8 was never a PyTorch-MPS option here. MLX's "
            "mx.quantize/mx.quantized_matmul run on the same GPU. Tests whether that gives MLX a real "
            "edge on the linear_qkv_proj_or_mlp category Phase 13 found to be the post-conv-fusion "
            "bottleneck, against pytorch_fp16 (Phase 14's existing lever for the same bottleneck)."
        ),
        "group_size": GROUP_SIZE,
        "bits": BITS,
        "warmup": WARMUP,
        "iters": ITERS,
        "blocks": blocks,
        "mean_speedup_vs_pytorch_fp32": summary,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="tiny_vit_5m_224", choices=list(MODEL_REGISTRY.keys()))
    args = ap.parse_args()

    result = run(args.variant)
    out_path = RAW_DIR / f"17_mlx_quantized_linear_microbench_{args.variant}.json"
    out_path.write_text(json.dumps(result, indent=2))
    s = result["mean_speedup_vs_pytorch_fp32"]
    print(f"\n[17] mean speedup vs pytorch_fp32:")
    for key in ("qkv_proj", "mlp_fc1"):
        print(f"     {key}: pytorch_fp16={s[key]['pytorch_fp16']:.2f}x  mlx_fp32={s[key]['mlx_fp32']:.2f}x  "
              f"mlx_int8={s[key]['mlx_int8']:.2f}x")
    print(f"[17] wrote {out_path}")
