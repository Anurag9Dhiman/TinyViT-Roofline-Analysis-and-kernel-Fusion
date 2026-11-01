#!/usr/bin/env python3
"""Phase 19: does MLX's int8 quantized matmul catch up at larger batch
sizes?

Phase 17 found MLX's int8 quantized matmul is slower than MLX's own fp32
matmul at batch=1, and reasoned that's because dequantization overhead
dominates at these small per-call sizes. That reasoning implies a testable
prediction: at larger batch sizes, where the matmul itself does
proportionally more work per dequantization pass, int8's relative
disadvantage should shrink -- maybe even reverse, since int8's bandwidth
savings should matter more once the op is big enough for bandwidth (not
per-call overhead) to dominate. This sweeps model-level batch_sizes=(1, 4,
16, 32) on `qkv_proj` for one representative block per stage, same real
captured shapes / ported weights / correctness-checking methodology as
Phase 17.

Note on what "batch size" means here: `qkv_proj`'s real input is windowed
tokens, shape (model_batch_size * num_windows, window_size^2, dim) -- not
(model_batch_size, N, dim). Stages with num_windows > 1 (e.g. stage 1 on
5M, num_windows=16) see a much bigger actual GEMM row count than the
model-level batch size suggests. Both are recorded per row
(`model_batch_size` and `effective_rows`) so the real variable the
hypothesis is about -- GEMM size -- isn't confused with the sweep knob
that was turned to vary it.
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
BATCH_SIZES = (1, 4, 16, 32)


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


def bench_linear_at_batch(linear: torch.nn.Linear, x_torch: torch.Tensor, model_batch_size: int) -> dict:
    with torch.no_grad():
        ref = linear(x_torch)
    t_fp32 = _time_torch(lambda: linear(x_torch))

    x_h, w_h, b_h = x_torch.half(), linear.weight.half(), linear.bias.half()
    t_fp16 = _time_torch(lambda: F.linear(x_h, w_h, b_h))

    w = port_linear_weights(linear)
    x_mx = torch_to_mx(x_torch)
    mlx_fp32_fn = mx.compile(lambda inp: inp @ w["weight"].T + w["bias"])
    t_mlx_fp32 = _time_mlx(lambda: mlx_fp32_fn(x_mx))

    wq, scales, biases = mx.quantize(w["weight"], group_size=GROUP_SIZE, bits=BITS)

    def _mlx_int8(inp):
        return mx.quantized_matmul(inp, wq, scales=scales, biases=biases, transpose=True,
                                    group_size=GROUP_SIZE, bits=BITS) + w["bias"]

    mlx_int8_fn = mx.compile(_mlx_int8)
    t_mlx_int8 = _time_mlx(lambda: mlx_int8_fn(x_mx))
    out_mlx_int8 = mlx_int8_fn(x_mx)
    diff_mlx_int8 = float((torch.from_numpy(np.array(out_mlx_int8, dtype=np.float32)).to("mps") - ref).abs().max())

    return {
        "model_batch_size": model_batch_size,
        "effective_rows": x_torch.shape[0],  # qkv_proj's real input leading dim: model_batch_size * num_windows
        "pytorch_fp32_us": t_fp32 * 1e6,
        "pytorch_fp16_us": t_fp16 * 1e6,
        "mlx_fp32_us": t_mlx_fp32 * 1e6,
        "mlx_int8_us": t_mlx_int8 * 1e6,
        "speedup_vs_fp32": {
            "pytorch_fp16": t_fp32 / t_fp16,
            "mlx_fp32": t_fp32 / t_mlx_fp32,
            "mlx_int8": t_fp32 / t_mlx_int8,
        },
        "mlx_int8_vs_mlx_fp32_speedup": t_mlx_fp32 / t_mlx_int8,
        "max_abs_diff_vs_pytorch_fp32": diff_mlx_int8,
    }


def bench_block(model, bi) -> dict:
    block = get_module_by_name(model, bi.module_name)
    rows = []
    for bs in BATCH_SIZES:
        x_full = torch.randn(bs, 3, 224, 224, device="mps")
        qkv_in = _capture_input(model, x_full, block.attn.qkv)
        rows.append(bench_linear_at_batch(block.attn.qkv, qkv_in, model_batch_size=bs))
        torch.mps.empty_cache()
    return {"stage_idx": bi.stage_idx, "dim": bi.dim, "qkv_proj_by_batch": rows}


def run(variant: str) -> dict:
    print(f"[19] loading {variant} ...")
    model = load_baseline_model(variant)
    infos = [bi for bi in enumerate_attention_blocks(model, variant) if bi.block_idx == 0]

    blocks = []
    for bi in infos:
        print(f"[19] stage {bi.stage_idx} (dim={bi.dim}) ...")
        r = bench_block(model, bi)
        for row in r["qkv_proj_by_batch"]:
            su = row["speedup_vs_fp32"]
            print(f"     model_bs={row['model_batch_size']:3d} (effective_rows={row['effective_rows']:4d}): "
                  f"pytorch_fp16={su['pytorch_fp16']:.2f}x  "
                  f"mlx_fp32={su['mlx_fp32']:.2f}x  mlx_int8={su['mlx_int8']:.2f}x  "
                  f"(int8 vs mlx_fp32: {row['mlx_int8_vs_mlx_fp32_speedup']:.2f}x)")
        blocks.append(r)

    by_batch_mean = {}
    for bs in BATCH_SIZES:
        rows = [row for b in blocks for row in b["qkv_proj_by_batch"] if row["model_batch_size"] == bs]
        by_batch_mean[bs] = {
            "mlx_int8_vs_mlx_fp32_speedup": sum(r["mlx_int8_vs_mlx_fp32_speedup"] for r in rows) / len(rows),
            "pytorch_fp16_speedup": sum(r["speedup_vs_fp32"]["pytorch_fp16"] for r in rows) / len(rows),
        }

    # The real variable the hypothesis is about is the GEMM's actual row count (model_batch_size *
    # num_windows, which differs by stage) -- not the model-level batch size. Sort all rows by that
    # directly for the cleanest view of "does int8 catch up as the matmul gets bigger."
    all_rows_by_effective_size = sorted(
        (row for b in blocks for row in b["qkv_proj_by_batch"]),
        key=lambda r: r["effective_rows"],
    )

    return {
        "phase": "19_mlx_quantized_linear_batch_sweep",
        "variant": variant,
        "methodology_note": (
            "Follow-up to Phase 17: tests whether MLX's int8 quantized matmul disadvantage vs. its own "
            "fp32 matmul shrinks at larger batch sizes, where dequantization overhead should matter less "
            "relative to the matmul's total work."
        ),
        "batch_sizes": list(BATCH_SIZES),
        "group_size": GROUP_SIZE,
        "bits": BITS,
        "warmup": WARMUP,
        "iters": ITERS,
        "blocks": blocks,
        "mean_by_batch_size": by_batch_mean,
        "all_rows_by_effective_size": all_rows_by_effective_size,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="tiny_vit_5m_224", choices=list(MODEL_REGISTRY.keys()))
    args = ap.parse_args()

    result = run(args.variant)
    out_path = RAW_DIR / f"19_mlx_quantized_linear_batch_sweep_{args.variant}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[19] mlx_int8 vs mlx_fp32 speedup, sorted by actual GEMM row count:")
    for row in result["all_rows_by_effective_size"]:
        print(f"     effective_rows={row['effective_rows']:5d} (model_bs={row['model_batch_size']}): "
              f"{row['mlx_int8_vs_mlx_fp32_speedup']:.2f}x")
    print(f"[19] wrote {out_path}")
