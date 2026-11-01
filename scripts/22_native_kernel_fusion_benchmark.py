#!/usr/bin/env python3
"""Phase 22: does kernel fusion actually beat baseline, if you bypass MLX
entirely?

Phases 6/15/16/17 established that MLX-based kernel fusion -- whether
MLX's own built-in primitives or a hand-written mx.fast.metal_kernel --
never beats PyTorch-MPS baseline at TinyViT's operator sizes, because
MLX's Python->Metal dispatch path has a large fixed per-call floor
(~300-400us) that dominates regardless of how fused the compute is.

This tests the one remaining lever: a hand-written Metal kernel (same
fused LayerNorm + QKV projection + attention + output projection as the
earlier MLX attempt), dispatched through PyTorch's OWN native MPS command
stream instead (fusion/native/fused_window_attention.mm, loaded via
torch.utils.cpp_extension). That avoids MLX's dispatch floor entirely by
reusing PyTorch's own low-overhead command encoding.

Fixed for window_size=4, dim=64 -- the ablation's (section 6) worst-case
config, where every prior fusion attempt lost ~2x to baseline. Swept
across the project's own batch_size grid (model-level batch_size=1,4,16,32
-> batch_windows=16,64,256,512, with REPRESENTATIVE_NUM_WINDOWS=16
matching run_ablation.py) plus a finer sweep near the crossover point
found during development (~bw=32-40), since the honest result is that
this wins at the smallest size and loses above it -- not an unconditional
win, and that boundary is itself the finding worth reporting precisely.

Also checks whether this win transfers to any REAL TinyViT block (not
just the synthetic ws=4/dim=64 ablation-grid point): tests 5M's actual
stage-1 attention module (window_size=7, dim=128). That config needs
135KB of threadgroup memory in the all-threadgroup-memory kernel design --
4x over the M4's hard 32768-byte limit (confirmed directly) -- so it uses
a second kernel variant with the large intermediates in device memory
instead. Spoiler: it doesn't transfer -- see run_real_config_check().
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from tinyvit_roofline.config import RAW_DIR
from tinyvit_roofline.microbench.window_attention_bench import build_attention
from tinyvit_roofline.fusion.native_metal_attn import NativeFusedWindowAttention, NativeFusedWindowAttentionV2
from tinyvit_roofline.models import load_baseline_model, enumerate_attention_blocks, get_module_by_name

WINDOW_SIZE = 4
DIM = 64
N = WINDOW_SIZE * WINDOW_SIZE
REPRESENTATIVE_NUM_WINDOWS = 16  # matches ablation/run_ablation.py
MODEL_BATCH_SIZES = (1, 4, 16, 32)  # matches config.py's BATCH_SIZES
WARMUP = 20
ITERS = 200


def _time_call(fn, warmup: int = WARMUP, iters: int = ITERS) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters


def run_real_config_check() -> dict:
    """Does the native-kernel-fusion win transfer to an actual TinyViT
    block, not just the synthetic window_size=4/dim=64 ablation-grid
    point? Tests 5M's real stage-1 attention module (window_size=7,
    dim=128, num_heads=4) -- the config doesn't fit the all-threadgroup-
    memory kernel (135KB needed vs. the M4's hard 32768-byte limit,
    confirmed directly via MTLDevice.maxThreadgroupMemoryLength), so this
    uses the device-memory-scratch v2 kernel instead."""
    model = load_baseline_model("tiny_vit_5m_224")
    infos = enumerate_attention_blocks(model, "tiny_vit_5m_224")
    bi = [b for b in infos if b.stage_idx == 1 and b.block_idx == 0][0]
    block = get_module_by_name(model, bi.module_name)
    attn = block.attn
    native_v2 = NativeFusedWindowAttentionV2(attn, bi.window_size, bi.dim)
    n = bi.window_size * bi.window_size

    x_check = torch.randn(16, n, bi.dim, device="mps").contiguous()
    with torch.no_grad():
        ref = attn(x_check)
        out = native_v2(x_check)
    torch.mps.synchronize()
    max_abs_diff = (out - ref).abs().max().item()
    print(f"[22]   correctness @ bw=16 (real config): max_abs_diff={max_abs_diff:.2e}")

    results = []
    for bw in (16, 32, 64, 128, 256, 512):
        x = torch.randn(bw, n, bi.dim, device="mps").contiguous()
        t_baseline = _time_call(lambda: attn(x))
        t_native = _time_call(lambda: native_v2(x))
        speedup = t_baseline / t_native
        print(f"[22]   batch_windows={bw:4d}: baseline={t_baseline*1e6:8.2f}us  "
              f"native_v2={t_native*1e6:8.2f}us  speedup={speedup:.2f}x")
        results.append({"batch_windows": bw, "baseline_us": t_baseline * 1e6,
                         "native_us": t_native * 1e6, "speedup": speedup})

    return {
        "config": {"window_size": bi.window_size, "dim": bi.dim, "num_heads": attn.num_heads,
                   "key_dim": attn.key_dim, "val_dim": attn.val_dim,
                   "threadgroup_memory_bytes_needed_all_threadgroup_design":
                       n * bi.dim * 4 + n * attn.num_heads * (2 * attn.key_dim + attn.val_dim) * 4 + n * n * 4 + n * attn.out_dim * 4,
                   "m4_threadgroup_memory_limit_bytes": 32768},
        "correctness": {"max_abs_diff": max_abs_diff},
        "results": results,
        "wins_at_any_size": any(r["speedup"] > 1.0 for r in results),
    }


def run() -> dict:
    print(f"[22] loading native extension (compiles on first run) ...")
    attn = build_attention(DIM, WINDOW_SIZE)
    native = NativeFusedWindowAttention(attn)

    x_check = torch.randn(16, N, DIM, device="mps").contiguous()
    with torch.no_grad():
        ref = attn(x_check)
        out = native(x_check)
    torch.mps.synchronize()
    max_abs_diff = (out - ref).abs().max().item()
    top1_match = bool((out.argmax(-1) == ref.argmax(-1)).all().item()) if out.ndim > 1 else None
    print(f"[22] correctness @ bw=16: max_abs_diff={max_abs_diff:.2e}")

    results = []
    batch_windows_points = sorted(set(
        [bs * REPRESENTATIVE_NUM_WINDOWS for bs in MODEL_BATCH_SIZES] + [20, 24, 28, 32, 40, 48, 64]
    ))
    for bw in batch_windows_points:
        x = torch.randn(bw, N, DIM, device="mps").contiguous()
        t_baseline = _time_call(lambda: attn(x))
        t_native = _time_call(lambda: native(x))
        speedup = t_baseline / t_native
        print(f"[22] batch_windows={bw:4d}: baseline={t_baseline*1e6:8.2f}us  "
              f"native={t_native*1e6:8.2f}us  speedup={speedup:.2f}x")
        results.append({
            "batch_windows": bw,
            "is_representative_grid_point": bw in [bs * REPRESENTATIVE_NUM_WINDOWS for bs in MODEL_BATCH_SIZES],
            "baseline_us": t_baseline * 1e6,
            "native_us": t_native * 1e6,
            "speedup": speedup,
        })

    crossover = next((r["batch_windows"] for r in results if r["speedup"] < 1.0), None)

    print("\n[22] does this win transfer to a REAL TinyViT block (not the synthetic ws=4/dim=64 "
          "ablation-grid point)? Testing 5M's stage-1 block: window_size=7, dim=128 ...")
    real_result = run_real_config_check()

    return {
        "phase": "22_native_kernel_fusion_benchmark",
        "config": {"window_size": WINDOW_SIZE, "dim": DIM, "num_heads": attn.num_heads,
                   "key_dim": attn.key_dim, "val_dim": attn.val_dim},
        "methodology_note": (
            "Native PyTorch MPS C++ extension (torch.utils.cpp_extension), not MLX -- see "
            "fusion/native/fused_window_attention.mm's header for why. Isolated microbenchmark, "
            "same warmup+iters+sync methodology as the rest of the project."
        ),
        "correctness": {"max_abs_diff": max_abs_diff, "top1_match": top1_match},
        "results": results,
        "crossover_batch_windows": crossover,
        "mean_speedup_at_representative_grid_points": sum(
            r["speedup"] for r in results if r["is_representative_grid_point"]
        ) / sum(1 for r in results if r["is_representative_grid_point"]),
        "real_config_check": real_result,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.parse_args()

    result = run()
    out_path = RAW_DIR / "22_native_kernel_fusion_benchmark.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[22] crossover point (first batch_windows where native loses): {result['crossover_batch_windows']}")
    print(f"[22] mean speedup at representative grid points (bw=16,64,256,512): "
          f"{result['mean_speedup_at_representative_grid_points']:.2f}x")
    print(f"[22] does the win transfer to a real TinyViT block? "
          f"{'yes' if result['real_config_check']['wins_at_any_size'] else 'no'}")
    print(f"[22] wrote {out_path}")
