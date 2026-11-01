#!/usr/bin/env python3
"""Phase 9: Conv+BatchNorm fusion -- a real, working optimization.

Follows directly from the roofline diagnostics (Phase 1: "conv" is the
single largest dispatch-time category) rather than from the window-attention
hypothesis the rest of the project centers on. Included because the
attention-focused optimizations (Phases 5/6: torch.compile and MLX-fused
attention) did not reliably beat the standard baseline -- this is the
project's actual positive speedup result, using a standard, exact,
well-understood technique instead of an experimental one.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from tinyvit_roofline.config import MODEL_REGISTRY, RAW_DIR
from tinyvit_roofline.models import load_baseline_model
from tinyvit_roofline.fusion.conv_bn_fusion import fuse_conv_norms


def time_forward(model, x, warmup=10, iters=50):
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters


def run(variant: str, batch_sizes) -> dict:
    print(f"[09] loading {variant} baseline + conv-BN-fused copy ...")
    baseline = load_baseline_model(variant)
    fused = load_baseline_model(variant)
    n_fused = fuse_conv_norms(fused)
    fused.to("mps")
    print(f"[09] fused {n_fused} ConvNorm modules (patch_embed, stage0 MBConv, PatchMerging, local_conv)")

    x_check = torch.randn(1, 3, 224, 224, device="mps")
    with torch.no_grad():
        out_b = baseline(x_check)
        out_f = fused(x_check)
    max_abs_diff = (out_b - out_f).abs().max().item()
    top1_match = bool((out_b.argmax(-1) == out_f.argmax(-1)).all().item())
    print(f"[09] correctness: max_abs_diff={max_abs_diff:.2e}  top1_match={top1_match}")

    results = []
    for bs in batch_sizes:
        x = torch.randn(bs, 3, 224, 224, device="mps")
        t_baseline = time_forward(baseline, x)
        t_fused = time_forward(fused, x)
        speedup = t_baseline / t_fused
        print(f"[09] batch={bs:3d}: baseline={t_baseline*1e3:8.3f}ms  conv_fused={t_fused*1e3:8.3f}ms  speedup={speedup:.2f}x")
        results.append({"batch_size": bs, "baseline_ms": t_baseline * 1e3, "conv_fused_ms": t_fused * 1e3, "speedup": speedup})

    return {
        "phase": "09_conv_fusion_benchmark",
        "variant": variant,
        "num_conv_norms_fused": n_fused,
        "correctness": {"max_abs_diff": max_abs_diff, "top1_match": top1_match},
        "results": results,
        "mean_speedup": sum(r["speedup"] for r in results) / len(results),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="tiny_vit_5m_224", choices=list(MODEL_REGISTRY.keys()))
    args = ap.parse_args()

    result = run(args.variant, [1, 4, 16, 32])
    out_path = RAW_DIR / f"09_conv_fusion_benchmark_{args.variant}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[09] mean speedup: {result['mean_speedup']:.2f}x")
    print(f"[09] wrote {out_path}")
