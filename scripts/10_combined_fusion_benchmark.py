#!/usr/bin/env python3
"""Phase 10: Does conv+BN fusion (Phase 9, ~1.1-1.3x) stack with MLX-fused
attention (fusion/hybrid_model.py, roughly parity/~0.96x standalone in the
full-model end-to-end check)? The two touch disjoint submodules -- conv+BN
fusion rewrites patch_embed/stage0/PatchMerging/local_conv, MLX routing
only replaces each TinyVitBlock's attn.forward -- so they should compose
without conflict. This measures whether they actually do, and whether the
combined speedup is roughly multiplicative, or whether MLX's per-call
PyTorch<->MLX round trip (fusion/hybrid_model.py's documented overhead)
eats into the gain conv fusion provides.

Uses the same single-process, two-model (baseline then combined), many-
iteration-per-batch-size methodology as scripts/09_conv_fusion_benchmark.py
-- already cross-validated against fully separate-process runs for that
exact pattern (see that script's/Phase 7's notes). Not the four-model
sequential pattern that triggered thermal-throttling confounds.
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
from tinyvit_roofline.fusion.hybrid_model import wrap_blocks_with_mlx_attention


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


def build_combined(variant: str):
    model = load_baseline_model(variant)
    n_conv = fuse_conv_norms(model)
    model.to("mps")
    wrapped = wrap_blocks_with_mlx_attention(model)
    return model, n_conv, len(wrapped)


def run(variant: str, batch_sizes) -> dict:
    print(f"[10] loading {variant} baseline + combined (conv-BN-fused + MLX-attention) copy ...")
    baseline = load_baseline_model(variant)
    combined, n_conv, n_mlx_blocks = build_combined(variant)
    print(f"[10] combined model: {n_conv} ConvNorm modules fused, {n_mlx_blocks} blocks routed through MLX attention")

    x_check = torch.randn(1, 3, 224, 224, device="mps")
    with torch.no_grad():
        out_b = baseline(x_check)
        out_c = combined(x_check)
    max_abs_diff = (out_b - out_c).abs().max().item()
    top1_match = bool((out_b.argmax(-1) == out_c.argmax(-1)).all().item())
    print(f"[10] correctness: max_abs_diff={max_abs_diff:.2e}  top1_match={top1_match}")

    results = []
    for bs in batch_sizes:
        x = torch.randn(bs, 3, 224, 224, device="mps")
        t_baseline = time_forward(baseline, x)
        t_combined = time_forward(combined, x)
        speedup = t_baseline / t_combined
        print(f"[10] batch={bs:3d}: baseline={t_baseline*1e3:8.3f}ms  combined={t_combined*1e3:8.3f}ms  speedup={speedup:.2f}x")
        results.append({"batch_size": bs, "baseline_ms": t_baseline * 1e3, "combined_ms": t_combined * 1e3, "speedup": speedup})

    return {
        "phase": "10_combined_fusion_benchmark",
        "variant": variant,
        "num_conv_norms_fused": n_conv,
        "num_blocks_mlx_attention": n_mlx_blocks,
        "correctness": {"max_abs_diff": max_abs_diff, "top1_match": top1_match},
        "results": results,
        "mean_speedup": sum(r["speedup"] for r in results) / len(results),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="tiny_vit_5m_224", choices=list(MODEL_REGISTRY.keys()))
    args = ap.parse_args()

    result = run(args.variant, [1, 4, 16, 32])
    out_path = RAW_DIR / f"10_combined_fusion_benchmark_{args.variant}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[10] mean speedup: {result['mean_speedup']:.2f}x")
    print(f"[10] wrote {out_path}")
