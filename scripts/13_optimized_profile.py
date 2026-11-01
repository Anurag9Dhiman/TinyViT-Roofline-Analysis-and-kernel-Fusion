#!/usr/bin/env python3
"""Phase 13: re-profile the OPTIMIZED model (conv+BN fusion + channels_last,
the project's validated best combination) with the exact same methodology
Phase 1 used on the standard baseline.

Closes the diagnostic loop this project is built around: Phase 1 found
"conv" was the single largest dispatch-time category in the baseline (more
than qkv/mlp/attention combined), which is what motivated conv+BN fusion in
the first place (Phase 9). Now that that bottleneck has been measurably
reduced (Phases 9/11), what dominates the optimized model's dispatch time
instead? Same op-category breakdown, same instrumentation
(profiling/op_classify.py), directly comparable to
results/raw/01_baseline_*.json.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from tinyvit_roofline.config import MODEL_REGISTRY, RAW_DIR, TRACE_DIR, TIMED_ITERS, WARMUP_ITERS
from tinyvit_roofline.models import load_baseline_model, enumerate_attention_blocks
from tinyvit_roofline.fusion.conv_bn_fusion import fuse_conv_norms
from tinyvit_roofline.profiling.torch_profile import time_forward, peak_memory_mb, profile_ops, export_chrome_trace
from tinyvit_roofline.profiling.op_classify import annotate_tinyvit_blocks, summarize_by_category


def build_optimized(variant: str) -> torch.nn.Module:
    model = load_baseline_model(variant)
    fuse_conv_norms(model)
    model.to("mps")
    return model.to(memory_format=torch.channels_last)


def run(variant: str, batch_size: int) -> dict:
    print(f"[13] loading {variant} (conv+BN fused + channels_last) ...")
    model = build_optimized(variant)
    x = torch.randn(batch_size, 3, 224, 224, device="mps").to(memory_format=torch.channels_last)

    print(f"[13] timing pure forward (warmup={WARMUP_ITERS}, iters={TIMED_ITERS}) ...")
    latency = time_forward(model, x, warmup=WARMUP_ITERS, iters=TIMED_ITERS)
    print(f"     mean={latency.mean_ms:.3f}ms p50={latency.p50_ms:.3f}ms p90={latency.p90_ms:.3f}ms")

    print("[13] measuring peak memory ...")
    peak_mb = peak_memory_mb(model, x)
    print(f"     peak allocated = {peak_mb:.1f} MB")

    print("[13] op-level breakdown (uninstrumented, name-based classification) ...")
    ops = profile_ops(model, x, iters=20)
    by_cat = summarize_by_category(ops["ops"])
    for cat, d in sorted(by_cat.items(), key=lambda kv: -kv[1]["self_cpu_time_us"]):
        print(f"     {cat:35s} self={d['self_cpu_time_us']/ops['iters']:8.1f}us/iter  calls/iter={d['count']/ops['iters']:6.1f}")

    print("[13] op-level breakdown (instrumented: qkv/proj/window-partition/etc. labeled) ...")
    model_annotated = build_optimized(variant)  # fresh copy so the pure-timing model stays untouched
    block_infos = enumerate_attention_blocks(model_annotated, variant)
    n_blocks = annotate_tinyvit_blocks(model_annotated, block_infos)
    print(f"     annotated {n_blocks} TinyVitBlocks")
    ops_annotated = profile_ops(model_annotated, x, iters=20)
    fine_rows = [
        {
            "name": r["name"],
            "self_cpu_time_us_per_iter": r["self_cpu_time_us"] / ops_annotated["iters"],
            "calls_per_iter": r["count"] / ops_annotated["iters"],
        }
        for r in ops_annotated["ops"]
        if r["name"].startswith("tinyvit/")
    ]
    fine_rows.sort(key=lambda r: -r["self_cpu_time_us_per_iter"])
    print("     fine-grained tinyvit/* regions:")
    for r in fine_rows:
        print(f"       {r['name']:40s} {r['self_cpu_time_us_per_iter']:8.1f}us/iter  calls/iter={r['calls_per_iter']:5.1f}")

    trace_path = str(TRACE_DIR / f"13_optimized_{variant}_bs{batch_size}.json")
    export_chrome_trace(model, x, trace_path)
    print(f"[13] chrome trace exported to {trace_path}")

    # Compare against Phase 1's baseline breakdown for the same variant/batch, if present.
    comparison = None
    baseline_path = RAW_DIR / f"01_baseline_{variant}_bs{batch_size}.json"
    if baseline_path.exists():
        base = json.loads(baseline_path.read_text())
        base_cat = base["op_breakdown_uninstrumented"]["by_category"]
        opt_cat = {k: {"self_cpu_time_us_per_iter": v["self_cpu_time_us"] / ops["iters"],
                       "calls_per_iter": v["count"] / ops["iters"]} for k, v in by_cat.items()}
        all_cats = sorted(set(base_cat) | set(opt_cat),
                           key=lambda c: -base_cat.get(c, {}).get("self_cpu_time_us_per_iter", 0))
        comparison = {
            "baseline_mean_ms": base["latency"]["mean_ms"],
            "optimized_mean_ms": latency.mean_ms,
            "end_to_end_speedup": base["latency"]["mean_ms"] / latency.mean_ms,
            "by_category": {
                c: {
                    "baseline_us_per_iter": base_cat.get(c, {}).get("self_cpu_time_us_per_iter", 0.0),
                    "optimized_us_per_iter": opt_cat.get(c, {}).get("self_cpu_time_us_per_iter", 0.0),
                }
                for c in all_cats
            },
        }
        print(f"\n[13] vs. baseline: {base['latency']['mean_ms']:.3f}ms -> {latency.mean_ms:.3f}ms "
              f"({comparison['end_to_end_speedup']:.2f}x)")
        print("     category            baseline(us)  optimized(us)   delta")
        for c, d in comparison["by_category"].items():
            delta = d["optimized_us_per_iter"] - d["baseline_us_per_iter"]
            print(f"     {c:20s} {d['baseline_us_per_iter']:10.1f}  {d['optimized_us_per_iter']:12.1f}  {delta:+8.1f}")

    result = {
        "phase": "13_optimized_profile",
        "variant": variant,
        "batch_size": batch_size,
        "device": "mps",
        "dtype": "float32",
        "optimization": "conv_bn_fused + channels_last",
        "latency": latency.to_dict(),
        "peak_memory_mb": peak_mb,
        "op_breakdown_uninstrumented": {
            "total_op_calls_per_iter": ops["op_calls_per_iter"],
            "by_category": {
                k: {"self_cpu_time_us_per_iter": v["self_cpu_time_us"] / ops["iters"],
                    "calls_per_iter": v["count"] / ops["iters"],
                    "ops": v["ops"]}
                for k, v in by_cat.items()
            },
            "top_ops": ops["ops"][:20],
        },
        "op_breakdown_instrumented_fine_grained": fine_rows,
        "chrome_trace_path": trace_path,
        "comparison_vs_baseline": comparison,
    }
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="tiny_vit_5m_224", choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    result = run(args.variant, args.batch_size)
    out_path = RAW_DIR / f"13_optimized_{args.variant}_bs{args.batch_size}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[13] wrote {out_path}")
