#!/usr/bin/env python3
"""Phase 1: standard-baseline profiling.

Loads the stock, pretrained timm TinyViT model UNMODIFIED and profiles it:
total latency, op-level breakdown, op-count-as-kernel-launch-proxy, peak
memory. This produces the "baseline" numbers every later phase compares
against, per the project's explicit requirement that the baseline reflect
standard, unmodified performance.
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from tinyvit_roofline.config import MODEL_REGISTRY, RAW_DIR, TRACE_DIR, TIMED_ITERS, WARMUP_ITERS
from tinyvit_roofline.models import load_baseline_model, enumerate_attention_blocks
from tinyvit_roofline.profiling.torch_profile import (
    time_forward,
    peak_memory_mb,
    profile_ops,
    export_chrome_trace,
)
from tinyvit_roofline.profiling.op_classify import annotate_tinyvit_blocks, summarize_by_category


def run(variant: str, batch_size: int) -> dict:
    print(f"[01] loading {variant} ...")
    model = load_baseline_model(variant)
    x = torch.randn(batch_size, 3, 224, 224, device="mps")

    print(f"[01] timing pure forward (warmup={WARMUP_ITERS}, iters={TIMED_ITERS}) ...")
    latency = time_forward(model, x, warmup=WARMUP_ITERS, iters=TIMED_ITERS)
    print(f"     mean={latency.mean_ms:.3f}ms p50={latency.p50_ms:.3f}ms p90={latency.p90_ms:.3f}ms")

    print("[01] measuring peak memory ...")
    peak_mb = peak_memory_mb(model, x)
    print(f"     peak allocated = {peak_mb:.1f} MB")

    print("[01] op-level breakdown (uninstrumented, name-based classification) ...")
    ops = profile_ops(model, x, iters=20)
    from tinyvit_roofline.profiling.op_classify import summarize_by_category as _s
    by_cat = _s(ops["ops"])
    for cat, d in sorted(by_cat.items(), key=lambda kv: -kv[1]["self_cpu_time_us"]):
        print(f"     {cat:35s} self={d['self_cpu_time_us']/ops['iters']:8.1f}us/iter  calls/iter={d['count']/ops['iters']:6.1f}")

    print("[01] op-level breakdown (instrumented: qkv/proj/window-partition/etc. labeled) ...")
    model_annotated = load_baseline_model(variant)  # fresh copy so the pure-timing model stays untouched
    block_infos = enumerate_attention_blocks(model_annotated, variant)
    n_blocks = annotate_tinyvit_blocks(model_annotated, block_infos)
    print(f"     annotated {n_blocks} TinyVitBlocks")
    ops_annotated = profile_ops(model_annotated, x, iters=20)
    by_cat_annotated = _s(ops_annotated["ops"])
    fine_grained = {k: v for k, v in by_cat_annotated.items() if k not in ("other",)}
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

    trace_path = str(TRACE_DIR / f"01_baseline_{variant}_bs{batch_size}.json")
    export_chrome_trace(model, x, trace_path)
    print(f"[01] chrome trace exported to {trace_path}")

    result = {
        "phase": "01_baseline_profile",
        "variant": variant,
        "timm_id": MODEL_REGISTRY[variant]["timm_id"],
        "batch_size": batch_size,
        "device": "mps",
        "dtype": "float32",
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
    }
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="tiny_vit_21m_224", choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    result = run(args.variant, args.batch_size)
    out_path = RAW_DIR / f"01_baseline_{args.variant}_bs{args.batch_size}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[01] wrote {out_path}")
