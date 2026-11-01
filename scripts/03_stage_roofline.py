#!/usr/bin/env python3
"""Phase 3: per-stage arithmetic intensity, mapped onto the empirical roofline.

Uses isolated microbenchmarks (roofline/op_bench.py) for per-op GPU timing,
not in-forward torch.profiler regions -- see op_bench.py's module docstring
for why (torch.mps.synchronize()'s fixed cost dominates sub-100us regions
when called once per region; isolated, warmed-up, many-iteration timing
amortizes that cost the same way profiling/torch_profile.time_forward and
roofline/hw_bench already do for whole-model and raw-matmul timing).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from tinyvit_roofline.config import MODEL_REGISTRY, RAW_DIR, PLOTS_DIR
from tinyvit_roofline.models import load_baseline_model, enumerate_attention_blocks, get_module_by_name
from tinyvit_roofline.roofline.ai_estimator import block_ai_breakdown, profile_conv_layers
from tinyvit_roofline.roofline.op_bench import benchmark_block_ops, benchmark_conv_stem, benchmark_patch_merging
from tinyvit_roofline.roofline.plot import plot_roofline


def run(variant: str, batch_size: int) -> dict:
    print(f"[03] loading {variant} ...")
    model = load_baseline_model(variant)
    block_infos = enumerate_attention_blocks(model, variant)
    x = torch.randn(batch_size, 3, 224, 224, device="mps")

    print("[03] computing real conv-layer FLOPs/bytes from captured shapes ...")
    conv_ai = profile_conv_layers(model, x)

    points = []

    print("[03] benchmarking conv stem (patch_embed + stage0 MBConv) ...")
    t = benchmark_conv_stem(model, x)
    stem_names = {n for n in conv_ai if n.startswith("patch_embed") or n.startswith("stages.0")}
    flops = sum(conv_ai[n].flops for n in stem_names)
    byts = sum(conv_ai[n].bytes_moved for n in stem_names)
    points.append({
        "block": "all", "stage_idx": -1, "region": "conv_stem", "category": "conv",
        "dim": None, "window_size": None, "resolution": None,
        "flops": flops, "bytes_moved": byts, "ai": flops / byts,
        "self_time_us_per_iter": t * 1e6, "achieved_gflops_per_s": flops / t / 1e9,
    })

    print("[03] benchmarking per-stage PatchMerging ...")
    for stage_idx in (1, 2, 3):
        t = benchmark_patch_merging(model, stage_idx, x)
        names = {n for n in conv_ai if n.startswith(f"stages.{stage_idx}.downsample")}
        flops = sum(conv_ai[n].flops for n in names)
        byts = sum(conv_ai[n].bytes_moved for n in names)
        points.append({
            "block": f"s{stage_idx}", "stage_idx": stage_idx, "region": "patch_merging", "category": "conv",
            "dim": None, "window_size": None, "resolution": None,
            "flops": flops, "bytes_moved": byts, "ai": flops / byts,
            "self_time_us_per_iter": t * 1e6, "achieved_gflops_per_s": flops / t / 1e9,
        })

    print("[03] benchmarking per-block attention/mlp/local_conv/layout ops ...")
    for bi in block_infos:
        prefix = f"s{bi.stage_idx}b{bi.block_idx}"
        block_module = get_module_by_name(model, bi.module_name)
        times_s = benchmark_block_ops(model, block_module, bi, x)
        breakdown = block_ai_breakdown(
            batch=batch_size, resolution=bi.input_resolution, window_size=bi.window_size,
            dim=bi.dim, num_heads=bi.num_heads,
        )
        for region_name, stage_ai in breakdown.items():
            t = times_s.get(region_name)
            if t is None or t <= 0:
                continue
            points.append({
                "block": prefix, "stage_idx": bi.stage_idx, "region": region_name, "category": region_name,
                "dim": bi.dim, "window_size": bi.window_size, "resolution": bi.input_resolution,
                "flops": stage_ai.flops, "bytes_moved": stage_ai.bytes_moved, "ai": stage_ai.arithmetic_intensity,
                "self_time_us_per_iter": t * 1e6,
                "achieved_gflops_per_s": stage_ai.flops / t / 1e9,
            })

    hw = json.loads((RAW_DIR / "02_hw_roofline.json").read_text())
    peak_gflops = hw["fp32"]["compute"]["peak_gflops_per_s"]
    peak_gb = hw["fp32"]["bandwidth"]["peak_gb_per_s"]

    for p in points:
        if p["flops"] == 0:
            # Pure data-movement ops (window partition/reverse): undefined on
            # a FLOPs roofline. Report bandwidth utilization instead.
            achieved_gbps = p["bytes_moved"] / (p["self_time_us_per_iter"] * 1e-6) / 1e9
            p["achieved_gb_per_s"] = achieved_gbps
            p["bandwidth_utilization"] = achieved_gbps / peak_gb
            p["roofline_bound_gflops_per_s"] = None
            p["regime"] = "memory_bound_zero_flops"
            p["utilization_of_bound"] = p["bandwidth_utilization"]
            continue
        p["roofline_bound_gflops_per_s"] = min(peak_gflops, peak_gb * p["ai"])
        p["regime"] = "compute_bound" if peak_gflops < peak_gb * p["ai"] else "memory_bound"
        p["utilization_of_bound"] = (
            p["achieved_gflops_per_s"] / p["roofline_bound_gflops_per_s"] if p["roofline_bound_gflops_per_s"] else 0.0
        )

    plot_path = str(PLOTS_DIR / f"03_roofline_{variant}_bs{batch_size}.png")
    plot_roofline(peak_gflops, peak_gb, points, plot_path,
                  title=f"{variant} on Apple M4 GPU -- Empirical Roofline (fp32, batch={batch_size})")
    print(f"[03] roofline plot saved to {plot_path}")

    print("\n[03] summary (region: AI, achieved GFLOP/s, regime, utilization of roofline bound):")
    for p in sorted(points, key=lambda r: -r["self_time_us_per_iter"]):
        print(f"     {p['block']:>6s}/{p['region']:22s} AI={p['ai']:8.3f}  "
              f"achieved={p['achieved_gflops_per_s']:9.2f} GFLOP/s  {p['regime']:14s}  "
              f"util={p['utilization_of_bound']*100:5.1f}%  time={p['self_time_us_per_iter']:7.2f}us/call")

    return {
        "phase": "03_stage_roofline",
        "variant": variant,
        "batch_size": batch_size,
        "methodology_note": (
            "Per-op achieved throughput comes from isolated microbenchmarks "
            "(roofline/op_bench.py: real captured shapes/weights, warmup + "
            "many iterations + one final torch.mps.synchronize()), not "
            "in-forward profiler regions -- see that module's docstring for "
            "why (torch.mps.synchronize()'s fixed ~200-300us cost dominates "
            "when paid once per region across ~60 regions/forward)."
        ),
        "hw_reference": {"peak_gflops_per_s": peak_gflops, "peak_gb_per_s": peak_gb,
                          "ridge_ai_flops_per_byte": peak_gflops * 1e9 / (peak_gb * 1e9)},
        "points": points,
        "plot_path": plot_path,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="tiny_vit_5m_224", choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    result = run(args.variant, args.batch_size)
    out_path = RAW_DIR / f"03_stage_roofline_{args.variant}_bs{args.batch_size}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[03] wrote {out_path}")
