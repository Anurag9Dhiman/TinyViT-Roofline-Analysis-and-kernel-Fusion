#!/usr/bin/env python3
"""Phase 20: does batching ever push TinyViT into the compute-bound regime?

Every benchmark in this project so far stops at batch_size=32. The
project's whole framing is the roofline model -- P(I) = min(P_max,
B_max * I) -- but nothing has actually checked whether real end-to-end
inference ever gets close to the compute ridge (Phase 2's empirically
measured P_max) at larger batch sizes, where bigger matmuls are known
(from Phase 2's own hw_roofline sweep) to achieve higher utilization.

Measures the optimized model (conv+BN fusion + channels_last, the
project's best combination) at batch_sizes=(1,4,16,32,64,128), using
`torch.utils.flop_counter.FlopCounterMode` for an exact total-model FLOP
count (not an estimate) at batch=1, scaled linearly for larger batches
(no batch-dependent control flow in this architecture, so this is exact,
not approximate). Achieved GFLOP/s = total_flops * batch_size / latency,
compared against Phase 2's measured fp32 P_max to get a real utilization
number for the *whole model*, not just isolated ops.

Each batch size in its own subprocess (the established fix for MPS
allocator fragmentation and thermal confounds across very different
memory footprints).
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tinyvit_roofline.config import CACHE_DIR, MODEL_REGISTRY, RAW_DIR

SCRATCH_DIR = CACHE_DIR
RUN_RESULT_PATH = SCRATCH_DIR / "phase20_run_result.json"

BATCH_SIZES = (1, 4, 16, 32, 64, 128)
COOLDOWN_S = 10


def build_optimized(variant: str):
    import torch
    from tinyvit_roofline.models import load_baseline_model
    from tinyvit_roofline.fusion.conv_bn_fusion import fuse_conv_norms

    model = load_baseline_model(variant)
    fuse_conv_norms(model)
    model.to("mps")
    return model.to(memory_format=torch.channels_last)


def run_one(variant: str, batch_size: int) -> dict:
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    model = build_optimized(variant)
    x1 = torch.randn(1, 3, 224, 224, device="mps").to(memory_format=torch.channels_last)
    with torch.no_grad():
        model(x1)  # warmup before counting, avoids any one-time lazy-init ops skewing the count
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x1)
    total_flops_bs1 = fc.get_total_flops()

    x = torch.randn(batch_size, 3, 224, 224, device="mps").to(memory_format=torch.channels_last)
    with torch.no_grad():
        for _ in range(10):
            model(x)
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            model(x)
        torch.mps.synchronize()
    latency_s = (time.perf_counter() - t0) / 30

    total_flops = total_flops_bs1 * batch_size
    achieved_gflops_per_s = total_flops / latency_s / 1e9
    images_per_s = batch_size / latency_s

    return {
        "batch_size": batch_size,
        "latency_ms": latency_s * 1e3,
        "total_flops_bs1": total_flops_bs1,
        "total_flops": total_flops,
        "achieved_gflops_per_s": achieved_gflops_per_s,
        "images_per_s": images_per_s,
    }


def run_all(variant: str) -> dict:
    hw_path = RAW_DIR / "02_hw_roofline.json"
    p_max_fp32 = None
    if hw_path.exists():
        hw = json.loads(hw_path.read_text())
        p_max_fp32 = hw.get("fp32", {}).get("compute", {}).get("peak_gflops_per_s")

    results = []
    for i, bs in enumerate(BATCH_SIZES):
        print(f"[20] [{i+1}/{len(BATCH_SIZES)}] batch={bs} in a fresh subprocess ...")
        if RUN_RESULT_PATH.exists():
            RUN_RESULT_PATH.unlink()
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--variant", variant, "--batch-size", str(bs)],
            capture_output=True, text=True,
        )
        if not RUN_RESULT_PATH.exists():
            print(f"[20] batch={bs} FAILED. stderr tail:\n{proc.stderr[-2000:]}")
            continue
        r = json.loads(RUN_RESULT_PATH.read_text())
        util = r["achieved_gflops_per_s"] / p_max_fp32 if p_max_fp32 else None
        r["utilization_of_p_max_fp32"] = util
        results.append(r)
        util_s = f"{util*100:.1f}%" if util is not None else "n/a"
        print(f"     latency={r['latency_ms']:.3f}ms  achieved={r['achieved_gflops_per_s']:.1f} GFLOP/s  "
              f"util={util_s}  images/s={r['images_per_s']:.1f}")
        if i < len(BATCH_SIZES) - 1:
            time.sleep(COOLDOWN_S)

    return {
        "phase": "20_ridge_point_batch_sweep",
        "variant": variant,
        "methodology_note": (
            "Each batch size runs in its own subprocess. total_flops is exact "
            "(torch.utils.flop_counter.FlopCounterMode at batch=1, scaled linearly -- this architecture "
            "has no batch-dependent control flow, so that scaling is exact, not an approximation). "
            "Model under test is the optimized combination (conv+BN fusion + channels_last)."
        ),
        "p_max_fp32_gflops_per_s": p_max_fp32,
        "results": results,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="tiny_vit_5m_224", choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    if args.batch_size is not None:
        r = run_one(args.variant, args.batch_size)
        RUN_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RUN_RESULT_PATH.write_text(json.dumps(r))
        sys.exit(0)

    result = run_all(args.variant)
    out_path = RAW_DIR / f"20_ridge_point_batch_sweep_{args.variant}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[20] wrote {out_path}")
