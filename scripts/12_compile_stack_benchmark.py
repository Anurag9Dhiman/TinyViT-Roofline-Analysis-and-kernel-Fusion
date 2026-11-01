#!/usr/bin/env python3
"""Phase 12: does torch.compile stack on top of the project's best
combination (conv+BN fusion + channels_last, Phase 11)?

Completes the set of "third lever on top of conv+BN fusion" tests:
  - MLX attention routing on top: does NOT stack (Phase 10, 0.46x -- worse
    than either alone).
  - channels_last on top: DOES stack (Phase 11, up to 1.24x).
  - torch.compile on top: tested here.

Exploratory single-process check (both orderings, order-independent) found
torch.compile roughly *halves* throughput on top of the already-optimized
model (164.5ms vs 91.1ms at batch=32, 5M) -- consistent with every other
torch.compile result in this project (ablation Phase 6: mean 0.90x alone;
Phase 7's pytorch_optimized arm: 0.44x on real images).

Extra wrinkle found while building this: calling the *same* torch.compile'd
model with multiple different batch sizes in one process crashes PyTorch's
MPS Inductor codegen outright (`TypeError: cannot determine truth value of
Relational: s39*s89 <= 1024` -- a dynamic-shape specialization bug, same
crash first seen in the Phase 6 ablation). A single fixed batch size per
process (forcing static-shape compilation) does not crash. So this script
runs each (arm, batch_size) pair in its own subprocess, not just each arm --
the dynamic-shape crash is itself a finding about torch.compile's MPS
backend, documented rather than worked around silently.
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
RUN_RESULT_PATH = SCRATCH_DIR / "phase12_run_result.json"

ARMS = ("baseline", "best", "best_compiled")
BATCH_SIZES = (1, 4, 16, 32)
COOLDOWN_S = 10


def build_model(arm: str, variant: str):
    import torch
    from tinyvit_roofline.models import load_baseline_model
    from tinyvit_roofline.fusion.conv_bn_fusion import fuse_conv_norms

    model = load_baseline_model(variant)
    if arm == "baseline":
        return model
    fuse_conv_norms(model)
    model.to("mps")
    model = model.to(memory_format=torch.channels_last)
    if arm == "best_compiled":
        model = torch.compile(model)
    return model


def run_one(arm: str, variant: str, batch_size: int) -> dict:
    """One (arm, batch_size) pair, in its own process -- torch.compile's MPS
    backend crashes on dynamic shapes if the same compiled model sees more
    than one batch size, so each size gets a fresh model/process."""
    import torch

    model = build_model(arm, variant)
    use_cl = arm != "baseline"

    def time_forward(x, warmup=10, iters=50):
        with torch.no_grad():
            for _ in range(warmup):
                model(x)
            torch.mps.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                model(x)
            torch.mps.synchronize()
        return (time.perf_counter() - t0) / iters

    torch.manual_seed(0)  # fixed seed across arms/subprocesses -- see Phase 11's note on this
    x = torch.randn(batch_size, 3, 224, 224, device="mps")
    if use_cl:
        x = x.to(memory_format=torch.channels_last)
    with torch.no_grad():
        out_check = model(x)

    t = time_forward(x)
    return {"arm": arm, "batch_size": batch_size, "ms": t * 1e3, "check_logits": out_check.to("cpu").tolist()}


def run_all(variant: str) -> dict:
    results = {arm: {} for arm in ARMS}
    errors = {}
    total = len(ARMS) * len(BATCH_SIZES)
    i = 0
    for arm in ARMS:
        for bs in BATCH_SIZES:
            i += 1
            print(f"[12] [{i}/{total}] evaluating {arm} batch={bs} in a fresh subprocess ...")
            if RUN_RESULT_PATH.exists():
                RUN_RESULT_PATH.unlink()
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--arm", arm,
                 "--variant", variant, "--batch-size", str(bs)],
                capture_output=True, text=True,
            )
            if not RUN_RESULT_PATH.exists():
                print(f"[12] {arm} batch={bs} FAILED. stderr tail:\n{proc.stderr[-1500:]}")
                errors[f"{arm}_bs{bs}"] = proc.stderr[-2000:]
                continue
            r = json.loads(RUN_RESULT_PATH.read_text())
            results[arm][bs] = r
            print(f"     {r['ms']:.3f}ms")
            if i < total:
                time.sleep(COOLDOWN_S)

    correctness = {}
    if results["baseline"]:
        base_bs = next(iter(results["baseline"]))
        base_logits = results["baseline"][base_bs]["check_logits"][0]
        for arm in ARMS:
            if arm == "baseline" or base_bs not in results[arm]:
                continue
            logits = results[arm][base_bs]["check_logits"][0]
            max_abs_diff = max(abs(a - b) for a, b in zip(base_logits, logits))
            top1_match = base_logits.index(max(base_logits)) == logits.index(max(logits))
            correctness[arm] = {"max_abs_diff": max_abs_diff, "top1_match": top1_match, "at_batch_size": base_bs}
            print(f"[12] {arm} correctness @ bs={base_bs}: max_abs_diff={max_abs_diff:.2e}  top1_match={top1_match}")

    mean_speedup = {}
    for arm in ARMS:
        if arm == "baseline":
            continue
        ratios = [
            results["baseline"][bs]["ms"] / results[arm][bs]["ms"]
            for bs in BATCH_SIZES
            if bs in results["baseline"] and bs in results[arm]
        ]
        if ratios:
            mean_speedup[arm] = sum(ratios) / len(ratios)
            print(f"[12] {arm} mean speedup: {mean_speedup[arm]:.2f}x ({len(ratios)}/{len(BATCH_SIZES)} batch sizes succeeded)")

    return {
        "phase": "12_compile_stack_benchmark",
        "variant": variant,
        "methodology_note": (
            "Each (arm, batch_size) pair runs in its own subprocess with a cooldown pause "
            "(Phases 7/11's fix for thermal/allocator confounds) -- and, specifically for "
            "torch.compile, because reusing one compiled model across batch sizes crashes "
            "PyTorch's MPS Inductor codegen on dynamic shapes (see module docstring)."
        ),
        "batch_sizes": list(BATCH_SIZES),
        "correctness": correctness,
        "results": {
            arm: [{"batch_size": bs, "ms": r["ms"]} for bs, r in sorted(results[arm].items())]
            for arm in ARMS
        },
        "errors": errors,
        "mean_speedup": mean_speedup,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="tiny_vit_5m_224", choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--arm", default=None, choices=list(ARMS))
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    if args.arm is not None:
        r = run_one(args.arm, args.variant, args.batch_size)
        RUN_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RUN_RESULT_PATH.write_text(json.dumps(r))
        sys.exit(0)

    result = run_all(args.variant)
    out_path = RAW_DIR / f"12_compile_stack_benchmark_{args.variant}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[12] wrote {out_path}")
