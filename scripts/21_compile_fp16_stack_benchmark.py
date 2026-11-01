#!/usr/bin/env python3
"""Phase 21: does torch.compile stack on top of `best_fp16` -- the actual
fastest arm in this project, not just `best`?

Phase 12 tested torch.compile on top of `best` (conv+BN fusion +
channels_last) and found it fails (0.72x, worse than baseline). But
`best_fp16` (Phase 14, `best` + fp16) is the actual fastest arm found
anywhere in this project, and it's never been tried with torch.compile on
top. There's a real reason this might go differently: compilers sometimes
do better with simpler, more uniform fp16 kernels than with fp32's wider
dynamic-range codegen paths, so this isn't just repeating Phase 12 for
completeness -- it's testing the compiler on a different kernel-generation
problem.

Same subprocess-per-(arm, batch_size) methodology as Phase 12 (torch.compile
crashes on PyTorch's MPS Inductor backend if the same compiled model sees
more than one batch size -- a fixed batch size per process avoids it).
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tinyvit_roofline.config import MODEL_REGISTRY, RAW_DIR

SCRATCH_DIR = Path("/private/tmp/claude-501/-Users-anuragdhiman-Documents-tinyViT/ac31620d-7359-4957-aff3-df1814dc89cd/scratchpad")
RUN_RESULT_PATH = SCRATCH_DIR / "phase21_run_result.json"

ARMS = ("baseline", "best_fp16", "best_fp16_compiled")
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
    model = model.to(memory_format=torch.channels_last).half()
    if arm == "best_fp16_compiled":
        model = torch.compile(model)
    return model


def run_one(arm: str, variant: str, batch_size: int) -> dict:
    import torch

    model = build_model(arm, variant)
    use_cl_fp16 = arm != "baseline"

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

    torch.manual_seed(0)  # fixed seed across arms/subprocesses -- see Phase 11's note
    x = torch.randn(batch_size, 3, 224, 224, device="mps")
    if use_cl_fp16:
        x = x.to(memory_format=torch.channels_last).half()
    with torch.no_grad():
        out_check = model(x)

    t = time_forward(x)
    return {
        "arm": arm, "batch_size": batch_size, "ms": t * 1e3,
        "check_logits": out_check.to("cpu").float().tolist(),
    }


def run_all(variant: str) -> dict:
    results = {arm: {} for arm in ARMS}
    errors = {}
    total = len(ARMS) * len(BATCH_SIZES)
    i = 0
    for arm in ARMS:
        for bs in BATCH_SIZES:
            i += 1
            print(f"[21] [{i}/{total}] evaluating {arm} batch={bs} in a fresh subprocess ...")
            if RUN_RESULT_PATH.exists():
                RUN_RESULT_PATH.unlink()
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--arm", arm,
                 "--variant", variant, "--batch-size", str(bs)],
                capture_output=True, text=True,
            )
            if not RUN_RESULT_PATH.exists():
                print(f"[21] {arm} batch={bs} FAILED. stderr tail:\n{proc.stderr[-1500:]}")
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
            print(f"[21] {arm} correctness @ bs={base_bs}: max_abs_diff={max_abs_diff:.2e}  top1_match={top1_match}")

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
            print(f"[21] {arm} mean speedup: {mean_speedup[arm]:.2f}x ({len(ratios)}/{len(BATCH_SIZES)} batch sizes succeeded)")

    return {
        "phase": "21_compile_fp16_stack_benchmark",
        "variant": variant,
        "methodology_note": (
            "Each (arm, batch_size) pair runs in its own subprocess with a cooldown pause -- same fix "
            "as Phase 12, including for the same torch.compile dynamic-shape MPS crash."
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
    out_path = RAW_DIR / f"21_compile_fp16_stack_benchmark_{args.variant}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[21] wrote {out_path}")
