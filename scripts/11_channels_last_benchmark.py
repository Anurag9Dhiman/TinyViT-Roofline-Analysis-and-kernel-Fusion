#!/usr/bin/env python3
"""Phase 11: channels_last (NHWC) memory format -- does it help, and does it
stack with conv+BN fusion?

Motivated by two things already established: (1) conv is the real bottleneck
(Phase 1), and (2) MLX's own conv2d is natively NHWC (fusion/mlx_fused_
block.py), suggesting this hardware's conv kernels may prefer that layout.
Unlike MLX attention fusion (Phase 10), this is a pure PyTorch memory-layout
change with no cross-framework round trip, so there's no fixed per-call tax
to interact adversarially with conv fusion's gain.

Each arm runs as a SEPARATE subprocess with a cooldown pause between them --
the same fix already validated for scripts/07_accuracy_eval.py. An earlier
single-process version of this script gave a trustworthy-looking result for
the smaller variants (5M, 11M: combined > conv-fused alone > channels_last
alone > baseline, consistently) but went backwards for TinyViT-21M at
batch=32 (every arm looked *slower* than baseline) -- isolated, order-
independent re-checks in separate processes showed combined was still
genuinely faster (1.12-1.27x) at that exact config, confirming the
in-process number was thermally confounded (21M is the biggest model here;
more sustained GPU load per arm means more heat by the time the later arms
and the largest batch size are reached). See finding #10 in report.md for
the first time this exact failure mode was diagnosed.
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
ARM_RESULT_PATH = SCRATCH_DIR / "phase11_arm_result.json"

ARMS = ("baseline", "channels_last", "conv_bn_fused", "conv_bn_fused_channels_last")
BATCH_SIZES = (1, 4, 16, 32)
COOLDOWN_S = 10


def build_model(arm: str, variant: str):
    import torch
    from tinyvit_roofline.models import load_baseline_model
    from tinyvit_roofline.fusion.conv_bn_fusion import fuse_conv_norms

    model = load_baseline_model(variant)
    if arm in ("conv_bn_fused", "conv_bn_fused_channels_last"):
        fuse_conv_norms(model)
        model.to("mps")
    if arm in ("channels_last", "conv_bn_fused_channels_last"):
        model = model.to(memory_format=torch.channels_last)
    return model


def run_one_arm(arm: str, variant: str, baseline_preds_path: str = None) -> dict:
    """Runs in its own process."""
    import torch

    model = build_model(arm, variant)
    use_cl = arm in ("channels_last", "conv_bn_fused_channels_last")

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

    # Fixed seed: each arm runs in its own subprocess with an independent
    # random state, so without this every arm would check correctness
    # against a *different* random input -- not a meaningful comparison.
    # (Caught empirically: an unseeded version of this produced max-abs-diff
    # ~0.2-0.7, which looked like a correctness failure but was actually
    # just four different random images.)
    torch.manual_seed(0)
    x_check = torch.randn(1, 3, 224, 224, device="mps")
    if use_cl:
        x_check = x_check.to(memory_format=torch.channels_last)
    with torch.no_grad():
        out_check = model(x_check)

    timings = []
    for bs in BATCH_SIZES:
        x = torch.randn(bs, 3, 224, 224, device="mps")
        if use_cl:
            x = x.to(memory_format=torch.channels_last)
        t = time_forward(x)
        timings.append({"batch_size": bs, "ms": t * 1e3})
        torch.mps.empty_cache()  # avoid MPS allocator fragmentation across distinct batch-size
        # shapes within this one process -- caught empirically: without this, the last batch
        # size's timing degraded 2x+ for no implementation reason (same root cause as Phases 4/6).

    return {"arm": arm, "timings": timings, "check_logits": out_check.to("cpu").tolist()}


def run_all_arms(variant: str) -> dict:
    results = {}
    for i, arm in enumerate(ARMS):
        print(f"[11] evaluating {arm} in a fresh subprocess ...")
        if ARM_RESULT_PATH.exists():
            ARM_RESULT_PATH.unlink()
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--arm", arm, "--variant", variant],
            capture_output=True, text=True,
        )
        if not ARM_RESULT_PATH.exists():
            print(f"[11] ERROR: {arm} subprocess produced no result. stderr tail:\n{proc.stderr[-2000:]}")
            return {"phase": "11_channels_last_benchmark", "error": f"{arm} subprocess failed", "stderr": proc.stderr[-4000:]}
        r = json.loads(ARM_RESULT_PATH.read_text())
        results[arm] = r
        for t in r["timings"]:
            print(f"     batch={t['batch_size']:3d}: {t['ms']:.3f}ms")
        if i < len(ARMS) - 1:
            print(f"[11] cooling down {COOLDOWN_S}s before next arm ...")
            time.sleep(COOLDOWN_S)

    import math
    base_logits = results["baseline"]["check_logits"][0]
    correctness = {}
    for arm in ARMS:
        if arm == "baseline":
            continue
        logits = results[arm]["check_logits"][0]
        max_abs_diff = max(abs(a - b) for a, b in zip(base_logits, logits))
        top1_match = base_logits.index(max(base_logits)) == logits.index(max(logits))
        correctness[arm] = {"max_abs_diff": max_abs_diff, "top1_match": top1_match}
        print(f"[11] {arm} correctness: max_abs_diff={max_abs_diff:.2e}  top1_match={top1_match}")

    per_batch = {arm: {t["batch_size"]: t["ms"] for t in results[arm]["timings"]} for arm in ARMS}
    mean_speedup = {}
    for arm in ARMS:
        if arm == "baseline":
            continue
        ratios = [per_batch["baseline"][bs] / per_batch[arm][bs] for bs in BATCH_SIZES]
        mean_speedup[arm] = sum(ratios) / len(ratios)
        print(f"[11] {arm} mean speedup: {mean_speedup[arm]:.2f}x")

    return {
        "phase": "11_channels_last_benchmark",
        "variant": variant,
        "methodology_note": (
            "Each arm evaluated in a separate subprocess with a cooldown pause between them "
            "(same fix as scripts/07_accuracy_eval.py), after an in-process version gave a "
            "thermally-confounded result for TinyViT-21M at batch=32 -- see this script's "
            "module docstring."
        ),
        "batch_sizes": list(BATCH_SIZES),
        "correctness": correctness,
        "results": {arm: results[arm]["timings"] for arm in ARMS},
        "mean_speedup": mean_speedup,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="tiny_vit_5m_224", choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--arm", default=None, choices=list(ARMS),
                     help="Internal: run a single arm in this process and write ARM_RESULT_PATH.")
    args = ap.parse_args()

    if args.arm is not None:
        r = run_one_arm(args.arm, args.variant)
        ARM_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        ARM_RESULT_PATH.write_text(json.dumps(r))
        sys.exit(0)

    result = run_all_arms(args.variant)
    out_path = RAW_DIR / f"11_channels_last_benchmark_{args.variant}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[11] wrote {out_path}")
