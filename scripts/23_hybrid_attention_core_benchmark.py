#!/usr/bin/env python3
"""Phase 23: does kernel fusion actually speed up REAL TinyViT model
inference, not just a synthetic config or an isolated block?

Phase 22 found full-block native fusion (LayerNorm + QKV + attention +
output projection, one dispatch) wins on a synthetic ablation point but
loses on the real stage-1 block (window_size=7, dim=128) -- the all-
threadgroup-memory design doesn't fit real block sizes (135KB needed vs.
the M4's 32768-byte limit), and the device-memory-scratch fallback that
does fit is slower than baseline throughout.

This tries a different design, motivated directly by that failure's root
cause: PyTorch's own Linear/LayerNorm are NOT the problem (Phase 17 showed
MLX's own matmul beats PyTorch's at comparable shapes) -- only the
attention CORE (QK^T*scale + bias, softmax, AV) is genuinely small and
dispatch-heavy. So this fuses ONLY that into one native kernel, leaving
LayerNorm and both Linear projections as PyTorch's own ops.

Four things tested, in order:
1. Isolated attention-core speedup on the real stage-1 block (window_size=7) --
   a genuine, reproducible win (this is new: Phase 22's full-fusion kernel
   never won on any real block at any batch size).
2. Whether patching window_size=14 blocks too (via row-tiling, since N=196's
   dense N*N score matrix doesn't fit on-chip either) helps or hurts.
3. Applied to the FULL model (patching every eligible block, leaving
   ineligible ones as baseline) against the standard baseline -- does
   end-to-end inference actually get faster?
4. The real question this project's own working optimization raises: does
   it add anything when stacked on top of `fusion/best_model.build_best_model`
   (conv+BN fusion + channels_last, the project's actual deployable win),
   not just against an unmodified baseline? Tests the fully-combined,
   fully-real stack end to end.
"""

import argparse
import json
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from tinyvit_roofline.config import MODEL_REGISTRY, RAW_DIR
from tinyvit_roofline.models import load_baseline_model, enumerate_attention_blocks, get_module_by_name
from tinyvit_roofline.fusion.native_metal_attn import (
    make_hybrid_attention_forward, patch_model_with_hybrid_attention, get_core_kernel,
)
from tinyvit_roofline.fusion.best_model import build_best_model

WARMUP = 20
ITERS = 100
MODEL_WARMUP = 15
MODEL_ITERS = 50


def _time_call(fn, warmup: int, iters: int) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters


def isolated_block_check(variant: str, stage_idx: int) -> dict:
    """Isolated attention-core speedup on one real block, across batch sizes."""
    model = load_baseline_model(variant)
    infos = enumerate_attention_blocks(model, variant)
    bi = [b for b in infos if b.stage_idx == stage_idx and b.block_idx == 0][0]
    block = get_module_by_name(model, bi.module_name)
    attn = block.attn
    N = bi.window_size * bi.window_size

    hybrid_fwd = make_hybrid_attention_forward(attn, bi.window_size)

    x_check = torch.randn(16, N, bi.dim, device="mps").contiguous()
    with torch.no_grad():
        ref = attn(x_check)
        out = hybrid_fwd(attn, x_check)
    torch.mps.synchronize()
    max_abs_diff = (out - ref).abs().max().item()

    results = []
    for bw in (16, 32, 64, 128, 256, 512):
        x = torch.randn(bw, N, bi.dim, device="mps").contiguous()
        t_baseline = _time_call(lambda: attn(x), WARMUP, ITERS)
        t_hybrid = _time_call(lambda: hybrid_fwd(attn, x), WARMUP, ITERS)
        results.append({"batch_windows": bw, "baseline_us": t_baseline * 1e6,
                         "hybrid_us": t_hybrid * 1e6, "speedup": t_baseline / t_hybrid})

    return {
        "stage_idx": bi.stage_idx, "window_size": bi.window_size, "dim": bi.dim,
        "correctness": {"max_abs_diff": max_abs_diff},
        "results": results,
        "mean_speedup": sum(r["speedup"] for r in results) / len(results),
    }


def full_model_check(variant: str, max_window_size: int) -> dict:
    """Patches every eligible block, measures real end-to-end speedup."""
    baseline_model = load_baseline_model(variant)
    hybrid_model = load_baseline_model(variant)
    blocks = enumerate_attention_blocks(hybrid_model, variant)
    n_patched, n_skipped = patch_model_with_hybrid_attention(hybrid_model, blocks, max_window_size)

    torch.manual_seed(0)
    x_check = torch.randn(1, 3, 224, 224, device="mps")
    with torch.no_grad():
        ref = baseline_model(x_check)
        out = hybrid_model(x_check)
    torch.mps.synchronize()
    max_abs_diff = (out - ref).abs().max().item()
    top1_match = bool((out.argmax(-1) == ref.argmax(-1)).all().item())

    results = []
    for bs in (1, 4, 16, 32):
        x = torch.randn(bs, 3, 224, 224, device="mps")
        t_baseline = _time_call(lambda: baseline_model(x), MODEL_WARMUP, MODEL_ITERS)
        t_hybrid = _time_call(lambda: hybrid_model(x), MODEL_WARMUP, MODEL_ITERS)
        results.append({"batch_size": bs, "baseline_ms": t_baseline * 1e3,
                         "hybrid_ms": t_hybrid * 1e3, "speedup": t_baseline / t_hybrid})

    return {
        "blocks_patched": n_patched, "blocks_skipped": n_skipped,
        "correctness": {"max_abs_diff": max_abs_diff, "top1_match": top1_match},
        "results": results,
        "mean_speedup": sum(r["speedup"] for r in results) / len(results),
    }


def combined_stack_check(variant: str, max_window_size: int) -> dict:
    """Does the attention-core fusion add anything when stacked on top of
    this project's actual deployable win (conv+BN fusion + channels_last,
    fusion/best_model.build_best_model) -- not just against an unmodified
    baseline? Three arms: baseline, best (conv+BN+channels_last, exact),
    best+kernel (best, with window_size<=max_window_size attention cores
    also fused)."""
    baseline_model = load_baseline_model(variant)
    best_model = build_best_model(variant, fp16=False)
    best_plus_kernel_model = build_best_model(variant, fp16=False)
    blocks = enumerate_attention_blocks(best_plus_kernel_model, variant)
    n_patched, n_skipped = patch_model_with_hybrid_attention(best_plus_kernel_model, blocks, max_window_size)

    torch.manual_seed(0)
    x_check = torch.randn(1, 3, 224, 224, device="mps")
    x_check_cl = x_check.to(memory_format=torch.channels_last)
    with torch.no_grad():
        ref = baseline_model(x_check)
        out_best = best_model(x_check_cl)
        out_combo = best_plus_kernel_model(x_check_cl)
    torch.mps.synchronize()
    correctness = {
        "best_vs_baseline": {"max_abs_diff": (out_best - ref).abs().max().item(),
                              "top1_match": bool((out_best.argmax(-1) == ref.argmax(-1)).all().item())},
        "best_plus_kernel_vs_baseline": {"max_abs_diff": (out_combo - ref).abs().max().item(),
                                          "top1_match": bool((out_combo.argmax(-1) == ref.argmax(-1)).all().item())},
    }

    results = []
    for bs in (1, 4, 16, 32):
        x = torch.randn(bs, 3, 224, 224, device="mps")
        x_cl = x.to(memory_format=torch.channels_last)
        t_baseline = _time_call(lambda: baseline_model(x), MODEL_WARMUP, MODEL_ITERS)
        t_best = _time_call(lambda: best_model(x_cl), MODEL_WARMUP, MODEL_ITERS)
        t_combo = _time_call(lambda: best_plus_kernel_model(x_cl), MODEL_WARMUP, MODEL_ITERS)
        results.append({
            "batch_size": bs, "baseline_ms": t_baseline * 1e3, "best_ms": t_best * 1e3,
            "best_plus_kernel_ms": t_combo * 1e3,
            "speedup_vs_baseline": {"best": t_baseline / t_best, "best_plus_kernel": t_baseline / t_combo},
            "speedup_of_kernel_over_best": t_best / t_combo,
        })

    return {
        "blocks_patched": n_patched, "blocks_skipped": n_skipped,
        "correctness": correctness,
        "results": results,
        "mean_speedup_vs_baseline": {
            "best": sum(r["speedup_vs_baseline"]["best"] for r in results) / len(results),
            "best_plus_kernel": sum(r["speedup_vs_baseline"]["best_plus_kernel"] for r in results) / len(results),
        },
        "mean_speedup_of_kernel_over_best": sum(r["speedup_of_kernel_over_best"] for r in results) / len(results),
    }


def run(variant: str) -> dict:
    print(f"[23] {variant}: isolated attention-core check on real stage-1 block (window_size=7) ...")
    stage1 = isolated_block_check(variant, stage_idx=1)
    for r in stage1["results"]:
        print(f"[23]   bw={r['batch_windows']:4d}: baseline={r['baseline_us']:8.2f}us  "
              f"hybrid={r['hybrid_us']:8.2f}us  speedup={r['speedup']:.2f}x")
    print(f"[23]   mean speedup: {stage1['mean_speedup']:.2f}x  "
          f"max_abs_diff={stage1['correctness']['max_abs_diff']:.2e}")

    print(f"\n[23] {variant}: does extending to window_size=14 (row-tiled) help or hurt? ...")
    stage2 = isolated_block_check(variant, stage_idx=2)
    for r in stage2["results"]:
        print(f"[23]   bw={r['batch_windows']:4d}: baseline={r['baseline_us']:8.2f}us  "
              f"hybrid={r['hybrid_us']:8.2f}us  speedup={r['speedup']:.2f}x")
    print(f"[23]   mean speedup: {stage2['mean_speedup']:.2f}x -- "
          f"{'hurts' if stage2['mean_speedup'] < 1.0 else 'helps'}, so full-model test below skips it")

    print(f"\n[23] {variant}: full end-to-end model, patching only window_size<=9 blocks ...")
    full = full_model_check(variant, max_window_size=9)
    print(f"[23]   patched {full['blocks_patched']} blocks, skipped {full['blocks_skipped']}")
    for r in full["results"]:
        print(f"[23]   batch={r['batch_size']:3d}: baseline={r['baseline_ms']:7.3f}ms  "
              f"hybrid={r['hybrid_ms']:7.3f}ms  speedup={r['speedup']:.2f}x")
    print(f"[23]   mean end-to-end speedup: {full['mean_speedup']:.2f}x  "
          f"max_abs_diff={full['correctness']['max_abs_diff']:.2e}  top1_match={full['correctness']['top1_match']}")

    print(f"\n[23] {variant}: the real question -- does this add anything stacked on "
          f"build_best_model (conv+BN fusion + channels_last), not just vs. unmodified baseline? ...")
    combo = combined_stack_check(variant, max_window_size=9)
    print(f"[23]   patched {combo['blocks_patched']} blocks, skipped {combo['blocks_skipped']}")
    for r in combo["results"]:
        print(f"[23]   batch={r['batch_size']:3d}: baseline={r['baseline_ms']:7.3f}ms  "
              f"best={r['best_ms']:7.3f}ms ({r['speedup_vs_baseline']['best']:.2f}x)  "
              f"best+kernel={r['best_plus_kernel_ms']:7.3f}ms "
              f"({r['speedup_vs_baseline']['best_plus_kernel']:.2f}x vs baseline, "
              f"{r['speedup_of_kernel_over_best']:.2f}x vs best)")
    print(f"[23]   mean speedup of kernel fusion OVER best (already-optimized model): "
          f"{combo['mean_speedup_of_kernel_over_best']:.2f}x")

    return {
        "phase": "23_hybrid_attention_core_benchmark",
        "variant": variant,
        "methodology_note": (
            "Fuses ONLY the attention core (QK^T*scale+bias, softmax, AV) into one native MPS "
            "dispatch (fusion/native/fused_attention_core_template.mm), leaving PyTorch's own "
            "LayerNorm and Linear for norm/QKV/proj -- unlike Phase 22's full-block fusion. "
            "window_size=14 blocks use a row-tiled variant (N=196's dense score matrix doesn't fit "
            "on-chip either) but are excluded from the full-model test since isolated testing found "
            "fusing them hurts, not just fails to help. The combined_stack check additionally tests "
            "whether the kernel fusion adds anything on top of fusion/best_model.build_best_model "
            "(this project's actual deployable optimization), not just against an unmodified baseline."
        ),
        "isolated_window_size_7": stage1,
        "isolated_window_size_14_tiled": stage2,
        "full_model": full,
        "combined_stack": combo,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="tiny_vit_5m_224", choices=list(MODEL_REGISTRY.keys()))
    args = ap.parse_args()

    result = run(args.variant)
    out_path = RAW_DIR / f"23_hybrid_attention_core_benchmark_{args.variant}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[23] isolated (ws=7) mean speedup: {result['isolated_window_size_7']['mean_speedup']:.2f}x")
    print(f"[23] full end-to-end mean speedup (vs. unmodified baseline): {result['full_model']['mean_speedup']:.2f}x")
    print(f"[23] kernel fusion's speedup OVER best_model (the real deployable win): "
          f"{result['combined_stack']['mean_speedup_of_kernel_over_best']:.2f}x")
    print(f"[23] wrote {out_path}")
