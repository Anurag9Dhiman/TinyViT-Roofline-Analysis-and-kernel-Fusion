#!/usr/bin/env python3
"""Phase 24: the K/V-cached fp16 flash-attention kernel -- the design that
finally beats baseline on window_size=14, where every earlier attempt
(Phase 23's scalar row-tiled kernel, and five further native-kernel designs
tried in this phase's own development that never got their own script)
either lost or merely reached noise-level parity. In order:

1. Naive one-simdgroup-per-8x8-tile `simdgroup_matrix` GEMM: correct, but
   0.80x (thin-K, QK^T-shaped) / 0.94x (big-K, AV-shaped) -- close to
   PyTorch's own GEMM but never past it.
2. A textbook-next-step shared-memory-blocked GEMM (4 simdgroups/threadgroup
   cooperating on a 16x16 tile): regressed HARD, 0.20-0.21x -- the scalar
   cooperative-load staging into threadgroup memory cost far more than the
   ~2x redundant-read reduction it bought at this problem size.
3. fp32 flash attention (online/streaming softmax, so the N x N score
   matrix never materializes in device memory): isolated microbenchmark
   win (1.49x vs. PyTorch's own fused SDPA) collapsed to 0.77-0.97x once
   real shapes, the real learned position bias, and N=196's boundary
   masking (196 isn't a multiple of 8) were added back in.
4. fp16 flash attention, same per-query-tile dispatch as #3: correctness
   held (logit max-abs-diff ~3e-4), speed improved to a real-but-narrow
   win only at batch_size=64 (~1.01x); noise or a loss everywhere else.
5. **This phase: fp16 flash attention with K/V caching.** The per-tile
   design in #3/#4 dispatches one threadgroup per 8-row query tile, and
   every one of the ~25 tiles (N=196 -> ceil(196/8) tiles) independently
   re-reads the SAME K/V data for its (window, head) from device memory --
   25x redundant global-memory traffic, identified directly (not assumed)
   as the actual remaining bottleneck, not matmul throughput (#1/#2 already
   showed GEMM tuning tops out near parity, never a win). Redesigned so
   ONE threadgroup handles an entire (window, head) -- all ~25 query tiles,
   each its own simdgroup, 800 threads/threadgroup total for N=196 -- with
   K and V staged into threadgroup memory exactly ONCE and shared by every
   simdgroup. This is the design verified here, on real models, with real
   weights, real ImageNet accuracy, and proper multi-trial statistics.

See src/tinyvit_roofline/fusion/native/fused_attention_core_flash_fp16_kvcache_template.mm
for the kernel and src/tinyvit_roofline/fusion/native_metal_attn.py's
get_flash_core_kernel_fp16_kvcache / make_hybrid_attention_forward_flash_fp16_kvcache
for how it's wired into a real TinyViT Attention module's forward().

This script reproduces, from a fresh run against the real pretrained
models (not copied from any prior scratch session):
  (a) correctness (logit-level max-abs-diff + top-1 match vs. unmodified
      baseline) on all 3 model scales,
  (b) end-to-end speed at batch sizes 1/4/16/32/64, multi-trial with
      cooldown pauses (this machine is a fanless MacBook Air; see Phase 7's
      module docstring for the thermal-contamination failure mode this
      guards against -- re-discovered here at bs=32/64 during this kernel's
      own development, same fix),
  (c) the 20k-image ImageNet-1k accuracy eval, reusing the existing cached
      stratified subset from Phase 7's 20k robustness run (the cache is
      keyed only by images_per_class, not by model -- the images and labels
      don't depend on which model will classify them),
  (d) real K/V memory-traffic bytes moved, old design vs. this one,
  (e) measured (torch.profiler) + source-verified kernel-launch/dispatch
      counts for the attention core specifically, old path vs. this one.
"""

import argparse
import json
import pickle
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from tinyvit_roofline.config import MODEL_REGISTRY, RAW_DIR

SCRATCH_DIR = Path("/private/tmp/claude-501/-Users-anuragdhiman-Documents-tinyViT/ac31620d-7359-4957-aff3-df1814dc89cd/scratchpad")
SUBSET_CACHE = SCRATCH_DIR / "imagenet_subset_cache.pkl"
ACCURACY_ARM_RESULT_PATH = SCRATCH_DIR / "phase24_accuracy_arm_result.json"
SPEED_RESULT_PATH = SCRATCH_DIR / "phase24_speed_result.json"
IMAGES_PER_CLASS = 20
ACCURACY_BATCH_SIZE = 32
ACCURACY_ARMS = ("pytorch_baseline", "kvcache_fp16")

MODEL_WARMUP, MODEL_ITERS = 15, 50
SPEED_BATCH_SIZES = (1, 4, 16, 32, 64)
# bs=16's trial count/cooldown were bumped to match bs=1's protocol after the
# first full run showed a within-run thermal drift at bs=16 for 5M/11M (a
# clear downward-then-recovering trend across the 12 sequential trials, not
# just elevated noise) that per-model subprocess isolation alone didn't
# prevent -- the per-model isolation guards against carryover BETWEEN
# models/batch sizes, not drift accumulating WITHIN one batch size's own
# back-to-back trials. 21M's bs=16 run (one low early trial, otherwise flat)
# didn't show this pattern and didn't need the fix.
TRIAL_PLAN = {1: 15, 4: 12, 16: 18, 32: 12, 64: 12}
START_COOLDOWN_S = 20
# bs=16's 3s inter-trial cooldown resolved 5M (flat, std 0.009) but NOT 11M --
# that run still drifted down from ~1.04x to ~0.84x over its last 8 trials,
# meaning 11M's heavier per-call compute generates more heat per trial than
# 3s dissipates. 6s is the value that actually holds for the larger models,
# not just a guess -- confirmed by re-running 11M with it.
INTER_TRIAL_COOLDOWN_S = {1: 3, 4: 1.5, 16: 6, 32: 1.5, 64: 1.5}


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


def _stats(xs):
    return {"mean": statistics.mean(xs), "std": statistics.stdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs), "max": max(xs), "n": len(xs)}


def patch_ws14_with_kvcache(model, variant: str):
    from tinyvit_roofline.models import enumerate_attention_blocks, get_module_by_name
    from tinyvit_roofline.fusion.native_metal_attn import make_hybrid_attention_forward_flash_fp16_kvcache

    blocks = enumerate_attention_blocks(model, variant)
    n_patched = 0
    real_shape = None
    for bi in blocks:
        if bi.window_size == 14:
            block = get_module_by_name(model, bi.module_name)
            attn = block.attn
            if real_shape is None:
                real_shape = {"N": bi.window_size * bi.window_size, "num_heads": attn.num_heads,
                               "key_dim": attn.key_dim, "val_dim": attn.val_dim, "out_dim": attn.out_dim}
            attn.forward = types.MethodType(make_hybrid_attention_forward_flash_fp16_kvcache(attn, bi.window_size), attn)
            n_patched += 1
    return n_patched, real_shape


def correctness_check(variant: str) -> dict:
    from tinyvit_roofline.models import load_baseline_model

    baseline_model = load_baseline_model(variant)
    kv_model = load_baseline_model(variant)
    n_patched, real_shape = patch_ws14_with_kvcache(kv_model, variant)

    torch.manual_seed(0)
    x_check = torch.randn(1, 3, 224, 224, device="mps")
    with torch.no_grad():
        ref = baseline_model(x_check)
        out = kv_model(x_check)
    torch.mps.synchronize()
    max_abs_diff = (out - ref).abs().max().item()
    top1_match = bool((out.argmax(-1) == ref.argmax(-1)).all().item())
    return {"n_patched": n_patched, "real_shape": real_shape,
            "max_abs_diff": max_abs_diff, "top1_match": top1_match}


def speed_benchmark(variant: str, batch_sizes=SPEED_BATCH_SIZES,
                     n_trials_override=None, cooldown_override=None) -> dict:
    """n_trials_override/cooldown_override: used for the lighter latency-only
    pass (see latency_benchmark_isolated) -- absolute latency is a single
    noisy quantity, not a difference of two noisy things like the speedup
    ratio, so it doesn't need the full per-batch-size trial counts/cooldowns
    TRIAL_PLAN/INTER_TRIAL_COOLDOWN_S were tuned for (resolving speedup
    ratios close to 1.0x against this machine's thermal drift). Default
    (None) keeps using those, unchanged, for the rigorous speedup-ratio path."""
    from tinyvit_roofline.models import load_baseline_model

    baseline_model = load_baseline_model(variant)
    kv_model = load_baseline_model(variant)
    patch_ws14_with_kvcache(kv_model, variant)

    results = {}
    for bs in batch_sizes:
        print(f"[24]   {variant} bs={bs}: cooldown {START_COOLDOWN_S}s ...", flush=True)
        time.sleep(START_COOLDOWN_S)
        x = torch.randn(bs, 3, 224, 224, device="mps")
        speedups, base_ms, kv_ms = [], [], []
        n_trials = n_trials_override if n_trials_override is not None else TRIAL_PLAN[bs]
        cooldown = cooldown_override if cooldown_override is not None else INTER_TRIAL_COOLDOWN_S[bs]
        for t in range(n_trials):
            t_base = _time_call(lambda: baseline_model(x), MODEL_WARMUP, MODEL_ITERS)
            t_kv = _time_call(lambda: kv_model(x), MODEL_WARMUP, MODEL_ITERS)
            speedups.append(t_base / t_kv)
            base_ms.append(t_base * 1e3)
            kv_ms.append(t_kv * 1e3)
            time.sleep(cooldown)
        s = _stats(speedups)
        base_stats, kv_stats = _stats(base_ms), _stats(kv_ms)
        print(f"[24]   {variant} bs={bs}: mean={s['mean']:.4f}x std={s['std']:.4f} (n={s['n']}) "
              f"baseline={base_stats['mean']:.2f}ms kvcache={kv_stats['mean']:.2f}ms", flush=True)
        results[bs] = {
            "speedups": speedups, "stats": s,
            "baseline_latency_ms_stats": base_stats,
            "kvcache_latency_ms_stats": kv_stats,
            "baseline_throughput_img_per_s": bs / (base_stats["mean"] / 1e3),
            "kvcache_throughput_img_per_s": bs / (kv_stats["mean"] / 1e3),
        }
    return results


LATENCY_PASS_N_TRIALS = 8
LATENCY_PASS_COOLDOWN_S = 2.5


def latency_benchmark_isolated(variant: str, batch_sizes=SPEED_BATCH_SIZES) -> dict:
    """Lighter pass, same subprocess isolation, just to get absolute
    latency_ms/throughput numbers for the report -- not re-resolving the
    speedup ratio (already done, rigorously, by speed_benchmark_isolated).
    Absolute latency is a single noisy quantity, not a difference of two
    noisy things close to 1.0x, so it converges with far fewer trials."""
    return speed_benchmark_isolated(variant, batch_sizes=batch_sizes,
                                     n_trials_override=LATENCY_PASS_N_TRIALS,
                                     cooldown_override=LATENCY_PASS_COOLDOWN_S)


def speed_benchmark_isolated(variant: str, batch_sizes=None,
                              n_trials_override=None, cooldown_override=None) -> dict:
    """Runs speed_benchmark(variant) in its OWN fresh subprocess. Needed
    because the first run of this phase script (in-process, one variant
    after another, right after each variant's own 20k-image accuracy eval)
    showed the exact cross-run contamination pattern this project has hit
    and fixed before (Phase 7's module docstring; re-discovered during this
    kernel's own development at bs=32/64): a batch size's mean coming back
    inconsistent with its neighbors and with a much larger std (e.g. 11M's
    bs=4 measured 0.88x -- a loss -- surrounded by bs=1/16/32/64 all >=1.0x
    in the same run). Isolating each variant's speed benchmark in a fresh
    process, the same fix already applied to the accuracy eval above,
    removes that carryover."""
    print(f"[24]   {variant}: speed benchmark in a fresh subprocess ...", flush=True)
    if SPEED_RESULT_PATH.exists():
        SPEED_RESULT_PATH.unlink()
    cmd = [sys.executable, str(Path(__file__).resolve()), "--speed-only", "--variant", variant]
    if batch_sizes is not None:
        cmd += ["--speed-only-bs", ",".join(str(b) for b in batch_sizes)]
    if n_trials_override is not None:
        cmd += ["--speed-only-n-trials", str(n_trials_override)]
    if cooldown_override is not None:
        cmd += ["--speed-only-cooldown", str(cooldown_override)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if not SPEED_RESULT_PATH.exists():
        print(f"[24]   ERROR: speed benchmark subprocess failed. stderr tail:\n{proc.stderr[-2000:]}")
        return {"error": "speed benchmark subprocess failed", "stderr": proc.stderr[-4000:]}
    with open(SPEED_RESULT_PATH, "rb") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_subset():
    with open(SUBSET_CACHE, "rb") as f:
        cached = pickle.load(f)
    assert cached["images_per_class"] == IMAGES_PER_CLASS, (
        f"cached subset has images_per_class={cached['images_per_class']}, expected {IMAGES_PER_CLASS}"
    )
    return cached["samples"]


def _preprocess_batches(transform, samples, batch_size):
    batches = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        x_cpu = torch.stack([transform(img) for img, _ in batch])
        labels = torch.tensor([lbl for _, lbl in batch])
        batches.append((x_cpu, labels))
    return batches


def run_accuracy_arm(arm: str, variant: str) -> dict:
    from timm.data import resolve_data_config, create_transform
    from tinyvit_roofline.models import load_baseline_model

    samples = load_subset()
    data_cfg = resolve_data_config({}, model=load_baseline_model(variant))
    transform = create_transform(**data_cfg, is_training=False)
    batches = _preprocess_batches(transform, samples, ACCURACY_BATCH_SIZE)

    model = load_baseline_model(variant)
    if arm == "kvcache_fp16":
        patch_ws14_with_kvcache(model, variant)

    with torch.no_grad():
        x_warm = batches[0][0].to("mps")
        for _ in range(5):
            model(x_warm)
        torch.mps.synchronize()

        correct1, correct5, n = 0, 0, 0
        all_preds = []
        t0 = time.time()
        for x_cpu, labels in batches:
            x = x_cpu.to("mps")
            logits = model(x).float().to("cpu")
            top5 = logits.topk(5, dim=-1).indices
            correct1 += (top5[:, 0] == labels).sum().item()
            correct5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
            n += len(labels)
            all_preds.append(top5[:, 0])
        torch.mps.synchronize()
        compute_time = time.time() - t0

    preds = torch.cat(all_preds).tolist()
    return {"arm": arm, "top1_acc": correct1 / n, "top5_acc": correct5 / n, "n": n,
            "compute_time_s": compute_time, "preds": preds}


def accuracy_eval_20k(variant: str) -> dict:
    results = {}
    for i, arm in enumerate(ACCURACY_ARMS):
        print(f"[24]   {variant}: evaluating {arm} on 20k images in a fresh subprocess ...", flush=True)
        if ACCURACY_ARM_RESULT_PATH.exists():
            ACCURACY_ARM_RESULT_PATH.unlink()
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--accuracy-arm", arm, "--variant", variant],
            capture_output=True, text=True,
        )
        if not ACCURACY_ARM_RESULT_PATH.exists():
            print(f"[24]   ERROR: {arm} subprocess failed. stderr tail:\n{proc.stderr[-2000:]}")
            return {"error": f"{arm} subprocess failed", "stderr": proc.stderr[-4000:]}
        r = json.loads(ACCURACY_ARM_RESULT_PATH.read_text())
        print(f"[24]     {arm}: top1={r['top1_acc']*100:.3f}% top5={r['top5_acc']*100:.3f}% "
              f"compute={r['compute_time_s']:.1f}s", flush=True)
        results[arm] = r
        if i < len(ACCURACY_ARMS) - 1:
            time.sleep(15)

    base_preds = results["pytorch_baseline"]["preds"]
    kv_preds = results["kvcache_fp16"]["preds"]
    agreement = sum(a == b for a, b in zip(base_preds, kv_preds)) / len(base_preds)
    delta_top1_pp = (results["kvcache_fp16"]["top1_acc"] - results["pytorch_baseline"]["top1_acc"]) * 100
    delta_top5_pp = (results["kvcache_fp16"]["top5_acc"] - results["pytorch_baseline"]["top5_acc"]) * 100
    speedup_20k_run = results["pytorch_baseline"]["compute_time_s"] / results["kvcache_fp16"]["compute_time_s"]
    n_images = results["pytorch_baseline"]["n"]
    del results["pytorch_baseline"]["preds"]
    del results["kvcache_fp16"]["preds"]

    return {
        "num_images": n_images, "images_per_class": IMAGES_PER_CLASS,
        "results": results,
        "top1_prediction_agreement_vs_baseline": agreement,
        "accuracy_difference_top1_pp": delta_top1_pp,
        "accuracy_difference_top5_pp": delta_top5_pp,
        "compute_speedup_over_20k_run": speedup_20k_run,
    }


def memory_traffic_analysis() -> dict:
    """K/V bytes moved between device and threadgroup/registers, per
    (window, head), for the old per-tile design (each of ceil(N/8) query
    tiles independently re-reads the full K/V for its (window, head)) vs.
    this phase's kvcache design (K/V staged into threadgroup memory
    exactly once, shared by every query tile's simdgroup). Pure arithmetic
    from the real compile-time shapes the kernel templates actually use
    (N=196 from window_size=14; D=32 from TinyViT's fixed key_dim=val_dim
    across all three model scales -- verified directly against the live
    model registry, not assumed) -- not measured at runtime, but not
    estimated either: these are the exact byte counts the kernel source
    computes and dispatches on.
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tinyvit_roofline.models import load_baseline_model, enumerate_attention_blocks, get_module_by_name

    BYTES_PER_HALF = 2
    out = {}
    for variant in MODEL_REGISTRY:
        model = load_baseline_model(variant)
        infos = enumerate_attention_blocks(model, variant)
        ws14 = [bi for bi in infos if bi.window_size == 14]
        bi = ws14[0]
        block = get_module_by_name(model, bi.module_name)
        attn = block.attn
        N = bi.window_size * bi.window_size
        D_k, D_v = attn.key_dim, attn.val_dim
        num_heads = attn.num_heads
        num_tiles = (N + 7) // 8  # same formula the kernel template uses

        kv_bytes_per_head = (N * D_k + N * D_v) * BYTES_PER_HALF
        old_bytes_per_head = kv_bytes_per_head * num_tiles  # re-read once per query tile
        new_bytes_per_head = kv_bytes_per_head  # staged once, shared

        old_total_bs1 = old_bytes_per_head * num_heads  # batch_size=1 -> 1 window at ws=14
        new_total_bs1 = new_bytes_per_head * num_heads

        out[variant] = {
            "N": N, "key_dim": D_k, "val_dim": D_v, "num_heads": num_heads,
            "num_query_tiles": num_tiles,
            "kv_bytes_per_window_head_fp16": kv_bytes_per_head,
            "old_design_bytes_per_window_head": old_bytes_per_head,
            "new_design_bytes_per_window_head": new_bytes_per_head,
            "redundancy_factor_removed": num_tiles,
            "pct_kv_traffic_reduction": (1 - 1 / num_tiles) * 100,
            "old_design_total_bytes_batch1": old_total_bs1,
            "new_design_total_bytes_batch1": new_total_bs1,
            "bytes_saved_batch1": old_total_bs1 - new_total_bs1,
        }
    return out


def kernel_launch_count_analysis(variant: str = "tiny_vit_5m_224") -> dict:
    """Measured (torch.profiler, CPU activity, aten-level op dispatch
    count) for the attention-core-specific portion of a real ws=14 block's
    forward pass, old (baseline SDPA) path vs. this phase's kvcache path --
    plus a source-verified Metal-level dispatch count for the kvcache
    kernel itself (it calls dispatchThreadgroups exactly once per
    forward(), confirmed by grepping the .mm source, since the custom
    kernel is invoked directly via pybind11 and bypasses ATen's operator
    dispatcher entirely -- invisible to torch.profiler, a concrete
    instance of this project's own previously-flagged limit on what
    torch.profiler can see on this backend).
    """
    from torch.profiler import profile, ProfilerActivity
    from tinyvit_roofline.models import load_baseline_model, enumerate_attention_blocks, get_module_by_name
    from tinyvit_roofline.fusion.native_metal_attn import make_hybrid_attention_forward_flash_fp16_kvcache

    model = load_baseline_model(variant)
    infos = enumerate_attention_blocks(model, variant)
    bi = [b for b in infos if b.window_size == 14][0]
    block = get_module_by_name(model, bi.module_name)
    attn = block.attn
    N = bi.window_size * bi.window_size
    x = torch.randn(16, N, bi.dim, device="mps")

    def profile_calls(fn, iters=10):
        with torch.no_grad():
            for _ in range(5):
                fn()
            torch.mps.synchronize()
            with profile(activities=[ProfilerActivity.CPU]) as prof:
                for _ in range(iters):
                    fn()
                torch.mps.synchronize()
        events = {e.key: e.count for e in prof.key_averages()}
        return events

    baseline_events = profile_calls(lambda: attn(x))
    kv_fwd = make_hybrid_attention_forward_flash_fp16_kvcache(attn, bi.window_size)
    kvcache_events = profile_calls(lambda: kv_fwd(attn, x))

    # Attention-core-specific events: the QKV-split-through-attention-output
    # scaffolding that SDPA's calling convention requires and this kernel's
    # direct-from-the-packed-qkv-buffer indexing does not. Everything else
    # (norm, qkv linear, proj linear, their own empty/view bookkeeping) is
    # identical cost in both paths, so it's excluded from this specific count.
    core_op_names = {"aten::split_with_sizes", "aten::expand", "aten::transpose",
                      "aten::reshape", "aten::clone", "aten::scaled_dot_product_attention",
                      "aten::_scaled_dot_product_attention_math_for_mps"}
    baseline_core_calls_per_iter = sum(v for k, v in baseline_events.items() if k in core_op_names) / 10
    # The kvcache path's custom kernel call is invisible to torch.profiler (direct pybind11
    # call, bypasses the ATen dispatcher) -- confirmed by inspecting its own event list for
    # zero matching entries, and cross-checked by source: exactly one dispatchThreadgroups
    # call per forward() in fused_attention_core_flash_fp16_kvcache_template.mm.
    kvcache_mm_path = (Path(__file__).resolve().parents[1] / "src" / "tinyvit_roofline" / "fusion" /
                        "native" / "fused_attention_core_flash_fp16_kvcache_template.mm")
    dispatch_calls_in_source = kvcache_mm_path.read_text().count("dispatchThreadgroups")

    return {
        "variant_measured_on": variant,
        "profiler_total_events_per_iter": {
            "baseline_whole_attention_module": sum(baseline_events.values()) / 10,
            "kvcache_whole_attention_module_aten_visible": sum(kvcache_events.values()) / 10,
        },
        "attention_core_specific_dispatch_count": {
            "baseline_aten_level_calls_per_iter": baseline_core_calls_per_iter,
            "kvcache_raw_metal_dispatches_per_call": dispatch_calls_in_source,
            "reduction_factor": baseline_core_calls_per_iter / dispatch_calls_in_source,
        },
        "methodology_note": (
            "baseline_aten_level_calls_per_iter counts the PyTorch-dispatcher-visible ops "
            "specific to feeding Q/K/V into SDPA and reading its output (split_with_sizes, "
            "expand, transpose, reshape, clone, scaled_dot_product_attention, and its MPS math "
            "fallback) -- SDPA's own internal Metal dispatch count is NOT visible to "
            "torch.profiler on this backend (flagged in this project's README methodology notes "
            "since Phase 1), so this is a dispatch-overhead proxy, not a verified Metal-level "
            "count, for the baseline side. kvcache_raw_metal_dispatches_per_call=1 is NOT a "
            "profiler measurement -- it is read directly from the kernel's own .mm source "
            "(exactly one dispatchThreadgroups call per forward()), because the custom kernel "
            "is called directly via pybind11 and is completely invisible to torch.profiler "
            "(zero matching events), itself a concrete instance of the same limitation."
        ),
    }


def run_all() -> dict:
    result = {
        "phase": "24_kvcache_flash_attention_benchmark",
        "methodology_note": (
            "Fuses the attention core (QK^T*scale+bias, softmax, AV) for window_size=14 blocks "
            "into one native MPS dispatch using online/streaming softmax (simdgroup_half8x8 tile "
            "ops) with K/V cached once per (window, head) in threadgroup memory and shared across "
            "all ~25 query-tile simdgroups, instead of each tile independently re-reading K/V from "
            "device memory. See this script's module docstring for the full 5-design history and "
            "why each earlier one failed."
        ),
    }

    print("[24] === memory traffic analysis ===", flush=True)
    result["memory_traffic"] = memory_traffic_analysis()
    for v, m in result["memory_traffic"].items():
        print(f"[24]   {v}: {m['redundancy_factor_removed']}x redundancy removed "
              f"({m['pct_kv_traffic_reduction']:.1f}% reduction), "
              f"{m['bytes_saved_batch1']/1024/1024:.2f} MiB saved per batch_size=1 forward pass", flush=True)

    print("\n[24] === kernel launch count analysis ===", flush=True)
    result["kernel_launch_count"] = kernel_launch_count_analysis()
    klc = result["kernel_launch_count"]["attention_core_specific_dispatch_count"]
    print(f"[24]   attention core: {klc['baseline_aten_level_calls_per_iter']:.1f} aten-level calls/iter "
          f"(baseline) -> {klc['kvcache_raw_metal_dispatches_per_call']} raw Metal dispatch (kvcache), "
          f"{klc['reduction_factor']:.1f}x reduction", flush=True)

    per_variant = {}
    for variant in MODEL_REGISTRY:
        print(f"\n[24] === {variant} ===", flush=True)
        print(f"[24] correctness check ...", flush=True)
        correctness = correctness_check(variant)
        print(f"[24]   max_abs_diff={correctness['max_abs_diff']:.3e} top1_match={correctness['top1_match']} "
              f"(patched {correctness['n_patched']} blocks, shape={correctness['real_shape']})", flush=True)

        print(f"[24] speed benchmark (bs={SPEED_BATCH_SIZES}) ...", flush=True)
        speed = speed_benchmark_isolated(variant)

        print(f"[24] 20k ImageNet accuracy eval ...", flush=True)
        accuracy = accuracy_eval_20k(variant)

        per_variant[variant] = {"correctness": correctness, "speed": speed, "accuracy_20k": accuracy}

        # write incrementally so partial progress survives any interruption
        result["per_variant"] = per_variant
        out_path = RAW_DIR / "24_kvcache_flash_attention_benchmark.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"[24] (incremental) wrote {out_path}", flush=True)

    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="tiny_vit_5m_224", choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--accuracy-arm", default=None, choices=list(ACCURACY_ARMS),
                     help="Internal: run a single accuracy arm in this process. Used by subprocess invocations.")
    ap.add_argument("--speed-only", action="store_true",
                     help="Internal: run just the speed benchmark for --variant in this process. Used by subprocess invocations.")
    ap.add_argument("--speed-only-bs", default=None,
                     help="Internal: comma-separated batch sizes to restrict --speed-only to (e.g. '16'). Used by subprocess invocations.")
    ap.add_argument("--speed-only-n-trials", type=int, default=None,
                     help="Internal: override trial count for --speed-only (used by the lighter latency pass). Used by subprocess invocations.")
    ap.add_argument("--speed-only-cooldown", type=float, default=None,
                     help="Internal: override inter-trial cooldown (seconds) for --speed-only. Used by subprocess invocations.")
    args = ap.parse_args()

    if args.accuracy_arm is not None:
        r = run_accuracy_arm(args.accuracy_arm, args.variant)
        ACCURACY_ARM_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        ACCURACY_ARM_RESULT_PATH.write_text(json.dumps(r))
        sys.exit(0)

    if args.speed_only:
        bs_filter = ([int(b) for b in args.speed_only_bs.split(",")]
                     if args.speed_only_bs else SPEED_BATCH_SIZES)
        r = speed_benchmark(args.variant, batch_sizes=bs_filter,
                             n_trials_override=args.speed_only_n_trials,
                             cooldown_override=args.speed_only_cooldown)
        SPEED_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SPEED_RESULT_PATH.write_text(json.dumps(r))
        sys.exit(0)

    result = run_all()
    out_path = RAW_DIR / "24_kvcache_flash_attention_benchmark.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[24] wrote {out_path}")
