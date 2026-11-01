# TinyViT Window Attention: Roofline Analysis & Kernel Fusion on Apple M4

Characterizing and reducing the compute, memory, and dispatch inefficiencies
of TinyViT's small-window attention on Apple Silicon through empirical
Roofline analysis and MLX/Metal kernel fusion.

Hardware: Apple M4 (10-core GPU), 16GB unified memory. Software: PyTorch
2.8.0 (MPS backend) for the standard baseline, MLX for the fused-attention
implementation.

## Setup

Requires an Apple Silicon Mac (MPS + Metal support) and Python 3.10+.

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The custom Metal kernels (`src/tinyvit_roofline/fusion/native/*.mm`) are
JIT-compiled on first use via `torch.utils.cpp_extension.load()` -- this
needs Xcode Command Line Tools installed (`xcode-select --install`) but no
separate build step.

**Headline results** (see `report.md` for the full writeup):

- **Window-attention fusion (the original hypothesis) didn't pay off.** Across
  a 60-config ablation, neither `torch.compile` (mean 0.90x) nor MLX-fused
  attention (mean 0.68x) beats the standard PyTorch baseline on average --
  both lose most at the smallest window size (4x4: MLX-fused ~2.0x slower,
  `torch.compile` ~1.5x slower). A follow-up
  test extending MLX fusion to the *whole* block (not just attention) pointed
  at why: `mx.fast.layer_norm` and MLX's depthwise `conv2d` both measured
  slower than PyTorch-MPS's equivalents at these operator sizes, so only the
  attention core (genuinely fused) ever had a chance. **That comparison sat
  as an unbacked claim for a while -- no script, no `results/raw/*.json` --
  which is itself a bug given this project's whole premise; Phase 15 fixed
  that with a proper isolated microbenchmark (real captured shapes, real
  ported weights, correctness-checked).** The real numbers: `mx.fast.layer_norm`
  is **~14x slower** (not ~10x), and MLX's depthwise `conv2d` is **~12x
  slower** (not ~3x) -- both worse than the original ad-hoc estimate,
  especially conv2d. Correctness holds throughout (max-abs-diff ~1e-6,
  section 6 of `report.md`, "Verifying why MLX fusion doesn't pay off").
  Neither implementation is wrong -- MLX-fused
  attention is numerically exact (15/15 configs, max diff ~1e-7) -- the
  baseline's fused-SDPA attention path was just already a high bar, and
  MLX's non-attention primitives are measurably worse than previously
  stated, not better.
- **But kernel fusion *does* beat baseline -- once MLX itself is taken out
  of the equation (Phase 22).** Every MLX-based fusion attempt above fails
  for the same root cause: MLX's Python->Metal dispatch path has a fixed
  ~300-400us floor per call, confirmed directly -- a custom
  `mx.fast.metal_kernel` doing the *exact same fused computation* as below,
  timed with literally zero compute inside it, cost the same ~370-400us as
  the full version. That floor is specific to MLX's Python bindings, not
  this hardware. Bypassing it -- the identical fused kernel (LayerNorm +
  QKV projection + attention + output projection, one Metal dispatch),
  compiled as a native PyTorch C++/Metal extension and encoded directly
  into PyTorch's own MPS command stream (`torch.utils.cpp_extension` +
  `at::mps::getCurrentMPSStream()`, no MLX at all) -- **beats baseline
  1.69x at batch_windows=16** (window_size=4, model batch_size=1), the
  *exact* operating point the original 60-config ablation found *worst*
  for every other approach (torch.compile and MLX-fused attention both
  lost ~2x there). Numerically exact (max-abs-diff ~1.5e-7). The win is
  real but narrow: it holds up to roughly batch_windows=40-56, crosses to
  parity somewhere around 48-64 (this exact crossover point has some
  run-to-run variance), and reverses into a clear, stable loss above
  ~64 (0.39-0.40x at 256/512)
  -- because this kernel's hand-written scalar/float4 loops are less
  efficient than Apple's vendor-tuned GEMM once raw compute, not dispatch
  overhead, becomes the bottleneck. Same mechanism as every finding above,
  working in the opposite direction: small ops are dispatch-bound (fusion
  wins), bigger ones are compute-bound (fusion needs real GEMM tiling,
  not attempted here, to keep winning). **But this win does *not* transfer
  to any real TinyViT block** -- window_size=4 never appears in any of
  this project's three model variants (all use (7,7,14,7)); it's a
  synthetic ablation-grid point. Tested directly on 5M's real stage-1
  module (window_size=7, dim=128): the native kernel loses at every batch
  size (0.29-0.45x), both because that config needs 4x the M4's hard
  32768-byte threadgroup memory limit (confirmed directly) -- forcing a
  slower device-memory-scratch variant -- and because N=49 tokens is
  already big enough that baseline's dispatch overhead stops dominating.
  **The honest takeaway: this is a real result about the mechanism
  (native dispatch beats MLX's floor; dispatch-bound vs. compute-bound
  regimes), not a deployable speedup for this model.** Section 17 of
  `report.md`.
- **Fixed that, with a better-motivated design, and it does win on a
  real block -- just not enough to matter end-to-end (section 18).**
  Section 17 failed on real blocks partly because full-block fusion
  doesn't fit real sizes on-chip. The actual fix: PyTorch's own
  `LayerNorm`/`Linear` were never the problem, so fuse *only* the
  attention core (QK^T + bias, softmax, AV) into one native dispatch and
  leave PyTorch's native ops for everything else. On 5M/11M/21M's real
  stage-1 block (window_size=7): a genuine, reproducible **1.13-1.34x**
  win at every scale, never losing, correctness verified. Extending to
  window_size=14 (via row-tiling, since N=196's score matrix doesn't fit
  on-chip either) actively **hurts** instead -- confirmed directly, not
  assumed -- so the full end-to-end model only patches window_size=7
  blocks (4 of 10 attention blocks). Against an unmodified baseline, that
  end-to-end effect is statistically noise (0.96-1.01x). **Tested one
  level further: does it add anything stacked on `build_best_model`
  (conv+BN fusion + channels_last, this project's actual deployable
  win), not just an unmodified baseline?** Also no -- 0.97-1.02x,
  correctness preserved throughout, same noise-level result in the most
  realistic context available. **At the time, the honest answer was: kernel
  fusion genuinely speeds up the exact operation it targets on real TinyViT
  inference -- it just never targets enough of the model's total time to
  show up end-to-end.** window_size=14's half of that story turned out to
  be fixable -- see the next finding.
- **Revisited window_size=14 with a fundamentally different kernel design,
  and it wins end-to-end (section 19).** The row-tiled kernel above lost at
  window_size=14 because N=196 is big enough that raw compute, not dispatch
  overhead, was the bottleneck -- and that diagnosis pointed at the real
  fix: a **flash-attention-style kernel (online/streaming softmax, so the
  196x196 score matrix never materializes in device memory) with K and V
  cached ONCE per (window, head) in threadgroup memory and shared across
  all ~25 query-tile simdgroups**, instead of each tile independently
  re-reading the same K/V from device memory 25 times over -- the actual
  bottleneck, identified directly, not assumed. Five designs were tried
  getting here (naive `simdgroup_matrix` GEMM, 0.80-0.94x; a textbook
  shared-memory-blocked GEMM, which regressed to 0.20x; fp32 flash
  attention, 0.77-0.97x; fp16 flash attention without caching, a narrow
  win only at batch_size=64) before this one. Verified on real pretrained
  weights, all three model scales: correctness holds (logit max-abs-diff
  2.8e-4 to 1.1e-3, top-1 match exact), and it's a **real, reproducible
  end-to-end win at every batch size on every model** (1.02-1.08x,
  multi-trial with cooldown pauses, the same thermal-contamination fix
  this project established in Phase 7 and had to re-apply here). K/V
  memory traffic drops 96% (25x fewer redundant reads, measured from the
  kernel's own tiling arithmetic) and the attention core's dispatch count
  drops 7x (measured via `torch.profiler`). Re-ran the full 20k-image
  ImageNet-1k accuracy eval on all three models: every accuracy delta is
  noise-level (±0.01pp), exactly like this project's other fp16 arm.
- **The same audit found a second unbacked claim, and this one went the
  other way.** `mx.compile`'s benefit over an uncompiled, re-tracing MLX
  call had been stated as "~4x" with no script behind it either -- Phase
  15 measured the real attention module and found **~1.3x**, not ~4x.
  Still a real, consistent win (never slower, across all three stages
  tested), just much smaller than claimed. The lesson generalizes past
  this one number: an unverified estimate can be wrong in either
  direction, and the only way to know which is to actually measure it --
  which is this project's whole methodology, now applied to its own
  previously-unchecked claims. **Re-run on 11M/21M, not just 5M:** the
  layer_norm/conv2d slowdowns hold roughly constant across scale (~12-15x
  and ~12-17x, no trend either direction), while `mx.compile`'s benefit is
  small *and* inconsistent (1.3x/1.0x/1.4x, with individual configs ranging
  0.9x-1.9x) -- not a lever worth relying on at any scale tested.
- **The empirical roofline (section 3) names `local_conv` as the worst-
  utilized region in the model at every scale tested** -- 5.5% (5M), 7.3%
  (11M), 9.8% (21M) of its own roofline bound, last place by a wide margin
  every time (next-worst region is 2-3x better utilized). It's the real-
  world counterpart to Phase 15's MLX conv2d finding above: `local_conv` is
  exactly the operator MLX's native-NHWC conv2d should in principle help
  with (no PyTorch layout permute needed), except MLX's own conv2d kernel
  is ~12-17x slower at these sizes on this hardware, so that theoretical
  edge never materializes in practice.
- **fp16 doesn't close MLX's primitive gap either (Phase 16).** Phase 14's
  biggest PyTorch-MPS speedup lever applied to MLX's own layer_norm/conv2d
  kernels, on the theory that Apple GPUs often favor fp16 over fp32 more
  than PyTorch does. It didn't help: the slowdown stays in the same
  10-18x range at fp16 across all three model scales, sometimes slightly
  worse (layer_norm, 5M: 14.3x -> 17.7x), sometimes slightly better
  (conv2d, every scale, but never by more than ~25%) -- no consistent
  direction, and nowhere close to closing a 10x+ gap either way. Whatever
  is costing MLX its disadvantage here, it isn't an fp32-specific
  artifact.
- **MLX's int8 quantized matmul doesn't help either, and PyTorch-MPS can't
  even try it (Phase 17).** PyTorch's native quantization ops are not
  implemented for the MPS backend at all (`aten::quantize_per_tensor`
  raises `NotImplementedError`, confirmed directly) -- a real capability
  gap favoring MLX, which ships `mx.quantize`/`mx.quantized_matmul`.
  Tested on exactly the bottleneck fp16 already targets
  (`linear_qkv_proj_or_mlp`'s `qkv_proj`/`mlp.fc1`): it doesn't matter.
  MLX's int8 matmul is *slower than MLX's own fp32 matmul*, at every
  config and every scale tested (dequantization overhead dominates these
  small per-call sizes) -- and PyTorch's fp16 beats both MLX arms outright
  everywhere (1.05-1.32x vs. MLX fp32/int8's 0.2-0.6x). A capability
  PyTorch-MPS genuinely lacks turns out not to matter here, because the
  thing it would have competed against was never close to winning anyway.
- **The actual working optimization was conv+BN fusion**, redirected there by
  Phase 1's own diagnostics ("conv" was the largest dispatch-time category,
  more than qkv/mlp/attention combined). Fusing every `ConvNorm`'s
  BatchNorm into its conv at inference -- a standard, exact technique, not
  an experimental one -- gives a real, reproducible **~1.2-1.3x speedup**
  (1.13-1.35x across batch sizes 1-32, isolated benchmark; **1.31x**
  end-to-end on 1000 real ImageNet-1k images with accuracy fully preserved
  -- 81.20% top-1 / 95.30% top-5, identical across every implementation
  tested, verified via raw logit comparison, not just accuracy).
- **This machine is a fanless MacBook Air, and it shows.** Getting a trustworthy
  end-to-end comparison required evaluating each implementation in its own
  subprocess with a cooldown pause between them -- an in-process,
  four-models-in-a-row version of the same eval gave numbers that flatly
  contradicted the isolated benchmark (every non-baseline arm looked ~3x
  *slower*, uniformly, regardless of how different the implementations
  actually are), and the exact same function with identical arguments
  timed 5.4x differently depending only on recent GPU load. See finding
  #10 in `report.md`.
- **Conv+BN fusion generalizes across model scale, but with diminishing
  returns.** Re-run on TinyViT-11M (1.12x) and TinyViT-21M (1.08x), not just
  5M (1.18x) -- positive everywhere, but shrinking as the model grows, since
  wider embedding dimensions shift relative compute from conv toward
  attention/MLP, which this optimization doesn't touch.
- **Conv+BN fusion and MLX attention fusion do *not* stack.** Combining both
  into one model gives a *mean 0.46x* -- more than 2x slower than baseline,
  worse than either optimization alone, despite correctness being preserved
  (section 9 of `report.md`). MLX's per-block PyTorch<->MLX round trip is a
  roughly fixed cost; making the rest of the model faster (via conv fusion)
  only makes that fixed cost a *larger* share of a *smaller* total latency.
  Conv+BN fusion alone is the right choice. **The failure shrinks with model
  scale, though it never disappears** -- mean speedup 0.46x (5M) -> 0.52x
  (11M) -> 0.62x (21M), nearly reaching parity at bs=32 on 21M (0.91x) --
  the same fixed-overhead mechanism, just observed along the model-size
  axis instead of the batch-size axis.
- **But `channels_last` (NHWC) memory format *does* stack with conv+BN
  fusion** -- it's a pure PyTorch layout change with no cross-framework
  round trip, so it doesn't hit the problem MLX did. Combined gives
  **1.24x (5M) / 1.17x (11M) / 1.10x (21M)**, beating conv-fusion alone at
  every batch size and every scale tested, correctness preserved throughout
  (section 10 of `report.md`). This result also needed two real fixes along
  the way: an unseeded correctness check that compared different random
  inputs across subprocesses (looked like huge logit differences, was
  actually just four different images), and the same MPS-allocator-
  fragmentation bug from Phases 4/6 showing up again across batch sizes
  within one arm's subprocess. Re-verified end-to-end on real ImageNet-1k
  images (section 7, `pytorch_conv_bn_fused_channels_last`) with accuracy
  fully preserved -- 81.20% top-1 / 95.30% top-5, identical across all five
  implementations tested, 100% prediction agreement with baseline.
- **`torch.compile` was the third lever tried on top of the best combination,
  and it also fails** -- 0.72x mean, slower than *baseline*, not just
  slower than the uncompiled optimized model (section 11). Also crashes
  outright (`TypeError: cannot determine truth value of Relational`) if the
  same compiled model is reused across different batch sizes -- a real
  PyTorch MPS-Inductor dynamic-shape bug, not something in this project's
  code. Three-for-three now: of the levers tried on top of conv+BN fusion,
  only `channels_last` (zero cross-framework/compilation overhead) stacks.
  **Re-run on 11M/21M, it gets worse with scale, not better** -- 0.72x
  (5M) -> 0.58x (11M) -> 0.51x (21M) -- the opposite trend from MLX's
  combined-fusion failure (section 9), which improves with scale because
  MLX's overhead is roughly fixed. `torch.compile`'s cost here scales
  *with* the model instead, the clearest sign yet that this isn't a lever
  worth revisiting at any size on this hardware.
- **Re-profiling the optimized model closes the loop.** `conv` dropped 68%
  (1919 -> 615 us/iter) and is no longer the largest dispatch-time category
  -- `linear_qkv_proj_or_mlp`, untouched by either optimization, now is
  (section 12). Exactly what the roofline diagnosis predicts: fixing the
  identified bottleneck moves the bottleneck, it doesn't make dispatch
  overhead vanish. The QKV/output/MLP linears are where a continuation of
  this project should look next. **This holds at every scale tested** (1.24x/1.13x/1.12x
  end-to-end on 5M/11M/21M) -- and `linear_qkv_proj_or_mlp`'s post-fusion
  dispatch time is nearly *constant* across all three (~1188us/iter) despite
  their very different matmul sizes, a reminder that this metric is CPU
  dispatch overhead, not GPU compute time (see Methodology notes below).
- **fp16 directly targets that new bottleneck and gives the biggest speedup
  in the project -- 1.49x on top of the best combination -- but it's the
  first optimization here with a real cost.** Unlike every other lever
  (all numerically exact, ~1e-6 max-abs-diff), fp16's max-abs-diff is
  ~0.025, and on real ImageNet-1k that shows up as a measurable accuracy
  drop: 81.20% -> 81.10% top-1 (top-5 unchanged, 99.8% prediction
  agreement with baseline). Fastest arm in the whole project, but a
  genuine speed/accuracy tradeoff -- know that before choosing it over the
  exact `conv_bn_fused + channels_last` (section 13 of `report.md`).
- **fp16's speedup also shrinks with model scale** -- 1.49x (5M) -> 1.38x
  (11M) -> 1.34x (21M), mirroring conv+BN fusion and channels_last rather
  than reversing them. This runs against the intuition that motivated
  trying fp16 in the first place (that larger models, with relatively more
  compute in the QKV/MLP linears fp16 targets, would benefit *more*) --
  the data says otherwise, though fp16 remains the single biggest lever in
  the project at every scale tested, and the accuracy cost shrinks too
  (max-abs-diff 1.18e-2 -> 7.33e-3 -> 4.98e-3).
- **And that shrinking max-abs-diff shows up in real ImageNet-1k accuracy
  too.** Re-ran Phase 7's 6-arm, 1000-image accuracy eval on 11M and 21M,
  not just 5M -- fp16's top-1 delta vs. baseline goes -0.10pp (5M) -> 0.00pp
  (11M) -> +0.10pp (21M), tracking the numerical trend exactly (prediction
  agreement 99.8% -> 99.7% -> 99.9%). Each delta is within a single-image's
  worth of noise at this sample size, so this doesn't *prove* fp16 gets
  safer at scale -- but it does rule out the opposite, that fp16's accuracy
  risk grows as the model does (section 7 of `report.md`). **Pushed this
  all the way to 20,000 images on 5M (20x the original sample, ~0.5-0.6pp
  95% CI vs. the original's ~2.5pp) and the delta lands at -0.01pp** --
  indistinguishable from zero. The progression (1000: -0.10pp, 5000:
  +0.02pp, 10000: -0.02pp, 20000: -0.01pp) makes the conclusion
  unambiguous: the original -0.10pp was sampling noise, not a real fp16
  accuracy cost at this scale. Two real bugs surfaced and fixed getting
  here: (1) a per-arm subprocess never threaded a non-default
  `--images-per-class` through to its own cache lookup, silently falling
  back to the 1-image default; (2) at 10,000+ images, holding every image
  at native resolution pushed memory high enough to trigger a kernel OOM
  kill (confirmed directly: exit code 137/SIGKILL, no traceback possible)
  on this machine's shared 16GB -- fixed by downsizing each image to
  256px at collection time instead of holding thousands at full
  resolution.
- **int8 catches up to MLX's own fp32 matmul at bigger GEMM sizes, but
  never beats it, and it doesn't matter anyway.** Phase 17's int8-vs-mlx_fp32
  comparison was measured at small batch sizes; sorted by the real matmul
  row count (not the nominal batch-size label, which undercounts windowed
  attention's actual GEMM size), int8's disadvantage shrinks from ~0.68x
  at the smallest size tested to ~1.00x (parity) at the largest --
  confirming dequantization overhead is a fixed-ish cost that matters less
  as the matmul grows. It never clearly wins, though, and MLX's fp32
  matmul itself stays ~2x slower than plain PyTorch fp32 across the whole
  range, so this doesn't change the conclusion that PyTorch's fp16 wins
  regardless of GEMM size.
- **bf16 ties fp16 on speed and is ~10x less numerically precise -- no
  reason to prefer it here.** An untested alternative half-precision
  format (Apple GPUs run it natively): same 1.47x mean speedup as fp16 on
  the best combination, but max-abs-diff of ~0.115 vs. fp16's ~0.012.
  bf16 trades mantissa bits for exponent range neural-net activations at
  this scale don't need; fp16 is strictly better on this backend for this
  model.
- **Batching never pushes TinyViT into the compute-bound regime, even at
  128.** Every benchmark in this project stopped at batch_size=32 --
  extending to 64/128 with an exact (not estimated) total-model FLOP count
  shows achieved throughput plateauing at just ~32% of the empirically
  measured compute peak (26.2% at batch=1 -> 32.1% at batch=128, with
  almost all the gain already captured by batch=16). TinyViT at these
  operator sizes stays memory/dispatch-bound at every practical batch
  size on this hardware -- batching further wouldn't meaningfully help.
- **`torch.compile` fails on `best_fp16` too, the actual fastest arm in
  the project, not just on the exact `best` combination.** Worth checking
  separately since compilers sometimes handle fp16's simpler, more
  uniform kernels differently than fp32's codegen paths -- it doesn't
  here: best_fp16 1.48x -> best_fp16_compiled 0.80x, a real regression
  (though less bad than compiling fp32 `best`, which drops to 0.72x).
  `torch.compile` now fails on every combination tried in this project,
  at every precision tested.

## Project layout

```
src/tinyvit_roofline/
  config.py                 Model registry, sweep grids, output paths
  models.py                 Standard timm TinyViT loading + block introspection
  profiling/                torch.profiler wrappers, op classification, .gputrace capture
  roofline/                 Empirical P_max/B_max, per-stage AI estimators, isolated op microbenchmarks, plotting
  microbench/                Controlled window-attention sweep
  fusion/                    pytorch_optimized (torch.compile), mlx_fused (MLX), conv_bn_fusion, weight porting, correctness checks, hybrid model
  ablation/                  Implementation-comparison ablation grid
  report/                    Aggregation + supporting plots
scripts/                     One CLI entry point per phase (01-24), each independently runnable
results/
  raw/       Timestamped JSON per phase
  traces/    torch.profiler chrome traces
  plots/     PNG figures
report.md     Generated final write-up
```

## Using the actual result

Everything below this is the research pipeline. If you just want the one
validated optimization this project produced, applied to a real model:

```python
from tinyvit_roofline.fusion.best_model import build_best_model
import torch

model = build_best_model("tiny_vit_5m_224")         # exact, ~1.1-1.3x
x = torch.randn(1, 3, 224, 224, device="mps").to(memory_format=torch.channels_last)
out = model(x)

model_fast = build_best_model("tiny_vit_5m_224", fp16=True)   # fastest, small accuracy cost
out_fast = model_fast(x.half())
```

This is conv+BN fusion + channels_last (+ optional fp16), the one
combination validated end-to-end on real ImageNet-1k images (sections
7/8/10/13 of `report.md`) to beat baseline with correctness preserved.
Every other lever in this project (MLX attention fusion, `torch.compile`,
MLX int8 quantization, bf16) either failed outright or doesn't generalize
past a narrow synthetic config.

**If your model has window_size=14 attention stages specifically** (not
yet stacked into `build_best_model` above -- a separate integration step),
there is a second real, verified win:

```python
import types
from tinyvit_roofline.models import load_baseline_model, enumerate_attention_blocks, get_module_by_name
from tinyvit_roofline.fusion.native_metal_attn import make_hybrid_attention_forward_flash_fp16_kvcache

model = load_baseline_model("tiny_vit_5m_224")
for bi in enumerate_attention_blocks(model, "tiny_vit_5m_224"):
    if bi.window_size == 14:
        attn = get_module_by_name(model, bi.module_name).attn
        attn.forward = types.MethodType(make_hybrid_attention_forward_flash_fp16_kvcache(attn, bi.window_size), attn)
```

A K/V-cached fp16 flash-attention kernel (`fusion/native_metal_attn.py`,
Metal source in `fusion/native/fused_attention_core_flash_fp16_kvcache_template.mm`)
-- the only native-kernel attempt in this project that generalizes past a
narrow synthetic config: a real 1.02-1.08x end-to-end win at
every batch size, on all three model scales, correctness and real
ImageNet-1k accuracy both verified (section 19 of `report.md`).

## Running

Each phase writes to `results/raw/` and can be run standalone:

```
python3 scripts/01_baseline_profile.py --variant tiny_vit_5m_224   # also run on 11m/21m (feeds section 12's comparison)
python3 scripts/02_hw_roofline.py
python3 scripts/03_stage_roofline.py --variant tiny_vit_5m_224   # also run on 11m/21m
python3 scripts/04_window_attention_microbench.py
python3 scripts/05_fusion_correctness.py
python3 scripts/06_ablation.py
python3 scripts/15_mlx_primitive_microbench.py --variant tiny_vit_5m_224   # verifies why MLX fusion doesn't pay off (section 6); also run on 11m/21m
python3 scripts/16_mlx_fp16_primitive_microbench.py --variant tiny_vit_5m_224   # does fp16 close MLX's primitive gap? (no); also run on 11m/21m
python3 scripts/17_mlx_quantized_linear_microbench.py --variant tiny_vit_5m_224   # MLX int8 quantized matmul vs pytorch fp16 (pytorch fp16 wins); also run on 11m/21m
python3 scripts/19_mlx_quantized_linear_batch_sweep.py --variant tiny_vit_5m_224   # does int8 catch up at bigger GEMM sizes? (converges to ~parity, never wins)
python3 scripts/07_accuracy_eval.py --variant tiny_vit_5m_224   # also run on 11m/21m (tests fp16's accuracy cost across scale); --images-per-class 20 --output-suffix _20k for the full-confidence robustness check (20,000 images, ~0.5pp 95% CI)
python3 scripts/09_conv_fusion_benchmark.py --variant tiny_vit_5m_224   # the working optimization; also run on 11m/21m
python3 scripts/10_combined_fusion_benchmark.py --variant tiny_vit_5m_224   # conv+BN fusion + MLX attention (doesn't stack); also run on 11m/21m
python3 scripts/11_channels_last_benchmark.py --variant tiny_vit_5m_224   # conv+BN fusion + channels_last (stacks); also run on 11m/21m
python3 scripts/12_compile_stack_benchmark.py --variant tiny_vit_5m_224   # best combination + torch.compile (doesn't stack); also run on 11m/21m
python3 scripts/13_optimized_profile.py --variant tiny_vit_5m_224   # re-profile the optimized model; also run on 11m/21m
python3 scripts/14_fp16_benchmark.py --variant tiny_vit_5m_224   # fp16 on top of the best combination (biggest speedup, real accuracy cost); also run on 11m/21m
python3 scripts/18_bf16_benchmark.py --variant tiny_vit_5m_224   # bf16 vs fp16 (tied on speed, much less precise -- fp16 wins)
python3 scripts/20_ridge_point_batch_sweep.py --variant tiny_vit_5m_224   # does batching ever cross into compute-bound? (no, plateaus ~32% of P_max)
python3 scripts/21_compile_fp16_stack_benchmark.py --variant tiny_vit_5m_224   # torch.compile on top of best_fp16 (also fails)
python3 scripts/22_native_kernel_fusion_benchmark.py   # kernel fusion via native dispatch, bypassing MLX -- wins at small batch on a synthetic config; doesn't transfer to a real block (also checked)
python3 scripts/23_hybrid_attention_core_benchmark.py --variant tiny_vit_5m_224   # fuse only the attention core on REAL blocks -- real, reproducible op-level win; ~noise end-to-end (also run on 11m/21m)
python3 scripts/24_kvcache_flash_attention_benchmark.py   # fixes window_size=14 with a K/V-cached fp16 flash-attention kernel -- real end-to-end win, all 3 models (correctness + speed + 20k accuracy, no --variant flag: runs all three in one script)
python3 scripts/08_report.py --variant tiny_vit_5m_224   # run last -- aggregates everything above
```

or all in order via `python3 scripts/run_all.py`.

Phase 7 (ImageNet-1k accuracy) streams the HF `ILSVRC/imagenet-1k`
validation split -- requires accepting that dataset's gated-access terms on
huggingface.co first.

## Methodology notes worth knowing before reading the results

- **"Standard baseline" means standard.** Every baseline number comes from
  timm's off-the-shelf, pretrained TinyViT forward pass, unmodified.
- **timm's SDPA path is already "fused" at the PyTorch level** (`use_fused_attn()`
  defaults to `True`), but on MPS it falls back to
  `aten::_scaled_dot_product_attention_math_for_mps` -- opaque to
  `torch.profiler` (no child matmul/softmax events visible), but not
  necessarily a single fused Metal kernel either. This is exactly why the
  project needed MLX/Metal-level fusion and supplementary GPU-level tracing,
  not just `torch.profiler`.
- **`torch.profiler`'s "self CPU time" on MPS measures dispatch/encode time,
  not GPU execution time** (MPS is asynchronous; a `torch.mps.synchronize()`
  call is needed to force real completion, and it has a large, ~200-300us
  fixed cost). This makes it a genuine kernel-launch/dispatch-overhead
  metric, but it is unusable for roofline placement without correction --
  `roofline/op_bench.py` uses isolated, warmed-up, many-iteration
  microbenchmarks instead (one amortized sync per op, not per op *dispatch*).
- **fp16 elementwise bandwidth measured lower than fp32 on this backend**
  despite moving half the bytes/element -- flagged as a probable PyTorch MPS
  backend kernel-efficiency artifact, not a hardware ceiling (see
  `results/raw/02_hw_roofline.json`'s `notes` field).
- **MLX is a lazy-graph framework**: an uncompiled call re-traces the whole
  op graph in Python every invocation. `mx.compile` is required for a fair
  comparison against PyTorch's persistent op dispatch / `torch.compile`.
- **PyTorch-MPS and MLX do not share memory.** The end-to-end hybrid model
  (`fusion/hybrid_model.py`) pays a real GPU->CPU->GPU round trip per
  attention call when routing through MLX -- measured and reported, not
  hidden.
