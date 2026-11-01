# TinyViT Window Attention: Roofline Analysis & Kernel Fusion on Apple M4

Characterizing and reducing the compute, memory, and dispatch inefficiencies
of TinyViT's small-window attention on Apple Silicon through empirical
Roofline analysis and MLX/Metal kernel fusion.

Hardware: Apple M4 (10-core GPU), 16GB unified memory. Software: PyTorch
2.8.0 (MPS backend) for the standard baseline, MLX for the fused-attention
implementation.

**[Final Report (PDF)](System_Engineering_Final_Report.pdf)** -- condensed
writeup (problem, prior work, implementation, evaluation vs. baseline).
See `report.md` below for the full, unabridged technical writeup.

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

**Headline results** (full detail, every sub-experiment, and all bug fixes
found along the way: `report.md` / the PDF above):

- **The original hypothesis -- fuse window attention via MLX -- doesn't pay
  off.** Across a 60-config ablation, neither `torch.compile` (mean 0.90x)
  nor MLX-fused attention (mean 0.68x) beats standard PyTorch/MPS on
  average. Root cause: MLX's Python->Metal dispatch path has a fixed
  ~300-400us floor per call, confirmed directly -- it doesn't shrink with
  problem size, and is already bigger than TinyViT's own per-window
  compute.
- **Bypassing MLX (native PyTorch C++/Metal extensions) can beat baseline,
  but only in narrow regimes at first** -- real wins on synthetic configs
  and on real `window_size=7` blocks (1.13-1.34x, every scale), but
  `window_size=14` actively loses with the same design, and the
  `window_size=7` win is too small a slice of total model time to show up
  end-to-end (0.96-1.01x against an unmodified baseline).
- **The actual fix for `window_size=14`: cache K/V once per window instead
  of re-reading it ~25x (one read per query tile).** A flash-attention-style
  kernel (online softmax, fp16, K/V cached once in threadgroup memory) cuts
  K/V traffic 96% and attention-core dispatch count 7x. Verified on real
  pretrained weights, all three model scales: correctness holds (logit
  max-abs-diff 2.8e-4 to 1.1e-3), and it's a real, reproducible end-to-end
  win at every batch size on every model (1.02-1.08x). Five earlier kernel
  designs failed first (naive GEMM, shared-memory-blocked GEMM, fp32/fp16
  flash attention without caching) -- each failure ruled out a different
  hypothesis about where the bottleneck actually was.
- **The one deployable, exact-numerics optimization for the rest of the
  model is conv+BN fusion + `channels_last` memory format.** 1.24x (5M) /
  1.17x (11M) / 1.10x (21M), correctness exact (~1e-6), verified end-to-end
  on real ImageNet-1k images with accuracy fully preserved. `torch.compile`
  and MLX attention fusion both fail to stack on top of it.
- **fp16 is the single biggest lever (1.49x on top of the best combination)
  but is the only one with a real cost.** Accuracy delta was pinned down
  precisely: pushed to a 20,000-image ImageNet-1k eval and the delta lands
  at -0.01pp -- indistinguishable from zero, but a real tradeoff to know
  about (smaller samples showed a noisier -0.10pp that didn't hold up).
- **`torch.compile` never helps anywhere in this project** -- fails on the
  baseline, on the best combination, and on the fastest (fp16) arm, at
  every model scale tested.
- **The empirical roofline (section 3) confirms TinyViT is memory/dispatch-
  bound, not compute-bound, at every stage and every practical batch size**
  -- batching up to 128 only reaches ~32% of the measured compute peak.
  `local_conv` is the single worst-utilized region in the model at every
  scale.
- **This machine is a fanless MacBook Air, and getting trustworthy
  comparisons required real methodological care**: each implementation
  benchmarked in its own subprocess with cooldown pauses between trials --
  an in-process, back-to-back version of the same eval gave numbers that
  flatly contradicted the isolated benchmark.

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
