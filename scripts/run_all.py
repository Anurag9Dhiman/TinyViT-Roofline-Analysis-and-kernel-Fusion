#!/usr/bin/env python3
"""Run the full pipeline in order. Each phase is also independently runnable
(python scripts/0N_*.py) and writes its own results/raw/*.json, so this is a
convenience wrapper, not a requirement."""

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

PHASES = [
    "01_baseline_profile.py",
    "02_hw_roofline.py",
    "03_stage_roofline.py",
    "04_window_attention_microbench.py",
    "05_fusion_correctness.py",
    "06_ablation.py",
    "15_mlx_primitive_microbench.py",     # verifies section 6's "why MLX fusion doesn't pay off" claim
    "16_mlx_fp16_primitive_microbench.py",  # does fp16 close MLX's primitive gap? (no)
    "17_mlx_quantized_linear_microbench.py",  # MLX int8 quantized matmul vs pytorch fp16 (pytorch fp16 wins)
    "19_mlx_quantized_linear_batch_sweep.py",  # does int8 catch up at bigger GEMM sizes? (converges to ~parity, never wins)
    "07_accuracy_eval.py",
    "09_conv_fusion_benchmark.py",       # the working optimization
    "10_combined_fusion_benchmark.py",   # conv+BN fusion stacked with MLX attention (doesn't stack)
    "11_channels_last_benchmark.py",     # conv+BN fusion stacked with channels_last (does stack)
    "12_compile_stack_benchmark.py",     # best combination stacked with torch.compile (doesn't stack)
    "13_optimized_profile.py",           # re-profile the optimized model -- what's the new bottleneck?
    "14_fp16_benchmark.py",              # fp16 on top of the best combination (biggest speedup, real accuracy cost)
    "18_bf16_benchmark.py",              # bf16 vs fp16 (tied on speed, much less precise -- fp16 wins)
    "20_ridge_point_batch_sweep.py",     # does batching ever cross into compute-bound? (no, plateaus ~32% of P_max)
    "21_compile_fp16_stack_benchmark.py",  # torch.compile on top of best_fp16 (also fails)
    "22_native_kernel_fusion_benchmark.py",  # kernel fusion via native dispatch (bypassing MLX) -- wins at small batch!
    "23_hybrid_attention_core_benchmark.py",  # fuse only the attention core on REAL blocks -- real op-level win, ~noise end-to-end
    "24_kvcache_flash_attention_benchmark.py",  # fixes window_size=14 with a K/V-cached fp16 flash-attention kernel -- real end-to-end win, all 3 models
    "08_report.py",                      # run last -- aggregates everything above
]

if __name__ == "__main__":
    for phase in PHASES:
        print(f"\n{'='*70}\nRunning {phase}\n{'='*70}")
        result = subprocess.run([sys.executable, str(SCRIPTS_DIR / phase)])
        if result.returncode != 0:
            print(f"[run_all] {phase} exited with code {result.returncode} -- continuing to next phase")
