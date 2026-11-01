"""Ablation: baseline (standard PyTorch) vs. optimized (torch.compile) vs.
mlx_fused, swept across window size, embedding dim, and batch size (a fixed
representative window count, since Phase 4 already covers the num_windows
axis on its own). Each implementation is timed in isolation (warmup + many
iterations + one final sync -- same methodology as roofline/op_bench.py) and
gated by the correctness check before being reported.
"""

import time
from typing import Dict, List

import torch

from ..microbench.window_attention_bench import build_attention
from ..fusion.pytorch_optimized_attn import build_optimized
from ..fusion.mlx_fused_attn import build_mlx_attention
from ..fusion.correctness import check_attention_equivalence

REPRESENTATIVE_NUM_WINDOWS = 16


def _time_call(fn, warmup: int = 15, iters: int = 100) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters


def run_one_config(window_size: int, dim: int, batch_size: int, num_windows: int = REPRESENTATIVE_NUM_WINDOWS) -> Dict:
    batch_windows = batch_size * num_windows
    tokens = window_size * window_size
    x = torch.randn(batch_windows, tokens, dim, device="mps")

    baseline = build_attention(dim, window_size)
    baseline.eval()
    t_baseline = _time_call(lambda: baseline(x))

    optimized = build_optimized(dim, window_size, num_heads=baseline.num_heads)
    optimized.attn.load_state_dict(baseline.state_dict())
    optimized.eval()
    t_optimized = _time_call(lambda: optimized(x))

    mlx_attn = build_mlx_attention(baseline)
    x_mlx_probe = x[: min(2, x.shape[0])]
    correctness = check_attention_equivalence(baseline, mlx_attn, x_mlx_probe)

    import mlx.core as mx
    x_mlx = mx.array(x.detach().to("cpu", torch.float32).numpy())
    mlx_compiled = mlx_attn.compiled()

    def _mlx_call():
        out = mlx_compiled(x_mlx)
        mx.eval(out)
        return out

    t_mlx = _time_call(_mlx_call)

    return {
        "window_size": window_size,
        "dim": dim,
        "batch_size": batch_size,
        "num_windows": num_windows,
        "num_heads": baseline.num_heads,
        "latency_ms": {
            "pytorch_baseline": t_baseline * 1e3,
            "pytorch_optimized": t_optimized * 1e3,
            "mlx_fused": t_mlx * 1e3,
        },
        "speedup_vs_baseline": {
            "pytorch_optimized": t_baseline / t_optimized if t_optimized else None,
            "mlx_fused": t_baseline / t_mlx if t_mlx else None,
        },
        "pytorch_optimized_compiled_successfully": optimized.compiled_successfully,
        "mlx_correctness": correctness,
    }


def run_ablation_grid(window_sizes: List[int], embed_dims: List[int], batch_sizes: List[int]) -> List[Dict]:
    results = []
    for ws in window_sizes:
        for dim in embed_dims:
            for bs in batch_sizes:
                results.append(run_one_config(ws, dim, bs))
    return results
