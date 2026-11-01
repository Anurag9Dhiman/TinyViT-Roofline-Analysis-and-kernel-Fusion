"""Numerical-equivalence gate: an implementation's speed is only reported in
the ablation if it passes this check against the PyTorch baseline on the
same input and the same (ported) weights.
"""

from typing import Dict

import mlx.core as mx
import numpy as np
import torch


def compare_arrays(torch_out: torch.Tensor, mlx_out: mx.array) -> Dict:
    a = torch_out.detach().to("cpu", torch.float32).numpy()
    b = np.array(mlx_out).astype(np.float32)
    diff = np.abs(a - b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    cos_sim = float(np.sum(a.astype(np.float64) * b.astype(np.float64)) / denom)
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "cosine_similarity": cos_sim,
    }


def check_attention_equivalence(torch_attn, mlx_attn, x_torch: torch.Tensor, atol: float = 2e-3) -> Dict:
    with torch.no_grad():
        torch_out = torch_attn(x_torch)
    x_mlx = mx.array(x_torch.detach().to("cpu", torch.float32).numpy())
    mlx_out = mlx_attn(x_mlx)
    mx.eval(mlx_out)
    stats = compare_arrays(torch_out, mlx_out)
    stats["atol"] = atol
    stats["passed"] = stats["max_abs_diff"] < atol
    return stats
