"""Empirical M4 GPU roofline parameters: P_max (achieved compute throughput)
and B_max (achieved unified-memory bandwidth), measured directly on this
machine rather than taken from Apple's marketing spec sheet.
"""

import time
from typing import Dict, List

import torch


def _time_op(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters


def measure_compute_peak(
    dtype=torch.float32,
    sizes: List[int] = (512, 1024, 2048, 3072, 4096),
    warmup: int = 5,
    iters: int = 20,
) -> Dict:
    """Sweep square-matmul sizes; achieved GFLOPs/s = 2*N^3 / time. Larger N
    amortizes dispatch overhead so the max over the sweep approximates
    sustained peak throughput for this dtype."""
    points = []
    for n in sizes:
        a = torch.randn(n, n, device="mps", dtype=dtype)
        b = torch.randn(n, n, device="mps", dtype=dtype)
        t = _time_op(lambda: a @ b, warmup, iters)
        gflops = (2.0 * n ** 3) / t / 1e9
        points.append({"n": n, "time_s": t, "gflops_per_s": gflops})
        del a, b
        torch.mps.empty_cache()
    peak = max(points, key=lambda p: p["gflops_per_s"])
    return {"dtype": str(dtype), "sweep": points, "peak_gflops_per_s": peak["gflops_per_s"], "peak_at_n": peak["n"]}


def measure_bandwidth_peak(
    dtype=torch.float32,
    sizes_mb: List[float] = (16, 64, 256, 1024),
    warmup: int = 5,
    iters: int = 30,
) -> Dict:
    """Sweep a single elementwise op (y = x * 2) tensor sizes; achieved GB/s =
    bytes_read_plus_written / time. Elementwise ops are pure memory-bound
    (AI ~ 0.1 FLOP/byte), so their achieved throughput approximates B_max.

    IMPORTANT: uses a single op (`x.mul(2.0)`), not a chained expression like
    `x*2.0+1.0`. Verified empirically on this machine that PyTorch's eager
    MPS backend does NOT fuse chained elementwise ops into one kernel (two
    aten dispatches, two full read/write passes) -- an earlier version of
    this benchmark used a 2-op chain assuming 1-kernel byte traffic and
    silently underestimated GB/s by ~2x. A single op removes the ambiguity.
    """
    itemsize = torch.tensor([], dtype=dtype).element_size()
    points = []
    for mb in sizes_mb:
        n_elems = int(mb * 1024 * 1024 / itemsize)
        x = torch.randn(n_elems, device="mps", dtype=dtype)
        t = _time_op(lambda: x.mul(2.0), warmup, iters)
        bytes_moved = 2 * n_elems * itemsize  # 1 read + 1 write
        gbps = bytes_moved / t / 1e9
        points.append({"size_mb": mb, "time_s": t, "gb_per_s": gbps})
        del x
        torch.mps.empty_cache()
    peak = max(points, key=lambda p: p["gb_per_s"])
    return {"dtype": str(dtype), "sweep": points, "peak_gb_per_s": peak["gb_per_s"], "peak_at_mb": peak["size_mb"]}


def compute_ridge_point(peak_gflops_per_s: float, peak_gb_per_s: float) -> float:
    """I_ridge = P_max / B_max, in FLOPs/byte."""
    return (peak_gflops_per_s * 1e9) / (peak_gb_per_s * 1e9)


def run_hw_roofline() -> Dict:
    fp32_compute = measure_compute_peak(dtype=torch.float32)
    fp16_compute = measure_compute_peak(dtype=torch.float16)
    fp32_bw = measure_bandwidth_peak(dtype=torch.float32)
    fp16_bw = measure_bandwidth_peak(dtype=torch.float16)

    notes = []
    if fp16_bw["peak_gb_per_s"] < 0.6 * fp32_bw["peak_gb_per_s"]:
        notes.append(
            "fp16 elementwise bandwidth measured lower than fp32 despite moving half the "
            "bytes/element -- reproducible on this machine, likely a PyTorch MPS backend "
            "kernel-efficiency artifact (e.g. internal upcast or non-vectorized fp16 path) "
            "rather than a hardware bandwidth limit. Treat fp16 ridge point as backend-specific, "
            "not a hardware ceiling; fp32 is used as the primary roofline since TinyViT's "
            "standard baseline runs in fp32."
        )

    return {
        "phase": "02_hw_roofline",
        "device": "mps",
        "notes": notes,
        "fp32": {
            "compute": fp32_compute,
            "bandwidth": fp32_bw,
            "ridge_point_flops_per_byte": compute_ridge_point(
                fp32_compute["peak_gflops_per_s"], fp32_bw["peak_gb_per_s"]
            ),
        },
        "fp16": {
            "compute": fp16_compute,
            "bandwidth": fp16_bw,
            "ridge_point_flops_per_byte": compute_ridge_point(
                fp16_compute["peak_gflops_per_s"], fp16_bw["peak_gb_per_s"]
            ),
        },
    }
