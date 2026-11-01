"""torch.profiler wrapper for latency, op-breakdown, and memory measurement on MPS.

Two-tier methodology (documented in report.md):
  1. `time_forward` times the model exactly as loaded -- zero instrumentation,
     zero code changes. This is the number reported as "baseline latency".
  2. `profile_ops` / block-level annotation (see op_classify.py) adds
     `record_function` labels for attribution only. Same tensor ops, same
     order, same weights -- just named regions for the profiler. Its own
     latency is reported separately and is expected to differ slightly from
     (1) due to instrumentation overhead.
"""

import statistics
import time
from dataclasses import dataclass, asdict
from typing import Dict, List

import torch


@dataclass
class LatencyStats:
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p90_ms: float
    n: int

    def to_dict(self) -> Dict:
        return asdict(self)


def time_forward(model, x, warmup: int = 10, iters: int = 50) -> LatencyStats:
    """Wall-clock latency of the model exactly as-is. No instrumentation."""
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.mps.synchronize()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            torch.mps.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    n = len(times)
    return LatencyStats(
        mean_ms=statistics.mean(times),
        std_ms=statistics.pstdev(times) if n > 1 else 0.0,
        min_ms=times[0],
        max_ms=times[-1],
        p50_ms=times[n // 2],
        p90_ms=times[min(n - 1, int(n * 0.9))],
        n=n,
    )


def peak_memory_mb(model, x, warmup: int = 5, iters: int = 20) -> float:
    """Best-effort peak MPS allocator usage. MPS has no max_memory_allocated()
    API (unlike CUDA), so we poll current_allocated_memory() every iteration."""
    torch.mps.empty_cache()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.mps.synchronize()
        peak = torch.mps.current_allocated_memory()
        for _ in range(iters):
            model(x)
            torch.mps.synchronize()
            peak = max(peak, torch.mps.current_allocated_memory())
    return peak / (1024.0 ** 2)


def profile_ops(model, x, warmup: int = 5, iters: int = 20) -> Dict:
    """Op-level CPU/dispatcher-side breakdown via torch.profiler.

    Note (empirically confirmed on this hardware): composite MPS ops such as
    aten::_scaled_dot_product_attention_math_for_mps do NOT decompose into
    child aten::matmul/softmax events under torch.profiler on MPS -- PyTorch's
    MPS composite ops call into MPSGraph beneath the dispatcher, so the
    profiler sees one opaque leaf op. Op-count here is therefore a *lower
    bound* / proxy for true GPU kernel-dispatch count, not an exact count.
    We cross-check with a torch.mps.profiler .gputrace capture separately
    (see mps_trace.py) on a few representative configs.
    """
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.mps.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
        ) as prof:
            for _ in range(iters):
                model(x)
            torch.mps.synchronize()

    rows: List[Dict] = []
    for a in prof.key_averages():
        rows.append(
            dict(
                name=a.key,
                count=a.count,
                self_cpu_time_us=a.self_cpu_time_total,
                cpu_time_total_us=a.cpu_time_total,
                self_cpu_time_avg_us=a.self_cpu_time_total / max(a.count, 1),
            )
        )
    rows.sort(key=lambda r: -r["self_cpu_time_us"])
    total_self = sum(r["self_cpu_time_us"] for r in rows)
    total_calls = sum(r["count"] for r in rows)
    return {
        "iters": iters,
        "total_self_cpu_time_us": total_self,
        "total_op_calls": total_calls,
        "op_calls_per_iter": total_calls / iters,
        "ops": rows,
    }


def export_chrome_trace(model, x, path: str, warmup: int = 5, iters: int = 5) -> None:
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.mps.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
        ) as prof:
            for _ in range(iters):
                model(x)
            torch.mps.synchronize()
    prof.export_chrome_trace(path)
