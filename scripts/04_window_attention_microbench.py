#!/usr/bin/env python3
"""Phase 4: controlled window-attention microbenchmark.

Design: one-axis-at-a-time around a representative baseline config, plus a
focused 2D grid for window_size x batch_size (the axis pair the roofline/
dispatch-overhead story cares about most). A full 4D cartesian product over
window_size x embed_dim x num_windows x batch_size was tried first and
abandoned: the largest corner (dim=576, num_windows=256, batch=32) alone
took ~80s, and window_size=7/14 (3x/12x more tokens/window than
window_size=4) would extrapolate to roughly an hour total -- and a full
cross product also confounds interaction effects, making "how does X affect
latency" harder to read out than a controlled one-axis-at-a-time sweep
anyway. This design isolates each axis's effect cleanly and finishes in
under a few minutes.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from tinyvit_roofline.config import RAW_DIR, WINDOW_SIZES, EMBED_DIMS, NUM_WINDOWS, BATCH_SIZES
from tinyvit_roofline.microbench.window_attention_bench import run_config

BASELINE = {"window_size": 7, "dim": 96, "num_windows": 16, "batch_size": 4}


def build_sweep():
    configs = []
    seen = set()

    def add(ws, dim, nw, bs):
        key = (ws, dim, nw, bs)
        if key not in seen:
            seen.add(key)
            configs.append(key)

    # 1D sweeps around the baseline, one axis at a time.
    for ws in WINDOW_SIZES:
        add(ws, BASELINE["dim"], BASELINE["num_windows"], BASELINE["batch_size"])
    for dim in EMBED_DIMS:
        add(BASELINE["window_size"], dim, BASELINE["num_windows"], BASELINE["batch_size"])
    for nw in NUM_WINDOWS:
        add(BASELINE["window_size"], BASELINE["dim"], nw, BASELINE["batch_size"])
    for bs in BATCH_SIZES:
        add(BASELINE["window_size"], BASELINE["dim"], BASELINE["num_windows"], bs)

    # Focused 2D grid: window_size x batch_size, at baseline dim/num_windows --
    # the pair the dispatch-overhead-by-window-size story depends on most.
    for ws in WINDOW_SIZES:
        for bs in BATCH_SIZES:
            add(ws, BASELINE["dim"], BASELINE["num_windows"], bs)

    return configs


def write_checkpoint(path, configs, results, complete: bool):
    out = {
        "phase": "04_window_attention_microbench",
        "baseline_config": BASELINE,
        "sweep_grid": {"window_sizes": list(WINDOW_SIZES), "embed_dims": list(EMBED_DIMS),
                       "num_windows": list(NUM_WINDOWS), "batch_sizes": list(BATCH_SIZES)},
        "num_configs": len(configs),
        "num_completed": len(results),
        "complete": complete,
        "results": results,
    }
    path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    configs = build_sweep()
    print(f"[04] sweeping {len(configs)} configs (one-axis-at-a-time around baseline={BASELINE}, "
          f"plus a window_size x batch_size grid) ...", flush=True)

    out_path = RAW_DIR / "04_window_attention_microbench.json"
    results = []
    t_start = time.time()
    for i, (ws, dim, nw, bs) in enumerate(configs):
        t_cfg0 = time.time()
        r = run_config(ws, dim, nw, bs)
        cfg_elapsed = time.time() - t_cfg0
        results.append(r)
        torch.mps.empty_cache()  # avoid MPS allocator fragmentation across many distinct shapes
        elapsed = time.time() - t_start
        print(f"[04] {i+1}/{len(configs)} ({elapsed:.1f}s total, {cfg_elapsed:.2f}s this config) -- "
              f"ws={ws} dim={dim} nW={nw} bs={bs} total={r['t_total_us']:.1f}us "
              f"dispatch_frac={r['dispatch_overhead_fraction_of_total']*100:.1f}%", flush=True)
        if (i + 1) % 5 == 0:
            write_checkpoint(out_path, configs, results, complete=False)

    write_checkpoint(out_path, configs, results, complete=True)
    print(f"[04] wrote {out_path}", flush=True)

    print("\n[04] dispatch-overhead fraction by window size (baseline dim/num_windows/batch, from the 1D sweep):", flush=True)
    for ws in WINDOW_SIZES:
        rows = [r for r in results if r["window_size"] == ws and r["dim"] == BASELINE["dim"]
                and r["num_windows"] == BASELINE["num_windows"] and r["batch_size"] == BASELINE["batch_size"]]
        if not rows:
            continue
        r = rows[0]
        print(f"     window_size={ws:3d}: dispatch_overhead_fraction={r['dispatch_overhead_fraction_of_total']*100:5.1f}%  "
              f"total={r['t_total_us']:8.1f}us", flush=True)
