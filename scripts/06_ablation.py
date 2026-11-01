#!/usr/bin/env python3
"""Phase 6: ablation -- pytorch_baseline vs pytorch_optimized (torch.compile)
vs mlx_fused, swept across window size, embedding dim, batch size."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from tinyvit_roofline.config import RAW_DIR, WINDOW_SIZES, EMBED_DIMS, BATCH_SIZES
from tinyvit_roofline.ablation.run_ablation import run_one_config


def write_checkpoint(path, configs, results, complete: bool):
    n_mlx_correct = sum(1 for r in results if r["mlx_correctness"]["passed"])
    mean_speedup_opt = sum(r["speedup_vs_baseline"]["pytorch_optimized"] for r in results) / max(len(results), 1)
    mean_speedup_mlx = (
        sum(r["speedup_vs_baseline"]["mlx_fused"] for r in results if r["mlx_correctness"]["passed"]) / max(n_mlx_correct, 1)
    )
    out = {
        "phase": "06_ablation",
        "grid": {"window_sizes": list(WINDOW_SIZES), "embed_dims": list(EMBED_DIMS), "batch_sizes": list(BATCH_SIZES)},
        "num_configs": len(configs),
        "num_completed": len(results),
        "complete": complete,
        "num_mlx_correct": n_mlx_correct,
        "mean_speedup_pytorch_optimized": mean_speedup_opt,
        "mean_speedup_mlx_fused_correctness_gated": mean_speedup_mlx,
        "results": results,
    }
    path.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    configs = [(ws, dim, bs) for ws in WINDOW_SIZES for dim in EMBED_DIMS for bs in BATCH_SIZES]
    print(f"[06] running ablation over {len(configs)} configs "
          f"(window_sizes={WINDOW_SIZES}, embed_dims={EMBED_DIMS}, batch_sizes={BATCH_SIZES}) ...", flush=True)

    out_path = RAW_DIR / "06_ablation.json"
    results = []
    t_start = time.time()
    for i, (ws, dim, bs) in enumerate(configs):
        t_cfg0 = time.time()
        r = run_one_config(ws, dim, bs)
        cfg_elapsed = time.time() - t_cfg0
        results.append(r)
        torch.mps.empty_cache()
        elapsed = time.time() - t_start
        su_opt = r["speedup_vs_baseline"]["pytorch_optimized"]
        su_mlx = r["speedup_vs_baseline"]["mlx_fused"]
        print(f"[06] {i+1}/{len(configs)} ({elapsed:.1f}s total, {cfg_elapsed:.2f}s this config) -- "
              f"ws={ws} dim={dim} bs={bs}  speedup: optimized={su_opt:.2f}x mlx={su_mlx:.2f}x  "
              f"mlx_correct={r['mlx_correctness']['passed']}", flush=True)
        if (i + 1) % 10 == 0:
            write_checkpoint(out_path, configs, results, complete=False)

    out = write_checkpoint(out_path, configs, results, complete=True)

    print(f"\n[06] summary: {out['num_mlx_correct']}/{len(results)} configs passed MLX correctness check", flush=True)
    print(f"[06] mean speedup (pytorch_optimized vs baseline): {out['mean_speedup_pytorch_optimized']:.2f}x", flush=True)
    print(f"[06] mean speedup (mlx_fused vs baseline, correctness-gated): {out['mean_speedup_mlx_fused_correctness_gated']:.2f}x", flush=True)

    print("\n[06] mean speedup by window size:", flush=True)
    for ws in WINDOW_SIZES:
        rows = [r for r in results if r["window_size"] == ws]
        su_opt = sum(r["speedup_vs_baseline"]["pytorch_optimized"] for r in rows) / len(rows)
        mlx_rows = [r for r in rows if r["mlx_correctness"]["passed"]]
        su_mlx = sum(r["speedup_vs_baseline"]["mlx_fused"] for r in mlx_rows) / max(len(mlx_rows), 1)
        print(f"     window_size={ws:3d}: optimized={su_opt:.2f}x  mlx={su_mlx:.2f}x", flush=True)

    print(f"\n[06] wrote {out_path}", flush=True)
