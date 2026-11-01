#!/usr/bin/env python3
"""Phase 2: empirical M4 GPU roofline parameters (P_max, B_max, ridge point)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tinyvit_roofline.config import RAW_DIR
from tinyvit_roofline.roofline.hw_bench import run_hw_roofline

if __name__ == "__main__":
    print("[02] sweeping matmul sizes (fp32, fp16) for peak compute throughput ...")
    print("[02] sweeping elementwise-op sizes (fp32, fp16) for peak memory bandwidth ...")
    result = run_hw_roofline()
    for dt in ("fp32", "fp16"):
        c = result[dt]["compute"]
        b = result[dt]["bandwidth"]
        print(f"[02] {dt}: P_max={c['peak_gflops_per_s']:.1f} GFLOP/s (N={c['peak_at_n']}), "
              f"B_max={b['peak_gb_per_s']:.1f} GB/s (size={b['peak_at_mb']}MB), "
              f"ridge={result[dt]['ridge_point_flops_per_byte']:.2f} FLOP/byte")
    out_path = RAW_DIR / "02_hw_roofline.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[02] wrote {out_path}")
