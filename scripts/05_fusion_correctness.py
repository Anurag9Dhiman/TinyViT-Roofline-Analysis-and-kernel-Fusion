#!/usr/bin/env python3
"""Phase 5: numerical-equivalence gate for the MLX-fused attention.

Builds a PyTorch baseline Attention module with real (random-init, since this
tests the attention op in isolation -- see note below) weights, ports those
exact weights into MLX, and checks max-abs-diff / cosine-similarity on
identical inputs across the same window-size/dim grid used elsewhere. The
MLX-fused arm is only allowed into the ablation (Phase 6) for configs that
pass here.

Note on weights: this checks the isolated attention operator, so weights are
the freshly-initialized ones `build_attention` creates (real nn.Linear /
nn.LayerNorm init, not zeros) -- sufficient to catch any real numerical bug
in the port or the MLX math. Phase 8's end-to-end accuracy eval additionally
checks the full pretrained model's real checkpoint weights.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from tinyvit_roofline.config import RAW_DIR, WINDOW_SIZES, EMBED_DIMS
from tinyvit_roofline.microbench.window_attention_bench import build_attention
from tinyvit_roofline.fusion.mlx_fused_attn import build_mlx_attention
from tinyvit_roofline.fusion.correctness import check_attention_equivalence

if __name__ == "__main__":
    results = []
    print(f"[05] checking numerical equivalence across window_sizes={WINDOW_SIZES}, embed_dims={EMBED_DIMS} ...")
    for ws in WINDOW_SIZES:
        for dim in EMBED_DIMS:
            torch_attn = build_attention(dim, ws)
            torch_attn.eval()
            mlx_attn = build_mlx_attention(torch_attn)
            x = torch.randn(2, ws * ws, dim, device="mps")
            stats = check_attention_equivalence(torch_attn, mlx_attn, x)
            stats.update({"window_size": ws, "dim": dim, "num_heads": torch_attn.num_heads})
            results.append(stats)
            status = "PASS" if stats["passed"] else "FAIL"
            print(f"     ws={ws:3d} dim={dim:4d}  max_abs_diff={stats['max_abs_diff']:.6f}  "
                  f"cos_sim={stats['cosine_similarity']:.6f}  [{status}]")

    n_pass = sum(1 for r in results if r["passed"])
    print(f"\n[05] {n_pass}/{len(results)} configs passed numerical-equivalence check (atol={results[0]['atol']})")

    out = {"phase": "05_fusion_correctness", "results": results, "num_pass": n_pass, "num_total": len(results)}
    out_path = RAW_DIR / "05_fusion_correctness.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[05] wrote {out_path}")

    if n_pass < len(results):
        print("[05] WARNING: not all configs passed -- Phase 6 ablation will exclude failing configs from the MLX-fused arm.")
