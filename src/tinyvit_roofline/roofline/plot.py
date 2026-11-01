"""Roofline scatter plot: log-log AI (x) vs achieved GFLOPs/s (y), with the
measured roofline curve min(P_max, B_max * I) and per-stage points overlaid.
"""

from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Brand-neutral placeholder categorical palette (swap for project branding).
CATEGORY_COLORS = {
    "conv": "#4C72B0",
    "qkv_proj": "#DD8452",
    "sdpa_qk_softmax_av": "#C44E52",
    "out_proj": "#8172B2",
    "window_partition": "#937860",
    "window_reverse": "#937860",
    "local_conv": "#55A868",
    "mlp": "#64B5CD",
    "other": "#999999",
}


def plot_roofline(
    peak_gflops_per_s: float,
    peak_gb_per_s: float,
    stage_points: List[Dict],
    out_path: str,
    title: str = "TinyViT on Apple M4 GPU -- Empirical Roofline (fp32)",
) -> None:
    ridge_ai = (peak_gflops_per_s * 1e9) / (peak_gb_per_s * 1e9)

    # Ops with 0 FLOPs (pure data movement, e.g. window partition/reverse)
    # have undefined AI on a log axis and don't belong on a FLOPs roofline --
    # they're excluded from the curve range and scatter here (reported
    # separately as bandwidth utilization by the caller).
    plottable = [p for p in stage_points if p["ai"] > 0 and p["achieved_gflops_per_s"] > 0]

    ai_min = min([p["ai"] for p in plottable] + [ridge_ai * 1e-3])
    ai_max = max([p["ai"] for p in plottable] + [ridge_ai * 1e2])
    ai_range = np.logspace(np.log10(ai_min * 0.5), np.log10(ai_max * 2), 200)
    roofline = np.minimum(peak_gflops_per_s, peak_gb_per_s * ai_range)

    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=150)
    ax.plot(ai_range, roofline, color="#333333", linewidth=2, zorder=2, label="Measured roofline")
    ax.axvline(ridge_ai, color="#333333", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(ridge_ai, peak_gflops_per_s * 1.05, f"ridge = {ridge_ai:.1f} FLOP/B",
            ha="center", fontsize=9, color="#333333")

    seen_categories = set()
    for p in plottable:
        cat = p.get("category", "other")
        color = CATEGORY_COLORS.get(cat, CATEGORY_COLORS["other"])
        label = cat if cat not in seen_categories else None
        seen_categories.add(cat)
        ax.scatter(p["ai"], p["achieved_gflops_per_s"], s=60, color=color, edgecolor="white",
                   linewidth=0.6, zorder=3, alpha=0.85, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    y_min = min([p["achieved_gflops_per_s"] for p in plottable] + [peak_gflops_per_s]) * 0.5
    ax.set_ylim(y_min, peak_gflops_per_s * 2)
    ax.set_xlabel("Arithmetic Intensity (FLOP / byte)")
    ax.set_ylabel("Achieved throughput (GFLOP/s)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
