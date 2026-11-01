"""Shared constants: model registry, sweep grids, output paths."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
TRACE_DIR = RESULTS_DIR / "traces"
PLOTS_DIR = RESULTS_DIR / "plots"
for _d in (RAW_DIR, TRACE_DIR, PLOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DEVICE = "mps"

# timm TinyViT variants used throughout. embed_dims/depths/num_heads/window_sizes
# are the architecture's own per-stage config (mirrors timm.models.tiny_vit), kept
# here so profiling/roofline code can label results per-stage without importing
# timm's private model-builder internals.
MODEL_REGISTRY = {
    "tiny_vit_5m_224": {
        "timm_id": "tiny_vit_5m_224.dist_in22k_ft_in1k",
        "embed_dims": (64, 128, 160, 320),
        "depths": (2, 2, 6, 2),
        "num_heads": (2, 4, 5, 10),
        "window_sizes": (7, 7, 14, 7),
    },
    "tiny_vit_11m_224": {
        "timm_id": "tiny_vit_11m_224.dist_in22k_ft_in1k",
        "embed_dims": (64, 128, 256, 448),
        "depths": (2, 2, 6, 2),
        "num_heads": (2, 4, 8, 14),
        "window_sizes": (7, 7, 14, 7),
    },
    "tiny_vit_21m_224": {
        "timm_id": "tiny_vit_21m_224.dist_in22k_ft_in1k",
        "embed_dims": (96, 192, 384, 576),
        "depths": (2, 2, 6, 2),
        "num_heads": (3, 6, 12, 18),
        "window_sizes": (7, 7, 14, 7),
    },
}

PRIMARY_MODEL = "tiny_vit_5m_224"

# Window-attention microbenchmark sweep grid (Phase 4).
WINDOW_SIZES = (4, 7, 14)
EMBED_DIMS = (64, 96, 192, 384, 576)
NUM_WINDOWS = (16, 64, 256)  # e.g. 4x4, 8x8, 16x16 windows tiled
BATCH_SIZES = (1, 4, 16, 32)

# Ablation matrix (Phase 7): implementations compared at each config point.
IMPLEMENTATIONS = ("pytorch_baseline", "pytorch_optimized", "mlx_fused")

# ImageNet-1k streamed-accuracy eval (Phase 8).
IMAGENET_HF_ID = "ILSVRC/imagenet-1k"
IMAGES_PER_CLASS = 1  # 1 * 1000 = 1000 images -- reduced from 5/class (5000 images) after
# streaming attempts filled the disk (an unrelated ~/.cache/huggingface buildup, not this
# project's data). Still stratified across all 1000 classes.
NUM_CLASSES = 1000

# Benchmark hygiene
WARMUP_ITERS = 10
TIMED_ITERS = 50
