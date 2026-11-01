"""The standard, unmodified window-attention module -- timm's own
`Attention` class, used as-is. This is the "baseline" arm of the ablation;
every other arm (pytorch_optimized, mlx_fused) is checked for numerical
equivalence against this exact module's output before its speed is trusted.
"""

from timm.models.tiny_vit import Attention

from ..microbench.window_attention_bench import build_attention, heads_for_dim

__all__ = ["Attention", "build_attention", "heads_for_dim"]
