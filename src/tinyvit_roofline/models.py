"""Standard (unmodified) TinyViT loading via timm, plus block introspection helpers.

Nothing here alters TinyViT's forward path -- this is the "standard baseline"
the whole project measures against, per the project's explicit requirement.
"""

from dataclasses import dataclass
from typing import List

import timm
import torch

from .config import DEVICE, MODEL_REGISTRY


def load_baseline_model(variant: str = "tiny_vit_21m_224", device: str = DEVICE) -> torch.nn.Module:
    """Load the stock, pretrained timm TinyViT model, unmodified, in eval mode."""
    timm_id = MODEL_REGISTRY[variant]["timm_id"]
    model = timm.create_model(timm_id, pretrained=True)
    model.eval()
    model.to(device)
    return model


@dataclass
class BlockInfo:
    stage_idx: int
    block_idx: int
    module_name: str
    dim: int
    num_heads: int
    window_size: int
    input_resolution: int  # H == W of the feature map entering this stage's blocks


def enumerate_attention_blocks(model: torch.nn.Module, variant: str) -> List[BlockInfo]:
    """Walk model.stages and return one BlockInfo per TinyVitBlock (stages 1-3; stage 0 is pure conv/MBConv)."""
    cfg = MODEL_REGISTRY[variant]
    infos: List[BlockInfo] = []
    # patch_embed stride is 4; each subsequent stage (index >= 1) halves resolution again via PatchMerging.
    input_size = 224
    resolution = input_size // 4
    for stage_idx, stage in enumerate(model.stages):
        if stage_idx > 0:
            resolution = resolution // 2  # PatchMerging halves H, W
        if not hasattr(stage, "blocks"):
            continue
        for block_idx, block in enumerate(stage.blocks):
            if type(block).__name__ != "TinyVitBlock":
                continue
            infos.append(
                BlockInfo(
                    stage_idx=stage_idx,
                    block_idx=block_idx,
                    module_name=f"stages.{stage_idx}.blocks.{block_idx}",
                    dim=cfg["embed_dims"][stage_idx],
                    num_heads=cfg["num_heads"][stage_idx],
                    window_size=cfg["window_sizes"][stage_idx],
                    input_resolution=resolution,
                )
            )
    return infos


def get_module_by_name(model: torch.nn.Module, name: str) -> torch.nn.Module:
    mod = model
    for part in name.split("."):
        mod = getattr(mod, part) if not part.isdigit() else mod[int(part)]
    return mod
