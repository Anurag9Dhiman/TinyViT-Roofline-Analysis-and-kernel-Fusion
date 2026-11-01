"""Classify profiler op names into TinyViT-relevant stage categories, and
(optionally) annotate TinyVitBlock/Attention forward passes with
`record_function` labels for accurate per-role attribution.

The annotation functions below are line-for-line re-implementations of
timm's `Attention.forward` / `TinyVitBlock.forward` (timm/models/tiny_vit.py,
current as of timm 1.0.24) with `torch.profiler.record_function(...)` context
managers inserted around each logical stage. The tensor ops, their order, and
the weights used are identical to the unmodified model -- this changes
*instrumentation* only, not computation, so numerics are unaffected. Used
only for the op-breakdown deliverable; the headline latency number comes
from an uninstrumented run (see torch_profile.time_forward).

Note on "self CPU time" here: MPS executes asynchronously relative to CPU
submission, so a `record_function` region's measured self-time is
dispatch/encode time (how long the CPU took to build+submit the Metal
command), NOT GPU execution time -- that's exactly the kernel-launch/
dispatch-overhead view classical roofline analysis can't capture on its own
(see torch_profile.profile_ops). It is NOT valid for placing points on a
roofline, which needs true GPU compute time -- for that, see
roofline/op_bench.py's isolated microbenchmarks, and that module's docstring
for why an earlier attempt at fixing this with a per-region
torch.mps.synchronize() didn't work either (the sync's fixed ~200-300us cost
dominated every region once paid per-region instead of amortized).
"""

import types
from typing import Dict

import torch.nn.functional as F
from torch.profiler import record_function

# Name-substring -> category, checked in order (first match wins).
_CATEGORY_RULES = [
    ("convolution", "conv"),
    ("batch_norm", "conv"),  # ConvNorm fuses BN right after each conv
    ("scaled_dot_product_attention", "attn_core_qk_softmax_av"),  # SDPA is opaque on MPS; see torch_profile.py
    ("linear", "linear_qkv_proj_or_mlp"),  # ambiguous by name alone; see annotate_* for exact attribution
    ("layer_norm", "norm"),
    ("gelu", "activation"),
    ("permute", "layout"),
    ("transpose", "layout"),
    ("reshape", "layout"),
    ("_reshape_alias", "layout"),
    ("_unsafe_view", "layout"),
    ("view", "layout"),
    ("as_strided", "layout"),
    ("contiguous", "layout"),
    ("clone", "layout"),
    ("copy_", "layout"),
    ("split_with_sizes", "layout"),
    ("expand", "layout"),
    ("pad", "layout"),
    ("add", "elementwise"),
    ("dropout", "elementwise"),
    ("empty", "alloc"),
    ("adaptive_avg_pool2d", "head"),
    ("mean", "head"),
    ("flatten", "head"),
]


def classify_op_name(name: str) -> str:
    n = name.replace("aten::", "")
    for substr, cat in _CATEGORY_RULES:
        if substr in n:
            return cat
    return "other"


def summarize_by_category(op_rows) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for r in op_rows:
        cat = classify_op_name(r["name"])
        d = out.setdefault(cat, {"self_cpu_time_us": 0.0, "count": 0, "ops": set()})
        d["self_cpu_time_us"] += r["self_cpu_time_us"]
        d["count"] += r["count"]
        d["ops"].add(r["name"])
    for d in out.values():
        d["ops"] = sorted(d["ops"])
    return out


# ---------------------------------------------------------------------------
# Precise annotation via monkeypatched forward (same ops, added labels only)
# ---------------------------------------------------------------------------

def _annotated_attention_forward(self, x):
    p = self._prof_prefix
    with record_function(f"tinyvit/{p}/attn/bias_gather"):
        attn_bias = self.get_attention_biases(x.device)
    B, N, _ = x.shape
    with record_function(f"tinyvit/{p}/attn/norm"):
        x = self.norm(x)
    with record_function(f"tinyvit/{p}/attn/qkv_proj"):
        qkv = self.qkv(x)
    with record_function(f"tinyvit/{p}/attn/split_permute"):
        q, k, v = qkv.view(B, N, self.num_heads, -1).split(
            [self.key_dim, self.key_dim, self.val_dim], dim=3
        )
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
    if self.fused_attn:
        with record_function(f"tinyvit/{p}/attn/sdpa_qk_softmax_av"):
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
    else:
        with record_function(f"tinyvit/{p}/attn/qk_score"):
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn + attn_bias
        with record_function(f"tinyvit/{p}/attn/softmax"):
            attn = attn.softmax(dim=-1)
        with record_function(f"tinyvit/{p}/attn/attn_value"):
            x = attn @ v
    with record_function(f"tinyvit/{p}/attn/merge_heads"):
        x = x.transpose(1, 2).reshape(B, N, self.out_dim)
    with record_function(f"tinyvit/{p}/attn/out_proj"):
        x = self.proj(x)
    return x


def _annotated_block_forward(self, x):
    p = self._prof_prefix
    B, H, W, C = x.shape
    L = H * W
    shortcut = x
    if H == self.window_size and W == self.window_size:
        with record_function(f"tinyvit/{p}/block/reshape_full_window"):
            x = x.reshape(B, L, C)
        x = self.attn(x)
        with record_function(f"tinyvit/{p}/block/reshape_full_window"):
            x = x.view(B, H, W, C)
    else:
        with record_function(f"tinyvit/{p}/block/window_partition"):
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0
            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size
            x = x.view(B, nH, self.window_size, nW, self.window_size, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_size * self.window_size, C
            )
        x = self.attn(x)
        with record_function(f"tinyvit/{p}/block/window_reverse"):
            x = x.view(B, nH, nW, self.window_size, self.window_size, C).transpose(2, 3).reshape(
                B, pH, pW, C
            )
            if padding:
                x = x[:, :H, :W].contiguous()
    with record_function(f"tinyvit/{p}/block/residual_add_1"):
        x = shortcut + self.drop_path1(x)
    with record_function(f"tinyvit/{p}/block/local_conv"):
        x = x.permute(0, 3, 1, 2)
        x = self.local_conv(x)
        x = x.reshape(B, C, L).transpose(1, 2)
    with record_function(f"tinyvit/{p}/block/mlp"):
        x = x + self.drop_path2(self.mlp(x))
    return x.view(B, H, W, C)


def annotate_tinyvit_blocks(model, block_infos=None) -> int:
    """Monkeypatch every TinyVitBlock/Attention forward with record_function
    labels, in place. If `block_infos` (from models.enumerate_attention_blocks)
    is given, each block's labels are prefixed with its stage/block index
    (e.g. "s1b0") so results can be matched to per-stage FLOPs/bytes exactly,
    instead of a model-wide aggregate. Returns the number of blocks
    annotated. Callers should treat an annotated model as profiling-only, not
    for latency numbers (same ops, added labels only)."""
    prefix_by_name = {}
    if block_infos is not None:
        prefix_by_name = {bi.module_name: f"s{bi.stage_idx}b{bi.block_idx}" for bi in block_infos}

    count = 0
    for name, module in model.named_modules():
        if type(module).__name__ == "TinyVitBlock":
            prefix = prefix_by_name.get(name, f"blk{count}")
            module._prof_prefix = prefix
            module.attn._prof_prefix = prefix
            module.forward = types.MethodType(_annotated_block_forward, module)
            module.attn.forward = types.MethodType(_annotated_attention_forward, module.attn)
            count += 1
    return count
