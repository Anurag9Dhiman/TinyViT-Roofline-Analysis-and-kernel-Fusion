"""Accurate per-op GPU throughput via isolated, warmed-up microbenchmarks.

Why not just synchronize after each labeled region inside the full forward
(profiling/op_classify.py's SYNC_MODE)? Empirically measured on this machine:
torch.mps.synchronize() carries a large, roughly constant fixed latency
(~200-300us). TinyViT's individual sub-ops (qkv_proj, sdpa, out_proj, ...)
run in single-digit-to-tens of microseconds, so synchronizing ~60 times per
forward (once per labeled region) makes every measurement dominated by fixed
sync overhead, not real compute time -- every region came back with
implausibly LOW roofline utilization (0.6-50%) uniformly, not just small
ops, which is the signature of a fixed per-call tax swamping the signal.

The fix used here is the standard microbenchmark pattern: capture each op's
REAL input tensor (correct shape/dtype/device, real weights) from one actual
forward pass via forward hooks, then time that op alone over many iterations
with a single synchronize() at the end -- amortizing the fixed sync cost
instead of paying it per op. This is the same pattern as
profiling/torch_profile.time_forward and roofline/hw_bench, just applied to
individual submodules instead of the whole model.
"""

import time
from typing import Dict

import torch
import torch.nn.functional as F

from ..models import BlockInfo


def _time_call(fn, warmup: int = 15, iters: int = 100) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters


def _capture_inputs(model, x, target_modules: Dict[str, torch.nn.Module]) -> Dict[str, torch.Tensor]:
    """Forward-hook each target module to grab its real input tensor from one
    live forward pass, then remove the hooks."""
    captured = {}
    handles = []

    def make_hook(name):
        def hook(mod, inp, out):
            captured[name] = inp[0].detach().clone()
        return hook

    for name, mod in target_modules.items():
        handles.append(mod.register_forward_hook(make_hook(name)))
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()
    return captured


def benchmark_block_ops(model, block_module, bi: BlockInfo, x_full, warmup: int = 15, iters: int = 100) -> Dict[str, float]:
    """Isolated GPU time (seconds/call) for qkv_proj, out_proj, local_conv,
    mlp, and the SDPA core, for one specific TinyVitBlock, using its real
    captured input shapes and real weights."""
    attn = block_module.attn
    targets = {
        "qkv_proj": attn.qkv,
        "out_proj": attn.proj,
        "local_conv": block_module.local_conv,
        "mlp": block_module.mlp,
    }
    captured = _capture_inputs(model, x_full, targets)

    times = {}
    times["qkv_proj"] = _time_call(lambda: attn.qkv(captured["qkv_proj"]), warmup, iters)
    times["out_proj"] = _time_call(lambda: attn.proj(captured["out_proj"]), warmup, iters)
    times["local_conv"] = _time_call(lambda: block_module.local_conv(captured["local_conv"]), warmup, iters)
    times["mlp"] = _time_call(lambda: block_module.mlp(captured["mlp"]), warmup, iters)

    # SDPA isn't a module we can hook directly; reconstruct q,k,v from the
    # captured qkv-linear input using the exact same reshape/split/permute as
    # timm's Attention.forward (see profiling/op_classify.py's annotated copy).
    with torch.no_grad():
        x_norm = captured["qkv_proj"]
        B, N, _ = x_norm.shape
        qkv = attn.qkv(x_norm)
        q, k, v = qkv.view(B, N, attn.num_heads, -1).split([attn.key_dim, attn.key_dim, attn.val_dim], dim=3)
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        attn_bias = attn.get_attention_biases(x_norm.device)
    times["sdpa_qk_softmax_av"] = _time_call(
        lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias), warmup, iters
    )

    # Window partition / reverse: pure reshape/transpose ops, timed on the
    # block's real captured pre-attention tensor (shortcut input, BHWC).
    if bi.input_resolution != bi.window_size:
        shortcut_capture = _capture_inputs(model, x_full, {"block_in": block_module})
        x_bhwc = shortcut_capture["block_in"]
        B, H, W, C = x_bhwc.shape
        ws = bi.window_size
        pad_b = (ws - H % ws) % ws
        pad_r = (ws - W % ws) % ws

        def partition():
            xp = F.pad(x_bhwc, (0, 0, 0, pad_r, 0, pad_b)) if (pad_b or pad_r) else x_bhwc
            pH, pW = H + pad_b, W + pad_r
            nH, nW = pH // ws, pW // ws
            return xp.view(B, nH, ws, nW, ws, C).transpose(2, 3).reshape(B * nH * nW, ws * ws, C)

        times["window_partition"] = _time_call(partition, warmup, iters)
        windows = partition()
        pH, pW = H + pad_b, W + pad_r
        nH, nW = pH // ws, pW // ws

        def reverse():
            return windows.view(B, nH, nW, ws, ws, C).transpose(2, 3).reshape(B, pH, pW, C)

        times["window_reverse"] = _time_call(reverse, warmup, iters)

    return times


def benchmark_conv_stem(model, x, warmup: int = 15, iters: int = 50) -> float:
    return _time_call(lambda: model.stages[0](model.patch_embed(x)), warmup, iters)


def benchmark_patch_merging(model, stage_idx: int, x_full, warmup: int = 15, iters: int = 100) -> float:
    stage = model.stages[stage_idx]
    captured = _capture_inputs(model, x_full, {"downsample": stage.downsample})
    return _time_call(lambda: stage.downsample(captured["downsample"]), warmup, iters)
