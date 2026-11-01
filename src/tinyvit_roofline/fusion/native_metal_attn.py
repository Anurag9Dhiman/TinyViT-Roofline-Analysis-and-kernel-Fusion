"""Native PyTorch MPS extensions that fuse window-attention computation,
bypassing MLX's Python dispatch overhead entirely (Phases 22/23).

Two designs, in the order they were tried:
- NativeFusedWindowAttention / V2 (Phase 22): fuses the WHOLE block
  (LayerNorm + QKV + attention + output projection) into one dispatch.
  Wins on a synthetic ablation config; loses on real TinyViT blocks.
- make_hybrid_attention_forward / patch_model_with_hybrid_attention
  (Phase 23): fuses ONLY the attention core, leaving PyTorch's own
  LayerNorm/Linear untouched. Wins on real blocks (window_size=7), but
  that win is too small a share of total model time to show up
  end-to-end -- see report.md sections 17/18 for the full story either way.
"""

import types
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_EXT_SOURCE = Path(__file__).parent / "native" / "fused_window_attention.mm"
_TEMPLATE_V2_SOURCE = Path(__file__).parent / "native" / "fused_window_attention_template_v2.mm"
_CORE_TEMPLATE_SOURCE = Path(__file__).parent / "native" / "fused_attention_core_template.mm"
_CORE_TILED_TEMPLATE_SOURCE = Path(__file__).parent / "native" / "fused_attention_core_tiled_template.mm"
_CORE_FLASH_TEMPLATE_SOURCE = Path(__file__).parent / "native" / "fused_attention_core_flash_template.mm"
_CORE_FLASH_FP16_TEMPLATE_SOURCE = Path(__file__).parent / "native" / "fused_attention_core_flash_fp16_template.mm"
_CORE_FLASH_FP16_KVCACHE_TEMPLATE_SOURCE = Path(__file__).parent / "native" / "fused_attention_core_flash_fp16_kvcache_template.mm"
_ext = None
_parameterized_cache = {}
_core_kernel_cache = {}
_flash_core_kernel_cache = {}
_flash_core_fp16_kernel_cache = {}
_flash_core_fp16_kvcache_kernel_cache = {}


def get_extension():
    global _ext
    if _ext is None:
        _ext = load(
            name="tinyvit_fused_window_attention",
            sources=[str(_EXT_SOURCE)],
            extra_cflags=["-std=c++17"],
            extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
        )
    return _ext


def build_parameterized_kernel(N: int, DIM: int, NUM_HEADS: int, KEY_DIM: int, VAL_DIM: int,
                                OUT_DIM: int, num_threads: int = 512):
    """v2 (device-memory-scratch) kernel for configs too big to fit the
    all-threadgroup-memory design (fused_window_attention.mm) under Apple
    M4's 32768-byte threadgroup memory limit -- e.g. any real TinyViT block
    (window_size=7 or 14), as opposed to the synthetic window_size=4
    ablation-grid point the all-threadgroup design was built for. See
    fused_window_attention_template_v2.mm's header: this trades some speed
    (global memory round trips between stages) for the ability to run at
    these real sizes at all."""
    key = (N, DIM, NUM_HEADS, KEY_DIM, VAL_DIM, OUT_DIM, num_threads)
    if key in _parameterized_cache:
        return _parameterized_cache[key]
    kernel_name = f"fused_window_attention_v2_n{N}_d{DIM}_h{NUM_HEADS}"
    template = _TEMPLATE_V2_SOURCE.read_text()
    src = template.format(N=N, DIM=DIM, NUM_HEADS=NUM_HEADS, KEY_DIM=KEY_DIM, VAL_DIM=VAL_DIM,
                           OUT_DIM=OUT_DIM, NUM_THREADS=num_threads, KERNEL_NAME=kernel_name)
    gen_dir = Path("/tmp/tinyvit_native_kernels")
    gen_dir.mkdir(exist_ok=True)
    gen_path = gen_dir / f"{kernel_name}.mm"
    gen_path.write_text(src)
    ext = load(
        name=kernel_name,
        sources=[str(gen_path)],
        extra_cflags=["-std=c++17"],
        extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
    )
    _parameterized_cache[key] = ext
    return ext


class NativeFusedWindowAttention:
    """Fixed for window_size=4, dim=64 (num_heads=2, key_dim=32, val_dim=32) --
    matches build_attention(64, 4) exactly. Call with the same real,
    windowed-token input shape (batch_windows, 16, 64) the PyTorch module
    expects."""

    def __init__(self, torch_attn):
        assert torch_attn.key_dim == 32 and torch_attn.val_dim == 32 and torch_attn.num_heads == 2, (
            "native kernel is hardcoded for window_size=4, dim=64 (key_dim=val_dim=32, num_heads=2); "
            f"got key_dim={torch_attn.key_dim} val_dim={torch_attn.val_dim} num_heads={torch_attn.num_heads}"
        )
        self.ext = get_extension()
        self.norm_weight = torch_attn.norm.weight.contiguous()
        self.norm_bias = torch_attn.norm.bias.contiguous()
        self.qkv_weight = torch_attn.qkv.weight.contiguous()
        self.qkv_bias = torch_attn.qkv.bias.contiguous()
        self.proj_weight = torch_attn.proj.weight.contiguous()
        self.proj_bias = torch_attn.proj.bias.contiguous()
        with torch.no_grad():
            self.attn_bias = torch_attn.attention_biases[:, torch_attn.attention_bias_idxs].contiguous()
        self.scale = torch_attn.scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.ext.fused_window_attention(
            x.contiguous(), self.norm_weight, self.norm_bias,
            self.qkv_weight, self.qkv_bias, self.attn_bias,
            self.proj_weight, self.proj_bias, self.scale,
        )


class NativeFusedWindowAttentionV2:
    """Parameterized (device-memory-scratch) version -- works for any real
    TinyViT block's (window_size, dim, num_heads, key_dim, val_dim) config,
    not just the ws=4/dim=64 point NativeFusedWindowAttention is fixed to.
    Compiles a fresh kernel per distinct config (cached after first call)."""

    def __init__(self, torch_attn, window_size: int, dim: int):
        self.N = window_size * window_size
        self.ext = build_parameterized_kernel(
            self.N, dim, torch_attn.num_heads, torch_attn.key_dim, torch_attn.val_dim, torch_attn.out_dim,
        )
        self.norm_weight = torch_attn.norm.weight.contiguous()
        self.norm_bias = torch_attn.norm.bias.contiguous()
        self.qkv_weight = torch_attn.qkv.weight.contiguous()
        self.qkv_bias = torch_attn.qkv.bias.contiguous()
        self.proj_weight = torch_attn.proj.weight.contiguous()
        self.proj_bias = torch_attn.proj.bias.contiguous()
        with torch.no_grad():
            self.attn_bias = torch_attn.attention_biases[:, torch_attn.attention_bias_idxs].contiguous()
        self.scale = torch_attn.scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.ext.forward(
            x.contiguous(), self.norm_weight, self.norm_bias,
            self.qkv_weight, self.qkv_bias, self.attn_bias,
            self.proj_weight, self.proj_bias, self.scale,
        )


def get_core_kernel(N: int, num_heads: int, key_dim: int, val_dim: int, out_dim: int,
                     num_threads: int = 256, tile_rows: int = 32):
    """Attention-CORE-only kernel (QK^T*scale + bias, softmax, AV) -- see
    fused_attention_core_template.mm's header. Used with PyTorch's own
    LayerNorm/Linear for everything else, which is the design that
    actually wins on real TinyViT blocks (Phase 23), unlike the full-block
    fusion in NativeFusedWindowAttention/V2 (Phase 22), which doesn't.

    Non-tiled for N*N*4 <= 32768 bytes (window_size <= ~9, covers real
    window_size=7); row-tiled (TILE_ROWS*N*4 <= 32768) otherwise (covers
    window_size=14's N=196) -- though Phase 23 found the tiled version
    actually LOSES on window_size=14, since N=196 is big enough that
    baseline's own tuned attention wins; included for completeness, not
    because it should be used."""
    key = (N, num_heads, key_dim, val_dim, out_dim, num_threads, tile_rows)
    if key in _core_kernel_cache:
        return _core_kernel_cache[key]
    needs_tiling = N * N * 4 > 32768
    if needs_tiling:
        kernel_name = f"fused_attn_core_tiled_n{N}_h{num_heads}"
        template = _CORE_TILED_TEMPLATE_SOURCE.read_text()
        src = template.format(N=N, TILE_ROWS=tile_rows, NUM_HEADS=num_heads, KEY_DIM=key_dim,
                               VAL_DIM=val_dim, OUT_DIM=out_dim, NUM_THREADS=num_threads, KERNEL_NAME=kernel_name)
    else:
        kernel_name = f"fused_attn_core_n{N}_h{num_heads}"
        template = _CORE_TEMPLATE_SOURCE.read_text()
        src = template.format(N=N, NUM_HEADS=num_heads, KEY_DIM=key_dim, VAL_DIM=val_dim,
                               OUT_DIM=out_dim, NUM_THREADS=num_threads, KERNEL_NAME=kernel_name)
    gen_dir = Path("/tmp/tinyvit_native_kernels")
    gen_dir.mkdir(exist_ok=True)
    gen_path = gen_dir / f"{kernel_name}.mm"
    gen_path.write_text(src)
    ext = load(
        name=kernel_name,
        sources=[str(gen_path)],
        extra_cflags=["-std=c++17"],
        extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
    )
    _core_kernel_cache[key] = ext
    return ext


def get_flash_core_kernel(N: int, num_heads: int, key_dim: int, val_dim: int, out_dim: int):
    """Attention-CORE-only kernel using online/streaming softmax (flash-
    attention style) with Metal simdgroup_matrix hardware tile ops for both
    QK^T and AV, instead of fused_attention_core_tiled_template.mm's scalar
    dot-product loops. Tiles both queries and keys in 8-row/col blocks (not
    just queries), carrying running max/sum/output statistics across key
    tiles so the (N,N) scores matrix never exists in full -- only one 8x8
    block at a time. Works for any N (boundary-masked internally), unlike
    the isolated microbenchmarks this was prototyped against which assumed
    N divisible by 8."""
    key = (N, num_heads, key_dim, val_dim, out_dim)
    if key in _flash_core_kernel_cache:
        return _flash_core_kernel_cache[key]
    kernel_name = f"fused_attn_core_flash_n{N}_h{num_heads}"
    template = _CORE_FLASH_TEMPLATE_SOURCE.read_text()
    src = template.format(N=N, NUM_HEADS=num_heads, KEY_DIM=key_dim, VAL_DIM=val_dim,
                           OUT_DIM=out_dim, KERNEL_NAME=kernel_name)
    gen_dir = Path("/tmp/tinyvit_native_kernels")
    gen_dir.mkdir(exist_ok=True)
    gen_path = gen_dir / f"{kernel_name}.mm"
    gen_path.write_text(src)
    ext = load(
        name=kernel_name,
        sources=[str(gen_path)],
        extra_cflags=["-std=c++17"],
        extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
    )
    _flash_core_kernel_cache[key] = ext
    return ext


def make_hybrid_attention_forward_flash(attn, window_size: int):
    """Same division of labor as make_hybrid_attention_forward (PyTorch's
    own LayerNorm/Linear for norm/QKV/proj), but with the attention core
    replaced by the flash-attention-style simdgroup_matrix kernel
    (get_flash_core_kernel) instead of the scalar row-tiled one."""
    N = window_size * window_size
    ext = get_flash_core_kernel(N, attn.num_heads, attn.key_dim, attn.val_dim, attn.out_dim)
    with torch.no_grad():
        attn_bias_dense = attn.attention_biases[:, attn.attention_bias_idxs].contiguous()
    scale = attn.scale

    def forward(self, x):
        xn = self.norm(x)
        qkv = self.qkv(xn).contiguous()
        attn_out = ext.forward(qkv, attn_bias_dense, scale)
        return self.proj(attn_out)

    return forward


def get_flash_core_kernel_fp16(N: int, num_heads: int, key_dim: int, val_dim: int, out_dim: int):
    """fp16 variant of get_flash_core_kernel: Q/K/V and the position bias
    are half precision (halving their device-memory traffic, the dominant
    data volume here), matmul'd via simdgroup_half8x8; the online-softmax
    running statistics stay float32 (mixed-precision flash-attention, the
    standard pattern) and the kernel's output stays float32 so the
    surrounding PyTorch LayerNorm/Linear ops need no changes."""
    key = (N, num_heads, key_dim, val_dim, out_dim)
    if key in _flash_core_fp16_kernel_cache:
        return _flash_core_fp16_kernel_cache[key]
    kernel_name = f"fused_attn_core_flash_fp16_n{N}_h{num_heads}"
    template = _CORE_FLASH_FP16_TEMPLATE_SOURCE.read_text()
    src = template.format(N=N, NUM_HEADS=num_heads, KEY_DIM=key_dim, VAL_DIM=val_dim,
                           OUT_DIM=out_dim, KERNEL_NAME=kernel_name)
    gen_dir = Path("/tmp/tinyvit_native_kernels")
    gen_dir.mkdir(exist_ok=True)
    gen_path = gen_dir / f"{kernel_name}.mm"
    gen_path.write_text(src)
    ext = load(
        name=kernel_name,
        sources=[str(gen_path)],
        extra_cflags=["-std=c++17"],
        extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
    )
    _flash_core_fp16_kernel_cache[key] = ext
    return ext


def make_hybrid_attention_forward_flash_fp16(attn, window_size: int):
    """Same division of labor as make_hybrid_attention_forward_flash, but
    qkv and the position bias are cast to half before reaching the kernel
    (get_flash_core_kernel_fp16) -- the actual memory-traffic-halving lever
    being tested, not just doing fp16 math on fp32-loaded data. Output
    comes back float32, so proj() (fp32 Linear) is untouched."""
    N = window_size * window_size
    ext = get_flash_core_kernel_fp16(N, attn.num_heads, attn.key_dim, attn.val_dim, attn.out_dim)
    with torch.no_grad():
        attn_bias_dense = attn.attention_biases[:, attn.attention_bias_idxs].contiguous().half()
    scale = attn.scale

    def forward(self, x):
        xn = self.norm(x)
        qkv = self.qkv(xn).contiguous().half()
        attn_out = ext.forward(qkv, attn_bias_dense, scale)
        return self.proj(attn_out)

    return forward


def get_flash_core_kernel_fp16_kvcache(N: int, num_heads: int, key_dim: int, val_dim: int, out_dim: int):
    """K/V-caching variant of get_flash_core_kernel_fp16: one threadgroup
    per (window, head) -- not per query tile -- with K/V staged into
    threadgroup memory ONCE and shared by every query tile's simdgroup,
    instead of each query tile independently re-reading the same K/V from
    device memory ~ceil(N/8) times. See
    fused_attention_core_flash_fp16_kvcache_template.mm's header."""
    key = (N, num_heads, key_dim, val_dim, out_dim)
    if key in _flash_core_fp16_kvcache_kernel_cache:
        return _flash_core_fp16_kvcache_kernel_cache[key]
    kernel_name = f"fused_attn_core_flash_fp16_kvcache_n{N}_h{num_heads}"
    template = _CORE_FLASH_FP16_KVCACHE_TEMPLATE_SOURCE.read_text()
    src = template.format(N=N, NUM_HEADS=num_heads, KEY_DIM=key_dim, VAL_DIM=val_dim,
                           OUT_DIM=out_dim, KERNEL_NAME=kernel_name)
    gen_dir = Path("/tmp/tinyvit_native_kernels")
    gen_dir.mkdir(exist_ok=True)
    gen_path = gen_dir / f"{kernel_name}.mm"
    gen_path.write_text(src)
    ext = load(
        name=kernel_name,
        sources=[str(gen_path)],
        extra_cflags=["-std=c++17"],
        extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
    )
    _flash_core_fp16_kvcache_kernel_cache[key] = ext
    return ext


def make_hybrid_attention_forward_flash_fp16_kvcache(attn, window_size: int):
    """Same division of labor as make_hybrid_attention_forward_flash_fp16,
    but using get_flash_core_kernel_fp16_kvcache (one threadgroup per
    (window, head), K/V cached once) instead of one threadgroup per query
    tile."""
    N = window_size * window_size
    ext = get_flash_core_kernel_fp16_kvcache(N, attn.num_heads, attn.key_dim, attn.val_dim, attn.out_dim)
    with torch.no_grad():
        attn_bias_dense = attn.attention_biases[:, attn.attention_bias_idxs].contiguous().half()
    scale = attn.scale

    def forward(self, x):
        xn = self.norm(x)
        qkv = self.qkv(xn).contiguous().half()
        attn_out = ext.forward(qkv, attn_bias_dense, scale)
        return self.proj(attn_out)

    return forward


def make_hybrid_attention_forward(attn, window_size: int):
    """A drop-in replacement for a real TinyViT Attention module's
    .forward(): PyTorch's own LayerNorm and Linear for norm/QKV/proj
    (already fast -- Phase 17 showed PyTorch's native ops beat anything
    hand-written here), with ONLY the attention core replaced by the
    native fused kernel. Returns a bound-method-compatible function; use
    via `attn.forward = types.MethodType(make_hybrid_attention_forward(attn, ws), attn)`."""
    N = window_size * window_size
    ext = get_core_kernel(N, attn.num_heads, attn.key_dim, attn.val_dim, attn.out_dim)
    with torch.no_grad():
        attn_bias_dense = attn.attention_biases[:, attn.attention_bias_idxs].contiguous()
    scale = attn.scale

    def forward(self, x):
        xn = self.norm(x)
        qkv = self.qkv(xn).contiguous()
        attn_out = ext.forward(qkv, attn_bias_dense, scale)
        return self.proj(attn_out)

    return forward


def patch_model_with_hybrid_attention(model, block_infos, max_window_size: int = 9):
    """Replaces every eligible TinyVitBlock's attention .forward() with the
    hybrid native-core version (see make_hybrid_attention_forward).
    block_infos: from models.enumerate_attention_blocks(model, variant).
    max_window_size=9 default skips window_size=14 blocks -- Phase 23 found
    fusing those actively hurts (baseline's own attention wins at that
    size), not just "doesn't help." Returns (num_patched, num_skipped)."""
    from ..models import get_module_by_name

    patched, skipped = 0, 0
    for bi in block_infos:
        if bi.window_size > max_window_size:
            skipped += 1
            continue
        block = get_module_by_name(model, bi.module_name)
        attn = block.attn
        attn.forward = types.MethodType(make_hybrid_attention_forward(attn, bi.window_size), attn)
        patched += 1
    return patched, skipped
