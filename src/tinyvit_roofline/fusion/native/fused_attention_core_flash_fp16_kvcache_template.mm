// K/V-caching variant of fused_attention_core_flash_fp16_template.mm.
// The per-query-tile design there dispatches one threadgroup PER 8-row
// query tile, each independently re-reading the SAME K/V data for its
// (window, head) from device memory -- for N=196, that's ~25x redundant
// global-memory traffic for K and V. This version dispatches one
// threadgroup per (window, head) instead: all ceil(N/8) query tiles for
// that unit are handled by separate simdgroups WITHIN one threadgroup,
// after a single cooperative load stages the full K and V (half
// precision) into threadgroup memory once, shared by every simdgroup.
//
// Thread layout: NUM_TILES simdgroups (32 threads each) per threadgroup,
// one simdgroup per query tile (grid has no query-tile dimension anymore).
// For N=196, NUM_TILES=25 -> 800 threads/threadgroup.
//
// Threadgroup memory budget (half unless noted): Ksh/Vsh cache the full
// (boundary-padded) K/V, NUM_TILES*64-element half scratch buffers hold
// each simdgroup's own ephemeral 8x8 QK^T/PV tile (reused every key-tile
// iteration, never persisted), and one small Qs scratch buffer is shared
// by whichever single simdgroup owns the boundary query tile (N not a
// multiple of 8). The running softmax state (m, l, acc) is NOT threadgroup
// memory at all -- it lives in thread-private registers, held only by the
// 8 lanes (one per query row) that do the per-row softmax bookkeeping,
// using simdgroup_barrier (intra-simdgroup only, not a full threadgroup
// sync) between each simdgroup_store and the scalar reads/writes that
// follow it within the same simdgroup's own scratch slice.
//
// Build note: compiled WITHOUT -fobjc-arc (see fused_window_attention.mm).

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/native/mps/OperationUtils.h>
#include <Metal/Metal.h>

static constexpr uint N = {N};
static constexpr uint NUM_HEADS = {NUM_HEADS};
static constexpr uint KEY_DIM = {KEY_DIM};
static constexpr uint VAL_DIM = {VAL_DIM};
static constexpr uint OUT_DIM = {OUT_DIM};
static constexpr uint PER_HEAD = 2 * KEY_DIM + VAL_DIM;
static constexpr uint TOTAL_QKV = NUM_HEADS * PER_HEAD;
static constexpr uint NUM_TILES = (N + 7) / 8;
static constexpr uint N_PAD = NUM_TILES * 8;
static constexpr uint NUM_THREADS = NUM_TILES * 32;

static const char* KERNEL_SRC = R"METAL(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint N = {N};
constant uint NUM_HEADS = {NUM_HEADS};
constant uint KEY_DIM = {KEY_DIM};
constant uint VAL_DIM = {VAL_DIM};
constant uint OUT_DIM = {OUT_DIM};
constant uint PER_HEAD = 2 * KEY_DIM + VAL_DIM;
constant uint TOTAL_QKV = NUM_HEADS * PER_HEAD;
constant uint NUM_TILES = (N + 7) / 8;
constant uint N_PAD = NUM_TILES * 8;
constant uint NUM_THREADS = NUM_TILES * 32;

kernel void {KERNEL_NAME}(
    device const half* qkv [[buffer(0)]],
    device const half* attn_bias [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{{
    uint h = tile.x;
    uint window = tile.y;

    device const half* qkv_w = qkv + window * N * TOTAL_QKV;
    device float* out_w = out + window * N * OUT_DIM;

    threadgroup half Ksh[N_PAD * KEY_DIM];
    threadgroup half Vsh[N_PAD * VAL_DIM];
    threadgroup half S_scratch[NUM_TILES * 64];
    threadgroup half delta_scratch[NUM_TILES * 64];
    threadgroup half Qs_scratch[8 * KEY_DIM];  // used only by the one boundary-tile simdgroup

    // cooperative load of this (window, head)'s full K/V, once, shared by
    // every simdgroup -- zero-padded past N (masked out in softmax below).
    for (uint idx = tid; idx < N_PAD * KEY_DIM; idx += NUM_THREADS) {{
        uint r = idx / KEY_DIM, c = idx % KEY_DIM;
        Ksh[idx] = (r < N) ? qkv_w[r * TOTAL_QKV + h * PER_HEAD + KEY_DIM + c] : (half)0.0;
    }}
    for (uint idx = tid; idx < N_PAD * VAL_DIM; idx += NUM_THREADS) {{
        uint r = idx / VAL_DIM, c = idx % VAL_DIM;
        Vsh[idx] = (r < N) ? qkv_w[r * TOTAL_QKV + h * PER_HEAD + 2 * KEY_DIM + c] : (half)0.0;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint qt = sg_id;
    uint q_row_valid = min((uint)8, N - qt * 8);
    bool q_full = (q_row_valid == 8);
    threadgroup half* S_mine = S_scratch + sg_id * 64;
    threadgroup half* delta_mine = delta_scratch + sg_id * 64;

    if (!q_full) {{
        for (uint idx = lane; idx < 8 * KEY_DIM; idx += 32) {{
            uint r = idx / KEY_DIM, c = idx % KEY_DIM;
            Qs_scratch[idx] = (r < q_row_valid) ? qkv_w[(qt * 8 + r) * TOTAL_QKV + h * PER_HEAD + c] : (half)0.0;
        }}
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }}

    float m_reg = -INFINITY, l_reg = 0.0f;
    float acc_row[VAL_DIM];
    for (uint c = 0; c < VAL_DIM; c++) acc_row[c] = 0.0f;

    for (uint kt = 0; kt < NUM_TILES; kt++) {{
        uint k_row_valid = min((uint)8, N - kt * 8);

        simdgroup_half8x8 sacc = simdgroup_half8x8(0.0h);
        for (uint k = 0; k < KEY_DIM; k += 8) {{
            simdgroup_half8x8 q8, k8;
            if (q_full) simdgroup_load(q8, qkv_w + (qt * 8) * TOTAL_QKV + h * PER_HEAD + k, TOTAL_QKV);
            else        simdgroup_load(q8, Qs_scratch + k, KEY_DIM);
            simdgroup_load(k8, Ksh + kt * 8 * KEY_DIM + k, KEY_DIM, ulong2(0, 0), /*transpose=*/true);
            simdgroup_multiply_accumulate(sacc, q8, k8, sacc);
        }}
        simdgroup_store(sacc, S_mine, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // bias + mask + online-softmax update, entirely in this lane's own
        // registers (only lane < 8, one lane per query row)
        if (lane < 8) {{
            uint i = lane;
            float srow[8];
            float rowmax = -INFINITY;
            for (uint j = 0; j < 8; j++) {{
                if (i < q_row_valid && j < k_row_valid) {{
                    srow[j] = float(S_mine[i * 8 + j]) * scale + float(attn_bias[h * N * N + (qt * 8 + i) * N + (kt * 8 + j)]);
                }} else {{
                    srow[j] = -INFINITY;
                }}
                rowmax = max(rowmax, srow[j]);
            }}
            float nm = max(m_reg, rowmax);
            float c0 = (m_reg == -INFINITY) ? 0.0f : exp(m_reg - nm);
            float rowsum = 0.0f;
            for (uint j = 0; j < 8; j++) {{
                float p = exp(srow[j] - nm);
                S_mine[i * 8 + j] = half(p);
                rowsum += p;
            }}
            l_reg = l_reg * c0 + rowsum;
            m_reg = nm;
            for (uint c = 0; c < VAL_DIM; c++) acc_row[c] *= c0;
        }}
        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (uint n = 0; n < VAL_DIM; n += 8) {{
            simdgroup_half8x8 pacc = simdgroup_half8x8(0.0h);
            simdgroup_half8x8 p8, v8;
            simdgroup_load(p8, S_mine, 8);
            simdgroup_load(v8, Vsh + kt * 8 * VAL_DIM + n, VAL_DIM);
            simdgroup_multiply_accumulate(pacc, p8, v8, pacc);
            simdgroup_store(pacc, delta_mine, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (lane < 8) {{
                for (uint c = 0; c < 8; c++) acc_row[n + c] += float(delta_mine[lane * 8 + c]);
            }}
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }}
    }}

    if (lane < 8 && lane < q_row_valid) {{
        uint i = lane;
        for (uint c = 0; c < VAL_DIM; c++) {{
            out_w[(qt * 8 + i) * OUT_DIM + h * VAL_DIM + c] = acc_row[c] / l_reg;
        }}
    }}
}}
)METAL";

static id<MTLComputePipelineState> g_pipeline = nil;

static id<MTLComputePipelineState> getPipeline() {{
    if (g_pipeline != nil) return g_pipeline;
    id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
    NSError* error = nil;
    NSString* src = [NSString stringWithUTF8String:KERNEL_SRC];
    id<MTLLibrary> library = [device newLibraryWithSource:src options:nil error:&error];
    TORCH_CHECK(library != nil, "Failed to compile Metal kernel: ",
                error ? error.localizedDescription.UTF8String : "unknown error");
    id<MTLFunction> fn = [library newFunctionWithName:@"{KERNEL_NAME}"];
    TORCH_CHECK(fn != nil, "Failed to find kernel function");
    g_pipeline = [device newComputePipelineStateWithFunction:fn error:&error];
    TORCH_CHECK(g_pipeline != nil, "Failed to create pipeline state: ",
                error ? error.localizedDescription.UTF8String : "unknown error");
    return g_pipeline;
}}

torch::Tensor {KERNEL_NAME}_fn(torch::Tensor qkv, torch::Tensor attn_bias, double scale)
{{
    TORCH_CHECK(qkv.is_mps(), "qkv must be an MPS tensor");
    TORCH_CHECK(qkv.scalar_type() == torch::kFloat16, "qkv must be float16");
    int64_t batch_windows = qkv.size(0);

    torch::Tensor out = torch::empty({{batch_windows, N, OUT_DIM}}, qkv.options().dtype(torch::kFloat32));
    float scale_f = (float)scale;

    id<MTLComputePipelineState> pipeline = getPipeline();
    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();

    dispatch_sync(stream->queue(), ^{{
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(qkv) offset:qkv.storage_offset() * qkv.element_size() atIndex:0];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(attn_bias) offset:attn_bias.storage_offset() * attn_bias.element_size() atIndex:1];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(out) offset:out.storage_offset() * out.element_size() atIndex:2];
        [encoder setBytes:&scale_f length:sizeof(float) atIndex:3];

        MTLSize gridSize = MTLSizeMake(NUM_HEADS, batch_windows, 1);
        MTLSize threadgroupSize = MTLSizeMake(NUM_THREADS, 1, 1);
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        stream->synchronize(at::mps::SyncType::COMMIT);
    }});

    return out;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("forward", &{KERNEL_NAME}_fn, "Fused attention core, flash-attention-style, fp16, K/V cached per-threadgroup (native MPS)");
}}
