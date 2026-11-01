// Flash-attention-style fused attention core: QK^T*scale+bias, softmax, AV
// computed via online/streaming softmax using Metal's simdgroup_matrix
// hardware tile ops -- the (N,N) scores matrix never materializes in
// device memory, only one 8x8 block of it exists at a time (in a
// threadgroup scratch buffer). This is the alternative to
// fused_attention_core_tiled_template.mm's row-tiled-but-still-scalar
// design: that one avoids the on-chip (N,N) buffer by tiling rows (still
// fits a dense N-wide row per tile) and uses plain scalar loops for the
// dot products; this one tiles BOTH rows and columns (needed for N=196,
// where even one N-wide row tile's score buffer is fine, but this
// kernel's point is to use real matrix-unit hardware ops instead of
// scalar loops for the QK^T and AV matmuls themselves) and uses
// simdgroup_matrix for both matmuls, carrying running (max, sum, weighted
// output) statistics across key-tiles (the standard flash-attention
// recurrence) instead of materializing a full row of scores at once.
//
// N is not necessarily a multiple of 8 (window_size=14 -> N=196, not
// divisible by 8), so every tile load/store is boundary-guarded: query/key
// rows past N are zero-padded when staging into threadgroup memory (safe,
// since the guard short-circuits the device-memory read), and the bias
// add explicitly masks any (query, key) pair where either index is >= N
// to -INFINITY before the softmax max/sum pass, which makes padded key
// columns contribute exactly 0 after exp() (not NaN) -- only the final
// store of an invalid query row is guarded separately (there, garbage/NaN
// statistics in an unused row never reach the output buffer).
//
// One threadgroup = one simdgroup (32 threads) = one (window, query-tile)
// pair; heads are looped inside the same threadgroup, each resetting its
// own running statistics. Grid: (ceil(N/8) query tiles, batch_windows, 1).
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
static constexpr uint NUM_Q_TILES = (N + 7) / 8;
static constexpr uint NUM_K_TILES = (N + 7) / 8;

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

kernel void {KERNEL_NAME}(
    device const float* qkv [[buffer(0)]],
    device const float* attn_bias [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{{
    uint qt = tile.x;
    uint window = tile.y;
    uint q_row_valid = min((uint)8, N - qt * 8);
    bool q_full = (q_row_valid == 8);

    device const float* qkv_w = qkv + window * N * TOTAL_QKV;
    device float* out_w = out + window * N * OUT_DIM;

    // Qs/Ks/Vs are only written and read on the (rare) boundary tile where
    // N isn't a multiple of 8 -- the common case loads straight from device
    // memory via simdgroup_load's own row-stride support (TOTAL_QKV),
    // skipping the staging copy entirely. Staging every tile regardless of
    // need was tried first and cost ~2x: the scalar guarded-copy loop is far
    // more expensive than the hardware simdgroup_load it was protecting.
    threadgroup float Qs[8 * KEY_DIM];
    threadgroup float Ks[8 * KEY_DIM];
    threadgroup float Vs[8 * VAL_DIM];
    threadgroup float S[8 * 8];
    threadgroup float m[8], l[8], newm[8], corr[8];
    threadgroup float acc[8 * VAL_DIM];

    for (uint h = 0; h < NUM_HEADS; h++) {{
        if (tid < 8) {{ m[tid] = -INFINITY; l[tid] = 0.0f; }}
        for (uint idx = tid; idx < 8 * VAL_DIM; idx += 32) acc[idx] = 0.0f;

        if (!q_full) {{
            for (uint idx = tid; idx < 8 * KEY_DIM; idx += 32) {{
                uint r = idx / KEY_DIM, c = idx % KEY_DIM;
                Qs[idx] = (r < q_row_valid) ? qkv_w[(qt * 8 + r) * TOTAL_QKV + h * PER_HEAD + c] : 0.0f;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        for (uint kt = 0; kt < (N + 7) / 8; kt++) {{
            uint k_row_valid = min((uint)8, N - kt * 8);
            bool k_full = (k_row_valid == 8);

            if (!k_full) {{
                for (uint idx = tid; idx < 8 * KEY_DIM; idx += 32) {{
                    uint r = idx / KEY_DIM, c = idx % KEY_DIM;
                    Ks[idx] = (r < k_row_valid) ? qkv_w[(kt * 8 + r) * TOTAL_QKV + h * PER_HEAD + KEY_DIM + c] : 0.0f;
                }}
                for (uint idx = tid; idx < 8 * VAL_DIM; idx += 32) {{
                    uint r = idx / VAL_DIM, c = idx % VAL_DIM;
                    Vs[idx] = (r < k_row_valid) ? qkv_w[(kt * 8 + r) * TOTAL_QKV + h * PER_HEAD + 2 * KEY_DIM + c] : 0.0f;
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}

            // S = Q_tile(8xKEY_DIM) @ K_tile(8xKEY_DIM)^T -> raw (unscaled, unbiased) dot products
            simdgroup_float8x8 sacc = simdgroup_float8x8(0.0f);
            for (uint k = 0; k < KEY_DIM; k += 8) {{
                simdgroup_float8x8 q8, k8;
                if (q_full) simdgroup_load(q8, qkv_w + (qt * 8) * TOTAL_QKV + h * PER_HEAD + k, TOTAL_QKV);
                else        simdgroup_load(q8, Qs + k, KEY_DIM);
                if (k_full) simdgroup_load(k8, qkv_w + (kt * 8) * TOTAL_QKV + h * PER_HEAD + KEY_DIM + k, TOTAL_QKV, ulong2(0, 0), /*transpose=*/true);
                else        simdgroup_load(k8, Ks + k, KEY_DIM, ulong2(0, 0), /*transpose=*/true);
                simdgroup_multiply_accumulate(sacc, q8, k8, sacc);
            }}
            simdgroup_store(sacc, S, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // scale + bias + boundary mask (2 elements/thread: 64 total / 32 threads)
            for (uint idx = tid; idx < 64; idx += 32) {{
                uint i = idx / 8, j = idx % 8;
                if (i < q_row_valid && j < k_row_valid) {{
                    S[idx] = S[idx] * scale + attn_bias[h * N * N + (qt * 8 + i) * N + (kt * 8 + j)];
                }} else {{
                    S[idx] = -INFINITY;
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // online softmax update, one thread per query row
            if (tid < 8) {{
                uint i = tid;
                float rowmax = -INFINITY;
                for (uint j = 0; j < 8; j++) rowmax = max(rowmax, S[i * 8 + j]);
                float nm = max(m[i], rowmax);
                newm[i] = nm;
                float c0 = (m[i] == -INFINITY) ? 0.0f : exp(m[i] - nm);
                corr[i] = c0;
                float rowsum = 0.0f;
                for (uint j = 0; j < 8; j++) {{
                    float p = exp(S[i * 8 + j] - nm);
                    S[i * 8 + j] = p;
                    rowsum += p;
                }}
                l[i] = l[i] * c0 + rowsum;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < 8) m[tid] = newm[tid];

            // acc = acc * corr + P(8x8) @ Vs(8xVAL_DIM)
            for (uint idx = tid; idx < 8 * VAL_DIM; idx += 32) {{
                uint i = idx / VAL_DIM;
                acc[idx] *= corr[i];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint n = 0; n < VAL_DIM; n += 8) {{
                simdgroup_float8x8 pacc = simdgroup_float8x8(0.0f);
                simdgroup_float8x8 p8, v8;
                simdgroup_load(p8, S, 8);
                if (k_full) simdgroup_load(v8, qkv_w + (kt * 8) * TOTAL_QKV + h * PER_HEAD + 2 * KEY_DIM + n, TOTAL_QKV);
                else        simdgroup_load(v8, Vs + n, VAL_DIM);
                simdgroup_multiply_accumulate(pacc, p8, v8, pacc);
                threadgroup float delta[64];
                simdgroup_store(pacc, delta, 8);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint t = tid; t < 64; t += 32) {{
                    uint r = t / 8, c = t % 8;
                    acc[r * VAL_DIM + n + c] += delta[t];
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
        }}

        // normalize and write out (only valid query rows)
        for (uint idx = tid; idx < 8 * VAL_DIM; idx += 32) {{
            uint i = idx / VAL_DIM, c = idx % VAL_DIM;
            if (i < q_row_valid) {{
                out_w[(qt * 8 + i) * OUT_DIM + h * VAL_DIM + c] = acc[idx] / l[i];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
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
    TORCH_CHECK(qkv.scalar_type() == torch::kFloat32, "qkv must be float32");
    int64_t batch_windows = qkv.size(0);

    torch::Tensor out = torch::empty({{batch_windows, N, OUT_DIM}}, qkv.options());
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

        MTLSize gridSize = MTLSizeMake((N + 7) / 8, batch_windows, 1);
        MTLSize threadgroupSize = MTLSizeMake(32, 1, 1);
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        stream->synchronize(at::mps::SyncType::COMMIT);
    }});

    return out;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("forward", &{KERNEL_NAME}_fn, "Fused attention core, flash-attention-style (native MPS)");
}}
