// fp16 variant of fused_attention_core_flash_template.mm: Q/K/V and the
// position bias are loaded and matmul'd in half precision
// (simdgroup_half8x8) -- halving the memory traffic for the dominant
// inputs -- while the online-softmax running statistics (m, l, acc) stay
// float32 for numerical stability, the standard mixed-precision
// flash-attention pattern. Each simdgroup matmul result (QK^T raw scores,
// then P@V) is produced in a half accumulator, staged through a
// threadgroup half buffer, then explicitly upcast to float when folded
// into the float running state -- same staging-then-accumulate pattern
// the fp32 version already uses for P@V, just with an explicit cast added.
// Output stays float32 (acc is float, so the final store already casts
// back) so the surrounding PyTorch proj() Linear needs no changes.
//
// Boundary handling (N not a multiple of 8) is identical to the fp32
// version: only the one partial tile stages through threadgroup memory
// (now half-typed); the other N/8 full tiles simdgroup_load directly from
// the half device buffer.
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
    device const half* qkv [[buffer(0)]],
    device const half* attn_bias [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{{
    uint qt = tile.x;
    uint window = tile.y;
    uint q_row_valid = min((uint)8, N - qt * 8);
    bool q_full = (q_row_valid == 8);

    device const half* qkv_w = qkv + window * N * TOTAL_QKV;
    device float* out_w = out + window * N * OUT_DIM;

    threadgroup half Qs[8 * KEY_DIM];
    threadgroup half Ks[8 * KEY_DIM];
    threadgroup half Vs[8 * VAL_DIM];
    threadgroup half Sh[8 * 8];   // raw half QK^T scores, later reused for half(P)
    threadgroup float S[8 * 8];   // scaled+biased+masked scores (fp32), then softmax probs
    threadgroup float m[8], l[8], newm[8], corr[8];
    threadgroup float acc[8 * VAL_DIM];
    threadgroup half delta[8 * 8];

    for (uint h = 0; h < NUM_HEADS; h++) {{
        if (tid < 8) {{ m[tid] = -INFINITY; l[tid] = 0.0f; }}
        for (uint idx = tid; idx < 8 * VAL_DIM; idx += 32) acc[idx] = 0.0f;

        if (!q_full) {{
            for (uint idx = tid; idx < 8 * KEY_DIM; idx += 32) {{
                uint r = idx / KEY_DIM, c = idx % KEY_DIM;
                Qs[idx] = (r < q_row_valid) ? qkv_w[(qt * 8 + r) * TOTAL_QKV + h * PER_HEAD + c] : (half)0.0;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        for (uint kt = 0; kt < (N + 7) / 8; kt++) {{
            uint k_row_valid = min((uint)8, N - kt * 8);
            bool k_full = (k_row_valid == 8);

            if (!k_full) {{
                for (uint idx = tid; idx < 8 * KEY_DIM; idx += 32) {{
                    uint r = idx / KEY_DIM, c = idx % KEY_DIM;
                    Ks[idx] = (r < k_row_valid) ? qkv_w[(kt * 8 + r) * TOTAL_QKV + h * PER_HEAD + KEY_DIM + c] : (half)0.0;
                }}
                for (uint idx = tid; idx < 8 * VAL_DIM; idx += 32) {{
                    uint r = idx / VAL_DIM, c = idx % VAL_DIM;
                    Vs[idx] = (r < k_row_valid) ? qkv_w[(kt * 8 + r) * TOTAL_QKV + h * PER_HEAD + 2 * KEY_DIM + c] : (half)0.0;
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}

            // S_raw = Q_tile(8xKEY_DIM) @ K_tile(8xKEY_DIM)^T, half matmul
            simdgroup_half8x8 sacc = simdgroup_half8x8(0.0h);
            for (uint k = 0; k < KEY_DIM; k += 8) {{
                simdgroup_half8x8 q8, k8;
                if (q_full) simdgroup_load(q8, qkv_w + (qt * 8) * TOTAL_QKV + h * PER_HEAD + k, TOTAL_QKV);
                else        simdgroup_load(q8, Qs + k, KEY_DIM);
                if (k_full) simdgroup_load(k8, qkv_w + (kt * 8) * TOTAL_QKV + h * PER_HEAD + KEY_DIM + k, TOTAL_QKV, ulong2(0, 0), /*transpose=*/true);
                else        simdgroup_load(k8, Ks + k, KEY_DIM, ulong2(0, 0), /*transpose=*/true);
                simdgroup_multiply_accumulate(sacc, q8, k8, sacc);
            }}
            simdgroup_store(sacc, Sh, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // upcast + scale + bias + boundary mask -> fp32 S
            for (uint idx = tid; idx < 64; idx += 32) {{
                uint i = idx / 8, j = idx % 8;
                if (i < q_row_valid && j < k_row_valid) {{
                    S[idx] = float(Sh[idx]) * scale + float(attn_bias[h * N * N + (qt * 8 + i) * N + (kt * 8 + j)]);
                }} else {{
                    S[idx] = -INFINITY;
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // online softmax update in fp32, one thread per query row; also
            // writes half(p) into Sh for the half P@V matmul below
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
                    Sh[i * 8 + j] = half(p);
                    rowsum += p;
                }}
                l[i] = l[i] * c0 + rowsum;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < 8) m[tid] = newm[tid];

            // acc (fp32) = acc * corr + float(P(half,8x8) @ Vs(half,8xVAL_DIM))
            for (uint idx = tid; idx < 8 * VAL_DIM; idx += 32) {{
                uint i = idx / VAL_DIM;
                acc[idx] *= corr[i];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint n = 0; n < VAL_DIM; n += 8) {{
                simdgroup_half8x8 pacc = simdgroup_half8x8(0.0h);
                simdgroup_half8x8 p8, v8;
                simdgroup_load(p8, Sh, 8);
                if (k_full) simdgroup_load(v8, qkv_w + (kt * 8) * TOTAL_QKV + h * PER_HEAD + 2 * KEY_DIM + n, TOTAL_QKV);
                else        simdgroup_load(v8, Vs + n, VAL_DIM);
                simdgroup_multiply_accumulate(pacc, p8, v8, pacc);
                simdgroup_store(pacc, delta, 8);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint t = tid; t < 64; t += 32) {{
                    uint r = t / 8, c = t % 8;
                    acc[r * VAL_DIM + n + c] += float(delta[t]);
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
        }}

        // normalize and write out (fp32 output; only valid query rows)
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

        MTLSize gridSize = MTLSizeMake((N + 7) / 8, batch_windows, 1);
        MTLSize threadgroupSize = MTLSizeMake(32, 1, 1);
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        stream->synchronize(at::mps::SyncType::COMMIT);
    }});

    return out;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("forward", &{KERNEL_NAME}_fn, "Fused attention core, flash-attention-style, fp16 matmul (native MPS)");
}}
