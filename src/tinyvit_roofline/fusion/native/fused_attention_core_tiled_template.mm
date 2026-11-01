// Row-tiled version of fused_attention_core_template.mm: tiles the QUERY
// dimension (not keys/values) so the on-chip scores buffer
// (TILE_ROWS * N * 4 bytes) fits the M4's 32768-byte threadgroup memory
// limit even when N itself is too big for a dense N*N matrix (e.g.
// window_size=14, N=196, where N*N*4=153664 bytes -- 4.7x over the
// limit). Softmax is row-independent (each row's max/sum only depends on
// that row's own N scores), so tiling over rows needs no cross-tile state
// (no "online softmax" -- that's only needed when tiling over the KEY
// dimension, not attempted here since K/V are read directly from device
// memory per tile, not staged on-chip).
//
// Same division of labor as the non-tiled version: PyTorch's own
// LayerNorm and Linear projections stay as native ops; only the
// attention core (QK^T*scale + bias, softmax, AV) is fused into one
// native dispatch.
//
// Build note: compiled WITHOUT -fobjc-arc (see fused_window_attention.mm).

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/native/mps/OperationUtils.h>
#include <Metal/Metal.h>

static constexpr uint N = {N};
static constexpr uint TILE_ROWS = {TILE_ROWS};
static constexpr uint NUM_HEADS = {NUM_HEADS};
static constexpr uint KEY_DIM = {KEY_DIM};
static constexpr uint VAL_DIM = {VAL_DIM};
static constexpr uint OUT_DIM = {OUT_DIM};
static constexpr uint PER_HEAD = 2 * KEY_DIM + VAL_DIM;
static constexpr uint TOTAL_QKV = NUM_HEADS * PER_HEAD;
static constexpr uint NUM_THREADS = {NUM_THREADS};

static const char* KERNEL_SRC = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant uint N = {N};
constant uint TILE_ROWS = {TILE_ROWS};
constant uint NUM_HEADS = {NUM_HEADS};
constant uint KEY_DIM = {KEY_DIM};
constant uint VAL_DIM = {VAL_DIM};
constant uint OUT_DIM = {OUT_DIM};
constant uint PER_HEAD = 2 * KEY_DIM + VAL_DIM;
constant uint TOTAL_QKV = NUM_HEADS * PER_HEAD;
constant uint NUM_THREADS = {NUM_THREADS};

kernel void {KERNEL_NAME}(
    device const float* qkv [[buffer(0)]],
    device const float* attn_bias [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    uint window [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{{
    threadgroup float scores[TILE_ROWS * N];

    device const float* qkv_w = qkv + window * N * TOTAL_QKV;
    device float* out_w = out + window * N * OUT_DIM;

    for (uint h = 0; h < NUM_HEADS; h++) {{
        for (uint i0 = 0; i0 < N; i0 += TILE_ROWS) {{
            uint rows = min(TILE_ROWS, N - i0);

            for (uint idx = tid; idx < rows * N; idx += NUM_THREADS) {{
                uint ri = idx / N, j = idx % N;
                uint i = i0 + ri;
                device const float* qv = qkv_w + i * TOTAL_QKV + h * PER_HEAD;
                device const float* kv = qkv_w + j * TOTAL_QKV + h * PER_HEAD + KEY_DIM;
                float acc = 0.0f;
                for (uint c = 0; c < KEY_DIM; c++) acc += qv[c] * kv[c];
                scores[ri * N + j] = acc * scale + attn_bias[h * N * N + i * N + j];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint ri = tid; ri < rows; ri += NUM_THREADS) {{
                float maxv = -INFINITY;
                for (uint j = 0; j < N; j++) maxv = max(maxv, scores[ri * N + j]);
                float sum = 0.0f;
                for (uint j = 0; j < N; j++) {{ float e = exp(scores[ri * N + j] - maxv); scores[ri * N + j] = e; sum += e; }}
                float inv = 1.0f / sum;
                for (uint j = 0; j < N; j++) scores[ri * N + j] *= inv;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint idx = tid; idx < rows * VAL_DIM; idx += NUM_THREADS) {{
                uint ri = idx / VAL_DIM, c = idx % VAL_DIM;
                uint i = i0 + ri;
                float acc = 0.0f;
                for (uint j = 0; j < N; j++) {{
                    device const float* vv = qkv_w + j * TOTAL_QKV + h * PER_HEAD + 2 * KEY_DIM;
                    acc += scores[ri * N + j] * vv[c];
                }}
                out_w[i * OUT_DIM + h * VAL_DIM + c] = acc;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
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

        MTLSize gridSize = MTLSizeMake(batch_windows, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(NUM_THREADS, 1, 1);
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        stream->synchronize(at::mps::SyncType::COMMIT);
    }});

    return out;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("forward", &{KERNEL_NAME}_fn, "Fused attention core, row-tiled (native MPS)");
}}
