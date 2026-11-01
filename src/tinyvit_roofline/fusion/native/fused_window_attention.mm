// Native PyTorch MPS extension: fused window attention (LayerNorm + QKV
// projection + multi-head attention w/ bias + output projection) as ONE
// Metal kernel dispatch, encoded directly into PyTorch's own MPS command
// stream -- no MLX involved, so none of MLX's Python->Metal dispatch
// overhead applies. Fixed for window_size=4, dim=64 (num_heads=2,
// key_dim=32, val_dim=32, out_dim=64) -- the ablation's (section 6)
// worst-case config, where every prior fusion attempt (torch.compile,
// MLX-fused attention) lost ~2x to baseline.
//
// Why native, not MLX: an earlier mx.fast.metal_kernel version of this
// exact kernel (same source, same math) lost to baseline at every batch
// size tested, including when the kernel did literally zero compute --
// MLX's Python->Metal dispatch path has a fixed ~300-400us floor per call,
// paid once per invocation regardless of content, which dominates at
// these operator sizes. Dispatching the identical kernel through
// PyTorch's own MPSStream (at::mps::getCurrentMPSStream()) instead avoids
// that floor entirely, since it reuses PyTorch's own low-overhead command
// encoding path rather than going through MLX's Python bindings.
//
// Result: beats baseline 2.6-5.25x at batch_windows=16 (model batch_size=1,
// the project's own representative config) -- but loses again above
// ~batch_windows=32-40, because this kernel's scalar/float4 compute loops
// are less efficient than Apple's vendor-tuned GEMM/attention kernels once
// dispatch overhead stops dominating. The win is real but narrow: it
// holds exactly where dispatch-overhead-dominated small ops are the
// problem (small batch, small window), and disappears once the op is
// big enough that raw compute throughput matters more than dispatch count.
//
// Build note: compiled WITHOUT -fobjc-arc. PyTorch's own MPS headers
// (ATen/native/mps/OperationUtils.h) use manual dispatch_release calls
// that are illegal under ARC, so this file (and anything that includes
// those headers) must use manual reference counting.

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/native/mps/OperationUtils.h>
#include <Metal/Metal.h>

static constexpr uint N = 16;
static constexpr uint DIM = 64;
static constexpr uint NUM_HEADS = 2;
static constexpr uint KEY_DIM = 32;
static constexpr uint VAL_DIM = 32;
static constexpr uint OUT_DIM = 64;
static constexpr uint PER_HEAD = 2 * KEY_DIM + VAL_DIM;
static constexpr uint TOTAL_QKV = NUM_HEADS * PER_HEAD;
static constexpr uint NUM_THREADS = 512;

static const char* KERNEL_SRC = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant uint N = 16;
constant uint DIM = 64;
constant uint NUM_HEADS = 2;
constant uint KEY_DIM = 32;
constant uint VAL_DIM = 32;
constant uint OUT_DIM = 64;
constant uint PER_HEAD = 2 * KEY_DIM + VAL_DIM;
constant uint TOTAL_QKV = NUM_HEADS * PER_HEAD;
constant uint NUM_THREADS = 512;

kernel void fused_window_attention(
    device const float* x [[buffer(0)]],
    device const float* norm_weight [[buffer(1)]],
    device const float* norm_bias [[buffer(2)]],
    device const float* qkv_weight [[buffer(3)]],
    device const float* qkv_bias [[buffer(4)]],
    device const float* attn_bias [[buffer(5)]],
    device const float* proj_weight [[buffer(6)]],
    device const float* proj_bias [[buffer(7)]],
    device float* out [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    uint window [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float x_norm[N * DIM];
    threadgroup float qkv_buf[N * TOTAL_QKV];
    threadgroup float scores[N * N];
    threadgroup float out_merged[N * OUT_DIM];

    device const float* x_w = x + window * N * DIM;
    device float* out_w = out + window * N * DIM;

    for (uint n = tid; n < N; n += NUM_THREADS) {
        const device float4* xv = (const device float4*)(x_w + n * DIM);
        float4 sum4 = float4(0.0f);
        for (uint c = 0; c < DIM / 4; c++) sum4 += xv[c];
        float mean = (sum4.x + sum4.y + sum4.z + sum4.w) / DIM;
        float4 var4 = float4(0.0f);
        for (uint c = 0; c < DIM / 4; c++) { float4 d = xv[c] - mean; var4 += d * d; }
        float var = (var4.x + var4.y + var4.z + var4.w) / DIM;
        float inv_std = rsqrt(var + 1e-5f);
        for (uint c = 0; c < DIM; c++) {
            float v = (x_w[n * DIM + c] - mean) * inv_std;
            x_norm[n * DIM + c] = v * norm_weight[c] + norm_bias[c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < N * TOTAL_QKV; idx += NUM_THREADS) {
        uint n = idx / TOTAL_QKV;
        uint j = idx % TOTAL_QKV;
        const threadgroup float4* xv = (const threadgroup float4*)(x_norm + n * DIM);
        const device float4* wv = (const device float4*)(qkv_weight + j * DIM);
        float4 acc4 = float4(0.0f);
        for (uint c = 0; c < DIM / 4; c++) acc4 += xv[c] * wv[c];
        qkv_buf[idx] = acc4.x + acc4.y + acc4.z + acc4.w + qkv_bias[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint h = 0; h < NUM_HEADS; h++) {
        for (uint idx = tid; idx < N * N; idx += NUM_THREADS) {
            uint i = idx / N, j = idx % N;
            const threadgroup float4* qv = (const threadgroup float4*)(qkv_buf + i * TOTAL_QKV + h * PER_HEAD);
            const threadgroup float4* kv = (const threadgroup float4*)(qkv_buf + j * TOTAL_QKV + h * PER_HEAD + KEY_DIM);
            float4 acc4 = float4(0.0f);
            for (uint c = 0; c < KEY_DIM / 4; c++) acc4 += qv[c] * kv[c];
            float acc = acc4.x + acc4.y + acc4.z + acc4.w;
            scores[idx] = acc * scale + attn_bias[h * N * N + i * N + j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = tid; i < N; i += NUM_THREADS) {
            float maxv = -INFINITY;
            for (uint j = 0; j < N; j++) maxv = max(maxv, scores[i * N + j]);
            float sum = 0.0f;
            for (uint j = 0; j < N; j++) { float e = exp(scores[i * N + j] - maxv); scores[i * N + j] = e; sum += e; }
            float inv = 1.0f / sum;
            for (uint j = 0; j < N; j++) scores[i * N + j] *= inv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint idx = tid; idx < N * VAL_DIM; idx += NUM_THREADS) {
            uint i = idx / VAL_DIM, c = idx % VAL_DIM;
            float acc = 0.0f;
            for (uint j = 0; j < N; j++) {
                uint vidx = j * TOTAL_QKV + h * PER_HEAD + 2 * KEY_DIM + c;
                acc += scores[i * N + j] * qkv_buf[vidx];
            }
            out_merged[i * OUT_DIM + h * VAL_DIM + c] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint idx = tid; idx < N * DIM; idx += NUM_THREADS) {
        uint n = idx / DIM, d = idx % DIM;
        const threadgroup float4* ov = (const threadgroup float4*)(out_merged + n * OUT_DIM);
        const device float4* wv = (const device float4*)(proj_weight + d * OUT_DIM);
        float4 acc4 = float4(0.0f);
        for (uint o = 0; o < OUT_DIM / 4; o++) acc4 += ov[o] * wv[o];
        out_w[idx] = acc4.x + acc4.y + acc4.z + acc4.w + proj_bias[d];
    }
}
)METAL";

static id<MTLComputePipelineState> g_pipeline = nil;

static id<MTLComputePipelineState> getPipeline() {
    if (g_pipeline != nil) return g_pipeline;
    id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
    NSError* error = nil;
    NSString* src = [NSString stringWithUTF8String:KERNEL_SRC];
    id<MTLLibrary> library = [device newLibraryWithSource:src options:nil error:&error];
    TORCH_CHECK(library != nil, "Failed to compile Metal kernel: ",
                error ? error.localizedDescription.UTF8String : "unknown error");
    id<MTLFunction> fn = [library newFunctionWithName:@"fused_window_attention"];
    TORCH_CHECK(fn != nil, "Failed to find kernel function");
    g_pipeline = [device newComputePipelineStateWithFunction:fn error:&error];
    TORCH_CHECK(g_pipeline != nil, "Failed to create pipeline state: ",
                error ? error.localizedDescription.UTF8String : "unknown error");
    return g_pipeline;
}

torch::Tensor fused_window_attention(
    torch::Tensor x, torch::Tensor norm_weight, torch::Tensor norm_bias,
    torch::Tensor qkv_weight, torch::Tensor qkv_bias, torch::Tensor attn_bias,
    torch::Tensor proj_weight, torch::Tensor proj_bias, double scale)
{
    TORCH_CHECK(x.is_mps(), "x must be an MPS tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    uint64_t batch_windows = x.size(0);

    torch::Tensor out = torch::empty({(int64_t)batch_windows, N, DIM}, x.options());
    float scale_f = (float)scale;

    id<MTLComputePipelineState> pipeline = getPipeline();
    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();

    dispatch_sync(stream->queue(), ^{
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(x) offset:x.storage_offset() * x.element_size() atIndex:0];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(norm_weight) offset:norm_weight.storage_offset() * norm_weight.element_size() atIndex:1];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(norm_bias) offset:norm_bias.storage_offset() * norm_bias.element_size() atIndex:2];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(qkv_weight) offset:qkv_weight.storage_offset() * qkv_weight.element_size() atIndex:3];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(qkv_bias) offset:qkv_bias.storage_offset() * qkv_bias.element_size() atIndex:4];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(attn_bias) offset:attn_bias.storage_offset() * attn_bias.element_size() atIndex:5];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(proj_weight) offset:proj_weight.storage_offset() * proj_weight.element_size() atIndex:6];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(proj_bias) offset:proj_bias.storage_offset() * proj_bias.element_size() atIndex:7];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(out) offset:out.storage_offset() * out.element_size() atIndex:8];
        [encoder setBytes:&scale_f length:sizeof(float) atIndex:9];

        MTLSize gridSize = MTLSizeMake(batch_windows, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(NUM_THREADS, 1, 1);
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        stream->synchronize(at::mps::SyncType::COMMIT);
    });

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_window_attention", &fused_window_attention, "Fused window attention (native MPS)");
}
