/******************************************************************************
 * Objective-C++ bridge for the Metal selective scan kernel.
 *
 * Loads the .metal shader, creates a compute pipeline, extracts MTLBuffers
 * from PyTorch MPS tensors, encodes the dispatch, and returns.
 *
 * SYNCHRONIZATION: This version integrates natively with PyTorch's MPSStream.
 * By using PyTorch's own computeCommandEncoder, we achieve zero-overhead
 * asynchronous dispatch. We do NOT need to synchronize or wait.
 *
 * Build: compiled as part of a PyTorch CppExtension via setup.py
 ******************************************************************************/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
    return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

// ---------------------------------------------------------------------------
// Cached pipeline state
// ---------------------------------------------------------------------------

struct MetalState {
    id<MTLDevice> device = nil;
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> pipeline = nil;
    bool initialized = false;
};

static MetalState& getState() {
    static MetalState state;
    return state;
}

static void ensureInitialized(const std::string& shader_path) {
    MetalState& state = getState();
    if (state.initialized) return;

    @autoreleasepool {
        state.device = MTLCreateSystemDefaultDevice();
        TORCH_CHECK(state.device != nil, "Metal is not available on this device");

        NSString* path = [NSString stringWithUTF8String:shader_path.c_str()];
        NSError* error = nil;
        NSString* source = [NSString stringWithContentsOfFile:path
                                                    encoding:NSUTF8StringEncoding
                                                       error:&error];
        TORCH_CHECK(error == nil, "Failed to load Metal shader: ",
                    [[error localizedDescription] UTF8String]);

        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.mathMode = MTLMathModeFast;
        state.library = [state.device newLibraryWithSource:source
                                                  options:options
                                                    error:&error];
        TORCH_CHECK(error == nil, "Failed to compile Metal shader: ",
                    [[error localizedDescription] UTF8String]);

        id<MTLFunction> function = [state.library newFunctionWithName:@"selective_scan_fwd"];
        TORCH_CHECK(function != nil, "Metal function 'selective_scan_fwd' not found");

        state.pipeline = [state.device newComputePipelineStateWithFunction:function
                                                                    error:&error];
        TORCH_CHECK(error == nil, "Failed to create compute pipeline: ",
                    [[error localizedDescription] UTF8String]);

        state.initialized = true;
    }
}

// ---------------------------------------------------------------------------
// Main dispatch function
// ---------------------------------------------------------------------------

std::vector<torch::Tensor> selective_scan_metal_fwd(
    torch::Tensor deltaA,       // [B, D, L, N] float32 on MPS
    torch::Tensor deltaB_u,     // [B, D, L, N]
    torch::Tensor C,            // [B, N, L]
    torch::Tensor u,            // [B, D, L]
    torch::Tensor D_param,      // [D] or empty
    torch::Tensor z,            // [B, D, L] or empty
    bool return_last_state,
    const std::string& shader_path
) {
    @autoreleasepool {
        ensureInitialized(shader_path);
        MetalState& state = getState();

        TORCH_CHECK(deltaA.is_mps(), "deltaA must be on MPS device");
        TORCH_CHECK(deltaA.is_contiguous(), "deltaA must be contiguous");
        TORCH_CHECK(deltaB_u.is_contiguous(), "deltaB_u must be contiguous");
        TORCH_CHECK(C.is_contiguous(), "C must be contiguous");
        TORCH_CHECK(u.is_contiguous(), "u must be contiguous");

        uint32_t B = deltaA.size(0);
        uint32_t L = deltaA.size(1);
        uint32_t N = deltaA.size(2);
        uint32_t D = deltaA.size(3);
        bool has_D = D_param.numel() > 0;
        bool has_z = z.numel() > 0;

        // Allocate outputs on MPS
        auto out = torch::empty({B, L, D}, deltaA.options().dtype(torch::kFloat32));
        auto last_state = return_last_state
            ? torch::empty({B, D, N}, deltaA.options().dtype(torch::kFloat32))
            : torch::empty({0}, deltaA.options().dtype(torch::kFloat32));
        auto dummy = torch::zeros({1}, deltaA.options().dtype(torch::kFloat32));

        uint32_t has_D_val = has_D ? 1 : 0;
        uint32_t has_z_val = has_z ? 1 : 0;
        uint32_t save_last = return_last_state ? 1 : 0;

        // Get the active PyTorch MPS compute encoder
        // (This guarantees strict ordering with previous/subsequent PyTorch ops)
        id<MTLComputeCommandEncoder> encoder = at::mps::getCurrentMPSStream()->commandEncoder();
        TORCH_CHECK(encoder != nil, "Failed to get PyTorch active compute encoder");

        [encoder setComputePipelineState:state.pipeline];

        // Bind input buffers
        [encoder setBuffer:getMTLBufferStorage(deltaA)
                    offset:deltaA.storage_offset() * sizeof(float) atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(deltaB_u)
                    offset:deltaB_u.storage_offset() * sizeof(float) atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(C)
                    offset:C.storage_offset() * sizeof(float) atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(u)
                    offset:u.storage_offset() * sizeof(float) atIndex:3];

        // Optional inputs
        [encoder setBuffer:has_D ? getMTLBufferStorage(D_param) : getMTLBufferStorage(dummy)
                    offset:has_D ? D_param.storage_offset() * sizeof(float) : 0
                   atIndex:4];
        [encoder setBuffer:has_z ? getMTLBufferStorage(z) : getMTLBufferStorage(dummy)
                    offset:has_z ? z.storage_offset() * sizeof(float) : 0
                   atIndex:5];

        // Output buffers
        [encoder setBuffer:getMTLBufferStorage(out)
                    offset:out.storage_offset() * sizeof(float) atIndex:6];
        [encoder setBuffer:return_last_state ? getMTLBufferStorage(last_state) : getMTLBufferStorage(dummy)
                    offset:return_last_state ? last_state.storage_offset() * sizeof(float) : 0
                   atIndex:7];

        // Scalar constants via setBytes
        [encoder setBytes:&B length:sizeof(uint32_t) atIndex:8];
        [encoder setBytes:&D length:sizeof(uint32_t) atIndex:9];
        [encoder setBytes:&L length:sizeof(uint32_t) atIndex:10];
        [encoder setBytes:&N length:sizeof(uint32_t) atIndex:11];
        [encoder setBytes:&has_D_val length:sizeof(uint32_t) atIndex:12];
        [encoder setBytes:&has_z_val length:sizeof(uint32_t) atIndex:13];
        [encoder setBytes:&save_last length:sizeof(uint32_t) atIndex:14];

        // Dispatch: one thread per (batch, dim) pair
        uint32_t total_threads = B * D;
        NSUInteger threadGroupSize = MIN(state.pipeline.maxTotalThreadsPerThreadgroup,
                                         (NSUInteger)total_threads);
        if (threadGroupSize > 256) threadGroupSize = 256;

        MTLSize gridSize = MTLSizeMake(total_threads, 1, 1);
        MTLSize groupSize = MTLSizeMake(threadGroupSize, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];

        // Note: we DO NOT call [encoder endEncoding] or [commandBuffer commit].
        // PyTorch manages this encoder and will end/commit it automatically
        // when a dependent synchronization or non-compute op requires it.

        return {out, last_state};
    }
}

// ---------------------------------------------------------------------------
// Pybind11 module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("selective_scan_metal_fwd", &selective_scan_metal_fwd,
          "Fused selective scan forward pass on Metal GPU",
          py::arg("deltaA"),
          py::arg("deltaB_u"),
          py::arg("C"),
          py::arg("u"),
          py::arg("D_param"),
          py::arg("z"),
          py::arg("return_last_state"),
          py::arg("shader_path"));
}
