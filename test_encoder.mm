#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>

void test_fn() {
    id<MTLComputeCommandEncoder> encoder = at::mps::getCurrentMPSStream()->commandEncoder();
}
