#include <neuronet/backends/metal/metal_ops.h>
#include <neuronet/backends/metal/metal_backend.h>
#include <neuronet/core/tensor.h>
#include <neuronet/utils/logging.h>

#if defined(__APPLE__) && defined(NEURONET_USE_METAL)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

namespace neuronet {
namespace ops {
namespace metal {

Tensor add(const Tensor& a, const Tensor& b) {
    // Check shapes match
    if (a.shape() != b.shape()) {
        log_error("Tensor shapes must match for addition");
        return Tensor();
    }
    
    // Create output tensor
    Tensor result(a.shape(), a.dtype(), DeviceType::Metal);
    
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)neuronet::metal::get_metal_device();
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)neuronet::metal::get_command_queue();
        id<MTLLibrary> library = (__bridge id<MTLLibrary>)neuronet::metal::get_metal_library();
        
        // Get function
        id<MTLFunction> addFunction = [library newFunctionWithName:@"add_kernel"];
        NSError* error = nil;
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:addFunction error:&error];
        
        if (!pipeline) {
            log_error("Failed to create compute pipeline: {}", error ? [[error localizedDescription] UTF8String] : "unknown error");
            return Tensor();
        }
        
        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pipeline];
        
        // Get Metal buffers from tensors
        id<MTLBuffer> a_buffer = (__bridge id<MTLBuffer>)a.data<void>();
        id<MTLBuffer> b_buffer = (__bridge id<MTLBuffer>)b.data<void>();
        id<MTLBuffer> result_buffer = (__bridge id<MTLBuffer>)result.data<void>();
        
        [encoder setBuffer:a_buffer offset:0 atIndex:0];
        [encoder setBuffer:b_buffer offset:0 atIndex:1];
        [encoder setBuffer:result_buffer offset:0 atIndex:2];
        
        // Dispatch threads
        NSUInteger size = a.size();
        NSUInteger threadsPerThreadgroup = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
        MTLSize threadgroupSize = MTLSizeMake(threadsPerThreadgroup, 1, 1);
        MTLSize gridSize = MTLSizeMake(size, 1, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
    
    return result;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    // Check dimensions
    const auto& a_shape = a.shape();
    const auto& b_shape = b.shape();
    
    if (a_shape.size() != 2 || b_shape.size() != 2 || a_shape[1] != b_shape[0]) {
        log_error("Invalid shapes for matrix multiplication");
        return Tensor();
    }
    
    // Create output tensor
    std::vector<int64_t> result_shape = {a_shape[0], b_shape[1]};
    Tensor result(result_shape, a.dtype(), DeviceType::Metal);
    
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)neuronet::metal::get_metal_device();
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)neuronet::metal::get_command_queue();
        
        // Use MPS for matrix multiplication
        MPSMatrixDescriptor* aDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:a_shape[0]
                                                                          columns:a_shape[1]
                                                                         rowBytes:a_shape[1] * sizeof(float)
                                                                         dataType:MPSDataTypeFloat32];
        
        MPSMatrixDescriptor* bDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:b_shape[0]
                                                                          columns:b_shape[1]
                                                                         rowBytes:b_shape[1] * sizeof(float)
                                                                         dataType:MPSDataTypeFloat32];
        
        MPSMatrixDescriptor* cDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:result_shape[0]
                                                                          columns:result_shape[1]
                                                                         rowBytes:result_shape[1] * sizeof(float)
                                                                         dataType:MPSDataTypeFloat32];
        
        // Get Metal buffers
        id<MTLBuffer> a_buffer = (__bridge id<MTLBuffer>)a.data<void>();
        id<MTLBuffer> b_buffer = (__bridge id<MTLBuffer>)b.data<void>();
        id<MTLBuffer> result_buffer = (__bridge id<MTLBuffer>)result.data<void>();
        
        MPSMatrix* matrixA = [[MPSMatrix alloc] initWithBuffer:a_buffer descriptor:aDesc];
        MPSMatrix* matrixB = [[MPSMatrix alloc] initWithBuffer:b_buffer descriptor:bDesc];
        MPSMatrix* matrixC = [[MPSMatrix alloc] initWithBuffer:result_buffer descriptor:cDesc];
        
        // Create MPS matrix multiplication kernel
        MPSMatrixMultiplication* matMul = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                        transposeLeft:NO
                                                                       transposeRight:NO
                                                                           resultRows:result_shape[0]
                                                                        resultColumns:result_shape[1]
                                                                      interiorColumns:a_shape[1]
                                                                                alpha:1.0
                                                                                 beta:0.0];
        
        // Create command buffer and encode operation
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        [matMul encodeToCommandBuffer:commandBuffer leftMatrix:matrixA rightMatrix:matrixB resultMatrix:matrixC];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
    
    return result;
}

Tensor relu(const Tensor& input) {
    // Create output tensor
    Tensor result(input.shape(), input.dtype(), DeviceType::Metal);
    
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)neuronet::metal::get_metal_device();
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)neuronet::metal::get_command_queue();
        id<MTLLibrary> library = (__bridge id<MTLLibrary>)neuronet::metal::get_metal_library();
        
        // Get function
        id<MTLFunction> reluFunction = [library newFunctionWithName:@"relu_kernel"];
        NSError* error = nil;
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:reluFunction error:&error];
        
        if (!pipeline) {
            log_error("Failed to create compute pipeline: {}", error ? [[error localizedDescription] UTF8String] : "unknown error");
            return Tensor();
        }
        
        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pipeline];
        
        // Get Metal buffers from tensors
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)input.data<void>();
        id<MTLBuffer> result_buffer = (__bridge id<MTLBuffer>)result.data<void>();
        
        [encoder setBuffer:input_buffer offset:0 atIndex:0];
        [encoder setBuffer:result_buffer offset:0 atIndex:1];
        
        // Dispatch threads
        NSUInteger size = input.size();
        NSUInteger threadsPerThreadgroup = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
        MTLSize threadgroupSize = MTLSizeMake(threadsPerThreadgroup, 1, 1);
        MTLSize gridSize = MTLSizeMake(size, 1, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
    
    return result;
}

} // namespace metal
} // namespace ops
} // namespace neuronet

#endif // defined(__APPLE__) && defined(NEURONET_USE_METAL)
