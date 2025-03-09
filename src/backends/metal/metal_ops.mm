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

// Use fully qualified names for metal logging functions
using ::neuronet::metal::metal_log_debug;
using ::neuronet::metal::metal_log_error;
using ::neuronet::metal::metal_log_warn;
using ::neuronet::metal::metal_log_info;

Tensor add(const Tensor& a, const Tensor& b) {
    // Check shapes match
    if (a.shape() != b.shape()) {
        metal_log_error("Tensor shapes must match for addition");
        return Tensor();
    }
    
    // Create output tensor
    Tensor result(a.shape(), a.dtype(), DeviceType::Metal);
    
    @autoreleasepool {
        // Get Metal objects directly without __bridge cast
        id<MTLDevice> device = ::neuronet::metal::get_metal_device();
        id<MTLCommandQueue> commandQueue = ::neuronet::metal::get_command_queue();
        id<MTLLibrary> library = ::neuronet::metal::get_metal_library();
        
        // Get function
        id<MTLFunction> addFunction = [library newFunctionWithName:@"add_kernel"];
        NSError* error = nil;
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:addFunction error:&error];
        
        if (!pipeline) {
            std::string errorMsg = error ? [[error localizedDescription] UTF8String] : "unknown error";
            metal_log_error("Failed to create compute pipeline: " + errorMsg);
            return Tensor();
        }
        
        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pipeline];
        
        // Get Metal buffers from tensors - fixed duplicate declaration
        id<MTLBuffer> a_buffer = (__bridge id<MTLBuffer>)a.data<void>();
        id<MTLBuffer> b_buffer = (__bridge id<MTLBuffer>)b.data<void>();
        id<MTLBuffer> result_buffer = (__bridge id<MTLBuffer>)result.data<void>();
        
        [encoder setBuffer:a_buffer offset:0 atIndex:0];
        [encoder setBuffer:b_buffer offset:0 atIndex:1];
        [encoder setBuffer:result_buffer offset:0 atIndex:2];
        
        // Dispatch threads - fixed the corrupted declarations
        NSUInteger size = a.size();
        NSUInteger threadsPerThreadgroup = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
        MTLSize threadgroupSize = MTLSizeMake(threadsPerThreadgroup, 1, 1);
        MTLSize gridSize = MTLSizeMake(size, 1, 1);
        
        metal_log_debug("Dispatching add kernel with size " + std::to_string(size));
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        metal_log_debug("Add operation completed");
    }
    
    return result;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    // Check dimensions
    const auto& a_shape = a.shape();
    const auto& b_shape = b.shape();
    
    if (a_shape.size() != 2 || b_shape.size() != 2 || a_shape[1] != b_shape[0]) {
        metal_log_error("Invalid shapes for matrix multiplication: [" + 
                      std::to_string(a_shape[0]) + "×" + std::to_string(a_shape[1]) + "] × [" +
                      std::to_string(b_shape[0]) + "×" + std::to_string(b_shape[1]) + "]");
        return Tensor();
    }
    
    // Create output tensor
    std::vector<int64_t> result_shape = {a_shape[0], b_shape[1]};
    Tensor result(result_shape, a.dtype(), DeviceType::Metal);
    
    @autoreleasepool {
        // Get Metal objects directly without __bridge cast
        id<MTLDevice> device = ::neuronet::metal::get_metal_device();
        id<MTLCommandQueue> commandQueue = ::neuronet::metal::get_command_queue();
        
        metal_log_debug("Performing matrix multiplication with MPS");
        
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
        
        metal_log_debug("Matrix multiplication completed: [" + 
                      std::to_string(a_shape[0]) + "×" + std::to_string(a_shape[1]) + "] × [" +
                      std::to_string(b_shape[0]) + "×" + std::to_string(b_shape[1]) + "] → [" +
                      std::to_string(result_shape[0]) + "×" + std::to_string(result_shape[1]) + "]");
    }
    
    return result;
}

Tensor relu(const Tensor& input) {
    // Create output tensor
    Tensor result(input.shape(), input.dtype(), DeviceType::Metal);
    
    @autoreleasepool {
        // Get Metal objects directly without __bridge cast
        id<MTLDevice> device = ::neuronet::metal::get_metal_device();
        id<MTLCommandQueue> commandQueue = ::neuronet::metal::get_command_queue();
        id<MTLLibrary> library = ::neuronet::metal::get_metal_library();
        
        // Get function
        id<MTLFunction> reluFunction = [library newFunctionWithName:@"relu_kernel"];
        NSError* error = nil;
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:reluFunction error:&error];
        
        if (!pipeline) {
            std::string errorMsg = error ? [[error localizedDescription] UTF8String] : "unknown error";
            metal_log_error("Failed to create compute pipeline: " + errorMsg);
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
        
        metal_log_debug("Dispatching ReLU kernel with size " + std::to_string(size));
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        metal_log_debug("ReLU operation completed");
    }
    
    return result;
}

// Add implementation of multiply
Tensor multiply(const Tensor& a, const Tensor& b) {
    // Check shapes match
    if (a.shape() != b.shape()) {
        metal_log_error("Tensor shapes must match for element-wise multiplication");
        return Tensor();
    }
    
    // Create output tensor
    Tensor result(a.shape(), a.dtype(), DeviceType::Metal);
    
    @autoreleasepool {
        id<MTLDevice> device = ::neuronet::metal::get_metal_device();
        id<MTLCommandQueue> commandQueue = ::neuronet::metal::get_command_queue();
        id<MTLLibrary> library = ::neuronet::metal::get_metal_library();
        
        // Get function
        id<MTLFunction> mulFunction = [library newFunctionWithName:@"mul_kernel"];
        NSError* error = nil;
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:mulFunction error:&error];
        
        if (!pipeline) {
            std::string errorMsg = error ? [[error localizedDescription] UTF8String] : "unknown error";
            metal_log_error("Failed to create multiply compute pipeline: " + errorMsg);
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
        
        metal_log_debug("Dispatching multiply kernel with size " + std::to_string(size));
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        metal_log_debug("Multiply operation completed");
    }
    
    return result;
}

// Add a implementation for mul_scalar
Tensor mul_scalar(const Tensor& tensor, float scalar) {
    // Create output tensor
    Tensor result(tensor.shape(), tensor.dtype(), DeviceType::Metal);
    
    @autoreleasepool {
        // Get Metal objects directly without __bridge cast
        id<MTLDevice> device = ::neuronet::metal::get_metal_device();
        id<MTLCommandQueue> commandQueue = ::neuronet::metal::get_command_queue();
        
        // For now, just use CPU implementation and transfer data
        // This will be replaced with a proper Metal implementation in the future
        metal_log_debug("Scalar multiplication - using CPU fallback");
        
        // Get data from device
        const float* tensor_data = tensor.data<float>();
        float* result_data = result.data<float>();
        
        // Get buffers for direct access
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)(void*)tensor_data;
        id<MTLBuffer> result_buffer = (__bridge id<MTLBuffer>)(void*)result_data;
        
        // Map the buffer contents
        float* input_ptr = (float*)[input_buffer contents];
        float* output_ptr = (float*)[result_buffer contents];
        
        // Perform scalar multiplication
        for (int64_t i = 0; i < tensor.size(); i++) {
            output_ptr[i] = input_ptr[i] * scalar;
        }
        
        metal_log_debug("Scalar multiplication completed");
    }
    
    return result;
}

} // namespace metal
} // namespace ops
} // namespace neuronet

#endif // defined(__APPLE__) && defined(NEURONET_USE_METAL)