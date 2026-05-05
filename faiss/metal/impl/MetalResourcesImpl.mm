#include <faiss/metal/impl/MetalResourcesImpl.h>

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <dispatch/dispatch.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/metal/impl/MetalKernelsData.h>

#include <cstring>
#include <unordered_map>

namespace faiss {
namespace metal {
namespace impl {

namespace {

void* retain_handle(id object) {
    return (__bridge_retained void*)object;
}

void release_handle(void* handle) {
    if (handle) {
        CFBridgingRelease(handle);
    }
}

id<MTLDevice> as_device(void* handle) {
    return (__bridge id<MTLDevice>)handle;
}

id<MTLCommandQueue> as_queue(void* handle) {
    return (__bridge id<MTLCommandQueue>)handle;
}

id<MTLLibrary> as_library(void* handle) {
    return (__bridge id<MTLLibrary>)handle;
}

id<MTLBuffer> as_buffer(void* handle) {
    return (__bridge id<MTLBuffer>)handle;
}

id<MTLComputePipelineState> as_pipeline(void* handle) {
    return (__bridge id<MTLComputePipelineState>)handle;
}

} // namespace

struct MetalResourcesImpl::State {
    int device = 0;
    bool initialized = false;
    bool apple_silicon = false;
    void* metal_device = nullptr;
    void* command_queue = nullptr;
    void* library = nullptr;
    std::unordered_map<std::string, void*> pipelines;

    ~State() {
        for (auto& kv : pipelines) {
            release_handle(kv.second);
        }
        release_handle(library);
        release_handle(command_queue);
        release_handle(metal_device);
    }
};

MetalBuffer::MetalBuffer(
        std::shared_ptr<MetalResourcesImpl> owner,
        void* handle,
        size_t size)
        : owner_(std::move(owner)), handle_(handle), size_(size) {}

MetalBuffer::MetalBuffer(MetalBuffer&& other) noexcept
        : owner_(std::move(other.owner_)),
          handle_(other.handle_),
          size_(other.size_) {
    other.handle_ = nullptr;
    other.size_ = 0;
}

MetalBuffer& MetalBuffer::operator=(MetalBuffer&& other) noexcept {
    if (this != &other) {
        reset();
        owner_ = std::move(other.owner_);
        handle_ = other.handle_;
        size_ = other.size_;
        other.handle_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

MetalBuffer::~MetalBuffer() {
    reset();
}

bool MetalBuffer::valid() const {
    return handle_ != nullptr;
}

void MetalBuffer::reset() {
    if (handle_ && owner_) {
        owner_->releaseBuffer(handle_);
    }
    handle_ = nullptr;
    size_ = 0;
    owner_.reset();
}

size_t MetalBuffer::size() const {
    return size_;
}

void* MetalBuffer::handle() const {
    return handle_;
}

void* MetalBuffer::contents() {
    return handle_ ? owner_->bufferContents(handle_) : nullptr;
}

const void* MetalBuffer::contents() const {
    return handle_ ? owner_->bufferContents(handle_) : nullptr;
}

MetalResourcesImpl::MetalResourcesImpl() : state_(std::make_shared<State>()) {}

MetalResourcesImpl::~MetalResourcesImpl() = default;

void MetalResourcesImpl::initializeForDevice(int device) {
    FAISS_THROW_IF_NOT_FMT(
            device == 0,
            "Metal backend only supports device 0 in v1 (got %d)",
            device);

    if (state_->initialized) {
        FAISS_THROW_IF_NOT(state_->device == device);
        return;
    }

    @autoreleasepool {
        id<MTLDevice> metal_device = nil;
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (devices != nil && devices.count > 0) {
            metal_device = [devices objectAtIndex:device];
        }
        if (metal_device == nil) {
            metal_device = MTLCreateSystemDefaultDevice();
        }
        FAISS_THROW_IF_NOT_MSG(
                metal_device != nil, "No Metal device is available");

        NSError* error = nil;
        id<MTLLibrary> library = nil;
        if (kMetalLibraryIsSource) {
            NSString* source = [[NSString alloc]
                    initWithBytes:kMetalLibraryData
                           length:kMetalLibrarySize
                         encoding:NSUTF8StringEncoding];
            library = [metal_device newLibraryWithSource:source
                                                 options:nil
                                                   error:&error];
        } else {
            dispatch_data_t library_data = dispatch_data_create(
                    kMetalLibraryData,
                    kMetalLibrarySize,
                    dispatch_get_main_queue(),
                    ^{
                    });
            library = [metal_device newLibraryWithData:library_data error:&error];
        }
        if (library == nil) {
            const char* message = error.localizedDescription.UTF8String;
            FAISS_THROW_FMT("Failed to load embedded Metal library: %s", message);
        }

        id<MTLCommandQueue> queue = [metal_device newCommandQueue];
        FAISS_THROW_IF_NOT_MSG(
                queue != nil, "Failed to create Metal command queue");

        state_->device = device;
        state_->apple_silicon =
                [metal_device respondsToSelector:@selector(supportsFamily:)] &&
                [metal_device supportsFamily:MTLGPUFamilyApple1];
        state_->metal_device = retain_handle(metal_device);
        state_->command_queue = retain_handle(queue);
        state_->library = retain_handle(library);
        state_->initialized = true;
    }
}

int MetalResourcesImpl::getDevice() const {
    return state_->device;
}

bool MetalResourcesImpl::isAppleSilicon() const {
    return state_->apple_silicon;
}

MetalBuffer MetalResourcesImpl::makeBuffer(size_t size) const {
    if (size == 0) {
        return MetalBuffer();
    }

    @autoreleasepool {
        id<MTLDevice> device = as_device(state_->metal_device);
        id<MTLBuffer> buffer = [device newBufferWithLength:size
                                                   options:MTLResourceStorageModeShared];
        FAISS_THROW_IF_NOT_MSG(buffer != nil, "Failed to allocate Metal buffer");
        return MetalBuffer(
                const_cast<MetalResourcesImpl*>(this)->shared_from_this(),
                retain_handle(buffer),
                size);
    }
}

void MetalResourcesImpl::copyBuffer(
        const MetalBuffer& src,
        size_t src_offset,
        const MetalBuffer& dst,
        size_t dst_offset,
        size_t size) const {
    std::memcpy(
            static_cast<uint8_t*>(bufferContents(dst.handle())) + dst_offset,
            static_cast<const uint8_t*>(bufferContents(src.handle())) +
                    src_offset,
            size);
}

void MetalResourcesImpl::matrixMultiply(
        const MetalBuffer& left,
        const MetalBuffer& right,
        const MetalBuffer& out,
        size_t rows,
        size_t cols,
        size_t inner,
        bool transpose_left,
        bool transpose_right) const {
    if (rows == 0 || cols == 0 || inner == 0) {
        return;
    }

    @autoreleasepool {
        id<MTLCommandQueue> queue = as_queue(state_->command_queue);
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];

        const NSUInteger left_rows = transpose_left ? inner : rows;
        const NSUInteger left_cols = transpose_left ? rows : inner;
        const NSUInteger right_rows = transpose_right ? cols : inner;
        const NSUInteger right_cols = transpose_right ? inner : cols;

        auto* left_desc = [MPSMatrixDescriptor
                matrixDescriptorWithRows:left_rows
                                   columns:left_cols
                                  rowBytes:left_cols * sizeof(float)
                                  dataType:MPSDataTypeFloat32];
        auto* right_desc = [MPSMatrixDescriptor
                matrixDescriptorWithRows:right_rows
                                   columns:right_cols
                                  rowBytes:right_cols * sizeof(float)
                                  dataType:MPSDataTypeFloat32];
        auto* out_desc = [MPSMatrixDescriptor
                matrixDescriptorWithRows:rows
                                   columns:cols
                                  rowBytes:cols * sizeof(float)
                                  dataType:MPSDataTypeFloat32];

        MPSMatrix* left_matrix =
                [[MPSMatrix alloc] initWithBuffer:as_buffer(left.handle())
                                       descriptor:left_desc];
        MPSMatrix* right_matrix =
                [[MPSMatrix alloc] initWithBuffer:as_buffer(right.handle())
                                       descriptor:right_desc];
        MPSMatrix* out_matrix =
                [[MPSMatrix alloc] initWithBuffer:as_buffer(out.handle())
                                       descriptor:out_desc];

        MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
                 initWithDevice:as_device(state_->metal_device)
                 transposeLeft:transpose_left
                transposeRight:transpose_right
                    resultRows:rows
                 resultColumns:cols
               interiorColumns:inner
                        alpha:1.0
                         beta:0.0];

        [mm encodeToCommandBuffer:command_buffer
                        leftMatrix:left_matrix
                       rightMatrix:right_matrix
                      resultMatrix:out_matrix];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
    }
}

void MetalResourcesImpl::runKernel(
        const std::string& name,
        size_t count,
        const std::vector<KernelArg>& args) const {
    if (count == 0) {
        return;
    }

    @autoreleasepool {
        auto it = state_->pipelines.find(name);
        if (it == state_->pipelines.end()) {
            NSError* error = nil;
            NSString* function_name =
                    [NSString stringWithUTF8String:name.c_str()];
            id<MTLFunction> function =
                    [as_library(state_->library) newFunctionWithName:function_name];
            if (function == nil) {
                NSArray<NSString*>* names = [as_library(state_->library) functionNames];
                NSMutableString* joined = [NSMutableString string];
                for (NSUInteger i = 0; i < names.count; ++i) {
                    if (i > 0) {
                        [joined appendString:@", "];
                    }
                    [joined appendString:names[i]];
                }
                FAISS_THROW_FMT(
                        "Requested Metal kernel %s was not found. Available functions: %s",
                        name.c_str(),
                        joined.UTF8String);
            }
            id<MTLComputePipelineState> pipeline =
                    [as_device(state_->metal_device)
                            newComputePipelineStateWithFunction:function
                                                          error:&error];
            if (pipeline == nil) {
                const char* message = error.localizedDescription.UTF8String;
                FAISS_THROW_FMT(
                        "Failed to create Metal pipeline for %s: %s",
                        name.c_str(),
                        message);
            }
            it = state_->pipelines.emplace(name, retain_handle(pipeline)).first;
        }

        id<MTLCommandBuffer> command_buffer =
                [as_queue(state_->command_queue) commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
                [command_buffer computeCommandEncoder];
        id<MTLComputePipelineState> pipeline = as_pipeline(it->second);
        [encoder setComputePipelineState:pipeline];

        for (const auto& arg : args) {
            if (arg.type == KernelArg::Type::Buffer) {
                [encoder setBuffer:as_buffer(const_cast<void*>(arg.data))
                            offset:0
                           atIndex:arg.index];
            } else {
                [encoder setBytes:arg.data length:arg.size atIndex:arg.index];
            }
        }

        const NSUInteger thread_width =
                std::max<NSUInteger>(1, pipeline.threadExecutionWidth);
        const NSUInteger max_threads =
                std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup);
        const NSUInteger tg = std::min<NSUInteger>(
                static_cast<NSUInteger>(count),
                std::min(max_threads, thread_width));

        [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
    }
}

void MetalResourcesImpl::releaseBuffer(void* handle) const {
    release_handle(handle);
}

void* MetalResourcesImpl::bufferContents(void* handle) const {
    return [as_buffer(handle) contents];
}

} // namespace impl
} // namespace metal
} // namespace faiss
