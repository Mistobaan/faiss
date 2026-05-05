#include <faiss/metal/StandardMetalResources.h>

#include <Metal/Metal.h>

#include <faiss/metal/impl/MetalResourcesImpl.h>

extern "C" bool faiss_metal_runtime_available() {
    @autoreleasepool {
        return MTLCreateSystemDefaultDevice() != nil;
    }
}

namespace faiss {
namespace metal {

StandardMetalResourcesImpl::StandardMetalResourcesImpl()
        : impl_(std::make_shared<impl::MetalResourcesImpl>()) {
    // Make resource capabilities observable immediately for Python/runtime
    // probing instead of only after the first index constructor touches them.
    impl_->initializeForDevice(0);
}

StandardMetalResourcesImpl::~StandardMetalResourcesImpl() = default;

void StandardMetalResourcesImpl::initializeForDevice(int device) {
    impl_->initializeForDevice(device);
}

int StandardMetalResourcesImpl::getDevice() const {
    return impl_->getDevice();
}

std::shared_ptr<impl::MetalResourcesImpl>
StandardMetalResourcesImpl::getImplShared() {
    return impl_;
}

std::shared_ptr<impl::MetalResourcesImpl>
StandardMetalResourcesImpl::getImplShared() const {
    return impl_;
}

bool StandardMetalResourcesImpl::isAppleSilicon() const {
    return impl_->isAppleSilicon();
}

StandardMetalResources::StandardMetalResources()
        : resources_(std::make_shared<StandardMetalResourcesImpl>()) {}

StandardMetalResources::~StandardMetalResources() = default;

std::shared_ptr<MetalResources> StandardMetalResources::getResources() {
    return resources_;
}

bool StandardMetalResources::isAppleSilicon() const {
    return resources_->isAppleSilicon();
}

int StandardMetalResources::getDevice() const {
    return resources_->getDevice();
}

} // namespace metal
} // namespace faiss
