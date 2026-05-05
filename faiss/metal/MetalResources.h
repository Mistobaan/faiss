#pragma once

#include <memory>

namespace faiss {
namespace metal {

namespace impl {
class MetalResourcesImpl;
}

class MetalResources {
   public:
    virtual ~MetalResources();

    virtual void initializeForDevice(int device) = 0;
    virtual int getDevice() const = 0;

    virtual std::shared_ptr<impl::MetalResourcesImpl> getImplShared() = 0;
    virtual std::shared_ptr<impl::MetalResourcesImpl> getImplShared() const = 0;
};

class MetalResourcesProvider {
   public:
    virtual ~MetalResourcesProvider();
    virtual std::shared_ptr<MetalResources> getResources() = 0;
};

} // namespace metal
} // namespace faiss
