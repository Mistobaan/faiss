#pragma once

#include <faiss/metal/MetalResources.h>
#include <memory>

namespace faiss {
namespace metal {

class StandardMetalResourcesImpl : public MetalResources {
   public:
    StandardMetalResourcesImpl();
    ~StandardMetalResourcesImpl() override;

    void initializeForDevice(int device) override;
    int getDevice() const override;

    std::shared_ptr<impl::MetalResourcesImpl> getImplShared() override;
    std::shared_ptr<impl::MetalResourcesImpl> getImplShared() const override;

    bool isAppleSilicon() const;

   private:
    std::shared_ptr<impl::MetalResourcesImpl> impl_;
};

class StandardMetalResources : public MetalResourcesProvider {
   public:
    StandardMetalResources();
    ~StandardMetalResources() override;

    std::shared_ptr<MetalResources> getResources() override;

    bool isAppleSilicon() const;
    int getDevice() const;

   private:
    std::shared_ptr<StandardMetalResourcesImpl> resources_;
};

} // namespace metal
} // namespace faiss
