#pragma once

#include <faiss/Index.h>
#include <faiss/metal/MetalResources.h>
#include <faiss/metal/impl/MetalResourcesImpl.h>

#include <memory>

namespace faiss {
namespace metal {
namespace impl {

class MetalFlatIndex {
   public:
    MetalFlatIndex(std::shared_ptr<MetalResources> resources, int dims);

    size_t getSize() const;
    size_t getCapacity() const;
    int getDims() const;

    void reset();
    void reserve(size_t n);
    void resize(size_t n);
    void add(const float* x, idx_t n);
    void setFromHost(const float* x, idx_t n);

    void reconstruct(idx_t key, idx_t n, float* out) const;
    void reconstruct_batch(idx_t n, const idx_t* keys, float* out) const;

    void query(
            idx_t n,
            const float* x,
            int k,
            MetricType metric,
            float metricArg,
            float* distances,
            idx_t* labels) const;

    float* hostData();
    const float* hostData() const;

    MetalBuffer& buffer();
    const MetalBuffer& buffer() const;

   private:
    void reserveImpl_(size_t n);

    std::shared_ptr<MetalResources> resources_;
    std::shared_ptr<MetalResourcesImpl> impl_;
    int dims_ = 0;
    size_t size_ = 0;
    size_t capacity_ = 0;
    MetalBuffer data_;
};

} // namespace impl
} // namespace metal
} // namespace faiss
