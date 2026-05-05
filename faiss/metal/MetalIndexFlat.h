#pragma once

#include <faiss/IndexFlat.h>
#include <faiss/metal/MetalIndex.h>

#include <memory>

namespace faiss {
namespace metal {

namespace impl {
class MetalFlatIndex;
}

struct MetalIndexFlatConfig : public MetalIndexConfig {
    bool useFloat16 = false;
    bool storeTransposed = false;
};

class MetalIndexFlat : public MetalIndex {
   public:
    MetalIndexFlat(
            MetalResourcesProvider* provider,
            const faiss::IndexFlat* index,
            MetalIndexFlatConfig config = MetalIndexFlatConfig());

    MetalIndexFlat(
            std::shared_ptr<MetalResources> resources,
            const faiss::IndexFlat* index,
            MetalIndexFlatConfig config = MetalIndexFlatConfig());

    MetalIndexFlat(
            MetalResourcesProvider* provider,
            int dims,
            faiss::MetricType metric,
            MetalIndexFlatConfig config = MetalIndexFlatConfig());

    MetalIndexFlat(
            std::shared_ptr<MetalResources> resources,
            int dims,
            faiss::MetricType metric,
            MetalIndexFlatConfig config = MetalIndexFlatConfig());

    ~MetalIndexFlat() override;

    void copyFrom(const faiss::IndexFlat* index);
    void copyTo(faiss::IndexFlat* index) const;

    size_t getNumVecs() const;

    void reset() override;
    void train(idx_t n, const float* x) override;
    void add(idx_t n, const float* x) override;

    void reconstruct(idx_t key, float* out) const override;
    void reconstruct_n(idx_t i0, idx_t num, float* out) const override;
    void reconstruct_batch(idx_t n, const idx_t* keys, float* out)
            const override;

   protected:
    void resetIndex_(int dims);

    bool addImplRequiresIDs_() const override;
    void addImpl_(idx_t n, const float* x, const idx_t* ids) override;
    void searchImpl_(
            idx_t n,
            const float* x,
            int k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override;

    const MetalIndexFlatConfig flatConfig_;
    std::unique_ptr<impl::MetalFlatIndex> data_;
};

class MetalIndexFlatL2 : public MetalIndexFlat {
   public:
    MetalIndexFlatL2(
            MetalResourcesProvider* provider,
            faiss::IndexFlatL2* index,
            MetalIndexFlatConfig config = MetalIndexFlatConfig());

    MetalIndexFlatL2(
            std::shared_ptr<MetalResources> resources,
            faiss::IndexFlatL2* index,
            MetalIndexFlatConfig config = MetalIndexFlatConfig());

    MetalIndexFlatL2(
            MetalResourcesProvider* provider,
            int dims,
            MetalIndexFlatConfig config = MetalIndexFlatConfig());

    MetalIndexFlatL2(
            std::shared_ptr<MetalResources> resources,
            int dims,
            MetalIndexFlatConfig config = MetalIndexFlatConfig());

    void copyFrom(faiss::IndexFlat* index);
    void copyTo(faiss::IndexFlat* index) const;
};

class MetalIndexFlatIP : public MetalIndexFlat {
   public:
    MetalIndexFlatIP(
            MetalResourcesProvider* provider,
            faiss::IndexFlatIP* index,
            MetalIndexFlatConfig config = MetalIndexFlatConfig());

    MetalIndexFlatIP(
            std::shared_ptr<MetalResources> resources,
            faiss::IndexFlatIP* index,
            MetalIndexFlatConfig config = MetalIndexFlatConfig());

    MetalIndexFlatIP(
            MetalResourcesProvider* provider,
            int dims,
            MetalIndexFlatConfig config = MetalIndexFlatConfig());

    MetalIndexFlatIP(
            std::shared_ptr<MetalResources> resources,
            int dims,
            MetalIndexFlatConfig config = MetalIndexFlatConfig());

    void copyFrom(faiss::IndexFlat* index);
    void copyTo(faiss::IndexFlat* index) const;
};

} // namespace metal
} // namespace faiss
