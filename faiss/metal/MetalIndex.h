#pragma once

#include <faiss/Index.h>
#include <faiss/metal/MetalResources.h>

#include <memory>

namespace faiss {
namespace metal {

struct MetalIndexConfig {
    int device = 0;
};

class MetalIndex : public faiss::Index {
   public:
    MetalIndex(
            MetalResourcesProvider* provider,
            int dims,
            faiss::MetricType metric,
            float metricArg,
            MetalIndexConfig config = MetalIndexConfig());

    MetalIndex(
            std::shared_ptr<MetalResources> resources,
            int dims,
            faiss::MetricType metric,
            float metricArg,
            MetalIndexConfig config = MetalIndexConfig());

    ~MetalIndex() override;

    int getDevice() const;
    std::shared_ptr<MetalResources> getResources();

    void add(idx_t n, const float* x) override;
    void add_ex(idx_t n, const void* x, NumericType numeric_type) override;
    void add_with_ids(idx_t n, const float* x, const idx_t* ids) override;
    void add_with_ids_ex(
            idx_t n,
            const void* x,
            NumericType numeric_type,
            const idx_t* ids) override;

    void assign(idx_t n, const float* x, idx_t* labels, idx_t k = 1)
            const override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void search_ex(
            idx_t n,
            const void* x,
            NumericType numeric_type,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void search_and_reconstruct(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            float* recons,
            const SearchParameters* params = nullptr) const override;

    void compute_residual(const float* x, float* residual, idx_t key)
            const override;

    void compute_residual_n(
            idx_t n,
            const float* xs,
            float* residuals,
            const idx_t* keys) const override;

   protected:
    void copyFrom(const faiss::Index* index);
    void copyTo(faiss::Index* index) const;

    virtual bool addImplRequiresIDs_() const = 0;
    virtual void addImpl_(idx_t n, const float* x, const idx_t* ids) = 0;
    virtual void searchImpl_(
            idx_t n,
            const float* x,
            int k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const = 0;

    std::shared_ptr<MetalResources> resources_;
    MetalIndexConfig config_;
};

} // namespace metal
} // namespace faiss
