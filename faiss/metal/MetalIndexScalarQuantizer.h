#pragma once

#include <faiss/IndexScalarQuantizer.h>
#include <faiss/metal/MetalIndex.h>

#include <memory>
#include <vector>

namespace faiss {
namespace metal {

namespace impl {
class MetalFlatIndex;
class MetalScalarQuantizer;
}

class MetalIndexScalarQuantizer : public MetalIndex {
   public:
    MetalIndexScalarQuantizer(
            MetalResourcesProvider* provider,
            const faiss::IndexScalarQuantizer* index,
            MetalIndexConfig config = MetalIndexConfig());

    MetalIndexScalarQuantizer(
            std::shared_ptr<MetalResources> resources,
            const faiss::IndexScalarQuantizer* index,
            MetalIndexConfig config = MetalIndexConfig());

    MetalIndexScalarQuantizer(
            MetalResourcesProvider* provider,
            idx_t d,
            faiss::ScalarQuantizer::QuantizerType qtype,
            MetricType metric = METRIC_L2,
            MetalIndexConfig config = MetalIndexConfig());

    MetalIndexScalarQuantizer(
            std::shared_ptr<MetalResources> resources,
            idx_t d,
            faiss::ScalarQuantizer::QuantizerType qtype,
            MetricType metric = METRIC_L2,
            MetalIndexConfig config = MetalIndexConfig());

    ~MetalIndexScalarQuantizer() override;

    static bool supportsQuantizerType(
            faiss::ScalarQuantizer::QuantizerType qtype);

    void copyFrom(const faiss::IndexScalarQuantizer* index);
    void copyTo(faiss::IndexScalarQuantizer* index) const;

    size_t getNumVecs() const;

    void reset() override;
    void train(idx_t n, const float* x) override;

    void reconstruct(idx_t key, float* out) const override;
    void reconstruct_n(idx_t i0, idx_t num, float* out) const override;
    void reconstruct_batch(idx_t n, const idx_t* keys, float* out)
            const override;

   protected:
    bool addImplRequiresIDs_() const override;
    void addImpl_(idx_t n, const float* x, const idx_t* ids) override;
    void searchImpl_(
            idx_t n,
            const float* x,
            int k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override;

   private:
    void resetIndex_();
    void syncCodec_();
    void appendCodes_(const uint8_t* codes, idx_t n);
    void rebuildDecodedStorage_();

    faiss::ScalarQuantizer sq_;
    std::vector<uint8_t> codes_;
    std::unique_ptr<impl::MetalFlatIndex> data_;
    std::unique_ptr<impl::MetalScalarQuantizer> codec_;
};

} // namespace metal
} // namespace faiss
