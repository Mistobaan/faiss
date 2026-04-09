#pragma once

#include <faiss/IndexTurboQuant.h>
#include <faiss/metal/MetalIndex.h>

#include <memory>
#include <vector>

namespace faiss {
namespace metal {

namespace impl {
class MetalFlatIndex;
class MetalTurboQuantizer;
}

class MetalIndexTurboQuantMSE : public MetalIndex {
   public:
    MetalIndexTurboQuantMSE(
            MetalResourcesProvider* provider,
            const faiss::IndexTurboQuantMSE* index,
            MetalIndexConfig config = MetalIndexConfig());

    MetalIndexTurboQuantMSE(
            std::shared_ptr<MetalResources> resources,
            const faiss::IndexTurboQuantMSE* index,
            MetalIndexConfig config = MetalIndexConfig());

    MetalIndexTurboQuantMSE(
            MetalResourcesProvider* provider,
            idx_t d,
            size_t nbits,
            MetricType metric = METRIC_L2,
            uint32_t seed = 12345,
            bool store_norm = true,
            MetalIndexConfig config = MetalIndexConfig());

    MetalIndexTurboQuantMSE(
            std::shared_ptr<MetalResources> resources,
            idx_t d,
            size_t nbits,
            MetricType metric = METRIC_L2,
            uint32_t seed = 12345,
            bool store_norm = true,
            MetalIndexConfig config = MetalIndexConfig());

    ~MetalIndexTurboQuantMSE() override;

    void copyFrom(const faiss::IndexTurboQuantMSE* index);
    void copyTo(faiss::IndexTurboQuantMSE* index) const;

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

    faiss::TurboQuantizer tq_;
    std::vector<uint8_t> codes_;
    std::unique_ptr<impl::MetalFlatIndex> data_;
    std::unique_ptr<impl::MetalTurboQuantizer> codec_;
};

} // namespace metal
} // namespace faiss
