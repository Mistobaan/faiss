#pragma once

#include <faiss/impl/TurboQuantizer.h>
#include <faiss/metal/MetalResources.h>
#include <faiss/metal/impl/MetalResourcesImpl.h>

#include <memory>

namespace faiss {
namespace metal {
namespace impl {

class MetalTurboQuantizer {
   public:
    MetalTurboQuantizer(
            std::shared_ptr<MetalResources> resources,
            const faiss::TurboQuantMSEQuantizer& quantizer);

    void update(const faiss::TurboQuantMSEQuantizer& quantizer);

    void computeCodes(const float* x, uint8_t* codes, size_t n) const;
    void decode(const uint8_t* codes, float* x, size_t n) const;
    void decodeToBuffer(
            const uint8_t* codes,
            size_t n,
            const MetalBuffer& out,
            size_t out_row_offset = 0) const;

   private:
    std::shared_ptr<MetalResources> resources_;
    std::shared_ptr<MetalResourcesImpl> impl_;

    size_t d_ = 0;
    size_t nbits_ = 0;
    bool store_norm_ = true;
    size_t packed_code_size_ = 0;
    size_t code_size_ = 0;

    MetalBuffer rotation_;
    MetalBuffer centroids_;
    MetalBuffer boundaries_;
};

} // namespace impl
} // namespace metal
} // namespace faiss
