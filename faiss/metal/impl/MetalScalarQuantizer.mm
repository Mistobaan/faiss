#include <faiss/metal/impl/MetalScalarQuantizer.h>

#include <faiss/impl/FaissAssert.h>

#include <algorithm>
#include <cstring>

namespace faiss {
namespace metal {
namespace impl {

namespace {

struct TurboQuantArgs {
    uint32_t d;
    uint32_t nbits;
    uint32_t packed_code_size;
    uint32_t code_size;
    uint32_t prefix_bytes;
};

size_t tqmse_bits(ScalarQuantizer::QuantizerType qtype) {
    switch (qtype) {
        case ScalarQuantizer::QT_1bit_tqmse:
            return 1;
        case ScalarQuantizer::QT_2bit_tqmse:
            return 2;
        case ScalarQuantizer::QT_3bit_tqmse:
            return 3;
        case ScalarQuantizer::QT_4bit_tqmse:
            return 4;
        case ScalarQuantizer::QT_8bit_tqmse:
            return 8;
        default:
            FAISS_THROW_MSG(
                    "MetalScalarQuantizer only supports TurboQuant SQ types");
    }
}

} // namespace

MetalScalarQuantizer::MetalScalarQuantizer(
        std::shared_ptr<MetalResources> resources,
        const faiss::ScalarQuantizer& quantizer)
        : resources_(std::move(resources)), impl_(resources_->getImplShared()) {
    update(quantizer);
}

void MetalScalarQuantizer::update(const faiss::ScalarQuantizer& quantizer) {
    d_ = quantizer.d;
    nbits_ = tqmse_bits(quantizer.qtype);
    packed_code_size_ = quantizer.code_size;
    code_size_ = quantizer.code_size;

    const size_t centroid_count = size_t(1) << nbits_;
    FAISS_THROW_IF_NOT_FMT(
            quantizer.trained.size() == 2 * centroid_count - 1,
            "invalid TurboQuant SQ trained size %zu for %zu bits",
            quantizer.trained.size(),
            nbits_);

    centroids_ = impl_->makeBuffer(sizeof(float) * centroid_count);
    std::memcpy(
            centroids_.contents(),
            quantizer.trained.data(),
            centroids_.size());

    const size_t boundary_count = std::max<size_t>(1, centroid_count - 1);
    boundaries_ = impl_->makeBuffer(sizeof(float) * boundary_count);
    std::memcpy(
            boundaries_.contents(),
            quantizer.trained.data() + centroid_count,
            sizeof(float) * (centroid_count - 1));
}

void MetalScalarQuantizer::computeCodes(
        const float* x,
        uint8_t* codes,
        size_t n) const {
    if (n == 0) {
        return;
    }

    auto input = impl_->makeBuffer(sizeof(float) * n * d_);
    auto norms = impl_->makeBuffer(sizeof(float) * n);
    auto code_buffer = impl_->makeBuffer(code_size_ * n);

    std::memcpy(input.contents(), x, input.size());

    const TurboQuantArgs args{
            static_cast<uint32_t>(d_),
            static_cast<uint32_t>(nbits_),
            static_cast<uint32_t>(packed_code_size_),
            static_cast<uint32_t>(code_size_),
            0};
    impl_->runKernel(
            "quantize_pack",
            n,
            {
                    KernelArg::buffer(0, input.handle()),
                    KernelArg::buffer(1, norms.handle()),
                    KernelArg::buffer(2, boundaries_.handle()),
                    KernelArg::buffer(3, code_buffer.handle()),
                    KernelArg::bytes(4, &args, sizeof(args)),
            });

    std::memcpy(codes, code_buffer.contents(), code_buffer.size());
}

void MetalScalarQuantizer::decode(
        const uint8_t* codes,
        float* x,
        size_t n) const {
    if (n == 0) {
        return;
    }

    auto out = impl_->makeBuffer(sizeof(float) * n * d_);
    decodeToBuffer(codes, n, out, 0);
    std::memcpy(x, out.contents(), out.size());
}

void MetalScalarQuantizer::decodeToBuffer(
        const uint8_t* codes,
        size_t n,
        const MetalBuffer& out,
        size_t out_row_offset) const {
    if (n == 0) {
        return;
    }

    auto code_buffer = impl_->makeBuffer(code_size_ * n);
    auto decoded = impl_->makeBuffer(sizeof(float) * n * d_);
    auto norms = impl_->makeBuffer(sizeof(float) * n);

    std::memcpy(code_buffer.contents(), codes, code_buffer.size());

    const TurboQuantArgs args{
            static_cast<uint32_t>(d_),
            static_cast<uint32_t>(nbits_),
            static_cast<uint32_t>(packed_code_size_),
            static_cast<uint32_t>(code_size_),
            0};
    impl_->runKernel(
            "unpack_codes",
            n,
            {
                    KernelArg::buffer(0, code_buffer.handle()),
                    KernelArg::buffer(1, norms.handle()),
                    KernelArg::buffer(2, centroids_.handle()),
                    KernelArg::buffer(3, decoded.handle()),
                    KernelArg::bytes(4, &args, sizeof(args)),
            });

    impl_->copyBuffer(
            decoded,
            0,
            out,
            out_row_offset * d_ * sizeof(float),
            decoded.size());
}

} // namespace impl
} // namespace metal
} // namespace faiss
