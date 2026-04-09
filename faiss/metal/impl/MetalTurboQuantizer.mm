#include <faiss/metal/impl/MetalTurboQuantizer.h>

#include <faiss/impl/FaissAssert.h>

#include <algorithm>
#include <cstring>

namespace faiss {
namespace metal {
namespace impl {

namespace {

struct NormalizeArgs {
    uint32_t d;
    uint32_t store_norm;
};

struct TurboQuantArgs {
    uint32_t d;
    uint32_t nbits;
    uint32_t packed_code_size;
    uint32_t code_size;
    uint32_t prefix_bytes;
};

struct ScaleArgs {
    uint32_t d;
    uint32_t out_row_offset;
};

} // namespace

MetalTurboQuantizer::MetalTurboQuantizer(
        std::shared_ptr<MetalResources> resources,
        const faiss::TurboQuantizer& quantizer)
        : resources_(std::move(resources)),
          impl_(resources_->getImplShared()) {
    update(quantizer);
}

void MetalTurboQuantizer::update(const faiss::TurboQuantizer& quantizer) {
    d_ = quantizer.d;
    nbits_ = quantizer.nbits;
    store_norm_ = quantizer.store_norm;
    packed_code_size_ = quantizer.packed_code_size;
    code_size_ = quantizer.code_size;

    rotation_ = impl_->makeBuffer(sizeof(float) * d_ * d_);
    std::memcpy(rotation_.contents(), quantizer.rotation.A.data(), rotation_.size());

    centroids_ = impl_->makeBuffer(sizeof(float) * quantizer.centroids.size());
    std::memcpy(
            centroids_.contents(),
            quantizer.centroids.data(),
            centroids_.size());

    const size_t boundary_count = std::max<size_t>(1, quantizer.boundaries.size());
    boundaries_ = impl_->makeBuffer(sizeof(float) * boundary_count);
    if (!quantizer.boundaries.empty()) {
        std::memcpy(
                boundaries_.contents(),
                quantizer.boundaries.data(),
                sizeof(float) * quantizer.boundaries.size());
    }
}

void MetalTurboQuantizer::computeCodes(
        const float* x,
        uint8_t* codes,
        size_t n) const {
    if (n == 0) {
        return;
    }

    auto input = impl_->makeBuffer(sizeof(float) * n * d_);
    auto normalized = impl_->makeBuffer(sizeof(float) * n * d_);
    auto rotated = impl_->makeBuffer(sizeof(float) * n * d_);
    auto norms = impl_->makeBuffer(sizeof(float) * n);
    auto code_buffer = impl_->makeBuffer(code_size_ * n);

    std::memcpy(input.contents(), x, input.size());

    const NormalizeArgs norm_args{
            static_cast<uint32_t>(d_), store_norm_ ? 1u : 0u};
    impl_->runKernel(
            "normalize_vectors",
            n,
            {
                    KernelArg::buffer(0, input.handle()),
                    KernelArg::buffer(1, normalized.handle()),
                    KernelArg::buffer(2, norms.handle()),
                    KernelArg::bytes(3, &norm_args, sizeof(norm_args)),
            });

    // The CPU path stores rotation.A in column-major form for the logical
    // matrix M. The same bytes look like M^T when interpreted as row-major, so
    // the forward multiplication uses transpose_right=true to recover M.
    impl_->matrixMultiply(
            normalized,
            rotation_,
            rotated,
            n,
            d_,
            d_,
            false,
            true);

    const TurboQuantArgs tq_args{
            static_cast<uint32_t>(d_),
            static_cast<uint32_t>(nbits_),
            static_cast<uint32_t>(packed_code_size_),
            static_cast<uint32_t>(code_size_),
            static_cast<uint32_t>(store_norm_ ? sizeof(float) : 0)};
    impl_->runKernel(
            "quantize_pack",
            n,
            {
                    KernelArg::buffer(0, rotated.handle()),
                    KernelArg::buffer(1, norms.handle()),
                    KernelArg::buffer(2, boundaries_.handle()),
                    KernelArg::buffer(3, code_buffer.handle()),
                    KernelArg::bytes(4, &tq_args, sizeof(tq_args)),
            });

    std::memcpy(codes, code_buffer.contents(), code_buffer.size());
}

void MetalTurboQuantizer::decode(
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

void MetalTurboQuantizer::decodeToBuffer(
        const uint8_t* codes,
        size_t n,
        const MetalBuffer& out,
        size_t out_row_offset) const {
    if (n == 0) {
        return;
    }

    auto code_buffer = impl_->makeBuffer(code_size_ * n);
    auto rotated = impl_->makeBuffer(sizeof(float) * n * d_);
    auto normalized = impl_->makeBuffer(sizeof(float) * n * d_);
    auto norms = impl_->makeBuffer(sizeof(float) * n);

    std::memcpy(code_buffer.contents(), codes, code_buffer.size());

    const TurboQuantArgs tq_args{
            static_cast<uint32_t>(d_),
            static_cast<uint32_t>(nbits_),
            static_cast<uint32_t>(packed_code_size_),
            static_cast<uint32_t>(code_size_),
            static_cast<uint32_t>(store_norm_ ? sizeof(float) : 0)};
    impl_->runKernel(
            "unpack_codes",
            n,
            {
                    KernelArg::buffer(0, code_buffer.handle()),
                    KernelArg::buffer(1, norms.handle()),
                    KernelArg::buffer(2, centroids_.handle()),
                    KernelArg::buffer(3, rotated.handle()),
                    KernelArg::bytes(4, &tq_args, sizeof(tq_args)),
            });

    // See the comment in computeCodes(): the inverse path needs Y * M^T, and
    // the embedded buffer bytes already look row-major like M^T.
    impl_->matrixMultiply(
            rotated,
            rotation_,
            normalized,
            n,
            d_,
            d_,
            false,
            false);

    const ScaleArgs scale_args{
            static_cast<uint32_t>(d_),
            static_cast<uint32_t>(out_row_offset)};
    impl_->runKernel(
            "rescale_vectors",
            n,
            {
                    KernelArg::buffer(0, normalized.handle()),
                    KernelArg::buffer(1, norms.handle()),
                    KernelArg::buffer(2, out.handle()),
                    KernelArg::bytes(3, &scale_args, sizeof(scale_args)),
            });
}

} // namespace impl
} // namespace metal
} // namespace faiss
