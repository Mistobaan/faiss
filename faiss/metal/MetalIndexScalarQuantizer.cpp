#include <faiss/metal/MetalIndexScalarQuantizer.h>

#include <cstring>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/maybe_owned_vector.h>
#include <faiss/metal/impl/MetalFlatIndex.h>
#include <faiss/metal/impl/MetalScalarQuantizer.h>

namespace faiss {
namespace metal {

namespace {

void validate_metric(MetricType metric) {
    FAISS_THROW_IF_NOT_MSG(
            metric == METRIC_L2 || metric == METRIC_INNER_PRODUCT,
            "MetalIndexScalarQuantizer only supports L2 and inner product");
}

} // namespace

bool MetalIndexScalarQuantizer::supportsQuantizerType(
        faiss::ScalarQuantizer::QuantizerType qtype) {
    switch (qtype) {
        case faiss::ScalarQuantizer::QT_1bit_tqmse:
        case faiss::ScalarQuantizer::QT_2bit_tqmse:
        case faiss::ScalarQuantizer::QT_3bit_tqmse:
        case faiss::ScalarQuantizer::QT_4bit_tqmse:
        case faiss::ScalarQuantizer::QT_8bit_tqmse:
            return true;
        default:
            return false;
    }
}

MetalIndexScalarQuantizer::MetalIndexScalarQuantizer(
        MetalResourcesProvider* provider,
        const faiss::IndexScalarQuantizer* index,
        MetalIndexConfig config)
        : MetalIndexScalarQuantizer(provider->getResources(), index, config) {}

MetalIndexScalarQuantizer::MetalIndexScalarQuantizer(
        std::shared_ptr<MetalResources> resources,
        const faiss::IndexScalarQuantizer* index,
        MetalIndexConfig config)
        : MetalIndex(
                  std::move(resources),
                  index->d,
                  index->metric_type,
                  index->metric_arg,
                  config),
          sq_(index->sq) {
    validate_metric(index->metric_type);
    FAISS_THROW_IF_NOT_MSG(
            supportsQuantizerType(index->sq.qtype),
            "MetalIndexScalarQuantizer only supports TurboQuant SQ types");
    copyFrom(index);
}

MetalIndexScalarQuantizer::MetalIndexScalarQuantizer(
        MetalResourcesProvider* provider,
        idx_t d,
        faiss::ScalarQuantizer::QuantizerType qtype,
        MetricType metric,
        MetalIndexConfig config)
        : MetalIndexScalarQuantizer(
                  provider->getResources(),
                  d,
                  qtype,
                  metric,
                  config) {}

MetalIndexScalarQuantizer::MetalIndexScalarQuantizer(
        std::shared_ptr<MetalResources> resources,
        idx_t d,
        faiss::ScalarQuantizer::QuantizerType qtype,
        MetricType metric,
        MetalIndexConfig config)
        : MetalIndex(std::move(resources), d, metric, 0.0f, config),
          sq_(d, qtype) {
    validate_metric(metric);
    FAISS_THROW_IF_NOT_MSG(
            supportsQuantizerType(qtype),
            "MetalIndexScalarQuantizer only supports TurboQuant SQ types");
    is_trained = false;
    resetIndex_();
}

MetalIndexScalarQuantizer::~MetalIndexScalarQuantizer() = default;

void MetalIndexScalarQuantizer::copyFrom(
        const faiss::IndexScalarQuantizer* index) {
    MetalIndex::copyFrom(index);
    sq_ = index->sq;
    FAISS_THROW_IF_NOT_MSG(
            supportsQuantizerType(sq_.qtype),
            "MetalIndexScalarQuantizer only supports TurboQuant SQ types");
    codes_.assign(index->codes.data(), index->codes.data() + index->codes.size());
    resetIndex_();
    if (is_trained) {
        syncCodec_();
        rebuildDecodedStorage_();
    }
}

void MetalIndexScalarQuantizer::copyTo(
        faiss::IndexScalarQuantizer* index) const {
    MetalIndex::copyTo(index);
    index->sq = sq_;
    index->code_size = sq_.code_size;
    index->codes = faiss::MaybeOwnedVector<uint8_t>(codes_);
}

size_t MetalIndexScalarQuantizer::getNumVecs() const {
    return ntotal;
}

void MetalIndexScalarQuantizer::reset() {
    ntotal = 0;
    codes_.clear();
    if (data_) {
        data_->reset();
    }
}

void MetalIndexScalarQuantizer::train(idx_t n, const float* x) {
    sq_.train(n, x);
    is_trained = true;
    syncCodec_();
    reset();
}

void MetalIndexScalarQuantizer::reconstruct(idx_t key, float* out) const {
    data_->reconstruct(key, 1, out);
}

void MetalIndexScalarQuantizer::reconstruct_n(
        idx_t i0,
        idx_t num,
        float* out) const {
    if (num == 0) {
        return;
    }
    data_->reconstruct(i0, num, out);
}

void MetalIndexScalarQuantizer::reconstruct_batch(
        idx_t n,
        const idx_t* keys,
        float* out) const {
    data_->reconstruct_batch(n, keys, out);
}

bool MetalIndexScalarQuantizer::addImplRequiresIDs_() const {
    return false;
}

void MetalIndexScalarQuantizer::addImpl_(
        idx_t n,
        const float* x,
        const idx_t* ids) {
    FAISS_THROW_IF_NOT_MSG(!ids, "add_with_ids not supported");
    FAISS_THROW_IF_NOT(is_trained);
    if (n == 0) {
        return;
    }

    std::vector<uint8_t> new_codes(size_t(n) * sq_.code_size);
    codec_->computeCodes(x, new_codes.data(), n);

    const idx_t old_ntotal = ntotal;
    appendCodes_(new_codes.data(), n);
    data_->reserve(size_t(old_ntotal + n));
    codec_->decodeToBuffer(new_codes.data(), n, data_->buffer(), size_t(old_ntotal));
    data_->resize(size_t(old_ntotal + n));
    ntotal += n;
}

void MetalIndexScalarQuantizer::searchImpl_(
        idx_t n,
        const float* x,
        int k,
        float* distances,
        idx_t* labels,
        const SearchParameters*) const {
    data_->query(n, x, k, metric_type, metric_arg, distances, labels);
}

void MetalIndexScalarQuantizer::resetIndex_() {
    data_ = std::make_unique<impl::MetalFlatIndex>(resources_, d);
}

void MetalIndexScalarQuantizer::syncCodec_() {
    if (!codec_) {
        codec_ = std::make_unique<impl::MetalScalarQuantizer>(resources_, sq_);
    } else {
        codec_->update(sq_);
    }
}

void MetalIndexScalarQuantizer::appendCodes_(const uint8_t* codes, idx_t n) {
    const size_t old_size = codes_.size();
    codes_.resize(old_size + size_t(n) * sq_.code_size);
    std::memcpy(codes_.data() + old_size, codes, size_t(n) * sq_.code_size);
}

void MetalIndexScalarQuantizer::rebuildDecodedStorage_() {
    if (ntotal == 0) {
        data_->reset();
        return;
    }
    data_->reserve(size_t(ntotal));
    codec_->decodeToBuffer(codes_.data(), ntotal, data_->buffer(), 0);
    data_->resize(size_t(ntotal));
}

} // namespace metal
} // namespace faiss
