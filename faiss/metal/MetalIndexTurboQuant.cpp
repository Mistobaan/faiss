#include <faiss/metal/MetalIndexTurboQuant.h>

#include <algorithm>
#include <cstring>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/maybe_owned_vector.h>
#include <faiss/metal/impl/MetalFlatIndex.h>
#include <faiss/metal/impl/MetalTurboQuantizer.h>

namespace faiss {
namespace metal {

namespace {

void validate_metric(MetricType metric) {
    FAISS_THROW_IF_NOT_MSG(
            metric == METRIC_L2 || metric == METRIC_INNER_PRODUCT,
            "MetalIndexTurboQuantMSE only supports L2 and inner product");
}

} // namespace

MetalIndexTurboQuantMSE::MetalIndexTurboQuantMSE(
        MetalResourcesProvider* provider,
        const faiss::IndexTurboQuantMSE* index,
        MetalIndexConfig config)
        : MetalIndexTurboQuantMSE(provider->getResources(), index, config) {}

MetalIndexTurboQuantMSE::MetalIndexTurboQuantMSE(
        std::shared_ptr<MetalResources> resources,
        const faiss::IndexTurboQuantMSE* index,
        MetalIndexConfig config)
        : MetalIndex(
                  std::move(resources),
                  index->d,
                  index->metric_type,
                  index->metric_arg,
                  config),
          tq_(index->tqmse) {
    validate_metric(index->metric_type);
    is_trained = index->is_trained;
    copyFrom(index);
}

MetalIndexTurboQuantMSE::MetalIndexTurboQuantMSE(
        MetalResourcesProvider* provider,
        idx_t d,
        size_t nbits,
        MetricType metric,
        uint32_t seed,
        bool store_norm,
        MetalIndexConfig config)
        : MetalIndexTurboQuantMSE(
                  provider->getResources(),
                  d,
                  nbits,
                  metric,
                  seed,
                  store_norm,
                  config) {}

MetalIndexTurboQuantMSE::MetalIndexTurboQuantMSE(
        std::shared_ptr<MetalResources> resources,
        idx_t d,
        size_t nbits,
        MetricType metric,
        uint32_t seed,
        bool store_norm,
        MetalIndexConfig config)
        : MetalIndex(std::move(resources), d, metric, 0.0f, config),
          tq_(d, nbits, seed, store_norm) {
    validate_metric(metric);
    tq_.initialize();
    is_trained = true;
    resetIndex_();
    syncCodec_();
}

MetalIndexTurboQuantMSE::~MetalIndexTurboQuantMSE() = default;

void MetalIndexTurboQuantMSE::copyFrom(const faiss::IndexTurboQuantMSE* index) {
    MetalIndex::copyFrom(index);
    tq_ = index->tqmse;
    is_trained = index->is_trained;
    codes_.assign(index->codes.data(), index->codes.data() + index->codes.size());
    resetIndex_();
    syncCodec_();
    rebuildDecodedStorage_();
}

void MetalIndexTurboQuantMSE::copyTo(faiss::IndexTurboQuantMSE* index) const {
    MetalIndex::copyTo(index);
    index->tqmse = tq_;
    index->code_size = tq_.code_size;
    index->codes = faiss::MaybeOwnedVector<uint8_t>(codes_);
}

size_t MetalIndexTurboQuantMSE::getNumVecs() const {
    return ntotal;
}

void MetalIndexTurboQuantMSE::reset() {
    ntotal = 0;
    codes_.clear();
    if (data_) {
        data_->reset();
    }
}

void MetalIndexTurboQuantMSE::train(idx_t n, const float* x) {
    tq_.train(n, x);
    is_trained = true;
    syncCodec_();
    reset();
}

void MetalIndexTurboQuantMSE::reconstruct(idx_t key, float* out) const {
    data_->reconstruct(key, 1, out);
}

void MetalIndexTurboQuantMSE::reconstruct_n(
        idx_t i0,
        idx_t num,
        float* out) const {
    if (num == 0) {
        return;
    }
    data_->reconstruct(i0, num, out);
}

void MetalIndexTurboQuantMSE::reconstruct_batch(
        idx_t n,
        const idx_t* keys,
        float* out) const {
    data_->reconstruct_batch(n, keys, out);
}

bool MetalIndexTurboQuantMSE::addImplRequiresIDs_() const {
    return false;
}

void MetalIndexTurboQuantMSE::addImpl_(
        idx_t n,
        const float* x,
        const idx_t* ids) {
    FAISS_THROW_IF_NOT_MSG(!ids, "add_with_ids not supported");
    FAISS_THROW_IF_NOT(is_trained);
    if (n == 0) {
        return;
    }

    std::vector<uint8_t> new_codes(size_t(n) * tq_.code_size);
    codec_->computeCodes(x, new_codes.data(), n);

    const idx_t old_ntotal = ntotal;
    appendCodes_(new_codes.data(), n);
    data_->reserve(size_t(old_ntotal + n));
    codec_->decodeToBuffer(new_codes.data(), n, data_->buffer(), size_t(old_ntotal));
    data_->resize(size_t(old_ntotal + n));
    ntotal += n;
}

void MetalIndexTurboQuantMSE::searchImpl_(
        idx_t n,
        const float* x,
        int k,
        float* distances,
        idx_t* labels,
        const SearchParameters*) const {
    data_->query(n, x, k, metric_type, metric_arg, distances, labels);
}

void MetalIndexTurboQuantMSE::resetIndex_() {
    data_ = std::make_unique<impl::MetalFlatIndex>(resources_, d);
}

void MetalIndexTurboQuantMSE::syncCodec_() {
    if (!codec_) {
        codec_ = std::make_unique<impl::MetalTurboQuantizer>(resources_, tq_);
    } else {
        codec_->update(tq_);
    }
}

void MetalIndexTurboQuantMSE::appendCodes_(const uint8_t* codes, idx_t n) {
    const size_t old_size = codes_.size();
    codes_.resize(old_size + size_t(n) * tq_.code_size);
    std::memcpy(codes_.data() + old_size, codes, size_t(n) * tq_.code_size);
}

void MetalIndexTurboQuantMSE::rebuildDecodedStorage_() {
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
