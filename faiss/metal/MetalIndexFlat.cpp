#include <faiss/metal/MetalIndexFlat.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/metal/impl/MetalFlatIndex.h>

namespace faiss {
namespace metal {

namespace {

void validate_metric(MetricType metric) {
    FAISS_THROW_IF_NOT_MSG(
            metric == METRIC_L2 || metric == METRIC_INNER_PRODUCT,
            "MetalIndexFlat only supports L2 and inner product metrics");
}

void validate_config(const MetalIndexFlatConfig& config) {
    FAISS_THROW_IF_NOT_MSG(
            !config.useFloat16,
            "MetalIndexFlat does not support float16 storage in v1");
}

} // namespace

MetalIndexFlat::MetalIndexFlat(
        MetalResourcesProvider* provider,
        const faiss::IndexFlat* index,
        MetalIndexFlatConfig config)
        : MetalIndexFlat(provider->getResources(), index, config) {}

MetalIndexFlat::MetalIndexFlat(
        std::shared_ptr<MetalResources> resources,
        const faiss::IndexFlat* index,
        MetalIndexFlatConfig config)
        : MetalIndex(
                  std::move(resources),
                  index->d,
                  index->metric_type,
                  index->metric_arg,
                  config),
          flatConfig_(config) {
    validate_config(flatConfig_);
    validate_metric(index->metric_type);
    is_trained = true;
    copyFrom(index);
}

MetalIndexFlat::MetalIndexFlat(
        MetalResourcesProvider* provider,
        int dims,
        faiss::MetricType metric,
        MetalIndexFlatConfig config)
        : MetalIndexFlat(provider->getResources(), dims, metric, config) {}

MetalIndexFlat::MetalIndexFlat(
        std::shared_ptr<MetalResources> resources,
        int dims,
        faiss::MetricType metric,
        MetalIndexFlatConfig config)
        : MetalIndex(std::move(resources), dims, metric, 0.0f, config),
          flatConfig_(config) {
    validate_config(flatConfig_);
    validate_metric(metric);
    is_trained = true;
    resetIndex_(dims);
}

MetalIndexFlat::~MetalIndexFlat() = default;

void MetalIndexFlat::copyFrom(const faiss::IndexFlat* index) {
    MetalIndex::copyFrom(index);
    resetIndex_(d);
    if (index->ntotal > 0) {
        data_->add(index->get_xb(), index->ntotal);
    }
}

void MetalIndexFlat::copyTo(faiss::IndexFlat* index) const {
    MetalIndex::copyTo(index);
    index->code_size = sizeof(float) * d;
    index->codes.resize(ntotal * index->code_size);
    if (ntotal > 0) {
        reconstruct_n(0, ntotal, index->get_xb());
    }
}

size_t MetalIndexFlat::getNumVecs() const {
    return ntotal;
}

void MetalIndexFlat::reset() {
    data_->reset();
    ntotal = 0;
}

void MetalIndexFlat::train(idx_t, const float*) {
    is_trained = true;
}

void MetalIndexFlat::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);
    if (n == 0) {
        return;
    }
    addImpl_(n, x, nullptr);
}

void MetalIndexFlat::reconstruct(idx_t key, float* out) const {
    data_->reconstruct(key, 1, out);
}

void MetalIndexFlat::reconstruct_n(idx_t i0, idx_t num, float* out) const {
    if (num == 0) {
        return;
    }
    data_->reconstruct(i0, num, out);
}

void MetalIndexFlat::reconstruct_batch(
        idx_t n,
        const idx_t* keys,
        float* out) const {
    data_->reconstruct_batch(n, keys, out);
}

void MetalIndexFlat::resetIndex_(int dims) {
    data_ = std::make_unique<impl::MetalFlatIndex>(resources_, dims);
}

bool MetalIndexFlat::addImplRequiresIDs_() const {
    return false;
}

void MetalIndexFlat::addImpl_(idx_t n, const float* x, const idx_t* ids) {
    FAISS_THROW_IF_NOT_MSG(!ids, "add_with_ids not supported");
    data_->add(x, n);
    ntotal += n;
}

void MetalIndexFlat::searchImpl_(
        idx_t n,
        const float* x,
        int k,
        float* distances,
        idx_t* labels,
        const SearchParameters*) const {
    data_->query(n, x, k, metric_type, metric_arg, distances, labels);
}

MetalIndexFlatL2::MetalIndexFlatL2(
        MetalResourcesProvider* provider,
        faiss::IndexFlatL2* index,
        MetalIndexFlatConfig config)
        : MetalIndexFlat(provider, index, config) {}

MetalIndexFlatL2::MetalIndexFlatL2(
        std::shared_ptr<MetalResources> resources,
        faiss::IndexFlatL2* index,
        MetalIndexFlatConfig config)
        : MetalIndexFlat(std::move(resources), index, config) {}

MetalIndexFlatL2::MetalIndexFlatL2(
        MetalResourcesProvider* provider,
        int dims,
        MetalIndexFlatConfig config)
        : MetalIndexFlat(provider, dims, METRIC_L2, config) {}

MetalIndexFlatL2::MetalIndexFlatL2(
        std::shared_ptr<MetalResources> resources,
        int dims,
        MetalIndexFlatConfig config)
        : MetalIndexFlat(std::move(resources), dims, METRIC_L2, config) {}

void MetalIndexFlatL2::copyFrom(faiss::IndexFlat* index) {
    MetalIndexFlat::copyFrom(index);
}

void MetalIndexFlatL2::copyTo(faiss::IndexFlat* index) {
    MetalIndexFlat::copyTo(index);
}

MetalIndexFlatIP::MetalIndexFlatIP(
        MetalResourcesProvider* provider,
        faiss::IndexFlatIP* index,
        MetalIndexFlatConfig config)
        : MetalIndexFlat(provider, index, config) {}

MetalIndexFlatIP::MetalIndexFlatIP(
        std::shared_ptr<MetalResources> resources,
        faiss::IndexFlatIP* index,
        MetalIndexFlatConfig config)
        : MetalIndexFlat(std::move(resources), index, config) {}

MetalIndexFlatIP::MetalIndexFlatIP(
        MetalResourcesProvider* provider,
        int dims,
        MetalIndexFlatConfig config)
        : MetalIndexFlat(provider, dims, METRIC_INNER_PRODUCT, config) {}

MetalIndexFlatIP::MetalIndexFlatIP(
        std::shared_ptr<MetalResources> resources,
        int dims,
        MetalIndexFlatConfig config)
        : MetalIndexFlat(
                  std::move(resources),
                  dims,
                  METRIC_INNER_PRODUCT,
                  config) {}

void MetalIndexFlatIP::copyFrom(faiss::IndexFlat* index) {
    MetalIndexFlat::copyFrom(index);
}

void MetalIndexFlatIP::copyTo(faiss::IndexFlat* index) {
    MetalIndexFlat::copyTo(index);
}

} // namespace metal
} // namespace faiss
