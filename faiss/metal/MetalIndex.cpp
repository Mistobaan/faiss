#include <faiss/metal/MetalIndex.h>

#include <algorithm>
#include <limits>
#include <vector>

#include <faiss/impl/FaissAssert.h>

namespace faiss {
namespace metal {

namespace {

void validate_device(int device) {
    FAISS_THROW_IF_NOT_FMT(
            device == 0,
            "Metal backend only supports device 0 in v1 (got %d)",
            device);
}

} // namespace

MetalIndex::MetalIndex(
        MetalResourcesProvider* provider,
        int dims,
        faiss::MetricType metric,
        float metricArg,
        MetalIndexConfig config)
        : MetalIndex(provider->getResources(), dims, metric, metricArg, config) {}

MetalIndex::MetalIndex(
        std::shared_ptr<MetalResources> resources,
        int dims,
        faiss::MetricType metric,
        float metricArg,
        MetalIndexConfig config)
        : faiss::Index(dims, metric),
          resources_(std::move(resources)),
          config_(config) {
    FAISS_THROW_IF_NOT(resources_);
    validate_device(config_.device);
    resources_->initializeForDevice(config_.device);
    metric_arg = metricArg;
}

MetalIndex::~MetalIndex() = default;

int MetalIndex::getDevice() const {
    return config_.device;
}

std::shared_ptr<MetalResources> MetalIndex::getResources() {
    return resources_;
}

void MetalIndex::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);
    if (n == 0) {
        return;
    }
    addImpl_(n, x, nullptr);
}

void MetalIndex::add_ex(idx_t n, const void* x, NumericType numeric_type) {
    if (numeric_type != NumericType::Float32) {
        FAISS_THROW_MSG("MetalIndex::add_ex only supports float32 inputs");
    }
    add(n, static_cast<const float*>(x));
}

void MetalIndex::add_with_ids(idx_t n, const float* x, const idx_t* ids) {
    FAISS_THROW_IF_NOT(is_trained);
    if (n == 0) {
        return;
    }
    if (!addImplRequiresIDs_()) {
        FAISS_THROW_IF_NOT_MSG(!ids, "add_with_ids not supported");
        addImpl_(n, x, nullptr);
        return;
    }
    addImpl_(n, x, ids);
}

void MetalIndex::add_with_ids_ex(
        idx_t n,
        const void* x,
        NumericType numeric_type,
        const idx_t* ids) {
    if (numeric_type != NumericType::Float32) {
        FAISS_THROW_MSG("MetalIndex::add_with_ids_ex only supports float32");
    }
    add_with_ids(n, static_cast<const float*>(x), ids);
}

void MetalIndex::assign(idx_t n, const float* x, idx_t* labels, idx_t k) const {
    std::vector<float> distances(n * k);
    search(n, x, k, distances.data(), labels);
}

void MetalIndex::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);
    if (n == 0) {
        return;
    }
    searchImpl_(n, x, static_cast<int>(k), distances, labels, params);
}

void MetalIndex::search_ex(
        idx_t n,
        const void* x,
        NumericType numeric_type,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    if (numeric_type != NumericType::Float32) {
        FAISS_THROW_MSG("MetalIndex::search_ex only supports float32");
    }
    search(n, static_cast<const float*>(x), k, distances, labels, params);
}

void MetalIndex::search_and_reconstruct(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        float* recons,
        const SearchParameters* params) const {
    search(n, x, k, distances, labels, params);
    for (idx_t i = 0; i < n * k; i++) {
        float* out = recons + i * d;
        if (labels[i] < 0) {
            std::fill(out, out + d, 0.0f);
        } else {
            reconstruct(labels[i], out);
        }
    }
}

void MetalIndex::compute_residual(
        const float* x,
        float* residual,
        idx_t key) const {
    std::vector<float> recon(d);
    reconstruct(key, recon.data());
    for (idx_t i = 0; i < d; i++) {
        residual[i] = x[i] - recon[i];
    }
}

void MetalIndex::compute_residual_n(
        idx_t n,
        const float* xs,
        float* residuals,
        const idx_t* keys) const {
    for (idx_t i = 0; i < n; i++) {
        compute_residual(xs + i * d, residuals + i * d, keys[i]);
    }
}

void MetalIndex::copyFrom(const faiss::Index* index) {
    d = index->d;
    ntotal = index->ntotal;
    metric_type = index->metric_type;
    metric_arg = index->metric_arg;
    is_trained = index->is_trained;
    verbose = index->verbose;
}

void MetalIndex::copyTo(faiss::Index* index) const {
    index->d = d;
    index->ntotal = ntotal;
    index->metric_type = metric_type;
    index->metric_arg = metric_arg;
    index->is_trained = is_trained;
    index->verbose = verbose;
}

} // namespace metal
} // namespace faiss
