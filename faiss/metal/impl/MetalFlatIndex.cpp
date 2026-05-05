#include <faiss/metal/impl/MetalFlatIndex.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include <faiss/impl/FaissAssert.h>

namespace faiss {
namespace metal {
namespace impl {

namespace {

struct SearchArgs {
    uint32_t d;
    uint32_t nb;
    uint32_t k;
    uint32_t metric;
};

} // namespace

MetalFlatIndex::MetalFlatIndex(std::shared_ptr<MetalResources> resources, int dims)
        : resources_(std::move(resources)),
          impl_(resources_->getImplShared()),
          dims_(dims) {}

size_t MetalFlatIndex::getSize() const {
    return size_;
}

size_t MetalFlatIndex::getCapacity() const {
    return capacity_;
}

int MetalFlatIndex::getDims() const {
    return dims_;
}

void MetalFlatIndex::reset() {
    size_ = 0;
}

void MetalFlatIndex::reserve(size_t n) {
    reserveImpl_(n);
}

void MetalFlatIndex::resize(size_t n) {
    FAISS_THROW_IF_NOT(n <= capacity_);
    size_ = n;
}

void MetalFlatIndex::add(const float* x, idx_t n) {
    if (n == 0) {
        return;
    }
    reserveImpl_(size_ + n);
    std::memcpy(
            hostData() + size_ * dims_, x, sizeof(float) * n * size_t(dims_));
    size_ += n;
}

void MetalFlatIndex::setFromHost(const float* x, idx_t n) {
    reset();
    add(x, n);
}

void MetalFlatIndex::reconstruct(idx_t key, idx_t n, float* out) const {
    FAISS_THROW_IF_NOT(key >= 0);
    FAISS_THROW_IF_NOT(size_t(key + n) <= size_);
    std::memcpy(
            out,
            hostData() + size_t(key) * dims_,
            sizeof(float) * size_t(n) * dims_);
}

void MetalFlatIndex::reconstruct_batch(
        idx_t n,
        const idx_t* keys,
        float* out) const {
    for (idx_t i = 0; i < n; i++) {
        reconstruct(keys[i], 1, out + i * dims_);
    }
}

void MetalFlatIndex::query(
        idx_t n,
        const float* x,
        int k,
        MetricType metric,
        float,
        float* distances,
        idx_t* labels) const {
    FAISS_THROW_IF_NOT_MSG(
            metric == METRIC_L2 || metric == METRIC_INNER_PRODUCT,
            "MetalFlatIndex only supports L2 and inner product metrics");

    const size_t query_bytes = sizeof(float) * size_t(n) * dims_;
    const size_t distance_bytes = sizeof(float) * size_t(n) * k;
    const size_t label_bytes = sizeof(idx_t) * size_t(n) * k;

    auto query_buffer = impl_->makeBuffer(query_bytes);
    auto distance_buffer = impl_->makeBuffer(distance_bytes);
    auto label_buffer = impl_->makeBuffer(label_bytes);

    std::memcpy(query_buffer.contents(), x, query_bytes);

    SearchArgs args{
            static_cast<uint32_t>(dims_),
            static_cast<uint32_t>(size_),
            static_cast<uint32_t>(k),
            metric == METRIC_L2 ? 0u : 1u};

    const auto count = static_cast<size_t>(n);
    impl_->runKernel(
            args.metric == 0 ? "flat_search_l2" : "flat_search_ip",
            count,
            {
                    KernelArg::buffer(0, query_buffer.handle()),
                    KernelArg::buffer(1, data_.handle()),
                    KernelArg::buffer(2, distance_buffer.handle()),
                    KernelArg::buffer(3, label_buffer.handle()),
                    KernelArg::bytes(4, &args, sizeof(args)),
            });

    std::memcpy(distances, distance_buffer.contents(), distance_bytes);
    std::memcpy(labels, label_buffer.contents(), label_bytes);
}

float* MetalFlatIndex::hostData() {
    return static_cast<float*>(data_.contents());
}

const float* MetalFlatIndex::hostData() const {
    return static_cast<const float*>(data_.contents());
}

MetalBuffer& MetalFlatIndex::buffer() {
    return data_;
}

const MetalBuffer& MetalFlatIndex::buffer() const {
    return data_;
}

void MetalFlatIndex::reserveImpl_(size_t n) {
    if (n <= capacity_) {
        return;
    }

    const size_t new_capacity = std::max<size_t>(n, std::max<size_t>(1, capacity_ * 2));
    auto new_buffer = impl_->makeBuffer(sizeof(float) * new_capacity * dims_);
    if (size_ > 0 && data_.valid()) {
        std::memcpy(
                new_buffer.contents(),
                data_.contents(),
                sizeof(float) * size_ * dims_);
    }
    data_ = std::move(new_buffer);
    capacity_ = new_capacity;
}

} // namespace impl
} // namespace metal
} // namespace faiss
