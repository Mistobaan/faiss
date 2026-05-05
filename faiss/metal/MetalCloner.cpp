#include <faiss/metal/MetalCloner.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/FaissAssert.h>

namespace faiss {
namespace metal {

Index* ToCPUCloner::clone_Index(const Index* index) {
    if (auto metal_l2 = dynamic_cast<const MetalIndexFlatL2*>(index)) {
        auto* out = new IndexFlatL2();
        metal_l2->copyTo(out);
        return out;
    } else if (auto metal_ip = dynamic_cast<const MetalIndexFlatIP*>(index)) {
        auto* out = new IndexFlatIP();
        metal_ip->copyTo(out);
        return out;
    } else if (auto metal_flat = dynamic_cast<const MetalIndexFlat*>(index)) {
        auto* out = new IndexFlat();
        metal_flat->copyTo(out);
        return out;
    } else if (
            auto metal_sq = dynamic_cast<const MetalIndexScalarQuantizer*>(
                    index)) {
        auto* out = new IndexScalarQuantizer();
        metal_sq->copyTo(out);
        return out;
    }
    return Cloner::clone_Index(index);
}

ToMetalCloner::ToMetalCloner(
        MetalResourcesProvider* provider_in,
        int device_in,
        const MetalClonerOptions& options)
        : MetalClonerOptions(options),
          provider(provider_in),
          device(device_in) {}

Index* ToMetalCloner::clone_Index(const Index* index) {
    FAISS_THROW_IF_NOT_FMT(
            device == 0,
            "Metal backend only supports device 0 in v1 (got %d)",
            device);

    if (auto flat_l2 = dynamic_cast<const IndexFlatL2*>(index)) {
        MetalIndexFlatConfig config;
        config.device = device;
        config.useFloat16 = useFloat16;
        return new MetalIndexFlatL2(
                provider, const_cast<IndexFlatL2*>(flat_l2), config);
    } else if (auto flat_ip = dynamic_cast<const IndexFlatIP*>(index)) {
        MetalIndexFlatConfig config;
        config.device = device;
        config.useFloat16 = useFloat16;
        return new MetalIndexFlatIP(
                provider, const_cast<IndexFlatIP*>(flat_ip), config);
    } else if (auto flat = dynamic_cast<const IndexFlat*>(index)) {
        MetalIndexFlatConfig config;
        config.device = device;
        config.useFloat16 = useFloat16;
        if (flat->metric_type == METRIC_L2) {
            auto* out = new MetalIndexFlatL2(provider, flat->d, config);
            out->copyFrom(const_cast<IndexFlat*>(flat));
            return out;
        }
        if (flat->metric_type == METRIC_INNER_PRODUCT) {
            auto* out = new MetalIndexFlatIP(provider, flat->d, config);
            out->copyFrom(const_cast<IndexFlat*>(flat));
            return out;
        }
        FAISS_THROW_MSG("Metal flat indexes only support L2 and inner product metrics.");
    } else if (
            auto sq = dynamic_cast<const IndexScalarQuantizer*>(index)) {
        FAISS_THROW_IF_NOT_MSG(
                MetalIndexScalarQuantizer::supportsQuantizerType(sq->sq.qtype),
                "Metal backend only supports TurboQuant scalar quantizers");
        MetalIndexConfig config;
        config.device = device;
        return new MetalIndexScalarQuantizer(provider, sq, config);
    } else if (
            auto pre = dynamic_cast<const IndexPreTransform*>(index)) {
        auto* out = new IndexPreTransform();
        out->d = pre->d;
        out->ntotal = pre->ntotal;
        out->is_trained = pre->is_trained;
        out->metric_type = pre->metric_type;
        out->metric_arg = pre->metric_arg;
        out->verbose = pre->verbose;
        for (VectorTransform* transform : pre->chain) {
            out->chain.push_back(clone_VectorTransform(transform));
        }
        out->index = clone_Index(pre->index);
        out->own_fields = true;
        return out;
    }

    FAISS_THROW_MSG("This index type is not implemented on Metal.");
}

faiss::Index* index_metal_to_cpu(const faiss::Index* metal_index) {
    ToCPUCloner cloner;
    return cloner.clone_Index(metal_index);
}

faiss::Index* index_cpu_to_metal(
        MetalResourcesProvider* provider,
        int device,
        const faiss::Index* index,
        const MetalClonerOptions* options) {
    MetalClonerOptions default_options;
    const MetalClonerOptions& use_options = options ? *options : default_options;
    ToMetalCloner cloner(provider, device, use_options);
    return cloner.clone_Index(index);
}

} // namespace metal
} // namespace faiss
