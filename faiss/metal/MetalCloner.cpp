#include <faiss/metal/MetalCloner.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexTurboQuant.h>
#include <faiss/impl/FaissAssert.h>

namespace faiss {
namespace metal {

Index* ToCPUCloner::clone_Index(const Index* index) {
    if (auto metal_flat = dynamic_cast<const MetalIndexFlat*>(index)) {
        auto* out = new IndexFlat();
        metal_flat->copyTo(out);
        return out;
    } else if (
            auto metal_tq = dynamic_cast<const MetalIndexTurboQuantMSE*>(
                    index)) {
        auto* out = new IndexTurboQuantMSE();
        metal_tq->copyTo(out);
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
            auto turbo = dynamic_cast<const IndexTurboQuantMSE*>(index)) {
        MetalIndexConfig config;
        config.device = device;
        return new MetalIndexTurboQuantMSE(provider, turbo, config);
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
