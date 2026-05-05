#pragma once

#include <faiss/Clustering.h>
#include <faiss/clone_index.h>
#include <faiss/metal/MetalClonerOptions.h>
#include <faiss/metal/MetalIndexFlat.h>
#include <faiss/metal/MetalIndexScalarQuantizer.h>

namespace faiss {
namespace metal {

struct ToCPUCloner : faiss::Cloner {
    Index* clone_Index(const Index* index) override;
};

struct ToMetalCloner : faiss::Cloner, MetalClonerOptions {
    MetalResourcesProvider* provider;
    int device;

    ToMetalCloner(
            MetalResourcesProvider* provider,
            int device,
            const MetalClonerOptions& options);

    Index* clone_Index(const Index* index) override;
};

faiss::Index* index_metal_to_cpu(const faiss::Index* metal_index);

faiss::Index* index_cpu_to_metal(
        MetalResourcesProvider* provider,
        int device,
        const faiss::Index* index,
        const MetalClonerOptions* options = nullptr);

} // namespace metal
} // namespace faiss
