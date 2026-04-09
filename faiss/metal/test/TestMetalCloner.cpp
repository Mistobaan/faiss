#include <gtest/gtest.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexTurboQuant.h>
#include <faiss/metal/MetalCloner.h>
#include <faiss/metal/MetalIndexFlat.h>
#include <faiss/metal/MetalIndexTurboQuant.h>
#include <faiss/metal/StandardMetalResources.h>
#include <faiss/utils/random.h>

#include <memory>
#include <vector>

namespace {

std::vector<float> make_data(size_t n, size_t d, int64_t seed) {
    std::vector<float> x(n * d);
    faiss::float_randn(x.data(), x.size(), seed);
    return x;
}

} // namespace

TEST(TestMetalCloner, FlatRoundTrip) {
    constexpr int d = 16;
    auto xb = make_data(64, d, 123);

    faiss::IndexFlatL2 cpu(d);
    cpu.add(64, xb.data());

    faiss::metal::StandardMetalResources resources;
    std::unique_ptr<faiss::Index> metal(
            faiss::metal::index_cpu_to_metal(&resources, 0, &cpu));
    ASSERT_NE(dynamic_cast<faiss::metal::MetalIndexFlat*>(metal.get()), nullptr);

    std::unique_ptr<faiss::Index> roundtrip(
            faiss::metal::index_metal_to_cpu(metal.get()));
    auto* flat = dynamic_cast<faiss::IndexFlat*>(roundtrip.get());
    ASSERT_NE(flat, nullptr);
    EXPECT_EQ(flat->ntotal, cpu.ntotal);
}

TEST(TestMetalCloner, TurboQuantRoundTrip) {
    constexpr int d = 32;
    auto xb = make_data(64, d, 321);

    faiss::IndexTurboQuantMSE cpu(d, 2);
    cpu.add(64, xb.data());

    faiss::metal::StandardMetalResources resources;
    std::unique_ptr<faiss::Index> metal(
            faiss::metal::index_cpu_to_metal(&resources, 0, &cpu));
    ASSERT_NE(
            dynamic_cast<faiss::metal::MetalIndexTurboQuantMSE*>(metal.get()),
            nullptr);

    std::unique_ptr<faiss::Index> roundtrip(
            faiss::metal::index_metal_to_cpu(metal.get()));
    auto* turbo = dynamic_cast<faiss::IndexTurboQuantMSE*>(roundtrip.get());
    ASSERT_NE(turbo, nullptr);
    EXPECT_EQ(turbo->ntotal, cpu.ntotal);
    EXPECT_EQ(turbo->codes.size(), cpu.codes.size());
}
