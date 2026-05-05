#include <gtest/gtest.h>

#include <faiss/IndexFlat.h>
#include <faiss/metal/MetalIndexFlat.h>
#include <faiss/metal/StandardMetalResources.h>
#include <faiss/utils/random.h>

#include <cmath>
#include <vector>

namespace {

std::vector<float> make_data(size_t n, size_t d, int64_t seed) {
    std::vector<float> x(n * d);
    faiss::float_randn(x.data(), x.size(), seed);
    return x;
}

void expect_same_results(
        const std::vector<float>& ref_distances,
        const std::vector<faiss::idx_t>& ref_labels,
        const std::vector<float>& test_distances,
        const std::vector<faiss::idx_t>& test_labels) {
    ASSERT_EQ(ref_distances.size(), test_distances.size());
    ASSERT_EQ(ref_labels.size(), test_labels.size());
    for (size_t i = 0; i < ref_labels.size(); i++) {
        EXPECT_EQ(ref_labels[i], test_labels[i]);
        if (ref_labels[i] >= 0) {
            EXPECT_NEAR(ref_distances[i], test_distances[i], 1e-4);
        }
    }
}

} // namespace

TEST(TestMetalIndexFlat, L2ParityAndReconstruct) {
    constexpr int d = 32;
    constexpr int nb = 96;
    constexpr int nq = 7;
    constexpr int k = 6;

    auto xb = make_data(nb, d, 1234);
    auto xq = make_data(nq, d, 2345);

    faiss::IndexFlatL2 cpu(d);
    cpu.add(nb, xb.data());

    faiss::metal::StandardMetalResources resources;
    faiss::metal::MetalIndexFlatL2 metal(&resources, &cpu);

    std::vector<float> cpu_distances(nq * k);
    std::vector<faiss::idx_t> cpu_labels(nq * k);
    std::vector<float> metal_distances(nq * k);
    std::vector<faiss::idx_t> metal_labels(nq * k);

    cpu.search(nq, xq.data(), k, cpu_distances.data(), cpu_labels.data());
    metal.search(nq, xq.data(), k, metal_distances.data(), metal_labels.data());

    expect_same_results(cpu_distances, cpu_labels, metal_distances, metal_labels);

    std::vector<float> recon_cpu(d);
    std::vector<float> recon_metal(d);
    cpu.reconstruct(5, recon_cpu.data());
    metal.reconstruct(5, recon_metal.data());
    for (int i = 0; i < d; i++) {
        EXPECT_FLOAT_EQ(recon_cpu[i], recon_metal[i]);
    }
}

TEST(TestMetalIndexFlat, IPCopyToAndEmptySearch) {
    constexpr int d = 16;
    constexpr int nb = 48;
    constexpr int nq = 4;
    constexpr int k = 5;

    auto xb = make_data(nb, d, 3456);
    auto xq = make_data(nq, d, 4567);

    faiss::IndexFlatIP cpu(d);
    cpu.add(nb, xb.data());

    faiss::metal::StandardMetalResources resources;
    faiss::metal::MetalIndexFlatIP metal(&resources, &cpu);

    std::vector<float> cpu_distances(nq * k);
    std::vector<faiss::idx_t> cpu_labels(nq * k);
    std::vector<float> metal_distances(nq * k);
    std::vector<faiss::idx_t> metal_labels(nq * k);

    cpu.search(nq, xq.data(), k, cpu_distances.data(), cpu_labels.data());
    metal.search(nq, xq.data(), k, metal_distances.data(), metal_labels.data());
    expect_same_results(cpu_distances, cpu_labels, metal_distances, metal_labels);

    faiss::IndexFlatIP roundtrip(d);
    metal.copyTo(&roundtrip);
    ASSERT_EQ(roundtrip.ntotal, nb);

    std::vector<float> copied(nb * d);
    roundtrip.reconstruct_n(0, nb, copied.data());
    for (size_t i = 0; i < copied.size(); i++) {
        EXPECT_FLOAT_EQ(copied[i], xb[i]);
    }

    faiss::metal::MetalIndexFlatL2 empty(&resources, d);
    std::vector<float> empty_distances(nq * k);
    std::vector<faiss::idx_t> empty_labels(nq * k);
    empty.search(nq, xq.data(), k, empty_distances.data(), empty_labels.data());
    for (int i = 0; i < nq * k; i++) {
        EXPECT_EQ(empty_labels[i], -1);
        EXPECT_TRUE(std::isinf(empty_distances[i]));
    }
}
