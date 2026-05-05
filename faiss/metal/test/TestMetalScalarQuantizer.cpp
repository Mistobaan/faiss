#include <gtest/gtest.h>

#include <faiss/IndexTurboQuant.h>
#include <faiss/metal/MetalIndexTurboQuant.h>
#include <faiss/metal/StandardMetalResources.h>
#include <faiss/utils/random.h>

#include <cmath>
#include <vector>

namespace {

std::vector<float> make_unit_vectors(size_t n, size_t d, int64_t seed) {
    std::vector<float> x(n * d);
    faiss::float_randn(x.data(), x.size(), seed);

    for (size_t i = 0; i < n; i++) {
        float norm2 = 0.0f;
        for (size_t j = 0; j < d; j++) {
            norm2 += x[i * d + j] * x[i * d + j];
        }
        const float inv = 1.0f / std::sqrt(norm2);
        for (size_t j = 0; j < d; j++) {
            x[i * d + j] *= inv;
        }
    }
    return x;
}

double mean_squared_error(
        const std::vector<float>& x,
        const std::vector<float>& y) {
    double err = 0.0;
    for (size_t i = 0; i < x.size(); i++) {
        const double diff = double(x[i]) - y[i];
        err += diff * diff;
    }
    return err / x.size();
}

} // namespace

TEST(TestMetalTurboQuant, ZeroVectorRoundTrip) {
    faiss::metal::StandardMetalResources resources;
    faiss::metal::MetalIndexTurboQuantMSE index(&resources, 32, 2);

    std::vector<float> x(32, 0.0f);
    index.add(1, x.data());

    std::vector<float> decoded(32);
    index.reconstruct(0, decoded.data());
    for (float v : decoded) {
        EXPECT_FLOAT_EQ(v, 0.0f);
    }
}

TEST(TestMetalTurboQuant, ReconstructionImprovesWithMoreBits) {
    constexpr size_t d = 64;
    constexpr size_t n = 256;

    auto x = make_unit_vectors(n, d, 1234);
    std::vector<float> decoded(n * d);

    faiss::metal::StandardMetalResources resources;
    faiss::metal::MetalIndexTurboQuantMSE one(&resources, d, 1);
    faiss::metal::MetalIndexTurboQuantMSE two(&resources, d, 2);
    faiss::metal::MetalIndexTurboQuantMSE four(&resources, d, 4);

    one.add(n, x.data());
    one.reconstruct_n(0, n, decoded.data());
    const double err1 = mean_squared_error(x, decoded);

    two.add(n, x.data());
    two.reconstruct_n(0, n, decoded.data());
    const double err2 = mean_squared_error(x, decoded);

    four.add(n, x.data());
    four.reconstruct_n(0, n, decoded.data());
    const double err4 = mean_squared_error(x, decoded);

    EXPECT_GT(err1, err2);
    EXPECT_GT(err2, err4);
}

TEST(TestMetalTurboQuant, CopyParityAndSearchParity) {
    constexpr int d = 32;
    constexpr int nb = 96;
    constexpr int nq = 6;
    constexpr int k = 5;

    auto xb = make_unit_vectors(nb, d, 3333);
    auto xq = make_unit_vectors(nq, d, 4444);

    faiss::IndexTurboQuantMSE cpu(d, 2, faiss::METRIC_L2, 12345, true);
    cpu.add(nb, xb.data());

    faiss::metal::StandardMetalResources resources;
    faiss::metal::MetalIndexTurboQuantMSE metal(&resources, &cpu);

    std::vector<float> cpu_distances(nq * k);
    std::vector<faiss::idx_t> cpu_labels(nq * k);
    std::vector<float> metal_distances(nq * k);
    std::vector<faiss::idx_t> metal_labels(nq * k);

    cpu.search(nq, xq.data(), k, cpu_distances.data(), cpu_labels.data());
    metal.search(nq, xq.data(), k, metal_distances.data(), metal_labels.data());

    for (int i = 0; i < nq * k; i++) {
        EXPECT_EQ(cpu_labels[i], metal_labels[i]);
        if (cpu_labels[i] >= 0) {
            EXPECT_NEAR(cpu_distances[i], metal_distances[i], 1e-4);
        }
    }

    faiss::IndexTurboQuantMSE roundtrip;
    metal.copyTo(&roundtrip);

    EXPECT_EQ(roundtrip.code_size, cpu.code_size);
    EXPECT_EQ(roundtrip.ntotal, cpu.ntotal);
    EXPECT_EQ(roundtrip.codes.size(), cpu.codes.size());
    EXPECT_EQ(roundtrip.tq.nbits, cpu.tq.nbits);
    EXPECT_EQ(roundtrip.tq.seed, cpu.tq.seed);
    EXPECT_EQ(roundtrip.tq.store_norm, cpu.tq.store_norm);
    roundtrip.tq.check_identical(cpu.tq);
    for (size_t i = 0; i < cpu.codes.size(); i++) {
        EXPECT_EQ(roundtrip.codes[i], cpu.codes[i]);
    }
}
