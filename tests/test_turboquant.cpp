/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <faiss/IndexTurboQuant.h>
#include <faiss/index_io.h>
#include <faiss/impl/io.h>
#include <faiss/utils/random.h>

namespace {

std::vector<float> make_unit_vectors(size_t n, size_t d, int64_t seed) {
    std::vector<float> x(n * d);
    faiss::float_randn(x.data(), x.size(), seed);

    for (size_t i = 0; i < n; i++) {
        float norm2 = 0.0f;
        for (size_t j = 0; j < d; j++) {
            const float v = x[i * d + j];
            norm2 += v * v;
        }
        const float inv_norm = 1.0f / std::sqrt(norm2);
        for (size_t j = 0; j < d; j++) {
            x[i * d + j] *= inv_norm;
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

double inner_product(
        const float* x,
        const float* y,
        size_t d) {
    double sum = 0.0;
    for (size_t i = 0; i < d; i++) {
        sum += double(x[i]) * y[i];
    }
    return sum;
}

double mean_inner_product_squared_error(
        const std::vector<float>& x,
        const std::vector<float>& y,
        const std::vector<float>& q,
        size_t d) {
    const size_t n = x.size() / d;
    double err = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double diff = inner_product(x.data() + i * d, q.data(), d) -
                inner_product(y.data() + i * d, q.data(), d);
        err += diff * diff;
    }
    return err / n;
}

} // namespace

TEST(TurboQuantMSE, ZeroVectorRoundTrip) {
    faiss::IndexTurboQuantMSE index(32, 2, faiss::METRIC_L2, 12345, true);

    std::vector<float> x(32, 0.0f);
    index.add(1, x.data());

    std::vector<float> decoded(32);
    index.reconstruct(0, decoded.data());

    for (float v : decoded) {
        EXPECT_FLOAT_EQ(v, 0.0f);
    }
}

TEST(TurboQuantMSE, ReconstructionImprovesWithMoreBits) {
    constexpr size_t d = 64;
    constexpr size_t n = 256;

    const std::vector<float> x = make_unit_vectors(n, d, 1234);
    std::vector<float> decoded(n * d);

    faiss::IndexTurboQuantMSE index_1bit(d, 1, faiss::METRIC_L2, 12345, true);
    index_1bit.add(n, x.data());
    index_1bit.reconstruct_n(0, n, decoded.data());
    const double err_1bit = mean_squared_error(x, decoded);

    faiss::IndexTurboQuantMSE index_2bit(d, 2, faiss::METRIC_L2, 12345, true);
    index_2bit.add(n, x.data());
    index_2bit.reconstruct_n(0, n, decoded.data());
    const double err_2bit = mean_squared_error(x, decoded);

    faiss::IndexTurboQuantMSE index_4bit(d, 4, faiss::METRIC_L2, 12345, true);
    index_4bit.add(n, x.data());
    index_4bit.reconstruct_n(0, n, decoded.data());
    const double err_4bit = mean_squared_error(x, decoded);

    EXPECT_GT(err_1bit, err_2bit);
    EXPECT_GT(err_2bit, err_4bit);
}

TEST(TurboQuantProd, ZeroVectorRoundTrip) {
    faiss::IndexTurboQuantProd index(
            32, 2, faiss::METRIC_INNER_PRODUCT, 12345, true);

    std::vector<float> x(32, 0.0f);
    index.add(1, x.data());

    std::vector<float> decoded(32);
    index.reconstruct(0, decoded.data());

    for (float v : decoded) {
        EXPECT_FLOAT_EQ(v, 0.0f);
    }
}

TEST(TurboQuantProd, UnbiasedInnerProductEstimate) {
    constexpr size_t d = 256;
    constexpr size_t nseed = 128;

    const std::vector<float> x = make_unit_vectors(1, d, 1234);
    std::vector<float> decoded(d);

    double mse_mean = 0.0;
    double prod_mean = 0.0;

    for (size_t i = 0; i < nseed; i++) {
        faiss::IndexTurboQuantMSE index_mse(
                d, 1, faiss::METRIC_INNER_PRODUCT, 1000 + i, false);
        index_mse.add(1, x.data());
        index_mse.reconstruct(0, decoded.data());
        mse_mean += inner_product(x.data(), decoded.data(), d);

        faiss::IndexTurboQuantProd index_prod(
                d, 2, faiss::METRIC_INNER_PRODUCT, 1000 + i, false);
        index_prod.add(1, x.data());
        index_prod.reconstruct(0, decoded.data());
        prod_mean += inner_product(x.data(), decoded.data(), d);
    }

    mse_mean /= nseed;
    prod_mean /= nseed;

    EXPECT_LT(mse_mean, 0.8);
    EXPECT_NEAR(prod_mean, 1.0, 0.08);
}

TEST(TurboQuantProd, InnerProductErrorImprovesWithMoreBits) {
    constexpr size_t d = 128;
    constexpr size_t n = 256;

    const std::vector<float> x = make_unit_vectors(n, d, 4321);
    const std::vector<float> q = make_unit_vectors(1, d, 8765);
    std::vector<float> decoded(n * d);

    faiss::IndexTurboQuantProd index_1bit(
            d, 1, faiss::METRIC_INNER_PRODUCT, 12345, false);
    index_1bit.add(n, x.data());
    index_1bit.reconstruct_n(0, n, decoded.data());
    const double err_1bit =
            mean_inner_product_squared_error(x, decoded, q, d);

    faiss::IndexTurboQuantProd index_2bit(
            d, 2, faiss::METRIC_INNER_PRODUCT, 12345, false);
    index_2bit.add(n, x.data());
    index_2bit.reconstruct_n(0, n, decoded.data());
    const double err_2bit =
            mean_inner_product_squared_error(x, decoded, q, d);

    faiss::IndexTurboQuantProd index_4bit(
            d, 4, faiss::METRIC_INNER_PRODUCT, 12345, false);
    index_4bit.add(n, x.data());
    index_4bit.reconstruct_n(0, n, decoded.data());
    const double err_4bit =
            mean_inner_product_squared_error(x, decoded, q, d);

    EXPECT_GT(err_1bit, err_2bit);
    EXPECT_GT(err_2bit, err_4bit);
}

TEST(TurboQuantProd, SerializationRoundTrip) {
    constexpr size_t d = 64;
    constexpr size_t n = 32;

    const std::vector<float> x = make_unit_vectors(n, d, 999);

    faiss::IndexTurboQuantProd index(
            d, 3, faiss::METRIC_INNER_PRODUCT, 12345, true);
    index.add(n, x.data());

    faiss::VectorIOWriter writer;
    faiss::write_index(&index, &writer);

    faiss::VectorIOReader reader;
    reader.data = writer.data;

    std::unique_ptr<faiss::Index> loaded(faiss::read_index(&reader));
    auto* loaded_prod = dynamic_cast<faiss::IndexTurboQuantProd*>(loaded.get());
    ASSERT_NE(loaded_prod, nullptr);

    std::vector<float> decoded_a(d);
    std::vector<float> decoded_b(d);
    index.reconstruct(0, decoded_a.data());
    loaded_prod->reconstruct(0, decoded_b.data());

    for (size_t i = 0; i < d; i++) {
        EXPECT_FLOAT_EQ(decoded_a[i], decoded_b[i]);
    }
}
