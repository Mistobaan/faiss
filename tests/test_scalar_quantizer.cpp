/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include <faiss/impl/ScalarQuantizer.h>

namespace {

std::vector<float> make_normalized_vectors(size_t n, size_t d) {
    std::mt19937 rng(1234);
    std::normal_distribution<float> distrib(0.0f, 1.0f);
    std::vector<float> x(n * d);

    for (size_t i = 0; i < n; i++) {
        float* xi = x.data() + i * d;
        double norm2 = 0.0;
        for (size_t j = 0; j < d; j++) {
            xi[j] = distrib(rng);
            norm2 += double(xi[j]) * xi[j];
        }
        const float inv_norm =
                norm2 > 0.0 ? 1.0f / std::sqrt(norm2) : 1.0f;
        for (size_t j = 0; j < d; j++) {
            xi[j] *= inv_norm;
        }
    }

    return x;
}

float mean_squared_error(
        const std::vector<float>& x,
        const std::vector<float>& y) {
    EXPECT_EQ(x.size(), y.size());
    double err = 0.0;
    for (size_t i = 0; i < x.size(); i++) {
        const double diff = double(x[i]) - y[i];
        err += diff * diff;
    }
    return err / x.size();
}

void check_tqmse_roundtrip(size_t d, faiss::ScalarQuantizer::QuantizerType qtype) {
    const size_t n = 128;
    std::vector<float> x = make_normalized_vectors(n, d);
    faiss::ScalarQuantizer sq(d, qtype);

    sq.train(n, x.data());

    std::vector<uint8_t> codes(sq.code_size * n, 0);
    sq.compute_codes(x.data(), codes.data(), n);

    std::vector<float> decoded(n * d);
    sq.decode(codes.data(), decoded.data(), n);

    for (float v : decoded) {
        EXPECT_TRUE(std::isfinite(v));
        EXPECT_LE(v, 1.0f);
        EXPECT_GE(v, -1.0f);
    }
}

} // namespace

TEST(ScalarQuantizer, RSQuantilesClamping) {
    int d = 8;
    int n = 100;

    std::vector<float> x(d * n);
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = static_cast<float>(i % 100);
    }

    faiss::ScalarQuantizer sq(d, faiss::ScalarQuantizer::QT_8bit);
    sq.rangestat = faiss::ScalarQuantizer::RS_quantiles;

    sq.rangestat_arg = 0.05f;
    ASSERT_NO_THROW(sq.train(n, x.data()));

    sq.rangestat_arg = 0.0f;
    ASSERT_NO_THROW(sq.train(n, x.data()));

    sq.rangestat_arg = -0.1f;
    ASSERT_NO_THROW(sq.train(n, x.data()));

    sq.rangestat_arg = 0.8f;
    ASSERT_NO_THROW(sq.train(n, x.data()));

    sq.rangestat_arg = 0.5f;
    ASSERT_NO_THROW(sq.train(n, x.data()));
}

TEST(ScalarQuantizer, RSQuantilesOddSize) {
    int d = 4;
    int n = 5;

    std::vector<float> x(d * n);
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = static_cast<float>(i);
    }

    faiss::ScalarQuantizer sq(d, faiss::ScalarQuantizer::QT_8bit);
    sq.rangestat = faiss::ScalarQuantizer::RS_quantiles;

    sq.rangestat_arg = 0.4f;
    ASSERT_NO_THROW(sq.train(n, x.data()));

    sq.rangestat_arg = 0.5f;
    ASSERT_NO_THROW(sq.train(n, x.data()));

    sq.rangestat_arg = 0.6f;
    ASSERT_NO_THROW(sq.train(n, x.data()));
}

TEST(ScalarQuantizer, RSQuantilesValidRange) {
    int d = 8;
    int n = 100;

    std::vector<float> x(d * n);
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = static_cast<float>(i);
    }

    faiss::ScalarQuantizer sq(d, faiss::ScalarQuantizer::QT_8bit);
    sq.rangestat = faiss::ScalarQuantizer::RS_quantiles;
    sq.rangestat_arg = 0.1f;

    sq.train(n, x.data());

    std::vector<uint8_t> codes(sq.code_size * n);
    ASSERT_NO_THROW(sq.compute_codes(x.data(), codes.data(), n));

    std::vector<float> decoded(d * n);
    ASSERT_NO_THROW(sq.decode(codes.data(), decoded.data(), n));
}

TEST(ScalarQuantizer, RSQuantilesSmallDataset) {
    int d = 2;
    int n = 2;

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};

    faiss::ScalarQuantizer sq(d, faiss::ScalarQuantizer::QT_8bit);
    sq.rangestat = faiss::ScalarQuantizer::RS_quantiles;
    sq.rangestat_arg = 0.1f;

    ASSERT_NO_THROW(sq.train(n, x.data()));
}

TEST(ScalarQuantizer, TQMSEEncodeDecode) {
    check_tqmse_roundtrip(32, faiss::ScalarQuantizer::QT_tqmse_4bit);
    check_tqmse_roundtrip(32, faiss::ScalarQuantizer::QT_tqmse_8bit);
}

TEST(ScalarQuantizer, TQMSEAccuracyOrdering) {
    const size_t d = 32;
    const size_t n = 256;
    std::vector<float> x = make_normalized_vectors(n, d);

    faiss::ScalarQuantizer sq4(d, faiss::ScalarQuantizer::QT_tqmse_4bit);
    faiss::ScalarQuantizer sq8(d, faiss::ScalarQuantizer::QT_tqmse_8bit);
    sq4.train(n, x.data());
    sq8.train(n, x.data());

    std::vector<uint8_t> codes4(sq4.code_size * n, 0);
    std::vector<uint8_t> codes8(sq8.code_size * n, 0);
    sq4.compute_codes(x.data(), codes4.data(), n);
    sq8.compute_codes(x.data(), codes8.data(), n);

    std::vector<float> decoded4(n * d);
    std::vector<float> decoded8(n * d);
    sq4.decode(codes4.data(), decoded4.data(), n);
    sq8.decode(codes8.data(), decoded8.data(), n);

    EXPECT_LT(mean_squared_error(x, decoded8), mean_squared_error(x, decoded4));
}

TEST(ScalarQuantizer, TQMSENonSimdDims) {
    check_tqmse_roundtrip(7, faiss::ScalarQuantizer::QT_tqmse_4bit);
    check_tqmse_roundtrip(33, faiss::ScalarQuantizer::QT_tqmse_8bit);
}
