/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexFlatCodes.h>
#include <faiss/impl/TurboQuantizer.h>

namespace faiss {

/** Flat index backed by the MSE-optimized TurboQuant codec. */
struct IndexTurboQuantMSE : IndexFlatCodes {
    TurboQuantMSEQuantizer tqmse;

    IndexTurboQuantMSE(
            idx_t d,
            size_t nbits,
            MetricType metric = METRIC_L2,
            uint32_t seed = 12345,
            bool store_norm = true);

    IndexTurboQuantMSE();

    void train(idx_t n, const float* x) override;

    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    void check_compatible_for_merge(const Index& otherIndex) const override;
};

/** Flat index backed by the inner-product-optimized TurboQuant codec. */
struct IndexTurboQuantProd : IndexFlatCodes {
    TurboQuantProdQuantizer tqprod;

    IndexTurboQuantProd(
            idx_t d,
            size_t nbits,
            MetricType metric = METRIC_INNER_PRODUCT,
            uint32_t seed = 12345,
            bool store_norm = true);

    IndexTurboQuantProd();

    void train(idx_t n, const float* x) override;

    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    void check_compatible_for_merge(const Index& otherIndex) const override;
};

} // namespace faiss
