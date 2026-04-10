/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <faiss/VectorTransform.h>
#include <faiss/impl/Quantizer.h>

namespace faiss {

#ifndef SWIG
/** Build the analytical TurboQuant MSE scalar codebook on [-1, 1].
 *
 * centroids.size() will be 2^nbits and boundaries.size() will be 2^nbits - 1.
 */
void build_TurboQuantMSECodebook(
        size_t d,
        size_t nbits,
        std::vector<float>& centroids,
        std::vector<float>& boundaries);
#endif

/** MSE-optimized TurboQuant quantizer from
 *
 *   TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
 *
 * This implementation follows the paper's random-rotation + optimal scalar
 * quantization recipe. For arbitrary-norm vectors it stores the vector norm in
 * the code and quantizes the normalized direction. A zero-bit configuration is
 * also supported for the degenerate all-zero codec used internally by
 * TurboQuantProd.
 */
struct TurboQuantMSEQuantizer : Quantizer {
    size_t nbits = 0;
    uint32_t seed = 12345;
    bool store_norm = true;
    bool is_trained = false;

    size_t packed_code_size = 0;

    RandomRotationMatrix rotation;

    /// sorted scalar codebook on [-1, 1]
    std::vector<float> centroids;

    /// Voronoi boundaries between consecutive centroids
    std::vector<float> boundaries;

    TurboQuantMSEQuantizer(
            size_t d = 0,
            size_t nbits = 0,
            uint32_t seed = 12345,
            bool store_norm = true);

    void train(size_t n, const float* x) override;

    void compute_codes(const float* x, uint8_t* codes, size_t n) const override;

    void decode(const uint8_t* codes, float* x, size_t n) const override;

    void validate() const;
    void set_derived_sizes();
    void init_rotation();
    void init_codebook();
    void update_boundaries();
    void initialize();
    void initialize_from_serialized_state();

    void check_identical(const TurboQuantMSEQuantizer& other) const;
};

/** Inner-product-optimized TurboQuant codec.
 *
 * TURBOQUANTprod uses a (b - 1)-bit TURBOQUANTmse stage and a 1-bit QJL stage
 * on the residual. For arbitrary-norm vectors it stores the original norm and
 * applies TURBOQUANTprod to the normalized direction.
 */
struct TurboQuantProdQuantizer : Quantizer {
    size_t nbits = 0;
    uint32_t seed = 12345;
    bool store_norm = true;
    bool is_trained = false;

    size_t qjl_code_size = 0;

    TurboQuantMSEQuantizer tqmse;
    LinearTransform qjl;

    TurboQuantProdQuantizer(
            size_t d = 0,
            size_t nbits = 0,
            uint32_t seed = 12345,
            bool store_norm = true);

    void train(size_t n, const float* x) override;

    void compute_codes(const float* x, uint8_t* codes, size_t n) const override;

    void decode(const uint8_t* codes, float* x, size_t n) const override;

    void validate() const;
    void set_derived_sizes();
    void init_qjl();
    void initialize();
    void initialize_from_serialized_state();

    void check_identical(const TurboQuantProdQuantizer& other) const;
};

} // namespace faiss
