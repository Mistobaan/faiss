/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/TurboQuantizer.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include <omp.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/random.h>

namespace faiss {

namespace {

constexpr size_t kTurboQuantMaxBits = 8;
constexpr size_t kTurboQuantProdMaxBits = kTurboQuantMaxBits + 1;
constexpr size_t kTurboQuantGridMin = 1 << 15;
constexpr size_t kTurboQuantGridPerCentroid = 512;
constexpr int kTurboQuantMaxIter = 100;
constexpr double kTurboQuantTol = 1e-8;
constexpr double kTurboQuantPi = 3.14159265358979323846;

float vector_norm(const float* x, size_t d) {
    double norm2 = 0.0;
    for (size_t i = 0; i < d; i++) {
        norm2 += double(x[i]) * x[i];
    }
    return std::sqrt(norm2);
}

} // namespace

void build_TurboQuantMSECodebook(
        size_t d,
        size_t nbits,
        std::vector<float>& centroids,
        std::vector<float>& boundaries) {
    FAISS_THROW_IF_NOT_FMT(
            nbits <= kTurboQuantMaxBits,
            "invalid TurboQuant nbits %zu (must be in [0, %zu])",
            nbits,
            kTurboQuantMaxBits);

    if (nbits == 0) {
        centroids.clear();
        boundaries.clear();
        return;
    }

    const size_t k = size_t(1) << nbits;

    if (d == 1) {
        centroids.resize(k);
        for (size_t i = 0; i < k; i++) {
            centroids[i] = i < k / 2 ? -1.0f : 1.0f;
        }
        boundaries.resize(k - 1);
        for (size_t i = 0; i + 1 < k; i++) {
            boundaries[i] = 0.5f * (centroids[i] + centroids[i + 1]);
        }
        return;
    }

    const size_t ngrid =
            std::max(kTurboQuantGridMin, k * kTurboQuantGridPerCentroid);
    const double step = 2.0 / ngrid;
    const double alpha = 0.5 * (double(d) - 3.0);

    std::vector<double> xs(ngrid);
    std::vector<double> prefix_w(ngrid + 1, 0.0);
    std::vector<double> prefix_wx(ngrid + 1, 0.0);

    for (size_t i = 0; i < ngrid; i++) {
        const double x = -1.0 + (i + 0.5) * step;
        const double one_minus_x2 = std::max(0.0, 1.0 - x * x);
        double w;
        if (alpha == 0.0) {
            w = 1.0;
        } else {
            w = std::pow(one_minus_x2, alpha);
        }
        if (!std::isfinite(w) || w < 0.0) {
            w = 0.0;
        }
        xs[i] = x;
        prefix_w[i + 1] = prefix_w[i] + w;
        prefix_wx[i + 1] = prefix_wx[i] + w * x;
    }

    auto range_mean = [&](size_t i0, size_t i1, double fallback) {
        const double w = prefix_w[i1] - prefix_w[i0];
        if (w <= 0.0) {
            return fallback;
        }
        return (prefix_wx[i1] - prefix_wx[i0]) / w;
    };

    const double total_w = prefix_w.back();
    std::vector<size_t> cuts(k + 1, 0);
    cuts[k] = ngrid;

    for (size_t i = 1; i < k; i++) {
        const double target = total_w * i / k;
        cuts[i] = std::lower_bound(prefix_w.begin(), prefix_w.end(), target) -
                prefix_w.begin();
        cuts[i] = std::min(cuts[i], ngrid);
    }

    std::vector<double> centroids_d(k);
    for (size_t i = 0; i < k; i++) {
        const double left = -1.0 + 2.0 * i / k;
        const double right = -1.0 + 2.0 * (i + 1) / k;
        centroids_d[i] =
                range_mean(cuts[i], cuts[i + 1], 0.5 * (left + right));
    }

    std::vector<double> boundaries_d(k > 0 ? k - 1 : 0);

    for (int iter = 0; iter < kTurboQuantMaxIter; iter++) {
        for (size_t i = 0; i + 1 < k; i++) {
            boundaries_d[i] = 0.5 * (centroids_d[i] + centroids_d[i + 1]);
        }

        cuts[0] = 0;
        cuts[k] = ngrid;
        for (size_t i = 1; i < k; i++) {
            cuts[i] = std::upper_bound(
                              xs.begin(), xs.end(), boundaries_d[i - 1]) -
                    xs.begin();
        }

        double max_delta = 0.0;
        for (size_t i = 0; i < k; i++) {
            const double left = i == 0 ? -1.0 : boundaries_d[i - 1];
            const double right = i + 1 == k ? 1.0 : boundaries_d[i];
            double c = range_mean(cuts[i], cuts[i + 1], 0.5 * (left + right));
            c = std::min(std::max(c, left), right);
            max_delta = std::max(max_delta, std::abs(c - centroids_d[i]));
            centroids_d[i] = c;
        }

        if (max_delta < kTurboQuantTol) {
            break;
        }
    }

    std::sort(centroids_d.begin(), centroids_d.end());

    centroids.resize(k);
    boundaries.resize(k - 1);
    for (size_t i = 0; i < k; i++) {
        centroids[i] = centroids_d[i];
    }
    for (size_t i = 0; i + 1 < k; i++) {
        boundaries[i] = 0.5f * (centroids[i] + centroids[i + 1]);
    }
}

TurboQuantMSEQuantizer::TurboQuantMSEQuantizer(
        size_t d_in,
        size_t nbits_in,
        uint32_t seed_in,
        bool store_norm_in)
        : Quantizer(d_in, 0),
          nbits(nbits_in),
          seed(seed_in),
          store_norm(store_norm_in),
          rotation(d_in, d_in) {
    set_derived_sizes();
}

void TurboQuantMSEQuantizer::validate() const {
    FAISS_THROW_IF_NOT_FMT(d > 0, "invalid TurboQuant dimension %zu", d);
    FAISS_THROW_IF_NOT_FMT(
            nbits <= kTurboQuantMaxBits,
            "invalid TurboQuant nbits %zu (must be in [0, %zu])",
            nbits,
            kTurboQuantMaxBits);
}

void TurboQuantMSEQuantizer::set_derived_sizes() {
    packed_code_size = (d * nbits + 7) / 8;
    code_size = packed_code_size + (store_norm ? sizeof(float) : 0);
}

void TurboQuantMSEQuantizer::init_rotation() {
    rotation = RandomRotationMatrix(d, d);
    rotation.init(seed);
}

void TurboQuantMSEQuantizer::update_boundaries() {
    boundaries.resize(centroids.empty() ? 0 : centroids.size() - 1);
    for (size_t i = 0; i + 1 < centroids.size(); i++) {
        boundaries[i] = 0.5f * (centroids[i] + centroids[i + 1]);
    }
}

void TurboQuantMSEQuantizer::init_codebook() {
    build_TurboQuantMSECodebook(d, nbits, centroids, boundaries);
}

void TurboQuantMSEQuantizer::initialize() {
    validate();
    set_derived_sizes();
    init_rotation();
    init_codebook();
    is_trained = true;
}

void TurboQuantMSEQuantizer::initialize_from_serialized_state() {
    validate();
    set_derived_sizes();
    FAISS_THROW_IF_NOT_FMT(
            centroids.size() == (nbits == 0 ? 0 : size_t(1) << nbits),
            "invalid TurboQuant centroid count %zu for nbits=%zu",
            centroids.size(),
            nbits);
    update_boundaries();
    init_rotation();
    is_trained = true;
}

void TurboQuantMSEQuantizer::train(size_t, const float*) {
    initialize();
}

void TurboQuantMSEQuantizer::compute_codes(
        const float* x,
        uint8_t* codes,
        size_t n) const {
    FAISS_THROW_IF_NOT(is_trained);

    const size_t prefix = store_norm ? sizeof(float) : 0;
    const size_t bs = 256;

#pragma omp parallel
    {
        std::vector<float> normalized(bs * d);
        std::vector<float> rotated(bs * d);
        std::vector<float> norms(bs);

#pragma omp for
        for (int64_t i0 = 0; i0 < static_cast<int64_t>(n); i0 += bs) {
            const size_t ni = std::min(bs, n - size_t(i0));

            for (size_t i = 0; i < ni; i++) {
                const float* xi = x + (i0 + i) * d;
                float* xn = normalized.data() + i * d;
                float norm = store_norm ? vector_norm(xi, d) : 1.0f;
                norms[i] = norm;
                if (norm > 0.0f) {
                    const float inv_norm = 1.0f / norm;
                    for (size_t j = 0; j < d; j++) {
                        xn[j] = xi[j] * inv_norm;
                    }
                } else {
                    std::fill(xn, xn + d, 0.0f);
                }
            }

            if (ni > 0) {
                rotation.apply_noalloc(ni, normalized.data(), rotated.data());
            }

            for (size_t i = 0; i < ni; i++) {
                uint8_t* code = codes + (i0 + i) * code_size;
                memset(code, 0, code_size);

                if (store_norm) {
                    memcpy(code, &norms[i], sizeof(float));
                }

                if (norms[i] == 0.0f || nbits == 0) {
                    continue;
                }

                BitstringWriter wr(code + prefix, packed_code_size);
                const float* yi = rotated.data() + i * d;
                for (size_t j = 0; j < d; j++) {
                    const size_t idx = std::upper_bound(
                                               boundaries.begin(),
                                               boundaries.end(),
                                               yi[j]) -
                            boundaries.begin();
                    wr.write(idx, nbits);
                }
            }
        }
    }
}

void TurboQuantMSEQuantizer::decode(
        const uint8_t* codes,
        float* x,
        size_t n) const {
    FAISS_THROW_IF_NOT(is_trained);

    const size_t prefix = store_norm ? sizeof(float) : 0;
    const size_t bs = 256;

#pragma omp parallel
    {
        std::vector<float> normalized(bs * d);
        std::vector<float> rotated(bs * d);
        std::vector<float> norms(bs);

#pragma omp for
        for (int64_t i0 = 0; i0 < static_cast<int64_t>(n); i0 += bs) {
            const size_t ni = std::min(bs, n - size_t(i0));

            for (size_t i = 0; i < ni; i++) {
                const uint8_t* code = codes + (i0 + i) * code_size;
                float norm = 1.0f;
                if (store_norm) {
                    memcpy(&norm, code, sizeof(float));
                }
                norms[i] = norm;

                float* yi = rotated.data() + i * d;
                if (norm == 0.0f || nbits == 0) {
                    std::fill(yi, yi + d, 0.0f);
                    continue;
                }

                BitstringReader rd(code + prefix, packed_code_size);
                for (size_t j = 0; j < d; j++) {
                    const uint64_t idx = rd.read(nbits);
                    yi[j] = centroids[idx];
                }
            }

            if (ni > 0) {
                rotation.transform_transpose(
                        ni, rotated.data(), normalized.data());
            }

            for (size_t i = 0; i < ni; i++) {
                float* xo = x + (i0 + i) * d;
                const float* xn = normalized.data() + i * d;
                if (norms[i] == 0.0f) {
                    std::fill(xo, xo + d, 0.0f);
                    continue;
                }
                for (size_t j = 0; j < d; j++) {
                    xo[j] = xn[j] * norms[i];
                }
            }
        }
    }
}

void TurboQuantMSEQuantizer::check_identical(
        const TurboQuantMSEQuantizer& other) const {
    FAISS_THROW_IF_NOT_MSG(d == other.d, "TurboQuant d mismatch");
    FAISS_THROW_IF_NOT_MSG(nbits == other.nbits, "TurboQuant nbits mismatch");
    FAISS_THROW_IF_NOT_MSG(seed == other.seed, "TurboQuant seed mismatch");
    FAISS_THROW_IF_NOT_MSG(
            store_norm == other.store_norm, "TurboQuant store_norm mismatch");
    FAISS_THROW_IF_NOT_MSG(
            centroids == other.centroids, "TurboQuant centroids mismatch");
}

TurboQuantProdQuantizer::TurboQuantProdQuantizer(
        size_t d_in,
        size_t nbits_in,
        uint32_t seed_in,
        bool store_norm_in)
        : Quantizer(d_in, 0),
          nbits(nbits_in),
          seed(seed_in),
          store_norm(store_norm_in),
          tqmse(d_in, nbits_in == 0 ? 0 : nbits_in - 1, seed_in, false),
          qjl(d_in, d_in, false) {
    set_derived_sizes();
}

void TurboQuantProdQuantizer::validate() const {
    FAISS_THROW_IF_NOT_FMT(d > 0, "invalid TurboQuantProd dimension %zu", d);
    FAISS_THROW_IF_NOT_FMT(
            nbits > 0 && nbits <= kTurboQuantProdMaxBits,
            "invalid TurboQuantProd nbits %zu (must be in [1, %zu])",
            nbits,
            kTurboQuantProdMaxBits);
    FAISS_THROW_IF_NOT_MSG(
            tqmse.d == d, "TurboQuantProd tqmse dimension mismatch");
    FAISS_THROW_IF_NOT_MSG(
            tqmse.nbits == nbits - 1,
            "TurboQuantProd tqmse bit-width mismatch");
}

void TurboQuantProdQuantizer::set_derived_sizes() {
    qjl_code_size = (d + 7) / 8;
    tqmse.d = d;
    tqmse.nbits = nbits == 0 ? 0 : nbits - 1;
    tqmse.seed = seed;
    tqmse.store_norm = false;
    tqmse.set_derived_sizes();
    code_size =
            (store_norm ? sizeof(float) : 0) + tqmse.code_size +
            sizeof(float) + qjl_code_size;
}

void TurboQuantProdQuantizer::init_qjl() {
    qjl = LinearTransform(d, d, false);
    qjl.A.resize(d * d);
    float_randn(qjl.A.data(), d * d, int64_t(seed) + 1);
    qjl.is_orthonormal = false;
    qjl.is_trained = true;
}

void TurboQuantProdQuantizer::initialize() {
    validate();
    set_derived_sizes();
    tqmse.initialize();
    init_qjl();
    is_trained = true;
}

void TurboQuantProdQuantizer::initialize_from_serialized_state() {
    validate();
    set_derived_sizes();
    tqmse.initialize_from_serialized_state();
    init_qjl();
    is_trained = true;
}

void TurboQuantProdQuantizer::train(size_t, const float*) {
    initialize();
}

void TurboQuantProdQuantizer::compute_codes(
        const float* x,
        uint8_t* codes,
        size_t n) const {
    FAISS_THROW_IF_NOT(is_trained);

    const size_t norm_offset = 0;
    const size_t mse_offset = store_norm ? sizeof(float) : 0;
    const size_t gamma_offset = mse_offset + tqmse.code_size;
    const size_t qjl_offset = gamma_offset + sizeof(float);
    const size_t bs = 256;

    std::vector<float> normalized(bs * d);
    std::vector<float> reconstructed(bs * d);
    std::vector<float> residual(bs * d);
    std::vector<float> projected(bs * d);
    std::vector<float> norms(bs);
    std::vector<float> residual_norms(bs);
    std::vector<uint8_t> mse_codes(bs * tqmse.code_size);

    for (size_t i0 = 0; i0 < n; i0 += bs) {
        const size_t ni = std::min(bs, n - i0);

        for (size_t i = 0; i < ni; i++) {
            const float* xi = x + (i0 + i) * d;
            float* xn = normalized.data() + i * d;
            float norm = store_norm ? vector_norm(xi, d) : 1.0f;
            norms[i] = norm;
            if (norm > 0.0f) {
                const float inv_norm = 1.0f / norm;
                for (size_t j = 0; j < d; j++) {
                    xn[j] = xi[j] * inv_norm;
                }
            } else {
                std::fill(xn, xn + d, 0.0f);
            }
        }

        if (tqmse.code_size > 0) {
            tqmse.compute_codes(normalized.data(), mse_codes.data(), ni);
            tqmse.decode(mse_codes.data(), reconstructed.data(), ni);
        } else {
            std::fill(reconstructed.begin(), reconstructed.begin() + ni * d, 0);
        }

        for (size_t i = 0; i < ni; i++) {
            const float* xn = normalized.data() + i * d;
            const float* xr = reconstructed.data() + i * d;
            float* rr = residual.data() + i * d;
            for (size_t j = 0; j < d; j++) {
                rr[j] = xn[j] - xr[j];
            }
            residual_norms[i] = vector_norm(rr, d);
        }

        if (ni > 0) {
            qjl.apply_noalloc(ni, residual.data(), projected.data());
        }

        for (size_t i = 0; i < ni; i++) {
            uint8_t* code = codes + (i0 + i) * code_size;
            memset(code, 0, code_size);

            if (store_norm) {
                memcpy(code + norm_offset, &norms[i], sizeof(float));
            }
            if (tqmse.code_size > 0) {
                memcpy(
                        code + mse_offset,
                        mse_codes.data() + i * tqmse.code_size,
                        tqmse.code_size);
            }
            memcpy(code + gamma_offset, &residual_norms[i], sizeof(float));

            if (residual_norms[i] == 0.0f) {
                continue;
            }

            BitstringWriter wr(code + qjl_offset, qjl_code_size);
            const float* qp = projected.data() + i * d;
            for (size_t j = 0; j < d; j++) {
                wr.write(qp[j] >= 0.0f ? 1 : 0, 1);
            }
        }
    }
}

void TurboQuantProdQuantizer::decode(
        const uint8_t* codes,
        float* x,
        size_t n) const {
    FAISS_THROW_IF_NOT(is_trained);

    const size_t norm_offset = 0;
    const size_t mse_offset = store_norm ? sizeof(float) : 0;
    const size_t gamma_offset = mse_offset + tqmse.code_size;
    const size_t qjl_offset = gamma_offset + sizeof(float);
    const size_t bs = 256;
    const float qjl_scale = std::sqrt(kTurboQuantPi / 2.0) / d;

    std::vector<float> reconstructed(bs * d);
    std::vector<float> qjl_signs(bs * d);
    std::vector<float> qjl_reconstruction(bs * d);
    std::vector<float> norms(bs);
    std::vector<float> residual_norms(bs);
    std::vector<uint8_t> mse_codes(bs * tqmse.code_size);

    for (size_t i0 = 0; i0 < n; i0 += bs) {
        const size_t ni = std::min(bs, n - i0);

        for (size_t i = 0; i < ni; i++) {
            const uint8_t* code = codes + (i0 + i) * code_size;
            float norm = 1.0f;
            if (store_norm) {
                memcpy(&norm, code + norm_offset, sizeof(float));
            }
            norms[i] = norm;
            memcpy(&residual_norms[i], code + gamma_offset, sizeof(float));

            if (tqmse.code_size > 0) {
                memcpy(
                        mse_codes.data() + i * tqmse.code_size,
                        code + mse_offset,
                        tqmse.code_size);
            }

            float* qs = qjl_signs.data() + i * d;
            BitstringReader rd(code + qjl_offset, qjl_code_size);
            for (size_t j = 0; j < d; j++) {
                qs[j] = rd.read(1) ? 1.0f : -1.0f;
            }
        }

        if (tqmse.code_size > 0) {
            tqmse.decode(mse_codes.data(), reconstructed.data(), ni);
        } else {
            std::fill(reconstructed.begin(), reconstructed.begin() + ni * d, 0);
        }
        if (ni > 0) {
            qjl.transform_transpose(
                    ni, qjl_signs.data(), qjl_reconstruction.data());
        }

        for (size_t i = 0; i < ni; i++) {
            float* xo = x + (i0 + i) * d;
            const float* xm = reconstructed.data() + i * d;
            const float* xq = qjl_reconstruction.data() + i * d;
            const float scale = qjl_scale * residual_norms[i] * norms[i];
            if (norms[i] == 0.0f) {
                std::fill(xo, xo + d, 0.0f);
                continue;
            }
            for (size_t j = 0; j < d; j++) {
                xo[j] = xm[j] * norms[i] + xq[j] * scale;
            }
        }
    }
}

void TurboQuantProdQuantizer::check_identical(
        const TurboQuantProdQuantizer& other) const {
    FAISS_THROW_IF_NOT_MSG(
            nbits == other.nbits, "TurboQuantProd nbits mismatch");
    FAISS_THROW_IF_NOT_MSG(seed == other.seed, "TurboQuantProd seed mismatch");
    FAISS_THROW_IF_NOT_MSG(d == other.d, "TurboQuantProd d mismatch");
    FAISS_THROW_IF_NOT_MSG(
            store_norm == other.store_norm,
            "TurboQuantProd store_norm mismatch");
    tqmse.check_identical(other.tqmse);
    qjl.check_identical(other.qjl);
}

} // namespace faiss
