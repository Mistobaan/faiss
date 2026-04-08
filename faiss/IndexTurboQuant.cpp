/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexTurboQuant.h>

#include <faiss/impl/FaissAssert.h>

namespace faiss {

IndexTurboQuantMSE::IndexTurboQuantMSE(
        idx_t d_in,
        size_t nbits,
        MetricType metric,
        uint32_t seed,
        bool store_norm)
        : IndexFlatCodes(0, d_in, metric),
          tqmse(d_in, nbits, seed, store_norm) {
    FAISS_THROW_IF_NOT_FMT(
            nbits > 0,
            "IndexTurboQuantMSE requires nbits >= 1 (got %zu)",
            nbits);
    tqmse.initialize();
    code_size = tqmse.code_size;
    is_trained = true;
}

IndexTurboQuantMSE::IndexTurboQuantMSE()
        : IndexFlatCodes(), tqmse() {}

void IndexTurboQuantMSE::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_FMT(
            tqmse.nbits > 0,
            "IndexTurboQuantMSE requires nbits >= 1 (got %zu)",
            tqmse.nbits);
    tqmse.train(n, x);
    code_size = tqmse.code_size;
    is_trained = true;
}

void IndexTurboQuantMSE::sa_encode(idx_t n, const float* x, uint8_t* bytes)
        const {
    FAISS_THROW_IF_NOT(is_trained);
    tqmse.compute_codes(x, bytes, n);
}

void IndexTurboQuantMSE::sa_decode(idx_t n, const uint8_t* bytes, float* x)
        const {
    FAISS_THROW_IF_NOT(is_trained);
    tqmse.decode(bytes, x, n);
}

void IndexTurboQuantMSE::check_compatible_for_merge(
        const Index& otherIndex) const {
    IndexFlatCodes::check_compatible_for_merge(otherIndex);
    const auto* other = dynamic_cast<const IndexTurboQuantMSE*>(&otherIndex);
    FAISS_THROW_IF_NOT(other);
    tqmse.check_identical(other->tqmse);
}

IndexTurboQuantProd::IndexTurboQuantProd(
        idx_t d_in,
        size_t nbits,
        MetricType metric,
        uint32_t seed,
        bool store_norm)
        : IndexFlatCodes(0, d_in, metric),
          tqprod(d_in, nbits, seed, store_norm) {
    tqprod.initialize();
    code_size = tqprod.code_size;
    is_trained = true;
}

IndexTurboQuantProd::IndexTurboQuantProd()
        : IndexFlatCodes(), tqprod() {}

void IndexTurboQuantProd::train(idx_t n, const float* x) {
    tqprod.train(n, x);
    code_size = tqprod.code_size;
    is_trained = true;
}

void IndexTurboQuantProd::sa_encode(idx_t n, const float* x, uint8_t* bytes)
        const {
    FAISS_THROW_IF_NOT(is_trained);
    tqprod.compute_codes(x, bytes, n);
}

void IndexTurboQuantProd::sa_decode(idx_t n, const uint8_t* bytes, float* x)
        const {
    FAISS_THROW_IF_NOT(is_trained);
    tqprod.decode(bytes, x, n);
}

void IndexTurboQuantProd::check_compatible_for_merge(
        const Index& otherIndex) const {
    IndexFlatCodes::check_compatible_for_merge(otherIndex);
    const auto* other = dynamic_cast<const IndexTurboQuantProd*>(&otherIndex);
    FAISS_THROW_IF_NOT(other);
    tqprod.check_identical(other->tqprod);
}

} // namespace faiss
