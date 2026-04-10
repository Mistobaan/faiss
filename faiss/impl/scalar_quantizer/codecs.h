/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>

#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/impl/simdlib/simdlib_dispatch.h>
#include <faiss/utils/simd_levels.h>

namespace faiss {

namespace scalar_quantizer {

/*******************************************************************
 * Codec: converts between values in [0, 1] and an index in a code
 * array. The "i" parameter is the vector component index (not byte
 * index).
 */

template <SIMDLevel SL>
struct Codec8bit {};

template <>
struct Codec8bit<SIMDLevel::NONE> {
    static FAISS_ALWAYS_INLINE void encode_component(
            float x,
            uint8_t* code,
            size_t i) {
        code[i] = (int)(255 * x);
    }

    static FAISS_ALWAYS_INLINE float decode_component(
            const uint8_t* code,
            size_t i) {
        return (code[i] + 0.5f) / 255.0f;
    }
};

template <SIMDLevel SL>
struct Codec4bit {};

template <>
struct Codec4bit<SIMDLevel::NONE> {
    static FAISS_ALWAYS_INLINE void encode_component(
            float x,
            uint8_t* code,
            size_t i) {
        code[i / 2] |= (int)(x * 15.0) << ((i & 1) << 2);
    }

    static FAISS_ALWAYS_INLINE float decode_component(
            const uint8_t* code,
            size_t i) {
        return (((code[i / 2] >> ((i & 1) << 2)) & 0xf) + 0.5f) / 15.0f;
    }
};

template <SIMDLevel SL>
struct Codec6bit {};

template <>
struct Codec6bit<SIMDLevel::NONE> {
    static FAISS_ALWAYS_INLINE void encode_component(
            float x,
            uint8_t* code,
            size_t i) {
        int bits = (int)(x * 63.0);
        code += (i >> 2) * 3;
        switch (i & 3) {
            case 0:
                code[0] |= bits;
                break;
            case 1:
                code[0] |= bits << 6;
                code[1] |= bits >> 2;
                break;
            case 2:
                code[1] |= bits << 4;
                code[2] |= bits >> 4;
                break;
            case 3:
                code[2] |= bits << 2;
                break;
            default:
                break;
        }
    }

    static FAISS_ALWAYS_INLINE float decode_component(
            const uint8_t* code,
            size_t i) {
        uint8_t bits = 0;
        code += (i >> 2) * 3;
        switch (i & 3) {
            case 0:
                bits = code[0] & 0x3f;
                break;
            case 1:
                bits = code[0] >> 6;
                bits |= (code[1] & 0xf) << 2;
                break;
            case 2:
                bits = code[1] >> 4;
                bits |= (code[2] & 3) << 4;
                break;
            case 3:
                bits = code[2] >> 2;
                break;
            default:
                break;
        }
        return (bits + 0.5f) / 63.0f;
    }
};

template <int NBITS, SIMDLevel SL>
struct CodecTurboQuantMSE;

template <>
struct CodecTurboQuantMSE<4, SIMDLevel::NONE> {
    static constexpr size_t kCentroids = 16;

    static FAISS_ALWAYS_INLINE void encode_component(
            float x,
            const float* boundaries,
            uint8_t* code,
            size_t i) {
        const float clamped = std::max(-1.0f, std::min(1.0f, x));
        const size_t bucket =
                std::upper_bound(boundaries, boundaries + kCentroids - 1, clamped) -
                boundaries;
        code[i / 2] |= static_cast<uint8_t>(bucket) << ((i & 1) << 2);
    }

    static FAISS_ALWAYS_INLINE uint8_t decode_index(
            const uint8_t* code,
            size_t i) {
        return (code[i / 2] >> ((i & 1) << 2)) & 0xf;
    }
};

template <SIMDLevel SL>
struct CodecTurboQuantMSE<4, SL> : CodecTurboQuantMSE<4, SIMDLevel::NONE> {};

template <>
struct CodecTurboQuantMSE<8, SIMDLevel::NONE> {
    static constexpr size_t kCentroids = 256;

    static FAISS_ALWAYS_INLINE void encode_component(
            float x,
            const float* boundaries,
            uint8_t* code,
            size_t i) {
        const float clamped = std::max(-1.0f, std::min(1.0f, x));
        const size_t bucket =
                std::upper_bound(boundaries, boundaries + kCentroids - 1, clamped) -
                boundaries;
        code[i] = static_cast<uint8_t>(bucket);
    }

    static FAISS_ALWAYS_INLINE uint8_t decode_index(
            const uint8_t* code,
            size_t i) {
        return code[i];
    }
};

template <SIMDLevel SL>
struct CodecTurboQuantMSE<8, SL> : CodecTurboQuantMSE<8, SIMDLevel::NONE> {};

} // namespace scalar_quantizer
} // namespace faiss
