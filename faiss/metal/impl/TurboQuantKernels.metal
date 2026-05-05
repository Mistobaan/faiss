#include <metal_stdlib>

using namespace metal;

struct NormalizeArgs {
    uint d;
    uint store_norm;
};

struct TurboQuantArgs {
    uint d;
    uint nbits;
    uint packed_code_size;
    uint code_size;
    uint prefix_bytes;
};

struct ScaleArgs {
    uint d;
    uint out_row_offset;
};

static inline uint read_bits(const device uchar* code, uint start_bit, uint nbit) {
    uint result = 0;
    uint bits_read = 0;
    uint bit = start_bit;

    while (bits_read < nbit) {
        const uint byte_index = bit >> 3;
        const uint shift = bit & 7u;
        const uint available = min(8u - shift, nbit - bits_read);
        const uint mask = (1u << available) - 1u;
        result |= ((uint(code[byte_index]) >> shift) & mask) << bits_read;
        bits_read += available;
        bit += available;
    }
    return result;
}

static inline void write_bits(device uchar* code, uint start_bit, uint value, uint nbit) {
    uint bits_written = 0;
    uint bit = start_bit;
    uint remaining = value;

    while (bits_written < nbit) {
        const uint byte_index = bit >> 3;
        const uint shift = bit & 7u;
        const uint available = min(8u - shift, nbit - bits_written);
        const uint mask = (1u << available) - 1u;
        code[byte_index] |= uchar((remaining & mask) << shift);
        remaining >>= available;
        bits_written += available;
        bit += available;
    }
}

kernel void normalize_vectors(
        const device float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        device float* norms [[buffer(2)]],
        constant NormalizeArgs& args [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    const device float* x = input + gid * args.d;
    device float* y = output + gid * args.d;

    float norm = 1.0f;
    if (args.store_norm != 0u) {
        float norm2 = 0.0f;
        for (uint i = 0; i < args.d; ++i) {
            norm2 += x[i] * x[i];
        }
        norm = sqrt(norm2);
    }
    norms[gid] = norm;

    if (args.store_norm != 0u && norm == 0.0f) {
        for (uint i = 0; i < args.d; ++i) {
            y[i] = 0.0f;
        }
        return;
    }

    const float inv = args.store_norm != 0u ? 1.0f / norm : 1.0f;
    for (uint i = 0; i < args.d; ++i) {
        y[i] = x[i] * inv;
    }
}

kernel void quantize_pack(
        const device float* rotated [[buffer(0)]],
        const device float* norms [[buffer(1)]],
        const device float* boundaries [[buffer(2)]],
        device uchar* codes [[buffer(3)]],
        constant TurboQuantArgs& args [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    const device float* x = rotated + gid * args.d;
    device uchar* code = codes + gid * args.code_size;

    for (uint i = 0; i < args.code_size; ++i) {
        code[i] = 0;
    }

    const float norm = norms[gid];
    if (args.prefix_bytes == 4u) {
        const uint bits = as_type<uint>(norm);
        code[0] = uchar(bits & 0xffu);
        code[1] = uchar((bits >> 8) & 0xffu);
        code[2] = uchar((bits >> 16) & 0xffu);
        code[3] = uchar((bits >> 24) & 0xffu);
    }

    if (args.prefix_bytes == 4u && norm == 0.0f) {
        return;
    }

    const uint centroid_count = 1u << args.nbits;
    uint bit_offset = 0;
    for (uint dim = 0; dim < args.d; ++dim) {
        uint idx = 0;
        while (idx + 1u < centroid_count && x[dim] >= boundaries[idx]) {
            ++idx;
        }
        write_bits(code + args.prefix_bytes, bit_offset, idx, args.nbits);
        bit_offset += args.nbits;
    }
}

kernel void unpack_codes(
        const device uchar* codes [[buffer(0)]],
        device float* norms [[buffer(1)]],
        const device float* centroids [[buffer(2)]],
        device float* rotated [[buffer(3)]],
        constant TurboQuantArgs& args [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    const device uchar* code = codes + gid * args.code_size;
    device float* out = rotated + gid * args.d;

    float norm = 1.0f;
    if (args.prefix_bytes == 4u) {
        const uint bits = uint(code[0]) | (uint(code[1]) << 8) |
                (uint(code[2]) << 16) | (uint(code[3]) << 24);
        norm = as_type<float>(bits);
    }
    norms[gid] = norm;

    if (args.prefix_bytes == 4u && norm == 0.0f) {
        for (uint i = 0; i < args.d; ++i) {
            out[i] = 0.0f;
        }
        return;
    }

    uint bit_offset = 0;
    for (uint dim = 0; dim < args.d; ++dim) {
        const uint idx = read_bits(code + args.prefix_bytes, bit_offset, args.nbits);
        out[dim] = centroids[idx];
        bit_offset += args.nbits;
    }
}

kernel void rescale_vectors(
        const device float* input [[buffer(0)]],
        const device float* norms [[buffer(1)]],
        device float* output [[buffer(2)]],
        constant ScaleArgs& args [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    const device float* x = input + gid * args.d;
    const float norm = norms[gid];
    device float* out = output + (gid + args.out_row_offset) * args.d;

    if (norm == 0.0f) {
        for (uint i = 0; i < args.d; ++i) {
            out[i] = 0.0f;
        }
        return;
    }

    for (uint i = 0; i < args.d; ++i) {
        out[i] = x[i] * norm;
    }
}
