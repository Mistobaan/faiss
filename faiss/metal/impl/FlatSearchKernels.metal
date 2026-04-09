#include <metal_stdlib>

using namespace metal;

struct SearchArgs {
    uint d;
    uint nb;
    uint k;
    uint metric;
};

static inline void init_results(
        device float* distances,
        device long* labels,
        uint k,
        bool larger_is_better) {
    const float init = larger_is_better ? -INFINITY : INFINITY;
    for (uint i = 0; i < k; ++i) {
        distances[i] = init;
        labels[i] = -1;
    }
}

static inline void insert_result(
        device float* distances,
        device long* labels,
        uint k,
        float value,
        long label,
        bool larger_is_better) {
    uint pos = k;
    for (uint i = 0; i < k; ++i) {
        if (larger_is_better ? (value > distances[i]) : (value < distances[i])) {
            pos = i;
            break;
        }
    }

    if (pos == k) {
        return;
    }

    for (uint i = k - 1; i > pos; --i) {
        distances[i] = distances[i - 1];
        labels[i] = labels[i - 1];
    }
    distances[pos] = value;
    labels[pos] = label;
}

kernel void flat_search_l2(
        const device float* queries [[buffer(0)]],
        const device float* database [[buffer(1)]],
        device float* out_distances [[buffer(2)]],
        device long* out_labels [[buffer(3)]],
        constant SearchArgs& args [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    const device float* query = queries + gid * args.d;
    device float* distances = out_distances + gid * args.k;
    device long* labels = out_labels + gid * args.k;
    init_results(distances, labels, args.k, false);

    for (uint j = 0; j < args.nb; ++j) {
        const device float* db = database + j * args.d;
        float distance = 0.0f;
        for (uint dim = 0; dim < args.d; ++dim) {
            const float diff = query[dim] - db[dim];
            distance += diff * diff;
        }
        insert_result(distances, labels, args.k, distance, long(j), false);
    }
}

kernel void flat_search_ip(
        const device float* queries [[buffer(0)]],
        const device float* database [[buffer(1)]],
        device float* out_distances [[buffer(2)]],
        device long* out_labels [[buffer(3)]],
        constant SearchArgs& args [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    const device float* query = queries + gid * args.d;
    device float* distances = out_distances + gid * args.k;
    device long* labels = out_labels + gid * args.k;
    init_results(distances, labels, args.k, true);

    for (uint j = 0; j < args.nb; ++j) {
        const device float* db = database + j * args.d;
        float distance = 0.0f;
        for (uint dim = 0; dim < args.d; ++dim) {
            distance += query[dim] * db[dim];
        }
        insert_result(distances, labels, args.k, distance, long(j), true);
    }
}
