#pragma once

namespace faiss {
namespace metal {

struct MetalClonerOptions {
    int device = 0;
    bool useFloat16 = false;
};

} // namespace metal
} // namespace faiss
