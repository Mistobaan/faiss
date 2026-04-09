#pragma once

#include <cstddef>
#include <cstdint>

namespace faiss {
namespace metal {
namespace impl {

extern const uint8_t kMetalLibraryData[];
extern const size_t kMetalLibrarySize;
extern const bool kMetalLibraryIsSource;

} // namespace impl
} // namespace metal
} // namespace faiss
