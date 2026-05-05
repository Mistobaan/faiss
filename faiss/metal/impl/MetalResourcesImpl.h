#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace faiss {
namespace metal {
namespace impl {

struct KernelArg {
    enum class Type {
        Buffer,
        Bytes,
    };

    Type type = Type::Bytes;
    size_t index = 0;
    const void* data = nullptr;
    size_t size = 0;

    static KernelArg buffer(size_t index, const void* handle) {
        return KernelArg{Type::Buffer, index, handle, 0};
    }

    static KernelArg bytes(size_t index, const void* data, size_t size) {
        return KernelArg{Type::Bytes, index, data, size};
    }
};

class MetalResourcesImpl;

class MetalBuffer {
   public:
    MetalBuffer() = default;
    MetalBuffer(
            std::shared_ptr<MetalResourcesImpl> owner,
            void* handle,
            size_t size);
    MetalBuffer(MetalBuffer&& other) noexcept;
    MetalBuffer& operator=(MetalBuffer&& other) noexcept;
    MetalBuffer(const MetalBuffer&) = delete;
    MetalBuffer& operator=(const MetalBuffer&) = delete;
    ~MetalBuffer();

    bool valid() const;
    void reset();

    size_t size() const;
    void* handle() const;
    void* contents();
    const void* contents() const;

   private:
    std::shared_ptr<MetalResourcesImpl> owner_;
    void* handle_ = nullptr;
    size_t size_ = 0;
};

class MetalResourcesImpl
        : public std::enable_shared_from_this<MetalResourcesImpl> {
   public:
    MetalResourcesImpl();
    ~MetalResourcesImpl();

    void initializeForDevice(int device);
    int getDevice() const;
    bool isAppleSilicon() const;

    MetalBuffer makeBuffer(size_t size) const;
    void copyBuffer(
            const MetalBuffer& src,
            size_t src_offset,
            const MetalBuffer& dst,
            size_t dst_offset,
            size_t size) const;

    void matrixMultiply(
            const MetalBuffer& left,
            const MetalBuffer& right,
            const MetalBuffer& out,
            size_t rows,
            size_t cols,
            size_t inner,
            bool transpose_left = false,
            bool transpose_right = false) const;

    void runKernel(
            const std::string& name,
            size_t count,
            const std::vector<KernelArg>& args) const;

    void releaseBuffer(void* handle) const;
    void* bufferContents(void* handle) const;

   private:
    struct State;
    std::shared_ptr<State> state_;
};

} // namespace impl
} // namespace metal
} // namespace faiss
