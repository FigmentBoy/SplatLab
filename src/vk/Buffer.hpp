#pragma once

#include "vk/VkCommon.hpp"

#include <memory>

class VulkanContext;

// A storage buffer with its own memory allocation. All buffers are usable as
// storage buffers, transfer source/destination and via device address.
class Buffer {
public:
    enum class Memory {
        DeviceLocal, // GPU-only; filled with upload() through a staging buffer
        Upload,      // host-visible, persistently mapped, CPU -> GPU
        Readback,    // host-visible (cached when possible), persistently mapped, GPU -> CPU
    };

    // `exportable` buffers get their own (dedicated) allocation that can be
    // shared with another API through exportHandle(); the context must have
    // been created with externalMemory.
    static std::unique_ptr<Buffer> create(VulkanContext& ctx, VkDeviceSize size, Memory memory,
                                          bool exportable = false);
    bool init(VulkanContext& ctx, VkDeviceSize size, Memory memory, bool exportable);
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    // Copies `bytes` from `data` into the buffer at `offset`.
    void upload(const void* data, VkDeviceSize bytes, VkDeviceSize offset = 0);

    VkBuffer handle() const { return m_buffer; }
    VkDeviceSize size() const { return m_size; }
    VkDeviceAddress address() const { return m_address; }
    void* mapped() const { return m_mapped; }

    // Size of the underlying memory allocation (>= size()).
    VkDeviceSize allocationSize() const { return m_allocationSize; }
    // Exports the allocation as a new OS handle (a Win32 NT handle on Windows,
    // a file descriptor elsewhere) that the caller owns. Returns false if the
    // buffer is not exportable or the export failed.
    bool exportHandle(void*& win32Handle, int& fd) const;

private:
    Buffer() = default;

    VulkanContext* m_ctx = nullptr;
    VkBuffer m_buffer = VK_NULL_HANDLE;
    VkDeviceMemory m_memory = VK_NULL_HANDLE;
    VkDeviceSize m_size = 0;
    VkDeviceSize m_allocationSize = 0;
    bool m_exportable = false;
    VkDeviceAddress m_address = 0;
    void* m_mapped = nullptr;
};
