#include "vk/Buffer.hpp"

#include "vk/VulkanContext.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>

namespace {
#ifdef _WIN32
constexpr VkExternalMemoryHandleTypeFlagBits kExportHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
constexpr VkExternalMemoryHandleTypeFlagBits kExportHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
} // namespace

std::unique_ptr<Buffer> Buffer::create(VulkanContext& ctx, VkDeviceSize size, Memory memory, bool exportable) {
    std::unique_ptr<Buffer> buffer(new Buffer());
    if (!buffer->init(ctx, size, memory, exportable)) return nullptr;
    return buffer;
}

bool Buffer::init(VulkanContext& ctx, VkDeviceSize size, Memory memory, bool exportable) {
    m_ctx = &ctx;
    m_size = std::max<VkDeviceSize>(size, 16);
    VkDevice device = ctx.device();
    if (exportable && !ctx.externalMemory()) {
        std::cerr << "Buffer: exportable buffers need a context created with externalMemory\n";
        return false;
    }
    m_exportable = exportable;
    VkExternalMemoryBufferCreateInfo externalInfo{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    externalInfo.handleTypes = kExportHandleType;

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = m_size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (exportable) bci.pNext = &externalInfo;
    if (vkCreateBuffer(device, &bci, nullptr, &m_buffer) != VK_SUCCESS) {
        std::cerr << "vkCreateBuffer failed (" << m_size << " bytes)\n";
        return false;
    }

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, m_buffer, &req);

    VkMemoryPropertyFlags required = 0, preferred = 0;
    switch (memory) {
    case Memory::DeviceLocal:
        required = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        break;
    case Memory::Upload:
        required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        break;
    case Memory::Readback:
        required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        preferred = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        break;
    }

    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.pNext = &flags;
    mai.allocationSize = req.size;

    // Exported memory: a dedicated allocation (CUDA imports it as such).
    VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicated.buffer = m_buffer;
    VkExportMemoryAllocateInfo exportInfo{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    exportInfo.handleTypes = kExportHandleType;
    exportInfo.pNext = &dedicated;
#ifdef _WIN32
    VkExportMemoryWin32HandleInfoKHR win32Info{VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR};
    win32Info.dwAccess = 0x80000000u | 0x1u; // DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE
    dedicated.pNext = &win32Info;
#endif
    if (exportable) {
        flags.pNext = &exportInfo;
    }
    m_allocationSize = req.size;
    mai.memoryTypeIndex = ctx.findMemoryType(req.memoryTypeBits, required, preferred);
    if (vkAllocateMemory(device, &mai, nullptr, &m_memory) != VK_SUCCESS) {
        std::cerr << "vkAllocateMemory failed (" << req.size << " bytes)\n";
        return false;
    }
    VK_CHECK(vkBindBufferMemory(device, m_buffer, m_memory, 0));

    if (memory != Memory::DeviceLocal) {
        VK_CHECK(vkMapMemory(device, m_memory, 0, VK_WHOLE_SIZE, 0, &m_mapped));
    }

    VkBufferDeviceAddressInfo addrInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    addrInfo.buffer = m_buffer;
    m_address = vkGetBufferDeviceAddress(device, &addrInfo);
    return true;
}

Buffer::~Buffer() {
    if (!m_ctx) return;
    VkDevice device = m_ctx->device();
    if (m_mapped) vkUnmapMemory(device, m_memory);
    if (m_buffer) vkDestroyBuffer(device, m_buffer, nullptr);
    if (m_memory) vkFreeMemory(device, m_memory, nullptr);
}

bool Buffer::exportHandle(void*& win32Handle, int& fd) const {
    win32Handle = nullptr;
    fd = -1;
    if (!m_exportable) return false;
#ifdef _WIN32
    VkMemoryGetWin32HandleInfoKHR info{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
    info.memory = m_memory;
    info.handleType = kExportHandleType;
    HANDLE handle = nullptr;
    if (vkGetMemoryWin32HandleKHR(m_ctx->device(), &info, &handle) != VK_SUCCESS) return false;
    win32Handle = handle;
    return handle != nullptr;
#else
    VkMemoryGetFdInfoKHR info{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    info.memory = m_memory;
    info.handleType = kExportHandleType;
    return vkGetMemoryFdKHR(m_ctx->device(), &info, &fd) == VK_SUCCESS && fd >= 0;
#endif
}

void Buffer::upload(const void* data, VkDeviceSize bytes, VkDeviceSize offset) {
    if (bytes == 0) return;
    if (offset + bytes > m_size) throw std::runtime_error("Buffer::upload out of range");
    if (m_mapped) {
        std::memcpy(static_cast<char*>(m_mapped) + offset, data, static_cast<size_t>(bytes));
        return;
    }
    std::unique_ptr<Buffer> staging = Buffer::create(*m_ctx, bytes, Memory::Upload);
    if (!staging) throw std::runtime_error("Buffer::upload: staging allocation failed");
    std::memcpy(staging->mapped(), data, static_cast<size_t>(bytes));
    m_ctx->submitAndWait([&](VkCommandBuffer cmd) {
        VkBufferCopy region{0, offset, bytes};
        vkCmdCopyBuffer(cmd, staging->handle(), m_buffer, 1, &region);
    });
}
