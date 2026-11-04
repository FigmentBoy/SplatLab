#pragma once

#include "vk/VkCommon.hpp"

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// Owns the Vulkan instance, the logical device and a single compute-capable
// queue (which is also the present queue when a surface is supplied).
class VulkanContext {
public:
    struct CreateInfo {
        std::vector<const char*> instanceExtensions;
        // When set, a surface is created right after the instance and the
        // device is chosen so that its queue can present to it.
        std::function<VkSurfaceKHR(VkInstance)> createSurface;
        bool enableValidation = false;
        // Buffers can be exported to other APIs (CUDA), see Buffer::exportHandle().
        bool externalMemory = false;
        // Only consider the physical device with this UUID (e.g. the CUDA device
        // whose memory will be shared).
        std::optional<std::array<uint8_t, VK_UUID_SIZE>> deviceUuid;
    };

    static std::unique_ptr<VulkanContext> create(const CreateInfo& info);
    bool init(const CreateInfo& info);
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    VkInstance instance() const { return m_instance; }
    VkPhysicalDevice physicalDevice() const { return m_physicalDevice; }
    VkDevice device() const { return m_device; }
    VkQueue queue() const { return m_queue; }
    uint32_t queueFamily() const { return m_queueFamily; }
    VkSurfaceKHR surface() const { return m_surface; }
    VkCommandPool commandPool() const { return m_commandPool; }
    const VkPhysicalDeviceProperties& properties() const { return m_properties; }
    const std::string& deviceName() const { return m_deviceName; }
    // Subgroup size of compute shaders (a divisor of 256, between 4 and 128).
    uint32_t subgroupSize() const { return m_subgroupSize; }
    // Features the backward pass needs: float atomic add on buffers and
    // subgroup vote operations.
    bool supportsBackward() const { return m_atomicFloatAdd && m_subgroupVote; }
    bool externalMemory() const { return m_externalMemory; }

    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags required,
                            VkMemoryPropertyFlags preferred = 0) const;

    VkCommandBuffer allocateCommandBuffer() const;

    // Records commands with `record`, submits them and blocks until finished.
    // Must not be nested.
    void submitAndWait(const std::function<void(VkCommandBuffer)>& record);

    void waitIdle() const;

private:
    VulkanContext() = default;

    void createInstance(const CreateInfo& info);
    void pickPhysicalDevice();
    void createDevice();

    VkInstance m_instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties m_properties{};
    VkPhysicalDeviceMemoryProperties m_memoryProperties{};
    std::string m_deviceName;
    uint32_t m_subgroupSize = 32;
    bool m_atomicFloatAdd = false;
    bool m_subgroupVote = false;
    bool m_externalMemory = false;
    std::optional<std::array<uint8_t, VK_UUID_SIZE>> m_wantedUuid;
    VkDevice m_device = VK_NULL_HANDLE;
    uint32_t m_queueFamily = 0;
    VkQueue m_queue = VK_NULL_HANDLE;
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    VkCommandBuffer m_immediateCmd = VK_NULL_HANDLE;
    VkFence m_immediateFence = VK_NULL_HANDLE;
};
