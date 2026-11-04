#include "vk/VulkanContext.hpp"

#include <cstring>
#include <iostream>

namespace {

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT,
                                             const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "[vulkan] " << data->pMessage << "\n";
    }
    return VK_FALSE;
}

bool hasLayer(const char* name) {
    uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());
    for (const auto& layer : layers) {
        if (std::strcmp(layer.layerName, name) == 0) return true;
    }
    return false;
}

} // namespace

std::unique_ptr<VulkanContext> VulkanContext::create(const CreateInfo& info) {
    std::unique_ptr<VulkanContext> ctx(new VulkanContext());
    if (!ctx->init(info)) return nullptr;
    return ctx;
}

bool VulkanContext::init(const CreateInfo& info) {
    try {
        VK_CHECK(volkInitialize());
        if (volkGetInstanceVersion() < VK_API_VERSION_1_2) {
            throw std::runtime_error("Vulkan 1.2 or newer is required");
        }
        createInstance(info);
        if (info.createSurface) {
            m_surface = info.createSurface(m_instance);
            if (!m_surface) throw std::runtime_error("failed to create the window surface");
        }
        pickPhysicalDevice();
        createDevice();
    } catch (const std::exception& e) {
        std::cerr << "Vulkan initialization failed: " << e.what() << "\n";
        return false;
    }
    return true;
}

VulkanContext::~VulkanContext() {
    if (m_device) {
        vkDeviceWaitIdle(m_device);
        vkDestroyFence(m_device, m_immediateFence, nullptr);
        vkDestroyCommandPool(m_device, m_commandPool, nullptr);
        vkDestroyDevice(m_device, nullptr);
    }
    if (m_surface) vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
    if (m_debugMessenger) vkDestroyDebugUtilsMessengerEXT(m_instance, m_debugMessenger, nullptr);
    if (m_instance) vkDestroyInstance(m_instance, nullptr);
}

void VulkanContext::createInstance(const CreateInfo& info) {
    std::vector<const char*> extensions = info.instanceExtensions;
    std::vector<const char*> layers;

    bool validation = info.enableValidation && hasLayer("VK_LAYER_KHRONOS_validation");
    if (info.enableValidation && !validation) {
        std::cerr << "Validation requested but VK_LAYER_KHRONOS_validation is not installed\n";
    }
    if (validation) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "splat_lab";
    app.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    ci.ppEnabledExtensionNames = extensions.data();
    ci.enabledLayerCount = static_cast<uint32_t>(layers.size());
    ci.ppEnabledLayerNames = layers.data();
    VK_CHECK(vkCreateInstance(&ci, nullptr, &m_instance));
    volkLoadInstance(m_instance);

    if (validation) {
        VkDebugUtilsMessengerCreateInfoEXT dci{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        dci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        dci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                          VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                          VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        dci.pfnUserCallback = debugCallback;
        VK_CHECK(vkCreateDebugUtilsMessengerEXT(m_instance, &dci, nullptr, &m_debugMessenger));
    }
}

void VulkanContext::pickPhysicalDevice() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(m_instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(m_instance, &count, devices.data());

    int bestScore = -1;
    for (VkPhysicalDevice pd : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pd, &props);
        if (props.apiVersion < VK_API_VERSION_1_2) continue;

        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f2.pNext = &f12;
        vkGetPhysicalDeviceFeatures2(pd, &f2);
        if (!f12.bufferDeviceAddress) continue;

        // The sort and scan kernels rank keys with subgroup ballots and scans.
        VkPhysicalDeviceSubgroupProperties sub{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &sub;
        vkGetPhysicalDeviceProperties2(pd, &p2);
        const VkSubgroupFeatureFlags neededOps =
            VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
        if (!(sub.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) || (sub.supportedOperations & neededOps) != neededOps ||
            sub.subgroupSize < 4 || sub.subgroupSize > 128 || 256 % sub.subgroupSize != 0) {
            continue;
        }

        uint32_t qfCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(qfCount);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, families.data());

        int family = -1;
        for (uint32_t i = 0; i < qfCount; ++i) {
            if (!(families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) continue;
            if (m_surface) {
                VkBool32 present = VK_FALSE;
                vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, m_surface, &present);
                if (!present) continue;
            }
            family = static_cast<int>(i);
            break;
        }
        if (family < 0) continue;

        int score = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2 :
                    props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 1 : 0;
        if (score > bestScore) {
            bestScore = score;
            m_physicalDevice = pd;
            m_queueFamily = static_cast<uint32_t>(family);
            m_subgroupSize = sub.subgroupSize;
        }
    }
    if (!m_physicalDevice) {
        throw std::runtime_error(
            "No Vulkan 1.2 GPU with bufferDeviceAddress, subgroup ballot/arithmetic and a compute queue found");
    }
    vkGetPhysicalDeviceProperties(m_physicalDevice, &m_properties);
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &m_memoryProperties);
    m_deviceName = m_properties.deviceName;
}

void VulkanContext::createDevice() {
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = m_queueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.bufferDeviceAddress = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f12;

    std::vector<const char*> extensions;
    if (m_surface) extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

    VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    ci.pNext = &f2;
    ci.queueCreateInfoCount = 1;
    ci.pQueueCreateInfos = &qci;
    ci.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    ci.ppEnabledExtensionNames = extensions.data();
    VK_CHECK(vkCreateDevice(m_physicalDevice, &ci, nullptr, &m_device));
    volkLoadDevice(m_device);
    vkGetDeviceQueue(m_device, m_queueFamily, 0, &m_queue);

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = m_queueFamily;
    VK_CHECK(vkCreateCommandPool(m_device, &pci, nullptr, &m_commandPool));

    m_immediateCmd = allocateCommandBuffer();
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(m_device, &fci, nullptr, &m_immediateFence));
}

uint32_t VulkanContext::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags required,
                                       VkMemoryPropertyFlags preferred) const {
    // First try required + preferred, then required only.
    for (VkMemoryPropertyFlags wanted : {required | preferred, required}) {
        for (uint32_t i = 0; i < m_memoryProperties.memoryTypeCount; ++i) {
            if ((typeBits & (1u << i)) &&
                (m_memoryProperties.memoryTypes[i].propertyFlags & wanted) == wanted) {
                return i;
            }
        }
    }
    throw std::runtime_error("No suitable Vulkan memory type");
}

VkCommandBuffer VulkanContext::allocateCommandBuffer() const {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = m_commandPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(m_device, &ai, &cmd));
    return cmd;
}

void VulkanContext::submitAndWait(const std::function<void(VkCommandBuffer)>& record) {
    VK_CHECK(vkResetCommandBuffer(m_immediateCmd, 0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(m_immediateCmd, &bi));
    record(m_immediateCmd);
    VK_CHECK(vkEndCommandBuffer(m_immediateCmd));

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &m_immediateCmd;
    VK_CHECK(vkQueueSubmit(m_queue, 1, &si, m_immediateFence));
    VK_CHECK(vkWaitForFences(m_device, 1, &m_immediateFence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(m_device, 1, &m_immediateFence));
}

void VulkanContext::waitIdle() const {
    vkDeviceWaitIdle(m_device);
}
