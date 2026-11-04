#include "vk/Swapchain.hpp"

#include "vk/VulkanContext.hpp"

#include <algorithm>
#include <iostream>

std::unique_ptr<Swapchain> Swapchain::create(VulkanContext& ctx, uint32_t width, uint32_t height, bool vsync) {
    std::unique_ptr<Swapchain> swapchain(new Swapchain());
    if (!swapchain->init(ctx, width, height, vsync)) return nullptr;
    return swapchain;
}

bool Swapchain::init(VulkanContext& ctx, uint32_t width, uint32_t height, bool vsync) {
    m_ctx = &ctx;
    m_vsync = vsync;
    return build(width, height);
}

Swapchain::~Swapchain() {
    if (m_ctx && m_swapchain) {
        m_ctx->waitIdle();
        destroyViews();
        vkDestroySwapchainKHR(m_ctx->device(), m_swapchain, nullptr);
    }
}

bool Swapchain::recreate(uint32_t width, uint32_t height) {
    m_ctx->waitIdle();
    return build(width, height);
}

bool Swapchain::isBgra() const {
    return m_format == VK_FORMAT_B8G8R8A8_UNORM || m_format == VK_FORMAT_B8G8R8A8_SRGB;
}

bool Swapchain::build(uint32_t width, uint32_t height) {
    try {
        buildOrThrow(width, height);
    } catch (const std::exception& e) {
        std::cerr << "Swapchain creation failed: " << e.what() << "\n";
        return false;
    }
    return true;
}

void Swapchain::buildOrThrow(uint32_t width, uint32_t height) {
    VkPhysicalDevice pd = m_ctx->physicalDevice();
    VkSurfaceKHR surface = m_ctx->surface();

    VkSurfaceCapabilitiesKHR caps;
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pd, surface, &caps));
    if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
        throw std::runtime_error("Surface does not support TRANSFER_DST swapchain images");
    }

    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface, &formatCount, formats.data());

    // 3DGS colours are already display-referred, so use a UNORM format to
    // avoid an extra sRGB encode.
    VkSurfaceFormatKHR chosen{VK_FORMAT_UNDEFINED, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    for (const auto& f : formats) {
        if ((f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosen = f;
            break;
        }
    }
    if (chosen.format == VK_FORMAT_UNDEFINED) {
        throw std::runtime_error("Surface supports neither B8G8R8A8_UNORM nor R8G8B8A8_UNORM");
    }

    uint32_t modeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(pd, surface, &modeCount, nullptr);
    std::vector<VkPresentModeKHR> modes(modeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(pd, surface, &modeCount, modes.data());
    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
    if (!m_vsync) {
        for (VkPresentModeKHR mode : {VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR}) {
            if (std::find(modes.begin(), modes.end(), mode) != modes.end()) {
                presentMode = mode;
                break;
            }
        }
    }

    VkExtent2D extent = caps.currentExtent;
    if (extent.width == UINT32_MAX) {
        extent.width = std::clamp(width, caps.minImageExtent.width, caps.maxImageExtent.width);
        extent.height = std::clamp(height, caps.minImageExtent.height, caps.maxImageExtent.height);
    }

    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0) imageCount = std::min(imageCount, caps.maxImageCount);

    destroyViews(); // views of the old images must go before the old swapchain
    VkSwapchainKHR old = m_swapchain;
    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface = surface;
    ci.minImageCount = imageCount;
    ci.imageFormat = chosen.format;
    ci.imageColorSpace = chosen.colorSpace;
    ci.imageExtent = extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = presentMode;
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = old;
    VK_CHECK(vkCreateSwapchainKHR(m_ctx->device(), &ci, nullptr, &m_swapchain));
    if (old) vkDestroySwapchainKHR(m_ctx->device(), old, nullptr);

    m_format = chosen.format;
    m_extent = extent;
    m_minImageCount = imageCount;
    uint32_t count = 0;
    vkGetSwapchainImagesKHR(m_ctx->device(), m_swapchain, &count, nullptr);
    m_images.resize(count);
    vkGetSwapchainImagesKHR(m_ctx->device(), m_swapchain, &count, m_images.data());

    for (VkImage image : m_images) {
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = m_format;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkImageView view = VK_NULL_HANDLE;
        VK_CHECK(vkCreateImageView(m_ctx->device(), &vci, nullptr, &view));
        m_views.push_back(view);
    }
}

void Swapchain::destroyViews() {
    for (VkImageView view : m_views) vkDestroyImageView(m_ctx->device(), view, nullptr);
    m_views.clear();
}

bool Swapchain::acquire(VkSemaphore signal, uint32_t& imageIndex) {
    VkResult result = vkAcquireNextImageKHR(m_ctx->device(), m_swapchain, UINT64_MAX, signal, VK_NULL_HANDLE,
                                            &imageIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) return false;
    if (result != VK_SUBOPTIMAL_KHR) VK_CHECK(result);
    return true;
}

bool Swapchain::present(VkSemaphore wait, uint32_t imageIndex) {
    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &wait;
    pi.swapchainCount = 1;
    pi.pSwapchains = &m_swapchain;
    pi.pImageIndices = &imageIndex;
    VkResult result = vkQueuePresentKHR(m_ctx->queue(), &pi);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) return false;
    VK_CHECK(result);
    return true;
}
