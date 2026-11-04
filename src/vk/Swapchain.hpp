#pragma once

#include "vk/VkCommon.hpp"

#include <memory>
#include <vector>

class VulkanContext;

// Swapchain whose images receive the splat image through a buffer->image copy
// (the splat renderer is pure compute); the UI is then drawn on top as a
// colour attachment.
class Swapchain {
public:
    static std::unique_ptr<Swapchain> create(VulkanContext& ctx, uint32_t width, uint32_t height, bool vsync);
    bool init(VulkanContext& ctx, uint32_t width, uint32_t height, bool vsync);
    ~Swapchain();

    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;

    bool recreate(uint32_t width, uint32_t height);

    // Returns false if the swapchain is out of date and must be recreated.
    bool acquire(VkSemaphore signal, uint32_t& imageIndex);
    bool present(VkSemaphore wait, uint32_t imageIndex);

    VkImage image(uint32_t index) const { return m_images[index]; }
    VkImageView view(uint32_t index) const { return m_views[index]; }
    uint32_t imageCount() const { return static_cast<uint32_t>(m_images.size()); }
    uint32_t minImageCount() const { return m_minImageCount; }
    VkExtent2D extent() const { return m_extent; }
    VkFormat format() const { return m_format; }
    // True when the image byte order is B, G, R, A.
    bool isBgra() const;

private:
    Swapchain() = default;

    bool build(uint32_t width, uint32_t height);
    void buildOrThrow(uint32_t width, uint32_t height);
    void destroyViews();

    VulkanContext* m_ctx = nullptr;
    bool m_vsync = true;
    VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
    VkFormat m_format = VK_FORMAT_UNDEFINED;
    VkExtent2D m_extent{};
    uint32_t m_minImageCount = 2;
    std::vector<VkImage> m_images;
    std::vector<VkImageView> m_views;
};
