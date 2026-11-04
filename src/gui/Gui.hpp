#pragma once

#include "vk/VkCommon.hpp"

#include <memory>
#include <vector>

class MenuModule;
class Swapchain;
class VulkanContext;
class Window;

// Dear ImGui integration: owns the ImGui context, the GLFW/Vulkan backends and
// a render pass that draws the UI on top of the already-filled swapchain
// image. The main menu bar is assembled from registered MenuModules.
class Gui {
public:
    static std::unique_ptr<Gui> create(VulkanContext& ctx, Window& window, Swapchain& swapchain);
    bool init(VulkanContext& ctx, Window& window, Swapchain& swapchain);
    ~Gui();

    Gui(const Gui&) = delete;
    Gui& operator=(const Gui&) = delete;

    // Takes ownership; returns a non-owning pointer for convenience.
    MenuModule* addModule(std::unique_ptr<MenuModule> module);

    // Starts a UI frame and draws the menu bar and module windows.
    void beginFrame();
    // Finalizes the UI frame's draw data.
    void endFrame();

    // Draws the UI into swapchain image `imageIndex`, which must be in
    // TRANSFER_DST_OPTIMAL layout; leaves it in PRESENT_SRC_KHR.
    void record(VkCommandBuffer cmd, uint32_t imageIndex);

    // Must be called after the swapchain was recreated.
    bool onSwapchainRecreated();

    // While disabled, the UI ignores the mouse (e.g. while the camera has it captured).
    void setMouseEnabled(bool enabled);
    bool wantsMouse() const;
    bool wantsKeyboard() const;

private:
    Gui() = default;

    bool createRenderPass();
    bool createFramebuffers();
    void destroyFramebuffers();

    VulkanContext* m_ctx = nullptr;
    Swapchain* m_swapchain = nullptr;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkRenderPass m_renderPass = VK_NULL_HANDLE;
    VkFormat m_renderPassFormat = VK_FORMAT_UNDEFINED;
    std::vector<VkFramebuffer> m_framebuffers;
    bool m_imguiContext = false;
    bool m_glfwBackend = false;
    bool m_vulkanBackend = false;

    std::vector<std::unique_ptr<MenuModule>> m_modules;
};
