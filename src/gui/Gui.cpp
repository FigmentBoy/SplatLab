#include "gui/Gui.hpp"

#include "app/Window.hpp"
#include "gui/MenuModule.hpp"
#include "vk/Swapchain.hpp"
#include "vk/VulkanContext.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <algorithm>
#include <cstring>
#include <iostream>

namespace {

void checkImguiVk(VkResult result) {
    if (result != VK_SUCCESS) std::cerr << "[imgui] Vulkan error: " << vkutil::resultString(result) << "\n";
}

} // namespace

std::unique_ptr<Gui> Gui::create(VulkanContext& ctx, Window& window, Swapchain& swapchain) {
    std::unique_ptr<Gui> gui(new Gui());
    if (!gui->init(ctx, window, swapchain)) return nullptr;
    return gui;
}

bool Gui::init(VulkanContext& ctx, Window& window, Swapchain& swapchain) {
    m_ctx = &ctx;
    m_swapchain = &swapchain;

    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = 16;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &poolSize;
    if (vkCreateDescriptorPool(ctx.device(), &dpci, nullptr, &m_descriptorPool) != VK_SUCCESS) {
        std::cerr << "Gui: descriptor pool creation failed\n";
        return false;
    }
    if (!createRenderPass() || !createFramebuffers()) return false;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    m_imguiContext = true;
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.IniFilename = nullptr; // no imgui.ini next to the executable
    ImGui::StyleColorsDark();

    // Installs ImGui's GLFW callbacks, chaining to the ones Window set up.
    if (!ImGui_ImplGlfw_InitForVulkan(window.handle(), true)) return false;
    m_glfwBackend = true;

    ImGui_ImplVulkan_InitInfo info{};
    info.Instance = ctx.instance();
    info.PhysicalDevice = ctx.physicalDevice();
    info.Device = ctx.device();
    info.QueueFamily = ctx.queueFamily();
    info.Queue = ctx.queue();
    info.DescriptorPool = m_descriptorPool;
    info.RenderPass = m_renderPass;
    info.MinImageCount = std::max(2u, swapchain.minImageCount());
    info.ImageCount = std::max(info.MinImageCount, swapchain.imageCount());
    info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    info.CheckVkResultFn = checkImguiVk;
    if (!ImGui_ImplVulkan_Init(&info)) return false;
    m_vulkanBackend = true;
    return true;
}

Gui::~Gui() {
    if (m_ctx) m_ctx->waitIdle();
    m_modules.clear();
    if (m_vulkanBackend) ImGui_ImplVulkan_Shutdown();
    if (m_glfwBackend) ImGui_ImplGlfw_Shutdown();
    if (m_imguiContext) ImGui::DestroyContext();
    if (!m_ctx) return;
    destroyFramebuffers();
    VkDevice device = m_ctx->device();
    if (m_renderPass) vkDestroyRenderPass(device, m_renderPass, nullptr);
    if (m_descriptorPool) vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
}

MenuModule* Gui::addModule(std::unique_ptr<MenuModule> module) {
    if (!module) return nullptr;
    m_modules.push_back(std::move(module));
    return m_modules.back().get();
}

void Gui::beginFrame() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (ImGui::BeginMainMenuBar()) {
        // One top-level menu per distinct name, in first-registration order.
        std::vector<const char*> menus;
        for (const auto& module : m_modules) {
            const char* name = module->menuName();
            bool seen = false;
            for (const char* m : menus) seen = seen || std::strcmp(m, name) == 0;
            if (!seen) menus.push_back(name);
        }
        for (const char* name : menus) {
            if (ImGui::BeginMenu(name)) {
                std::vector<MenuModule*> items;
                for (const auto& module : m_modules) {
                    if (std::strcmp(module->menuName(), name) == 0) items.push_back(module.get());
                }
                std::stable_sort(items.begin(), items.end(), [](const MenuModule* a, const MenuModule* b) {
                    return a->menuOrder() < b->menuOrder();
                });
                for (MenuModule* item : items) item->drawMenuItems();
                ImGui::EndMenu();
            }
        }
        ImGui::EndMainMenuBar();
    }

    for (const auto& module : m_modules) module->drawWindows();
}

void Gui::endFrame() {
    ImGui::Render();
}

void Gui::record(VkCommandBuffer cmd, uint32_t imageIndex) {
    VkExtent2D extent = m_swapchain->extent();
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass = m_renderPass;
    rbi.framebuffer = m_framebuffers[imageIndex];
    rbi.renderArea = {{0, 0}, extent};
    vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRenderPass(cmd);
}

bool Gui::onSwapchainRecreated() {
    destroyFramebuffers();
    if (m_swapchain->format() != m_renderPassFormat) {
        // Practically never happens; the pipeline would also need rebuilding.
        std::cerr << "Gui: swapchain format changed, which is not supported\n";
        return false;
    }
    if (m_vulkanBackend) ImGui_ImplVulkan_SetMinImageCount(std::max(2u, m_swapchain->minImageCount()));
    return createFramebuffers();
}

void Gui::setMouseEnabled(bool enabled) {
    ImGuiIO& io = ImGui::GetIO();
    if (enabled) {
        io.ConfigFlags &= ~ImGuiConfigFlags_NoMouse;
    } else {
        io.ConfigFlags |= ImGuiConfigFlags_NoMouse;
    }
}

bool Gui::wantsMouse() const {
    return ImGui::GetIO().WantCaptureMouse;
}

bool Gui::wantsKeyboard() const {
    return ImGui::GetIO().WantCaptureKeyboard;
}

bool Gui::createRenderPass() {
    m_renderPassFormat = m_swapchain->format();

    VkAttachmentDescription color{};
    color.format = m_renderPassFormat;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; // keep the splat image underneath
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &ref;

    // Wait for the buffer->image copy before drawing the UI.
    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    dep.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpci.attachmentCount = 1;
    rpci.pAttachments = &color;
    rpci.subpassCount = 1;
    rpci.pSubpasses = &subpass;
    rpci.dependencyCount = 1;
    rpci.pDependencies = &dep;
    if (vkCreateRenderPass(m_ctx->device(), &rpci, nullptr, &m_renderPass) != VK_SUCCESS) {
        std::cerr << "Gui: render pass creation failed\n";
        return false;
    }
    return true;
}

bool Gui::createFramebuffers() {
    VkExtent2D extent = m_swapchain->extent();
    for (uint32_t i = 0; i < m_swapchain->imageCount(); ++i) {
        VkImageView view = m_swapchain->view(i);
        VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fci.renderPass = m_renderPass;
        fci.attachmentCount = 1;
        fci.pAttachments = &view;
        fci.width = extent.width;
        fci.height = extent.height;
        fci.layers = 1;
        VkFramebuffer fb = VK_NULL_HANDLE;
        if (vkCreateFramebuffer(m_ctx->device(), &fci, nullptr, &fb) != VK_SUCCESS) {
            std::cerr << "Gui: framebuffer creation failed\n";
            return false;
        }
        m_framebuffers.push_back(fb);
    }
    return true;
}

void Gui::destroyFramebuffers() {
    for (VkFramebuffer fb : m_framebuffers) vkDestroyFramebuffer(m_ctx->device(), fb, nullptr);
    m_framebuffers.clear();
}
