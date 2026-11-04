#include "app/Viewer.hpp"

#include "app/CameraController.hpp"
#include "app/Options.hpp"
#include "app/Window.hpp"
#include "gui/CamerasPanel.hpp"
#include "gui/ExitMenu.hpp"
#include "gui/FileMenu.hpp"
#include "gui/Gui.hpp"
#include "gui/LoadingOverlay.hpp"
#include "gui/ModelMenu.hpp"
#include "io/CameraSet.hpp"
#include "io/ModelLibrary.hpp"
#include "io/ImageWriter.hpp"
#include "scene/Camera.hpp"
#include "scene/GaussianCloud.hpp"
#include "vk/Buffer.hpp"
#include "vk/Swapchain.hpp"
#include "vk/VulkanContext.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>

namespace {
// Converts the OpenGL-convention camera matrix back to the CV convention
// used by cameras.json.
const glm::mat4 kFlipYZ(1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1);
} // namespace

std::unique_ptr<Viewer> Viewer::create(const Options& options, std::unique_ptr<CameraSet> cameras) {
    std::unique_ptr<Viewer> viewer(new Viewer());
    if (!viewer->init(options, std::move(cameras))) return nullptr;
    return viewer;
}

bool Viewer::init(const Options& options, std::unique_ptr<CameraSet> cameras) {
    m_options = &options;

    // volk must be initialized before GLFW is pointed at vkGetInstanceProcAddr.
    if (volkInitialize() != VK_SUCCESS) {
        std::cerr << "Vulkan loader not found\n";
        return false;
    }

    uint32_t width = options.width ? options.width : 1280;
    uint32_t height = options.height ? options.height : 720;
    m_window = Window::create(width, height, "splat_lab");
    if (!m_window) return false;

    VulkanContext::CreateInfo info;
    info.instanceExtensions = Window::requiredInstanceExtensions();
    info.createSurface = [this](VkInstance instance) { return m_window->createSurface(instance); };
    info.enableValidation = options.validation;
    m_ctx = VulkanContext::create(info);
    if (!m_ctx) return false;
    std::cout << "GPU: " << m_ctx->deviceName() << "\n";

    glm::uvec2 fb = m_window->framebufferSize();
    m_swapchain = Swapchain::create(*m_ctx, fb.x, fb.y, options.vsync);
    m_renderer = SplatRenderer::create(*m_ctx);
    if (!m_swapchain || !m_renderer) return false;

    m_gui = Gui::create(*m_ctx, *m_window, *m_swapchain);
    if (!m_gui) return false;
    auto modelMenu = ModelMenu::create([this](const ModelEntry& model) {
        requestLoad(LoadKind::Model, model.plyPath, model.camerasPath);
    });
    if (!modelMenu) return false;
    m_gui->addModule(std::move(modelMenu));
    auto fileMenu = FileMenu::create(
        [this](const std::string& path) { requestLoad(LoadKind::Ply, path, std::string()); },
        [this](const std::string& path) { requestLoad(LoadKind::Cameras, std::string(), path); });
    if (!fileMenu) return false;
    m_gui->addModule(std::move(fileMenu));
    auto exitMenu = ExitMenu::create([this] { m_window->requestClose(); });
    if (!exitMenu) return false;
    m_gui->addModule(std::move(exitMenu));
    auto camerasPanel = CamerasPanel::create([this](size_t index) {
        if (m_cameras && index < m_cameras->size()) applyDatasetCamera(index);
    });
    if (!camerasPanel) return false;
    m_camerasPanel = camerasPanel.get();
    m_gui->addModule(std::move(camerasPanel));
    m_loadingOverlay = LoadingOverlay::create();
    if (!m_loadingOverlay) return false;

    m_camera = Camera::create("ViewerCamera");
    m_controller = CameraController::create(*m_camera, *m_window);
    if (!m_camera || !m_controller) return false;

    m_settings.background = options.background;
    m_settings.shDegree = options.shDegree;
    m_settings.scaleModifier = options.scaleModifier;
    m_settings.tightTiles = !options.referenceBinning;

    m_presentCmd = m_ctx->allocateCommandBuffer();
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VK_CHECK(vkCreateFence(m_ctx->device(), &fci, nullptr, &m_presentFence));
    VK_CHECK(vkCreateSemaphore(m_ctx->device(), &sci, nullptr, &m_imageAvailable));

    if (cameras) {
        setCameras(std::move(cameras));
        if (options.cameraIndex >= 0 && static_cast<size_t>(options.cameraIndex) < m_cameras->size()) {
            m_cameraIndex = static_cast<size_t>(options.cameraIndex);
        }
        // With a scene to load, the panel opens once the PLY is in.
        if (options.scenePath.empty() && !m_cameras->empty()) m_camerasPanel->setVisible(true);
    }

    resetCamera();
    if (!options.scenePath.empty()) requestLoad(LoadKind::Ply, options.scenePath, std::string());
    return true;
}

Viewer::~Viewer() {
    m_loader.reset(); // cancels and joins a load in progress
    if (!m_ctx) return;
    m_ctx->waitIdle();
    VkDevice device = m_ctx->device();
    for (VkSemaphore s : m_renderFinished) vkDestroySemaphore(device, s, nullptr);
    if (m_imageAvailable) vkDestroySemaphore(device, m_imageAvailable, nullptr);
    if (m_presentFence) vkDestroyFence(device, m_presentFence, nullptr);
    // GPU objects before the context, the context before the window.
    m_gui.reset();
    m_renderer.reset();
    m_swapchain.reset();
    m_ctx.reset();
}

bool Viewer::setScene(std::unique_ptr<GaussianCloud> cloud) {
    if (!m_renderer->setScene(*cloud)) return false;
    m_cloud = std::move(cloud);
    return true;
}

void Viewer::setCameras(std::unique_ptr<CameraSet> cameras) {
    m_cameras = std::move(cameras);
    m_cameraIndex = 0;
    m_camerasPanel->setCameras(m_cameras.get());
    if (m_cameras && !m_cameras->empty()) {
        // One up axis for the whole set, so the horizontal movement plane does
        // not tilt from camera to camera.
        m_sceneUp = m_cameras->averageUp(m_options->up);
    } else {
        m_camerasPanel->setVisible(false);
    }
}

void Viewer::requestLoad(LoadKind kind, const std::string& plyPath, const std::string& camerasPath) {
    m_pending = LoadRequest{kind, plyPath, camerasPath};
}

void Viewer::updateLoading() {
    // Second frame after the worker finished: the upload message has been
    // presented, so do the blocking GPU upload now.
    if (m_loaded && m_loadedFramesShown > 0) {
        SceneLoader::Result result = std::move(*m_loaded);
        m_loaded.reset();
        m_loader.reset();
        finishLoad(m_loadKind, std::move(result));
    }

    if (m_loader && !m_loaded && m_loader->done()) {
        SceneLoader::Result result = m_loader->takeResult();
        if (result.cloud) {
            m_loaded = std::move(result);
            m_loadedFramesShown = 0;
        } else {
            m_loader.reset();
            finishLoad(m_loadKind, std::move(result));
        }
    }

    if (!m_loader && m_pending) {
        LoadRequest request = std::move(*m_pending);
        m_pending.reset();
        m_loadKind = request.kind;
        m_loader = SceneLoader::create(request.plyPath, request.camerasPath);
    }
}

void Viewer::finishLoad(LoadKind kind, SceneLoader::Result result) {
    if (result.failed) return; // already reported; keep the current scene

    if (kind == LoadKind::Cameras) {
        if (!result.cameras) return;
        setCameras(std::move(result.cameras));
        if (!m_cameras->empty()) {
            applyDatasetCamera(0);
            m_camerasPanel->setVisible(true);
        }
        return;
    }

    bool hadScene = m_cloud != nullptr;
    std::string path = result.cloud->name();
    if (!setScene(std::move(result.cloud))) {
        std::cerr << "Failed to upload " << path << " to the GPU\n";
        return;
    }

    if (kind == LoadKind::Model) {
        setCameras(std::move(result.cameras));
        if (m_cameras && !m_cameras->empty()) {
            applyDatasetCamera(0);
        } else {
            resetCamera();
        }
    } else if (!m_options->eye) {
        // Cameras loaded for the previous scene don't belong to this one, so
        // frame the new scene instead. Cameras loaded before any scene are kept.
        if (hadScene) setCameras(nullptr);
        resetCamera();
    }
    if (m_cameras && !m_cameras->empty()) m_camerasPanel->setVisible(true);
}

void Viewer::drawLoadingOverlay() {
    if (!m_loader) return;
    if (m_loaded) {
        char text[96];
        std::snprintf(text, sizeof(text), "Uploading %zu Gaussians to the GPU...", m_loaded->cloud->size());
        m_loadingOverlay->draw(text, -1.0f);
    } else {
        m_loadingOverlay->draw(m_loader->status(), m_loader->progress());
    }
}

void Viewer::resetCamera() {
    if (m_options->eye) {
        m_camera->setPerspective(glm::radians(m_options->fovYDegrees));
        m_camera->setPosition(*m_options->eye);
        m_camera->lookAt(m_options->target, m_options->up);
        m_controller->setWorldUp(m_options->up);
    } else if (m_cameras && !m_cameras->empty()) {
        applyDatasetCamera(m_cameraIndex);
    } else {
        // Look at the scene from outside its bounding box.
        glm::vec3 lo(-1.0f), hi(1.0f);
        glm::mat4 model(1.0f);
        if (m_cloud) {
            m_cloud->bounds(lo, hi);
            model = m_cloud->worldMatrix();
        }
        glm::vec3 centre = glm::vec3(model * glm::vec4(0.5f * (lo + hi), 1.0f));
        float radius = std::max(0.5f * glm::length(hi - lo), 1e-3f);
        m_camera->setPerspective(glm::radians(m_options->fovYDegrees));
        m_camera->setPosition(centre + glm::vec3(0.0f, 0.0f, -2.0f * radius));
        m_camera->lookAt(centre, m_options->up);
        m_controller->setWorldUp(m_options->up);
        m_controller->setMoveSpeed(std::max(radius * 0.5f, 0.1f));
    }
}

void Viewer::applyDatasetCamera(size_t index) {
    m_cameraIndex = index;
    const CameraRecord& rec = (*m_cameras)[index];
    rec.applyTo(*m_camera);
    // Keep the dataset's vertical FOV but adapt to the window's aspect ratio.
    m_camera->clearFocalLengths();
    m_controller->setWorldUp(m_sceneUp);
    m_camerasPanel->setCurrent(index);
    std::cout << "Camera " << index << "/" << m_cameras->size() << ": " << rec.imageName << "\n";
}

void Viewer::handleHotkeys() {
    if (m_gui->wantsKeyboard()) return;
    if (m_window->keyPressed(GLFW_KEY_R)) resetCamera();
    if (m_cameras && !m_cameras->empty()) {
        if (m_window->keyPressed(GLFW_KEY_RIGHT_BRACKET)) {
            applyDatasetCamera((m_cameraIndex + 1) % m_cameras->size());
        }
        if (m_window->keyPressed(GLFW_KEY_LEFT_BRACKET)) {
            applyDatasetCamera((m_cameraIndex + m_cameras->size() - 1) % m_cameras->size());
        }
    }
    if (m_window->keyPressed(GLFW_KEY_C)) printCameraPose();
}

bool Viewer::run() {
    using Clock = std::chrono::steady_clock;
    auto last = Clock::now();
    auto titleTimer = last;
    int frames = 0;
    double frameMsSum = 0.0;

    while (!m_window->shouldClose()) {
        m_window->pollEvents();
        auto now = Clock::now();
        float dt = std::chrono::duration<float>(now - last).count();
        last = now;

        updateLoading();

        glm::uvec2 fb = m_window->framebufferSize();
        if (fb.x == 0 || fb.y == 0) {
            glfwWaitEvents(); // minimized
            continue;
        }
        VkExtent2D extent = m_swapchain->extent();
        if (m_window->consumeResized() || extent.width != fb.x || extent.height != fb.y) {
            if (!m_swapchain->recreate(fb.x, fb.y) || !m_gui->onSwapchainRecreated()) return false;
        }

        // The UI only sees the mouse while the camera does not have it captured.
        m_gui->setMouseEnabled(!m_controller->mouseCaptured());
        m_gui->beginFrame();
        drawLoadingOverlay();
        m_gui->endFrame();

        handleHotkeys();
        if (m_controller->update(dt, m_gui->wantsMouse())) m_camerasPanel->clearCurrent();

        auto frameStart = Clock::now();
        if (!drawFrame()) return false;
        if (m_loaded) ++m_loadedFramesShown;
        frameMsSum += std::chrono::duration<double, std::milli>(Clock::now() - frameStart).count();
        ++frames;

        if (!m_gui->wantsKeyboard() && m_window->keyPressed(GLFW_KEY_P)) saveScreenshot();
        // Presses nobody consumed this frame must not trigger later.
        m_window->clearPresses();

        double elapsed = std::chrono::duration<double>(Clock::now() - titleTimer).count();
        if (elapsed > 0.5) {
            updateTitle(frames / elapsed, frameMsSum / frames);
            titleTimer = Clock::now();
            frames = 0;
            frameMsSum = 0.0;
        }
    }
    return true;
}

void Viewer::updateTitle(double fps, double frameMs) {
    char title[256];
    if (m_cloud) {
        const RenderStats& stats = m_renderer->stats();
        std::snprintf(title, sizeof(title),
                      "splat_lab - %ux%u - %.1f fps (%.2f ms) - %u Gaussians, %u instances", stats.width,
                      stats.height, fps, frameMs, stats.numGaussians, stats.numInstances);
    } else {
        std::snprintf(title, sizeof(title), "splat_lab - no scene loaded (File > Load Scene or Load PLY)");
    }
    m_window->setTitle(title);
}

bool Viewer::drawFrame() {
    uint32_t imageIndex = 0;
    if (!m_swapchain->acquire(m_imageAvailable, imageIndex)) {
        glm::uvec2 fb = m_window->framebufferSize();
        return m_swapchain->recreate(fb.x, fb.y) && m_gui->onSwapchainRecreated();
    }

    VkDevice device = m_ctx->device();
    while (m_renderFinished.size() < m_swapchain->imageCount()) {
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VkSemaphore s = VK_NULL_HANDLE;
        VK_CHECK(vkCreateSemaphore(device, &sci, nullptr, &s));
        m_renderFinished.push_back(s);
    }

    VkExtent2D extent = m_swapchain->extent();
    m_hasImage = false;
    if (m_cloud) {
        if (!m_renderer->render(*m_camera, extent.width, extent.height, m_settings, m_swapchain->isBgra())) {
            return false;
        }
        m_hasImage = true;
    }

    VkImage image = m_swapchain->image(imageIndex);
    VK_CHECK(vkResetCommandBuffer(m_presentCmd, 0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(m_presentCmd, &bi));

    VkImageMemoryBarrier toTransfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toTransfer.srcAccessMask = 0;
    toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.image = image;
    toTransfer.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(m_presentCmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &toTransfer);

    if (m_hasImage) {
        // Splat image -> swapchain image.
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {extent.width, extent.height, 1};
        vkCmdCopyBufferToImage(m_presentCmd, m_renderer->packedImage().handle(), image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    } else {
        VkClearColorValue clear{{m_settings.background.r, m_settings.background.g, m_settings.background.b, 1.0f}};
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(m_presentCmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &range);
    }

    // UI on top; the render pass transitions the image to PRESENT_SRC.
    m_gui->record(m_presentCmd, imageIndex);
    VK_CHECK(vkEndCommandBuffer(m_presentCmd));

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &m_imageAvailable;
    si.pWaitDstStageMask = &waitStage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &m_presentCmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &m_renderFinished[imageIndex];
    VK_CHECK(vkQueueSubmit(m_ctx->queue(), 1, &si, m_presentFence));
    VK_CHECK(vkWaitForFences(device, 1, &m_presentFence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(device, 1, &m_presentFence));

    if (!m_swapchain->present(m_renderFinished[imageIndex], imageIndex)) {
        glm::uvec2 fb = m_window->framebufferSize();
        return m_swapchain->recreate(fb.x, fb.y) && m_gui->onSwapchainRecreated();
    }
    return true;
}

void Viewer::saveScreenshot() {
    if (!m_hasImage) return;
    const RenderStats& stats = m_renderer->stats();
    std::vector<uint8_t> pixels(size_t(stats.width) * stats.height * 4);
    m_renderer->download(m_renderer->packedImage(), pixels.data(), pixels.size());
    if (m_swapchain->isBgra()) {
        for (size_t i = 0; i < pixels.size(); i += 4) std::swap(pixels[i], pixels[i + 2]);
    }
    char name[64];
    std::snprintf(name, sizeof(name), "screenshot_%03d.png", m_screenshotCounter++);
    ImageWriter::writePngRgba(name, stats.width, stats.height, pixels.data());
    std::cout << "Saved " << std::filesystem::absolute(name).string() << " (splat only, without UI)\n";
}

void Viewer::printCameraPose() const {
    // Same layout as an entry of the reference cameras.json, so poses picked
    // in the viewer can be rendered offline with --cameras.
    glm::mat4 c2w = m_camera->worldMatrix() * kFlipYZ;
    VkExtent2D extent = m_swapchain->extent();
    CameraIntrinsics k = m_camera->intrinsics(extent.width, extent.height);
    std::printf("{\"id\": 0, \"img_name\": \"view_%03d\", \"width\": %u, \"height\": %u, "
                "\"position\": [%.6f, %.6f, %.6f], \"rotation\": [[%.6f, %.6f, %.6f], [%.6f, %.6f, %.6f], "
                "[%.6f, %.6f, %.6f]], \"fx\": %.4f, \"fy\": %.4f}\n",
                m_screenshotCounter, extent.width, extent.height, c2w[3][0], c2w[3][1], c2w[3][2], c2w[0][0],
                c2w[1][0], c2w[2][0], c2w[0][1], c2w[1][1], c2w[2][1], c2w[0][2], c2w[1][2], c2w[2][2], k.fx, k.fy);
}
