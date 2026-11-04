#pragma once

#include "app/TrainingSession.hpp"
#include "io/SceneLoader.hpp"
#include "render/SplatRenderer.hpp"
#include "vk/VkCommon.hpp"

#include <memory>
#include <optional>
#include <string>
#include <vector>

class Camera;
class CameraController;
class CameraSet;
class CamerasPanel;
class FileMenu;
class ModelMenu;
class TrainPanel;
class GaussianCloud;
class Gui;
class LoadingOverlay;
class Options;
class Swapchain;
class VulkanContext;
class Window;

// Interactive mode: a GLFW window showing the splat from a fly camera, with
// an ImGui menu bar on top. The scene may be empty at start and loaded later
// through File > Load Scene / Load PLY / Load Cameras. Files are read on a
// worker thread while a progress window is shown; options.scenePath is
// loaded that way too.
class Viewer {
public:
    static std::unique_ptr<Viewer> create(const Options& options, std::unique_ptr<CameraSet> cameras);
    bool init(const Options& options, std::unique_ptr<CameraSet> cameras);
    ~Viewer();

    bool run();

private:
    Viewer() = default;

    bool setScene(std::unique_ptr<GaussianCloud> cloud);
    // Replaces the dataset cameras (nullptr clears them).
    void setCameras(std::unique_ptr<CameraSet> cameras);

    // Snapshot: a scene from the trainer; replaces the Gaussians but keeps
    // the view (and loads the run's cameras the first time).
    enum class LoadKind { Ply, Cameras, Model, Snapshot };
    struct LoadRequest {
        LoadKind kind;
        std::string plyPath;
        std::string camerasPath;
        std::string label;
    };
    // Queues a load; the latest request wins if one is already waiting.
    void requestLoad(LoadKind kind, const std::string& plyPath, const std::string& camerasPath,
                     const std::string& label = std::string());
    // Loads requested from the menus are ignored while training.
    void requestUserLoad(LoadKind kind, const std::string& plyPath, const std::string& camerasPath);
    // Starts queued loads and finishes completed ones (called once per frame).
    void updateLoading();
    void finishLoad(LoadKind kind, SceneLoader::Result result);
    void drawLoadingOverlay();

    // Training (Train > Train from Video): runs the Python trainer and shows
    // its intermediate results as they are written.
    void startTraining(const TrainingRequest& request);
    void updateTraining();
    bool trainingActive() const;
    // "Quit while training?" dialog shown when the window is closed during a run.
    void drawExitConfirmation();
    void handleHotkeys();
    void resetCamera();
    void applyDatasetCamera(size_t index);
    bool drawFrame();
    void saveScreenshot();
    void printCameraPose() const;
    void updateTitle(double fps, double frameMs);

    const Options* m_options = nullptr;
    std::unique_ptr<GaussianCloud> m_cloud;
    std::unique_ptr<CameraSet> m_cameras;
    size_t m_cameraIndex = 0;
    glm::vec3 m_sceneUp{0.0f, -1.0f, 0.0f}; // fly-camera world up while dataset cameras are loaded
    CamerasPanel* m_camerasPanel = nullptr; // owned by m_gui
    FileMenu* m_fileMenu = nullptr;         // owned by m_gui
    ModelMenu* m_modelMenu = nullptr;       // owned by m_gui
    TrainPanel* m_trainPanel = nullptr;     // owned by m_gui

    std::unique_ptr<TrainingSession> m_training;
    std::string m_trainingCameras; // cameras.json of the run, until loaded
    bool m_confirmExit = false;
    bool m_exitConfirmed = false;

    // Loading: a request waits in m_pending until the previous load is done.
    // Once the worker finishes, the result waits in m_loaded for one frame so
    // the "uploading" message is on screen before the (blocking) GPU upload.
    std::optional<LoadRequest> m_pending;
    std::unique_ptr<SceneLoader> m_loader;
    LoadKind m_loadKind = LoadKind::Ply;
    std::string m_loadLabel;
    std::optional<SceneLoader::Result> m_loaded;
    int m_loadedFramesShown = 0;
    std::unique_ptr<LoadingOverlay> m_loadingOverlay;
    bool m_hasImage = false;  // renderer output is valid for the current frame

    // The last splat image is reused while nothing that affects it changed
    // (keeps the GPU free for the trainer, and saves power).
    uint64_t m_sceneVersion = 0;
    uint64_t m_renderedVersion = ~0ull;
    glm::mat4 m_renderedView{0.0f};
    glm::vec2 m_renderedFocal{0.0f};
    VkExtent2D m_renderedExtent{0, 0};

    std::unique_ptr<Window> m_window;
    std::unique_ptr<VulkanContext> m_ctx;
    std::unique_ptr<Swapchain> m_swapchain;
    std::unique_ptr<SplatRenderer> m_renderer;
    std::unique_ptr<Gui> m_gui;
    std::unique_ptr<Camera> m_camera;
    std::unique_ptr<CameraController> m_controller;
    RenderSettings m_settings;

    VkCommandBuffer m_presentCmd = VK_NULL_HANDLE;
    VkFence m_presentFence = VK_NULL_HANDLE;
    VkSemaphore m_imageAvailable = VK_NULL_HANDLE;
    std::vector<VkSemaphore> m_renderFinished; // one per swapchain image
    int m_screenshotCounter = 0;
};
