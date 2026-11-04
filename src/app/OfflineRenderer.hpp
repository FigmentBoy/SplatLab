#pragma once

#include <memory>

class CameraSet;
class GaussianCloud;
class Options;
class SplatRenderer;
class VulkanContext;

// Headless mode: renders the scene from each requested camera and writes PNGs.
class OfflineRenderer {
public:
    static std::unique_ptr<OfflineRenderer> create(const Options& options, GaussianCloud& cloud,
                                                   const CameraSet* cameras);
    bool init(const Options& options, GaussianCloud& cloud, const CameraSet* cameras);
    ~OfflineRenderer();

    bool run();

private:
    OfflineRenderer() = default;

    const Options* m_options = nullptr;
    GaussianCloud* m_cloud = nullptr;
    const CameraSet* m_cameras = nullptr;
    std::unique_ptr<VulkanContext> m_ctx;
    std::unique_ptr<SplatRenderer> m_renderer;
};
