#include "app/OfflineRenderer.hpp"

#include "app/Options.hpp"
#include "io/CameraSet.hpp"
#include "io/ImageWriter.hpp"
#include "render/SplatRenderer.hpp"
#include "scene/Camera.hpp"
#include "scene/GaussianCloud.hpp"
#include "vk/Buffer.hpp"
#include "vk/VulkanContext.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <vector>

namespace {

struct View {
    std::string name;
    std::unique_ptr<Camera> camera;
};

} // namespace

std::unique_ptr<OfflineRenderer> OfflineRenderer::create(const Options& options, GaussianCloud& cloud,
                                                         const CameraSet* cameras) {
    std::unique_ptr<OfflineRenderer> app(new OfflineRenderer());
    if (!app->init(options, cloud, cameras)) return nullptr;
    return app;
}

bool OfflineRenderer::init(const Options& options, GaussianCloud& cloud, const CameraSet* cameras) {
    m_options = &options;
    m_cloud = &cloud;
    m_cameras = cameras;

    VulkanContext::CreateInfo info;
    info.enableValidation = options.validation;
    m_ctx = VulkanContext::create(info);
    if (!m_ctx) return false;
    std::cout << "GPU: " << m_ctx->deviceName() << "\n";

    m_renderer = SplatRenderer::create(*m_ctx);
    if (!m_renderer) return false;
    m_renderer->setProfiling(options.profile);
    return m_renderer->setScene(cloud);
}

OfflineRenderer::~OfflineRenderer() = default;

bool OfflineRenderer::run() {
    const Options& opt = *m_options;

    std::vector<View> views;
    if (opt.eye) {
        View view{"custom", Camera::create("custom")};
        view.camera->setPerspective(glm::radians(opt.fovYDegrees));
        view.camera->setPosition(*opt.eye);
        view.camera->lookAt(opt.target, opt.up);
        views.push_back(std::move(view));
    } else if (m_cameras && !m_cameras->empty()) {
        for (size_t i = 0; i < m_cameras->size(); ++i) {
            if (opt.cameraIndex >= 0 && static_cast<size_t>(opt.cameraIndex) != i) continue;
            const CameraRecord& rec = (*m_cameras)[i];
            char fallback[32];
            std::snprintf(fallback, sizeof(fallback), "%05zu", i);
            View view{rec.imageName.empty() ? fallback : rec.imageName, Camera::create(rec.imageName)};
            rec.applyTo(*view.camera);
            views.push_back(std::move(view));
        }
        if (views.empty()) {
            std::cerr << "Camera index " << opt.cameraIndex << " is out of range (" << m_cameras->size()
                      << " cameras)\n";
            return false;
        }
    } else {
        std::cerr << "Offline rendering needs --cameras or --eye\n";
        return false;
    }

    std::error_code ec;
    std::filesystem::create_directories(opt.outputDir, ec);
    if (ec) {
        std::cerr << "Cannot create output directory '" << opt.outputDir << "': " << ec.message() << "\n";
        return false;
    }

    RenderSettings settings;
    settings.background = opt.background;
    settings.shDegree = opt.shDegree;
    settings.scaleModifier = opt.scaleModifier;
    settings.tightTiles = !opt.referenceBinning;
    size_t binningMismatches = 0;

    // Stage label -> summed milliseconds, in first-seen order.
    std::vector<std::pair<std::string, double>> stageTotals;
    double totalMs = 0.0;

    for (View& view : views) {
        // Resolution: explicit size, else the camera's native size, else 720p.
        glm::uvec2 size(1280, 720);
        if (auto native = view.camera->referenceResolution()) size = *native;
        if (opt.width && opt.height) {
            size = {opt.width, opt.height};
        } else if (opt.width) {
            size = {opt.width, std::max(1u, static_cast<uint32_t>(uint64_t(size.y) * opt.width / size.x))};
        } else if (opt.height) {
            size = {std::max(1u, static_cast<uint32_t>(uint64_t(size.x) * opt.height / size.y)), opt.height};
        }

        // When profiling, render repeatedly for ~0.3 s first so allocations are
        // done and the GPU (laptop GPUs especially) has left its idle clocks,
        // then report the median of several timed renders.
        if (opt.profile) {
            auto warmStart = std::chrono::steady_clock::now();
            do {
                if (!m_renderer->render(*view.camera, size.x, size.y, settings)) return false;
            } while (std::chrono::steady_clock::now() - warmStart < std::chrono::milliseconds(300));
        }
        const int timedRuns = opt.profile ? 7 : 1;
        std::vector<std::pair<double, std::vector<std::pair<std::string, double>>>> runs;
        for (int run = 0; run < timedRuns; ++run) {
            auto start = std::chrono::steady_clock::now();
            if (!m_renderer->render(*view.camera, size.x, size.y, settings)) {
                std::cerr << "Render failed for view " << view.name << "\n";
                return false;
            }
            double runMs =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            runs.emplace_back(runMs, m_renderer->stageTimings());
        }
        std::sort(runs.begin(), runs.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        const auto& median = runs[runs.size() / 2];
        double ms = median.first;
        totalMs += ms;
        if (opt.profile) {
            for (const auto& [label, stageMs] : median.second) {
                auto it = std::find_if(stageTotals.begin(), stageTotals.end(),
                                       [&](const auto& e) { return e.first == label; });
                if (it == stageTotals.end()) {
                    stageTotals.emplace_back(label, stageMs);
                } else {
                    it->second += stageMs;
                }
            }
        }

        const size_t pixels = size_t(size.x) * size.y;
        std::vector<uint8_t> rgba(pixels * 4);
        m_renderer->download(m_renderer->packedImage(), rgba.data(), rgba.size());

        std::string base = (std::filesystem::path(opt.outputDir) / view.name).string();
        ImageWriter::writePngRgba(base + ".png", size.x, size.y, rgba.data());

        if (opt.saveAux) {
            std::vector<float> finalT(pixels);
            m_renderer->download(m_renderer->finalTransmittance(), finalT.data(), pixels * sizeof(float));
            ImageWriter::writePngGray(base + "_transmittance.png", size.x, size.y, finalT.data());
        }

        const RenderStats stats = m_renderer->stats();
        std::printf("%s.png  %ux%u  %.2f ms  (%u Gaussians, %u visible, %u tile instances)\n", base.c_str(), size.x,
                    size.y, ms, stats.numGaussians, stats.numVisible, stats.numInstances);

        if (opt.checkBinning) {
            // Colour (CHW) and transmittance as raw bits, for both binnings.
            auto grab = [&]() {
                std::vector<uint32_t> bits(pixels * 4);
                m_renderer->download(m_renderer->colorImage(), bits.data(), pixels * 3 * sizeof(float));
                m_renderer->download(m_renderer->finalTransmittance(), bits.data() + pixels * 3,
                                     pixels * sizeof(float));
                return bits;
            };
            std::vector<uint32_t> current = grab();
            RenderSettings other = settings;
            other.tightTiles = !settings.tightTiles;
            if (!m_renderer->render(*view.camera, size.x, size.y, other)) return false;
            std::vector<uint32_t> alternative = grab();
            size_t differing = 0;
            for (size_t i = 0; i < current.size(); ++i) differing += current[i] != alternative[i];
            binningMismatches += differing;
            const RenderStats otherStats = m_renderer->stats();
            const RenderStats& tight = settings.tightTiles ? stats : otherStats;
            const RenderStats& reference = settings.tightTiles ? otherStats : stats;
            std::printf("  binning check: %zu of %zu floats differ; %u tight vs %u reference instances (%.1f%%)\n",
                        differing, current.size(), tight.numInstances, reference.numInstances,
                        100.0 * tight.numInstances / std::max(1u, reference.numInstances));
        }
    }

    if (opt.checkBinning) {
        std::printf("Binning check: %zu differing floats over %zu views\n", binningMismatches, views.size());
    }
    if (opt.profile && !views.empty()) {
        const double count = double(views.size());
        double gpuTotal = 0.0;
        std::printf("\nMean GPU time per view over %zu views:\n", views.size());
        for (const auto& [label, sum] : stageTotals) {
            std::printf("  %-12s %8.3f ms\n", label.c_str(), sum / count);
            gpuTotal += sum;
        }
        std::printf("  %-12s %8.3f ms\n  %-12s %8.3f ms\n", "GPU total", gpuTotal / count, "wall",
                    totalMs / count);
    }
    return true;
}
