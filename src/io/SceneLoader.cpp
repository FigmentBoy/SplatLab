#include "io/SceneLoader.hpp"

#include "io/CameraSet.hpp"
#include "scene/GaussianCloud.hpp"

#include <filesystem>
#include <iostream>

std::unique_ptr<SceneLoader> SceneLoader::create(const std::string& plyPath, const std::string& camerasPath) {
    std::unique_ptr<SceneLoader> loader(new SceneLoader());
    if (!loader->init(plyPath, camerasPath)) return nullptr;
    return loader;
}

bool SceneLoader::init(const std::string& plyPath, const std::string& camerasPath) {
    if (plyPath.empty() && camerasPath.empty()) return false;
    m_plyPath = plyPath;
    m_camerasPath = camerasPath;
    setStatus("Loading...");
    m_thread = std::thread([this] { run(); });
    return true;
}

SceneLoader::~SceneLoader() {
    m_cancel = true;
    if (m_thread.joinable()) m_thread.join();
}

std::string SceneLoader::status() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_status;
}

void SceneLoader::setStatus(const std::string& status) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_status = status;
}

SceneLoader::Result SceneLoader::takeResult() {
    return std::move(m_result);
}

void SceneLoader::run() {
    namespace fs = std::filesystem;
    if (!m_plyPath.empty()) {
        // Show the model folder for point_cloud/iteration_N/point_cloud.ply.
        fs::path ply(m_plyPath);
        std::string label = ply.filename().string();
        if (label == "point_cloud.ply" && ply.parent_path().has_parent_path()) {
            label = ply.parent_path().parent_path().parent_path().filename().string() + "/" + label;
        }
        setStatus("Reading " + label);
        m_progress = 0.0f;
        m_result.cloud = GaussianCloud::createFromPly(m_plyPath, [this](float fraction) {
            m_progress = fraction;
            return !m_cancel.load();
        });
        if (!m_result.cloud) m_result.failed = true;
        else {
            std::cout << "Loaded " << m_result.cloud->size() << " Gaussians (SH degree "
                      << m_result.cloud->shDegree << ") from " << m_plyPath << "\n";
        }
    }
    if (!m_camerasPath.empty() && !m_result.failed && !m_cancel) {
        setStatus("Reading " + fs::path(m_camerasPath).filename().string());
        m_progress = -1.0f;
        m_result.cameras = CameraSet::create(m_camerasPath);
        if (m_result.cameras) {
            std::cout << "Loaded " << m_result.cameras->size() << " cameras from " << m_camerasPath << "\n";
        } else if (m_plyPath.empty()) {
            m_result.failed = true; // a model without usable cameras still loads
        }
    }
    m_progress = 1.0f;
    m_done = true;
}
