#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

class CameraSet;
class GaussianCloud;

// Loads a PLY and/or a cameras.json on a worker thread so the viewer can keep
// drawing (and show progress) meanwhile. Only CPU work happens here; the GPU
// upload is left to the caller on the main thread.
class SceneLoader {
public:
    struct Result {
        std::unique_ptr<GaussianCloud> cloud;
        std::unique_ptr<CameraSet> cameras;
        bool failed = false; // a requested file could not be loaded
    };

    // Either path may be empty (nothing to load for it).
    static std::unique_ptr<SceneLoader> create(const std::string& plyPath, const std::string& camerasPath);
    bool init(const std::string& plyPath, const std::string& camerasPath);
    // Cancels a PLY load in progress and waits for the worker.
    ~SceneLoader();

    SceneLoader(const SceneLoader&) = delete;
    SceneLoader& operator=(const SceneLoader&) = delete;

    const std::string& plyPath() const { return m_plyPath; }
    const std::string& camerasPath() const { return m_camerasPath; }

    bool done() const { return m_done.load(); }
    // 0..1, or negative while the amount of work is unknown.
    float progress() const { return m_progress.load(); }
    std::string status() const;

    // Valid once done(); moves the loaded objects out.
    Result takeResult();

private:
    SceneLoader() = default;

    void run();
    void setStatus(const std::string& status);

    std::string m_plyPath;
    std::string m_camerasPath;
    std::thread m_thread;
    std::atomic<bool> m_done{false};
    std::atomic<bool> m_cancel{false};
    std::atomic<float> m_progress{-1.0f};
    mutable std::mutex m_mutex; // guards m_status
    std::string m_status;
    Result m_result; // written by the worker before m_done is set
};
