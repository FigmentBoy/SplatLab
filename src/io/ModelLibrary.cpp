#include "io/ModelLibrary.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>

namespace fs = std::filesystem;

namespace {

fs::path findModelsDir() {
    std::error_code ec;
    fs::path dir = fs::current_path(ec);
    if (ec) return {};
    for (int depth = 0; depth < 4 && !dir.empty(); ++depth) {
        fs::path candidate = dir / "data" / "models";
        if (fs::is_directory(candidate, ec)) return candidate;
        fs::path parent = dir.parent_path();
        if (parent == dir) break;
        dir = parent;
    }
    return {};
}

// point_cloud/iteration_<N>/point_cloud.ply with the largest N.
fs::path latestPly(const fs::path& modelDir) {
    std::error_code ec;
    fs::path best;
    long bestIter = -1;
    for (const auto& entry : fs::directory_iterator(modelDir / "point_cloud", ec)) {
        std::string name = entry.path().filename().string();
        const std::string prefix = "iteration_";
        if (!entry.is_directory(ec) || name.rfind(prefix, 0) != 0) continue;
        long iter = std::strtol(name.c_str() + prefix.size(), nullptr, 10);
        fs::path ply = entry.path() / "point_cloud.ply";
        if (iter > bestIter && fs::is_regular_file(ply, ec)) {
            bestIter = iter;
            best = ply;
        }
    }
    return best;
}

} // namespace

std::unique_ptr<ModelLibrary> ModelLibrary::create() {
    std::unique_ptr<ModelLibrary> library(new ModelLibrary());
    if (!library->init()) return nullptr;
    return library;
}

bool ModelLibrary::init() {
    refresh();
    return true; // a missing models directory is not an error
}

void ModelLibrary::refresh() {
    m_models.clear();
    fs::path root = findModelsDir();
    m_root = root.empty() ? std::string() : root.string();
    if (root.empty()) return;

    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(root, ec)) {
        if (!entry.is_directory(ec)) continue;
        fs::path ply = latestPly(entry.path());
        if (ply.empty()) continue;
        ModelEntry model;
        model.name = entry.path().filename().string();
        model.plyPath = ply.string();
        fs::path cameras = entry.path() / "cameras.json";
        if (fs::is_regular_file(cameras, ec)) model.camerasPath = cameras.string();
        m_models.push_back(std::move(model));
    }
    std::sort(m_models.begin(), m_models.end(),
              [](const ModelEntry& a, const ModelEntry& b) { return a.name < b.name; });
}
