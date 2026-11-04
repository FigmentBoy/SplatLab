#include "scene/GaussianCloud.hpp"

#include "io/PlyLoader.hpp"

#include <cmath>
#include <iostream>
#include <limits>

std::unique_ptr<GaussianCloud> GaussianCloud::create(const std::string& name) {
    std::unique_ptr<GaussianCloud> cloud(new GaussianCloud());
    if (!cloud->init(name)) return nullptr;
    return cloud;
}

std::unique_ptr<GaussianCloud> GaussianCloud::createFromPly(const std::string& path,
                                                           const PlyLoader::ProgressCallback& progress) {
    std::unique_ptr<GaussianCloud> cloud(new GaussianCloud());
    if (!cloud->initFromPly(path, progress)) return nullptr;
    return cloud;
}

bool GaussianCloud::init(const std::string& name) {
    return Node::init(name);
}

bool GaussianCloud::initFromPly(const std::string& path, const PlyLoader::ProgressCallback& progress) {
    if (!init(path)) return false;
    std::string error;
    if (!PlyLoader::load(path, *this, error, progress)) {
        std::cerr << "Failed to load '" << path << "': " << error << "\n";
        return false;
    }
    return true;
}

void GaussianCloud::resize(size_t count) {
    position.resize(count, glm::vec3(0.0f));
    scaleLog.resize(count, glm::vec3(0.0f));
    rotation.resize(count, glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
    opacityLogit.resize(count, 0.0f);
    sh.resize(count * kMaxShCoeffs * 3, 0.0f);
}

glm::vec3 GaussianCloud::activatedScale(size_t i) const {
    return glm::exp(scaleLog[i]);
}

glm::vec4 GaussianCloud::activatedRotation(size_t i) const {
    const glm::vec4& q = rotation[i];
    float len = glm::length(q);
    return len > 0.0f ? q / len : glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
}

float GaussianCloud::activatedOpacity(size_t i) const {
    return 1.0f / (1.0f + std::exp(-opacityLogit[i]));
}

void GaussianCloud::bounds(glm::vec3& minOut, glm::vec3& maxOut) const {
    minOut = glm::vec3(std::numeric_limits<float>::max());
    maxOut = glm::vec3(std::numeric_limits<float>::lowest());
    for (const glm::vec3& p : position) {
        minOut = glm::min(minOut, p);
        maxOut = glm::max(maxOut, p);
    }
    if (position.empty()) minOut = maxOut = glm::vec3(0.0f);
}
