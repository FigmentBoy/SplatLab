#include "io/CameraSet.hpp"

#include "scene/Camera.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>

void CameraRecord::applyTo(Camera& camera) const {
    camera.setPoseFromCameraToWorldCV(cameraToWorldCV);
    camera.setFocalLengths(fx, fy, width, height);
}

std::unique_ptr<CameraSet> CameraSet::create(const std::string& jsonPath) {
    std::unique_ptr<CameraSet> set(new CameraSet());
    if (!set->init(jsonPath)) return nullptr;
    return set;
}

glm::vec3 CameraSet::averageUp(const glm::vec3& fallback) const {
    glm::vec3 sum(0.0f);
    for (const CameraRecord& rec : m_cameras) sum -= glm::vec3(rec.cameraToWorldCV[1]);
    float len = glm::length(sum);
    return len > 1e-6f ? sum / len : glm::normalize(fallback);
}

bool CameraSet::init(const std::string& jsonPath) {
    m_path = jsonPath;
    std::ifstream file(jsonPath);
    if (!file) {
        std::cerr << "Cannot open camera file '" << jsonPath << "'\n";
        return false;
    }
    try {
        nlohmann::json root;
        file >> root;
        if (!root.is_array()) throw std::runtime_error("expected a JSON array of cameras");

        m_cameras.clear();
        for (const auto& entry : root) {
            CameraRecord rec;
            rec.id = entry.value("id", static_cast<int>(m_cameras.size()));
            rec.imageName = entry.value("img_name", std::string());
            rec.width = entry.at("width").get<uint32_t>();
            rec.height = entry.at("height").get<uint32_t>();
            rec.fx = entry.at("fx").get<float>();
            rec.fy = entry.at("fy").get<float>();

            const auto& pos = entry.at("position");
            const auto& rot = entry.at("rotation");
            glm::mat4 c2w(1.0f);
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) c2w[c][r] = rot.at(r).at(c).get<float>(); // glm is column-major
                c2w[3][r] = pos.at(r).get<float>();
            }
            rec.cameraToWorldCV = c2w;
            m_cameras.push_back(rec);
        }
    } catch (const std::exception& e) {
        std::cerr << "Invalid camera file '" << jsonPath << "': " << e.what() << "\n";
        return false;
    }
    return true;
}
