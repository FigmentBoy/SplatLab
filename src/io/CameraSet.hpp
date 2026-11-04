#pragma once

#include <glm/glm.hpp>

#include <memory>
#include <string>
#include <vector>

class Camera;

// A camera pose + intrinsics as stored in the reference implementation's
// cameras.json (written next to the trained point_cloud/ folder):
//   { "id", "img_name", "width", "height",
//     "position": [x,y,z], "rotation": [[3x3 rows]], "fx", "fy" }
// `rotation` and `position` form the camera-to-world transform in the
// computer-vision convention (x right, y down, z forward).
struct CameraRecord {
    int id = 0;
    std::string imageName;
    uint32_t width = 0;
    uint32_t height = 0;
    glm::mat4 cameraToWorldCV{1.0f};
    float fx = 0.0f;
    float fy = 0.0f;

    // Applies this record's pose and focal lengths to `camera`.
    void applyTo(Camera& camera) const;
};

class CameraSet {
public:
    // Loads a cameras.json file; returns nullptr on failure.
    static std::unique_ptr<CameraSet> create(const std::string& jsonPath);
    bool init(const std::string& jsonPath);

    const std::vector<CameraRecord>& cameras() const { return m_cameras; }
    size_t size() const { return m_cameras.size(); }
    bool empty() const { return m_cameras.empty(); }
    const CameraRecord& operator[](size_t i) const { return m_cameras[i]; }
    const std::string& path() const { return m_path; }

    // Scene "up" estimated as the mean of the cameras' up vectors (-y in the
    // CV convention). Falls back to `fallback` if they cancel out.
    glm::vec3 averageUp(const glm::vec3& fallback) const;

private:
    CameraSet() = default;

    std::vector<CameraRecord> m_cameras;
    std::string m_path;
};
