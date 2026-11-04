#include "scene/Camera.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <cmath>

namespace {
// Flips Y and Z: converts between the OpenGL and CV camera conventions.
const glm::mat4 kFlipYZ(1, 0, 0, 0,
                        0, -1, 0, 0,
                        0, 0, -1, 0,
                        0, 0, 0, 1);
} // namespace

std::unique_ptr<Camera> Camera::create(const std::string& name) {
    std::unique_ptr<Camera> camera(new Camera());
    if (!camera->init(name)) return nullptr;
    return camera;
}

bool Camera::init(const std::string& name) {
    return Node::init(name);
}

void Camera::setPerspective(float fovYRadians, float nearPlane, float farPlane) {
    m_fovY = fovYRadians;
    m_near = nearPlane;
    m_far = farPlane;
    m_focal.reset();
}

void Camera::setFocalLengths(float fx, float fy, uint32_t referenceWidth, uint32_t referenceHeight) {
    m_focal = Focal{fx, fy, referenceWidth, referenceHeight};
    m_fovY = 2.0f * std::atan(0.5f * static_cast<float>(referenceHeight) / fy);
}

std::optional<glm::uvec2> Camera::referenceResolution() const {
    if (!m_focal) return std::nullopt;
    return glm::uvec2(m_focal->refWidth, m_focal->refHeight);
}

CameraIntrinsics Camera::intrinsics(uint32_t width, uint32_t height) const {
    CameraIntrinsics k;
    k.width = width;
    k.height = height;
    if (m_focal) {
        k.fx = m_focal->fx * static_cast<float>(width) / static_cast<float>(m_focal->refWidth);
        k.fy = m_focal->fy * static_cast<float>(height) / static_cast<float>(m_focal->refHeight);
    } else {
        k.fy = 0.5f * static_cast<float>(height) / std::tan(0.5f * m_fovY);
        k.fx = k.fy;
    }
    k.cx = 0.5f * static_cast<float>(width);
    k.cy = 0.5f * static_cast<float>(height);
    k.tanFovX = 0.5f * static_cast<float>(width) / k.fx;
    k.tanFovY = 0.5f * static_cast<float>(height) / k.fy;
    return k;
}

glm::mat4 Camera::viewMatrix() const {
    return glm::inverse(worldMatrix());
}

glm::mat4 Camera::viewMatrixCV() const {
    return kFlipYZ * viewMatrix();
}

glm::mat4 Camera::projectionMatrix(uint32_t width, uint32_t height) const {
    CameraIntrinsics k = intrinsics(width, height);
    glm::mat4 p = glm::perspectiveRH_ZO(2.0f * std::atan(k.tanFovY), k.tanFovX / k.tanFovY, m_near, m_far);
    p[1][1] *= -1.0f; // Vulkan clip space has +Y pointing down
    return p;
}

void Camera::setPoseFromCameraToWorldCV(const glm::mat4& c2wCV) {
    setWorldMatrix(c2wCV * kFlipYZ);
}
