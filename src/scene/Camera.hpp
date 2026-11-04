#pragma once

#include "core/Node.hpp"

#include <optional>

// Pixel-space pinhole intrinsics for a specific output resolution.
struct CameraIntrinsics {
    uint32_t width = 0;
    uint32_t height = 0;
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f; // principal point, image-corner origin (centre of pixel 0 is 0.5)
    float cy = 0.0f;
    float tanFovX = 0.0f;
    float tanFovY = 0.0f;
};

// Standard perspective camera. As a Node it uses the OpenGL convention
// (looks down -Z, +Y up); viewMatrixCV() provides the computer-vision
// convention (+Z forward, +Y down) used by 3DGS / COLMAP.
//
// By default the camera is defined by a vertical field of view and square
// pixels. A camera loaded from a dataset can instead carry explicit focal
// lengths measured at a reference resolution; they are rescaled to the output
// resolution in intrinsics().
class Camera : public Node {
public:
    static std::unique_ptr<Camera> create(const std::string& name = "Camera");
    bool init(const std::string& name) override;

    void setPerspective(float fovYRadians, float nearPlane = 0.01f, float farPlane = 1000.0f);
    void setFocalLengths(float fx, float fy, uint32_t referenceWidth, uint32_t referenceHeight);
    void clearFocalLengths() { m_focal.reset(); }

    float fovY() const { return m_fovY; }
    float nearPlane() const { return m_near; }
    float farPlane() const { return m_far; }

    // Native resolution of a dataset camera, if it has one.
    std::optional<glm::uvec2> referenceResolution() const;

    CameraIntrinsics intrinsics(uint32_t width, uint32_t height) const;

    // World -> camera, OpenGL convention.
    glm::mat4 viewMatrix() const;
    // World -> camera, computer-vision convention (x right, y down, z forward).
    glm::mat4 viewMatrixCV() const;
    // Vulkan-style perspective projection (depth 0..1) matching intrinsics().
    glm::mat4 projectionMatrix(uint32_t width, uint32_t height) const;

    // Sets the world pose from a camera-to-world matrix in the CV convention
    // (as stored by COLMAP / the 3DGS cameras.json).
    void setPoseFromCameraToWorldCV(const glm::mat4& c2wCV);

private:
    Camera() = default;

    struct Focal {
        float fx, fy;
        uint32_t refWidth, refHeight;
    };

    float m_fovY = glm::radians(60.0f);
    float m_near = 0.01f;
    float m_far = 1000.0f;
    std::optional<Focal> m_focal;
};
