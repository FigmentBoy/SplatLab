#pragma once

#include <glm/glm.hpp>

#include <memory>

class Camera;
class Window;

// First-person fly controls for a Camera node:
//   left click on the scene - capture the mouse (cursor hidden)
//   Esc                     - release the mouse (cursor shown; camera ignores input)
// While captured: mouse pans/tilts, WASD move in the horizontal plane (the
// plane perpendicular to the world up axis), Space/Shift move up/down, Ctrl
// moves faster, scroll changes movement speed.
// Yaw turns about a fixed world "up" axis so the horizon stays level.
class CameraController {
public:
    static std::unique_ptr<CameraController> create(Camera& camera, Window& window);
    bool init(Camera& camera, Window& window);

    // uiWantsMouse: the UI is using the mouse this frame (a click there must
    // not capture it). Returns true if the camera moved.
    bool update(float dt, bool uiWantsMouse);

    bool mouseCaptured() const;
    void releaseMouse();

    void setWorldUp(const glm::vec3& up) { m_worldUp = glm::normalize(up); }
    void setMoveSpeed(float speed) { m_moveSpeed = speed; }
    float moveSpeed() const { return m_moveSpeed; }

private:
    CameraController() = default;

    Camera* m_camera = nullptr;
    Window* m_window = nullptr;
    glm::vec3 m_worldUp{0.0f, 1.0f, 0.0f};
    float m_moveSpeed = 1.0f;
    float m_lookSensitivity = 0.003f;
    glm::dvec2 m_lastCursor{0.0};
    bool m_skipNextDelta = false;
};
