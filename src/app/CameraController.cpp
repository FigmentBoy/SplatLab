#include "app/CameraController.hpp"

#include "app/Window.hpp"
#include "scene/Camera.hpp"

#include <cmath>

std::unique_ptr<CameraController> CameraController::create(Camera& camera, Window& window) {
    std::unique_ptr<CameraController> controller(new CameraController());
    if (!controller->init(camera, window)) return nullptr;
    return controller;
}

bool CameraController::init(Camera& camera, Window& window) {
    m_camera = &camera;
    m_window = &window;
    return true;
}

bool CameraController::mouseCaptured() const {
    return m_window->cursorCaptured();
}

void CameraController::releaseMouse() {
    m_window->setCursorCaptured(false);
}

bool CameraController::update(float dt, bool uiWantsMouse) {
    bool moved = false;

    // --- capture / release ------------------------------------------------
    if (mouseCaptured()) {
        if (m_window->keyPressed(GLFW_KEY_ESCAPE)) releaseMouse();
    } else if (m_window->mousePressed(GLFW_MOUSE_BUTTON_LEFT) && !uiWantsMouse) {
        m_window->setCursorCaptured(true);
        m_skipNextDelta = true; // the cursor jumps when it gets locked
    }

    // --- look (only while captured) --------------------------------------
    glm::dvec2 cursor = m_window->cursorPosition();
    if (mouseCaptured()) {
        if (!m_skipNextDelta) {
            glm::dvec2 delta = cursor - m_lastCursor;
            if (delta.x != 0.0 || delta.y != 0.0) {
                float yaw = -static_cast<float>(delta.x) * m_lookSensitivity;
                float pitch = -static_cast<float>(delta.y) * m_lookSensitivity;
                m_camera->rotateGlobal(glm::angleAxis(yaw, m_worldUp));
                m_camera->rotateLocal(glm::angleAxis(pitch, glm::vec3(1, 0, 0)));
                moved = true;
            }
        }
        m_skipNextDelta = false;
    }
    m_lastCursor = cursor;

    double scroll = m_window->consumeScroll();
    if (scroll != 0.0 && mouseCaptured()) {
        m_moveSpeed *= static_cast<float>(std::pow(1.2, scroll));
    }

    // --- move (only while captured) -------------------------------------
    if (!mouseCaptured()) return moved;
    // Game-style: W/S/A/D stay in the horizontal plane (perpendicular to the
    // world up axis) whatever the pitch, and Space/Shift move straight up/down.
    glm::vec3 right = m_camera->right() - glm::dot(m_camera->right(), m_worldUp) * m_worldUp;
    if (glm::dot(right, right) < 1e-8f) right = glm::cross(m_camera->forward(), m_worldUp); // rolled 90 degrees
    right = glm::normalize(right);
    glm::vec3 forward = glm::cross(m_worldUp, right);

    glm::vec3 move(0.0f);
    if (m_window->keyDown(GLFW_KEY_W)) move += forward;
    if (m_window->keyDown(GLFW_KEY_S)) move -= forward;
    if (m_window->keyDown(GLFW_KEY_D)) move += right;
    if (m_window->keyDown(GLFW_KEY_A)) move -= right;
    if (m_window->keyDown(GLFW_KEY_SPACE)) move += m_worldUp;
    if (m_window->keyDown(GLFW_KEY_LEFT_SHIFT) || m_window->keyDown(GLFW_KEY_RIGHT_SHIFT)) move -= m_worldUp;
    if (glm::dot(move, move) > 0.0f) {
        bool sprint = m_window->keyDown(GLFW_KEY_LEFT_CONTROL) || m_window->keyDown(GLFW_KEY_RIGHT_CONTROL);
        float speed = m_moveSpeed * (sprint ? 4.0f : 1.0f);
        m_camera->translate(glm::normalize(move) * speed * dt);
        moved = true;
    }
    return moved;
}
