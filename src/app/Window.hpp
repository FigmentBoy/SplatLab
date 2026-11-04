#pragma once

#include <volk.h>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include <memory>
#include <string>
#include <vector>

// GLFW window without a client API (Vulkan presents to it).
class Window {
public:
    static std::unique_ptr<Window> create(uint32_t width, uint32_t height, const std::string& title);
    bool init(uint32_t width, uint32_t height, const std::string& title);
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    // Instance extensions GLFW needs for surface creation.
    static std::vector<const char*> requiredInstanceExtensions();
    VkSurfaceKHR createSurface(VkInstance instance) const;

    bool shouldClose() const;
    void requestClose();
    void pollEvents();
    void setTitle(const std::string& title);

    glm::uvec2 framebufferSize() const;
    // True once after the framebuffer was resized.
    bool consumeResized();

    bool keyDown(int key) const;
    // True once per physical key press.
    bool keyPressed(int key);
    bool mouseDown(int button) const;
    // True once per physical button press.
    bool mousePressed(int button);
    glm::dvec2 cursorPosition() const;
    // Hides and locks the cursor (unbounded, raw motion when available) or restores it.
    void setCursorCaptured(bool captured);
    bool cursorCaptured() const { return m_cursorCaptured; }
    // Drops queued key/button presses (e.g. ones the UI consumed).
    void clearPresses();
    // Accumulated scroll since the last call.
    double consumeScroll();

    GLFWwindow* handle() const { return m_window; }

private:
    Window() = default;

    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    static void scrollCallback(GLFWwindow* window, double dx, double dy);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);

    GLFWwindow* m_window = nullptr;
    bool m_resized = false;
    bool m_cursorCaptured = false;
    double m_scroll = 0.0;
    std::vector<int> m_pressedKeys;
    std::vector<int> m_pressedButtons;
};
