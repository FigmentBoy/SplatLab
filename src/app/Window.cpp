#include "app/Window.hpp"

#include <algorithm>
#include <iostream>

std::unique_ptr<Window> Window::create(uint32_t width, uint32_t height, const std::string& title) {
    std::unique_ptr<Window> window(new Window());
    if (!window->init(width, height, title)) return nullptr;
    return window;
}

bool Window::init(uint32_t width, uint32_t height, const std::string& title) {
    // Route GLFW's Vulkan calls through the loader volk already opened.
    glfwInitVulkanLoader(vkGetInstanceProcAddr);
    if (!glfwInit()) {
        std::cerr << "glfwInit failed\n";
        return false;
    }
    if (!glfwVulkanSupported()) {
        std::cerr << "GLFW reports no Vulkan support\n";
        return false;
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    m_window = glfwCreateWindow(static_cast<int>(width), static_cast<int>(height), title.c_str(), nullptr, nullptr);
    if (!m_window) {
        std::cerr << "glfwCreateWindow failed\n";
        return false;
    }
    glfwSetWindowUserPointer(m_window, this);
    glfwSetFramebufferSizeCallback(m_window, framebufferSizeCallback);
    glfwSetScrollCallback(m_window, scrollCallback);
    glfwSetKeyCallback(m_window, keyCallback);
    glfwSetMouseButtonCallback(m_window, mouseButtonCallback);
    return true;
}

Window::~Window() {
    if (m_window) glfwDestroyWindow(m_window);
    glfwTerminate();
}

std::vector<const char*> Window::requiredInstanceExtensions() {
    uint32_t count = 0;
    const char** names = glfwGetRequiredInstanceExtensions(&count);
    return std::vector<const char*>(names, names + count);
}

VkSurfaceKHR Window::createSurface(VkInstance instance) const {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (glfwCreateWindowSurface(instance, m_window, nullptr, &surface) != VK_SUCCESS) return VK_NULL_HANDLE;
    return surface;
}

bool Window::shouldClose() const {
    return glfwWindowShouldClose(m_window);
}

void Window::requestClose() {
    glfwSetWindowShouldClose(m_window, GLFW_TRUE);
}

void Window::cancelClose() {
    glfwSetWindowShouldClose(m_window, GLFW_FALSE);
}

void Window::waitEvents(double timeout) {
    glfwWaitEventsTimeout(timeout);
}

void Window::pollEvents() {
    glfwPollEvents();
}

void Window::setTitle(const std::string& title) {
    glfwSetWindowTitle(m_window, title.c_str());
}

glm::uvec2 Window::framebufferSize() const {
    int w = 0, h = 0;
    glfwGetFramebufferSize(m_window, &w, &h);
    return glm::uvec2(static_cast<uint32_t>(std::max(w, 0)), static_cast<uint32_t>(std::max(h, 0)));
}

bool Window::consumeResized() {
    bool r = m_resized;
    m_resized = false;
    return r;
}

bool Window::keyDown(int key) const {
    return glfwGetKey(m_window, key) == GLFW_PRESS;
}

bool Window::keyPressed(int key) {
    auto it = std::find(m_pressedKeys.begin(), m_pressedKeys.end(), key);
    if (it == m_pressedKeys.end()) return false;
    m_pressedKeys.erase(it);
    return true;
}

bool Window::mouseDown(int button) const {
    return glfwGetMouseButton(m_window, button) == GLFW_PRESS;
}

bool Window::mousePressed(int button) {
    auto it = std::find(m_pressedButtons.begin(), m_pressedButtons.end(), button);
    if (it == m_pressedButtons.end()) return false;
    m_pressedButtons.erase(it);
    return true;
}

void Window::setCursorCaptured(bool captured) {
    if (captured == m_cursorCaptured) return;
    m_cursorCaptured = captured;
    glfwSetInputMode(m_window, GLFW_CURSOR, captured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    if (glfwRawMouseMotionSupported()) {
        glfwSetInputMode(m_window, GLFW_RAW_MOUSE_MOTION, captured ? GLFW_TRUE : GLFW_FALSE);
    }
}

void Window::clearPresses() {
    m_pressedKeys.clear();
    m_pressedButtons.clear();
}

glm::dvec2 Window::cursorPosition() const {
    double x = 0, y = 0;
    glfwGetCursorPos(m_window, &x, &y);
    return {x, y};
}

double Window::consumeScroll() {
    double s = m_scroll;
    m_scroll = 0.0;
    return s;
}

void Window::framebufferSizeCallback(GLFWwindow* window, int, int) {
    static_cast<Window*>(glfwGetWindowUserPointer(window))->m_resized = true;
}

void Window::scrollCallback(GLFWwindow* window, double, double dy) {
    static_cast<Window*>(glfwGetWindowUserPointer(window))->m_scroll += dy;
}

void Window::keyCallback(GLFWwindow* window, int key, int, int action, int) {
    if (action == GLFW_PRESS) {
        static_cast<Window*>(glfwGetWindowUserPointer(window))->m_pressedKeys.push_back(key);
    }
}

void Window::mouseButtonCallback(GLFWwindow* window, int button, int action, int) {
    if (action == GLFW_PRESS) {
        static_cast<Window*>(glfwGetWindowUserPointer(window))->m_pressedButtons.push_back(button);
    }
}
