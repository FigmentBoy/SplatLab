#include "core/Node.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include <algorithm>

std::unique_ptr<Node> Node::create(const std::string& name) {
    std::unique_ptr<Node> node(new Node());
    if (!node->init(name)) return nullptr;
    return node;
}

bool Node::init(const std::string& name) {
    m_name = name;
    return true;
}

Node::~Node() {
    setParent(nullptr);
    for (Node* child : m_children) child->m_parent = nullptr;
}

void Node::setParent(Node* parent, bool keepWorldTransform) {
    if (parent == m_parent) return;
    for (Node* p = parent; p; p = p->m_parent) {
        if (p == this) return; // would create a cycle
    }

    glm::mat4 world = worldMatrix();
    if (m_parent) {
        auto& siblings = m_parent->m_children;
        siblings.erase(std::remove(siblings.begin(), siblings.end(), this), siblings.end());
    }
    m_parent = parent;
    if (m_parent) m_parent->m_children.push_back(this);
    if (keepWorldTransform) setWorldMatrix(world);
}

void Node::lookAt(const glm::vec3& target, const glm::vec3& up) {
    glm::vec3 dir = target - m_position;
    if (glm::dot(dir, dir) < 1e-12f) return;
    // glm::quatLookAt returns a rotation whose -Z axis faces `dir`.
    setRotation(glm::quatLookAt(glm::normalize(dir), up));
}

void Node::setLocalMatrix(const glm::mat4& m) {
    glm::vec3 scale, translation, skew;
    glm::vec4 perspective;
    glm::quat rotation;
    if (glm::decompose(m, scale, rotation, translation, skew, perspective)) {
        m_position = translation;
        m_rotation = glm::normalize(rotation);
        m_scale = scale;
    }
}

void Node::setWorldMatrix(const glm::mat4& m) {
    setLocalMatrix(m_parent ? glm::inverse(m_parent->worldMatrix()) * m : m);
}

glm::mat4 Node::localMatrix() const {
    glm::mat4 t = glm::translate(glm::mat4(1.0f), m_position);
    glm::mat4 r = glm::mat4_cast(m_rotation);
    glm::mat4 s = glm::scale(glm::mat4(1.0f), m_scale);
    return t * r * s;
}

glm::mat4 Node::worldMatrix() const {
    return m_parent ? m_parent->worldMatrix() * localMatrix() : localMatrix();
}

glm::vec3 Node::worldPosition() const {
    return glm::vec3(worldMatrix()[3]);
}

glm::vec3 Node::right() const {
    return glm::normalize(glm::vec3(worldMatrix()[0]));
}

glm::vec3 Node::up() const {
    return glm::normalize(glm::vec3(worldMatrix()[1]));
}

glm::vec3 Node::forward() const {
    return -glm::normalize(glm::vec3(worldMatrix()[2]));
}
