#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <memory>
#include <string>
#include <vector>

// Base class for every 3D object. A node has a local TRS transform relative to
// its parent; its world transform is the product of all ancestors' transforms.
//
// Nodes are owned through std::unique_ptr by whoever created them. The
// parent/child links are non-owning, and a node detaches itself from its
// parent and children when destroyed.
//
// Local axes follow the usual OpenGL convention: +X right, +Y up, -Z forward.
class Node {
public:
    static std::unique_ptr<Node> create(const std::string& name = "Node");
    virtual bool init(const std::string& name);
    virtual ~Node();

    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

    // --- hierarchy -------------------------------------------------------
    // Re-parents this node. With keepWorldTransform the local transform is
    // adjusted so the node does not move in world space.
    void setParent(Node* parent, bool keepWorldTransform = false);
    Node* parent() const { return m_parent; }
    const std::vector<Node*>& children() const { return m_children; }
    void addChild(Node* child, bool keepWorldTransform = false) { child->setParent(this, keepWorldTransform); }

    const std::string& name() const { return m_name; }
    void setName(const std::string& name) { m_name = name; }

    // --- local transform -------------------------------------------------
    const glm::vec3& position() const { return m_position; }
    const glm::quat& rotation() const { return m_rotation; }
    const glm::vec3& scale() const { return m_scale; }
    void setPosition(const glm::vec3& p) { m_position = p; }
    void setRotation(const glm::quat& q) { m_rotation = glm::normalize(q); }
    void setScale(const glm::vec3& s) { m_scale = s; }

    void translate(const glm::vec3& delta) { m_position += delta; }
    // Rotation applied in the parent's frame (pre-multiplied).
    void rotateGlobal(const glm::quat& q) { setRotation(q * m_rotation); }
    // Rotation applied in the node's own frame (post-multiplied).
    void rotateLocal(const glm::quat& q) { setRotation(m_rotation * q); }

    // Orients the node so its -Z axis points at `target` (in parent space).
    void lookAt(const glm::vec3& target, const glm::vec3& up = glm::vec3(0, 1, 0));

    // Sets the local transform from a matrix (decomposed into TRS; shear is lost).
    void setLocalMatrix(const glm::mat4& m);
    void setWorldMatrix(const glm::mat4& m);

    glm::mat4 localMatrix() const;
    glm::mat4 worldMatrix() const;

    glm::vec3 worldPosition() const;
    // World-space directions of the local axes.
    glm::vec3 right() const;
    glm::vec3 up() const;
    glm::vec3 forward() const;

protected:
    // Protected rather than private so derived nodes can construct their base.
    Node() = default;

private:
    std::string m_name;
    Node* m_parent = nullptr;
    std::vector<Node*> m_children;

    glm::vec3 m_position{0.0f};
    glm::quat m_rotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 m_scale{1.0f};
};
