#pragma once

#include "core/Node.hpp"
#include "io/PlyLoader.hpp"

#include <cstdint>
#include <vector>

// A set of 3D Gaussians stored in the reference 3DGS parameterization (the
// same values the PLY file holds):
//   position  - object-space mean
//   scaleLog  - log of the per-axis standard deviation (activation: exp)
//   rotation  - unnormalized quaternion (w, x, y, z)  (activation: normalize)
//   opacityLogit - (activation: sigmoid)
//   sh        - 16 RGB spherical-harmonic coefficients per Gaussian, laid out
//               [gaussian][coefficient][channel]; unused bands are zero.
//
// The cloud's Node transform places it in the world, so a splat can be
// parented to other nodes.
class GaussianCloud : public Node {
public:
    static constexpr uint32_t kMaxShCoeffs = 16;

    // Creates an empty cloud.
    static std::unique_ptr<GaussianCloud> create(const std::string& name = "GaussianCloud");
    // Creates a cloud from a 3DGS PLY file; returns nullptr on failure.
    // `progress` (optional) is told the fraction loaded and can cancel.
    static std::unique_ptr<GaussianCloud> createFromPly(const std::string& path,
                                                        const PlyLoader::ProgressCallback& progress = {});
    bool init(const std::string& name) override;
    bool initFromPly(const std::string& path, const PlyLoader::ProgressCallback& progress = {});

    size_t size() const { return position.size(); }
    bool empty() const { return position.empty(); }
    void resize(size_t count);

    // SH degree stored in the data (0..3).
    uint32_t shDegree = 0;

    std::vector<glm::vec3> position;
    std::vector<glm::vec3> scaleLog;
    std::vector<glm::vec4> rotation;
    std::vector<float> opacityLogit;
    std::vector<float> sh;

    // Activated values, as consumed by the renderer.
    glm::vec3 activatedScale(size_t i) const;
    glm::vec4 activatedRotation(size_t i) const;
    float activatedOpacity(size_t i) const;

    // Object-space bounds of the Gaussian means.
    void bounds(glm::vec3& minOut, glm::vec3& maxOut) const;

private:
    GaussianCloud() = default;
};
