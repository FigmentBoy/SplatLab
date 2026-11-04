#pragma once

#include "vk/VkCommon.hpp"

#include <glm/glm.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

class Buffer;
class Camera;
class ComputePipeline;
class GaussianCloud;
class GpuRadixSort;
class GpuScan;
class GpuTimer;
class VulkanContext;

struct RenderSettings {
    glm::vec3 background{0.0f};
    float scaleModifier = 1.0f;
    int shDegree = -1;     // -1: use every band stored in the scene
    float nearCull = 0.2f; // view-space depth below which Gaussians are culled (reference value)
    // Bin Gaussians only into tiles where they can reach alpha >= 1/255
    // (output is identical; see shaders/common.glsl). false bins them into the
    // reference rasterizer's 3-sigma squares.
    bool tightTiles = true;
};

struct RenderStats {
    uint32_t numGaussians = 0;
    uint32_t numVisible = 0;   // Gaussians binned into at least one tile
    uint32_t numInstances = 0; // Gaussian/tile pairs that were sorted and composited
    uint32_t width = 0;
    uint32_t height = 0;
};

// Mirrors FrameParamsRef in shaders/common.glsl (std430).
struct FrameParams {
    glm::mat4 modelView;
    glm::vec4 camPosObj;
    glm::vec4 intrinsics;
    glm::vec4 frustum;
    glm::vec4 background;
    glm::uvec4 dims;
    glm::uvec4 misc;
};

// Forward-only tile-based Gaussian splat rasterizer, implemented entirely
// with Vulkan compute shaders and recorded as a single submission:
//   1. preprocess   - project Gaussians, 2D conics, SH colour, tile counts
//   2. compact      - visible Gaussians -> (depth, index) pairs
//   3. depth sort   - stable radix sort of the visible Gaussians by depth
//   4. offsets      - tile counts in depth order, prefix sum
//   5. duplicate    - one (tile, Gaussian) instance per overlap, in depth order
//   6. tile sort    - stable radix sort of the instances by tile id only
//   7. tile ranges  - start/end of each tile's list
//   8. render       - per-tile front-to-back alpha compositing
//   9. pack         - float CHW image -> RGBA8 for display / PNG
// Sorting by depth, then stably by tile, yields the same per-tile order as the
// reference's single sort of (tile | depth) keys, ties included, while the
// expensive 32-bit depth passes run over Gaussians instead of instances.
// Intermediates the backward pass will need (per-Gaussian projections,
// sorted per-tile lists, final transmittance, last contributor) are kept in
// GPU buffers after each render.
class SplatRenderer {
public:
    static std::unique_ptr<SplatRenderer> create(VulkanContext& ctx);
    bool init(VulkanContext& ctx);
    ~SplatRenderer();

    SplatRenderer(const SplatRenderer&) = delete;
    SplatRenderer& operator=(const SplatRenderer&) = delete;

    // Uploads the Gaussians (activated) to the GPU. The cloud must outlive
    // subsequent render() calls, since its world transform is read per frame.
    bool setScene(const GaussianCloud& cloud);

    // Renders the scene from `camera` at width x height and waits for the GPU.
    // swapRedBlue makes packedImage() BGRA instead of RGBA.
    bool render(const Camera& camera, uint32_t width, uint32_t height, const RenderSettings& settings,
                bool swapRedBlue = false);

    // Outputs of the last render().
    const Buffer& colorImage() const { return *m_color; }       // float, planar CHW
    const Buffer& finalTransmittance() const { return *m_finalT; } // float, H*W
    const Buffer& lastContributor() const { return *m_nContrib; }  // uint,  H*W
    const Buffer& packedImage() const { return *m_packed; }     // RGBA8 (or BGRA8), H*W
    const RenderStats& stats() const { return m_stats; }

    // Copies GPU buffer contents to host memory.
    void download(const Buffer& src, void* dst, VkDeviceSize bytes);

    // Per-stage GPU timings (label, ms) of the last render(), when enabled.
    void setProfiling(bool enabled) { m_profiling = enabled; }
    std::vector<std::pair<std::string, double>> stageTimings() const;

private:
    SplatRenderer() = default;

    bool ensurePixelBuffers(uint32_t width, uint32_t height);
    bool ensureInstanceBuffers(uint32_t count);
    void recordFrame(VkCommandBuffer cmd, uint32_t tilesX, uint32_t tilesY, bool swapRedBlue);

    VulkanContext* m_ctx = nullptr;
    const GaussianCloud* m_cloud = nullptr;
    uint32_t m_numGaussians = 0;
    uint32_t m_sceneShDegree = 0;

    std::unique_ptr<GpuScan> m_scan;
    std::unique_ptr<GpuRadixSort> m_sort;
    std::unique_ptr<ComputePipeline> m_preprocess;
    std::unique_ptr<ComputePipeline> m_compact;
    std::unique_ptr<ComputePipeline> m_gatherTiles;
    std::unique_ptr<ComputePipeline> m_duplicate;
    std::unique_ptr<ComputePipeline> m_tileRanges;
    std::unique_ptr<ComputePipeline> m_render;
    std::unique_ptr<ComputePipeline> m_pack;

    // Frame constants and readback of the visible and instance counts.
    std::unique_ptr<Buffer> m_frameParams;
    std::unique_ptr<Buffer> m_countReadback;

    // Scene (per Gaussian).
    std::unique_ptr<Buffer> m_positionOpacity;
    std::unique_ptr<Buffer> m_scales;
    std::unique_ptr<Buffer> m_rotations;
    std::unique_ptr<Buffer> m_sh;

    // Geometry state (per Gaussian).
    std::unique_ptr<Buffer> m_means2D;
    std::unique_ptr<Buffer> m_conicOpacity;
    std::unique_ptr<Buffer> m_colorDepth;
    std::unique_ptr<Buffer> m_radii;
    std::unique_ptr<Buffer> m_tilesTouched;
    std::unique_ptr<Buffer> m_visibleScan;  // visible flags, scanned in place (N + 1; [N] = visible count)
    std::unique_ptr<Buffer> m_depthKeys[2]; // depth sort ping-pong
    std::unique_ptr<Buffer> m_depthOrder[2];
    std::unique_ptr<Buffer> m_tileOffsets;  // sorted tile counts, scanned in place (N + 1; [N] = instances)
    int m_depthSortedSet = 0;

    // Binning state (per instance), ping-pong pair for the tile sort.
    std::unique_ptr<Buffer> m_tileKeys[2];
    std::unique_ptr<Buffer> m_values[2];
    uint32_t m_instanceCapacity = 0;
    int m_sortedSet = 0;

    // Image state (per tile / pixel).
    std::unique_ptr<Buffer> m_tileRangesBuf;
    std::unique_ptr<Buffer> m_color;
    std::unique_ptr<Buffer> m_finalT;
    std::unique_ptr<Buffer> m_nContrib;
    std::unique_ptr<Buffer> m_packed;
    uint32_t m_pixelWidth = 0;
    uint32_t m_pixelHeight = 0;

    RenderStats m_stats;
    std::unique_ptr<GpuTimer> m_timer;
    bool m_profiling = false;
};
