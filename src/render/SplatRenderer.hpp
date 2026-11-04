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

// A pinhole view in the computer-vision convention (x right, y down, z forward).
struct ViewParams {
    glm::mat4 worldToCamera{1.0f}; // object/world space -> camera space (CV convention)
    glm::vec3 cameraPosition{0.0f}; // camera centre in the same space as the Gaussians
    float fx = 0.0f, fy = 0.0f;
    float cx = 0.0f, cy = 0.0f;     // principal point, image-corner origin (pixel 0 centre at 0.5)
    uint32_t width = 0, height = 0;
};

// Raw (unactivated) Gaussian parameters owned by the caller, as the trainer
// optimizes them. All arrays are tightly packed float32:
//   means [N][3], scaleLog [N][3] (exp), rotation [N][4] raw (w,x,y,z) (normalize),
//   opacityLogit [N] (sigmoid), shDc [N][3], shRest [N][15][3].
struct TrainingParams {
    VkDeviceAddress means = 0;
    VkDeviceAddress scaleLog = 0;
    VkDeviceAddress rotation = 0;
    VkDeviceAddress opacityLogit = 0;
    VkDeviceAddress shDc = 0;
    VkDeviceAddress shRest = 0;
    uint32_t count = 0;
    uint32_t shDegree = 0; // active SH degree (0..3)
};

// Outputs of backward(): gradients with the layouts of TrainingParams, plus
// dL/d(mean2D) in pixels ([N][2]) for densification statistics.
struct TrainingGrads {
    VkDeviceAddress means = 0;
    VkDeviceAddress scaleLog = 0;
    VkDeviceAddress rotation = 0;
    VkDeviceAddress opacityLogit = 0;
    VkDeviceAddress shDc = 0;
    VkDeviceAddress shRest = 0;
    VkDeviceAddress means2D = 0;
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

// Differentiable tile-based Gaussian splat rasterizer, implemented entirely
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
// Intermediates the backward pass needs (per-Gaussian projections, sorted
// per-tile lists, final transmittance, last contributor) are kept in GPU
// buffers after each render.
//
// Training (renderTraining / backward) reads raw parameters from caller-owned
// buffers (e.g. memory shared with PyTorch) and writes gradients back:
//   10. render_backward     - per-pixel reverse traversal -> dL/d(mean2D,
//                             conic, opacity, colour), subgroup-reduced atomics
//   11. preprocess_backward - -> dL/d(raw parameters)
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

    // --- training -------------------------------------------------------
    // Whether the device supports backward() (float atomics, subgroup vote).
    bool supportsBackward() const { return m_renderBackward != nullptr; }

    // Renders the caller's raw parameters. The planar CHW float image goes to
    // `outImage` (>= 3*W*H floats) and, if given, the per-Gaussian screen
    // radii (int32, 0 = not rendered) to `outRadii`. Waits for the GPU.
    bool renderTraining(const TrainingParams& scene, const ViewParams& view, const RenderSettings& settings,
                        const Buffer& outImage, const Buffer* outRadii);

    // Backward pass of the last renderTraining() call, which must have used
    // the same `scene`. `gradImage` holds dL/d(image) (planar CHW floats).
    // Writes every gradient entry of the scene's Gaussians. Waits for the GPU.
    bool backward(const TrainingParams& scene, const TrainingGrads& grads, const Buffer& gradImage);

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

    struct FrameTarget {
        VkDeviceAddress color = 0;       // planar CHW float output
        bool pack = false;               // also write packedImage()
        bool swapRedBlue = false;
        const Buffer* radiiOut = nullptr; // copy of the radii (training)
    };

    bool ensurePixelBuffers(uint32_t width, uint32_t height);
    bool ensureGaussianBuffers(uint32_t count);
    bool ensureInstanceBuffers(uint32_t count);
    bool renderView(const ViewParams& view, const RenderSettings& settings, uint32_t shDegree,
                    const FrameTarget& target);
    void recordFrame(VkCommandBuffer cmd, uint32_t tilesX, uint32_t tilesY, const FrameTarget& target);

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
    std::unique_ptr<ComputePipeline> m_preprocessTrain;
    std::unique_ptr<ComputePipeline> m_renderBackward;
    std::unique_ptr<ComputePipeline> m_preprocessBackward;

    // Which preprocess runs: the uploaded scene or the caller's raw parameters.
    bool m_trainingInputs = false;
    TrainingParams m_trainingParams;
    uint32_t m_lastTilesX = 0, m_lastTilesY = 0;
    bool m_backwardReady = false; // last frame was a successful renderTraining()

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
    std::unique_ptr<Buffer> m_gradAccum;    // backward: 12 floats per Gaussian
    uint32_t m_gaussianCapacity = 0;
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
