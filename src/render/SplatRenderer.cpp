#include "render/SplatRenderer.hpp"

#include "render/GpuRadixSort.hpp"
#include "render/GpuScan.hpp"
#include "scene/Camera.hpp"
#include "scene/GaussianCloud.hpp"
#include "vk/Buffer.hpp"
#include "vk/ComputePipeline.hpp"
#include "vk/GpuTimer.hpp"
#include "vk/VulkanContext.hpp"

#include "shaders/compact.spv.h"
#include "shaders/duplicate.spv.h"
#include "shaders/gather_tiles.spv.h"
#include "shaders/pack.spv.h"
#include "shaders/preprocess.spv.h"
#include "shaders/preprocess_backward.spv.h"
#include "shaders/preprocess_train.spv.h"
#include "shaders/render.spv.h"
#include "shaders/render_backward.spv.h"
#include "shaders/tile_ranges.spv.h"

#include <algorithm>
#include <cstring>
#include <iostream>

namespace {

constexpr uint32_t kTileSize = 16;
constexpr uint32_t kGroupSize = 256;

struct PreprocessPush {
    VkDeviceAddress params;
    VkDeviceAddress positionOpacity;
    VkDeviceAddress scales;
    VkDeviceAddress rotations;
    VkDeviceAddress sh;
    VkDeviceAddress means2D;
    VkDeviceAddress conicOpacity;
    VkDeviceAddress colorDepth;
    VkDeviceAddress radii;
    VkDeviceAddress tilesTouched;
    VkDeviceAddress visible;
};

struct PreprocessTrainPush {
    VkDeviceAddress params;
    VkDeviceAddress means;
    VkDeviceAddress scaleLog;
    VkDeviceAddress rotation;
    VkDeviceAddress opacityLogit;
    VkDeviceAddress shDc;
    VkDeviceAddress shRest;
    VkDeviceAddress means2D;
    VkDeviceAddress conicOpacity;
    VkDeviceAddress colorDepth;
    VkDeviceAddress radii;
    VkDeviceAddress tilesTouched;
    VkDeviceAddress visible;
};

struct RenderBackwardPush {
    VkDeviceAddress params;
    VkDeviceAddress ranges;
    VkDeviceAddress values;
    VkDeviceAddress means2D;
    VkDeviceAddress conicOpacity;
    VkDeviceAddress colorDepth;
    VkDeviceAddress finalT;
    VkDeviceAddress nContrib;
    VkDeviceAddress gradImage;
    VkDeviceAddress gradAccum;
};

struct PreprocessBackwardPush {
    VkDeviceAddress params;
    VkDeviceAddress means;
    VkDeviceAddress scaleLog;
    VkDeviceAddress rotation;
    VkDeviceAddress opacityLogit;
    VkDeviceAddress shDc;
    VkDeviceAddress shRest;
    VkDeviceAddress radii;
    VkDeviceAddress gradAccum;
    VkDeviceAddress gMeans;
    VkDeviceAddress gScaleLog;
    VkDeviceAddress gRotation;
    VkDeviceAddress gOpacityLogit;
    VkDeviceAddress gShDc;
    VkDeviceAddress gShRest;
    VkDeviceAddress gMeans2D;
};

// Per-Gaussian accumulator of render_backward: mean2D xy, conic (xx, xy, yy),
// opacity, 2 unused, rgb, 1 unused.
constexpr uint32_t kGradAccumFloats = 12;

struct CompactPush {
    VkDeviceAddress params;
    VkDeviceAddress visibleScan;
    VkDeviceAddress colorDepth;
    VkDeviceAddress depthKeys;
    VkDeviceAddress indices;
};

struct GatherTilesPush {
    VkDeviceAddress params;
    VkDeviceAddress visibleCount;
    VkDeviceAddress sortedIndices;
    VkDeviceAddress tilesTouched;
    VkDeviceAddress counts;
};

struct DuplicatePush {
    VkDeviceAddress params;
    VkDeviceAddress visibleCount;
    VkDeviceAddress sortedIndices;
    VkDeviceAddress offsets;
    VkDeviceAddress means2D;
    VkDeviceAddress conicOpacity;
    VkDeviceAddress radii;
    VkDeviceAddress tileKeys;
    VkDeviceAddress values;
    uint32_t capacity;
    uint32_t sentinel;
};

struct TileRangesPush {
    VkDeviceAddress keysTile;
    VkDeviceAddress ranges;
    VkDeviceAddress countBuf;
    uint32_t capacity;
    uint32_t numTiles;
};

struct RenderPush {
    VkDeviceAddress params;
    VkDeviceAddress ranges;
    VkDeviceAddress values;
    VkDeviceAddress means2D;
    VkDeviceAddress conicOpacity;
    VkDeviceAddress colorDepth;
    VkDeviceAddress outColor;
    VkDeviceAddress outFinalT;
    VkDeviceAddress outNContrib;
};

struct PackPush {
    VkDeviceAddress outPacked;
    VkDeviceAddress color;
    uint32_t width;
    uint32_t height;
    uint32_t swapRedBlue;
};

uint32_t divUp(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

uint32_t bitWidth(uint32_t v) {
    uint32_t bits = 0;
    while (v) {
        ++bits;
        v >>= 1;
    }
    return bits;
}

std::unique_ptr<Buffer> deviceBuffer(VulkanContext& ctx, VkDeviceSize bytes) {
    return Buffer::create(ctx, bytes, Buffer::Memory::DeviceLocal);
}

} // namespace

std::unique_ptr<SplatRenderer> SplatRenderer::create(VulkanContext& ctx) {
    std::unique_ptr<SplatRenderer> renderer(new SplatRenderer());
    if (!renderer->init(ctx)) return nullptr;
    return renderer;
}

bool SplatRenderer::init(VulkanContext& ctx) {
    m_ctx = &ctx;
    m_scan = GpuScan::create(ctx);
    if (!m_scan) return false;
    m_sort = GpuRadixSort::create(ctx, *m_scan);
    m_preprocess = ComputePipeline::create(ctx, kSpv_preprocess, kSpv_preprocess_size, sizeof(PreprocessPush));
    m_compact = ComputePipeline::create(ctx, kSpv_compact, kSpv_compact_size, sizeof(CompactPush));
    m_gatherTiles = ComputePipeline::create(ctx, kSpv_gather_tiles, kSpv_gather_tiles_size, sizeof(GatherTilesPush));
    m_duplicate = ComputePipeline::create(ctx, kSpv_duplicate, kSpv_duplicate_size, sizeof(DuplicatePush));
    m_tileRanges = ComputePipeline::create(ctx, kSpv_tile_ranges, kSpv_tile_ranges_size, sizeof(TileRangesPush));
    m_render = ComputePipeline::create(ctx, kSpv_render, kSpv_render_size, sizeof(RenderPush));
    m_pack = ComputePipeline::create(ctx, kSpv_pack, kSpv_pack_size, sizeof(PackPush));
    m_frameParams = Buffer::create(ctx, sizeof(FrameParams), Buffer::Memory::Upload);
    m_countReadback = Buffer::create(ctx, 2 * sizeof(uint32_t), Buffer::Memory::Readback);
    m_timer = GpuTimer::create(ctx);
    m_preprocessTrain =
        ComputePipeline::create(ctx, kSpv_preprocess_train, kSpv_preprocess_train_size, sizeof(PreprocessTrainPush));
    if (!m_preprocessTrain) return false;
    // The backward shaders need float atomics and subgroup votes.
    if (ctx.supportsBackward()) {
        m_renderBackward = ComputePipeline::create(ctx, kSpv_render_backward, kSpv_render_backward_size,
                                                   sizeof(RenderBackwardPush));
        m_preprocessBackward = ComputePipeline::create(ctx, kSpv_preprocess_backward,
                                                       kSpv_preprocess_backward_size, sizeof(PreprocessBackwardPush));
        if (!m_renderBackward || !m_preprocessBackward) return false;
    }
    return m_sort && m_preprocess && m_compact && m_gatherTiles && m_duplicate && m_tileRanges && m_render &&
           m_pack && m_frameParams && m_countReadback;
}

SplatRenderer::~SplatRenderer() {
    if (m_ctx) m_ctx->waitIdle();
}

bool SplatRenderer::setScene(const GaussianCloud& cloud) {
    m_ctx->waitIdle();
    const size_t n = cloud.size();
    if (n == 0) {
        std::cerr << "SplatRenderer: scene is empty\n";
        return false;
    }
    if (n >= (1ull << 31)) {
        std::cerr << "SplatRenderer: too many Gaussians\n";
        return false;
    }

    std::vector<glm::vec4> positionOpacity(n), scales(n), rotations(n);
    for (size_t i = 0; i < n; ++i) {
        positionOpacity[i] = glm::vec4(cloud.position[i], cloud.activatedOpacity(i));
        scales[i] = glm::vec4(cloud.activatedScale(i), 0.0f);
        rotations[i] = cloud.activatedRotation(i);
    }

    const VkDeviceSize vec4Bytes = n * sizeof(glm::vec4);
    m_positionOpacity = deviceBuffer(*m_ctx, vec4Bytes);
    m_scales = deviceBuffer(*m_ctx, vec4Bytes);
    m_rotations = deviceBuffer(*m_ctx, vec4Bytes);
    m_sh = deviceBuffer(*m_ctx, cloud.sh.size() * sizeof(float));
    if (!m_positionOpacity || !m_scales || !m_rotations || !m_sh) return false;

    m_positionOpacity->upload(positionOpacity.data(), vec4Bytes);
    m_scales->upload(scales.data(), vec4Bytes);
    m_rotations->upload(rotations.data(), vec4Bytes);
    m_sh->upload(cloud.sh.data(), cloud.sh.size() * sizeof(float));

    m_cloud = &cloud;
    m_trainingInputs = false;
    m_backwardReady = false;
    m_numGaussians = static_cast<uint32_t>(n);
    m_sceneShDegree = cloud.shDegree;
    // Start with room for two tiles per Gaussian; render() grows it on demand.
    return ensureGaussianBuffers(m_numGaussians) &&
           ensureInstanceBuffers(std::max<uint32_t>(2 * m_numGaussians, 1u << 20));
}

bool SplatRenderer::ensureGaussianBuffers(uint32_t count) {
    const bool wantGrad = m_trainingInputs && m_renderBackward;
    if (count <= m_gaussianCapacity && (!wantGrad || m_gradAccum)) return true;
    m_ctx->waitIdle();
    // Grow-only; once grown, keep headroom for the trainer's densification.
    uint32_t capacity = std::max(count, m_gaussianCapacity);
    if (count > m_gaussianCapacity && m_gaussianCapacity > 0) capacity = count + count / 8;
    const VkDeviceSize n = capacity;
    const VkDeviceSize vec4Bytes = n * sizeof(glm::vec4);
    const VkDeviceSize uintBytes = n * sizeof(uint32_t);
    m_gaussianCapacity = 0;
    m_means2D = deviceBuffer(*m_ctx, n * sizeof(glm::vec2));
    m_conicOpacity = deviceBuffer(*m_ctx, vec4Bytes);
    m_colorDepth = deviceBuffer(*m_ctx, vec4Bytes);
    m_radii = deviceBuffer(*m_ctx, uintBytes);
    m_tilesTouched = deviceBuffer(*m_ctx, uintBytes);
    m_visibleScan = deviceBuffer(*m_ctx, uintBytes + sizeof(uint32_t));
    m_tileOffsets = deviceBuffer(*m_ctx, uintBytes + sizeof(uint32_t));
    bool ok = m_means2D && m_conicOpacity && m_colorDepth && m_radii && m_tilesTouched && m_visibleScan &&
              m_tileOffsets;
    for (int i = 0; i < 2 && ok; ++i) {
        m_depthKeys[i] = deviceBuffer(*m_ctx, uintBytes);
        m_depthOrder[i] = deviceBuffer(*m_ctx, uintBytes);
        ok = m_depthKeys[i] && m_depthOrder[i];
    }
    m_gradAccum.reset();
    if (ok && wantGrad) {
        m_gradAccum = deviceBuffer(*m_ctx, n * kGradAccumFloats * sizeof(float));
        ok = m_gradAccum != nullptr;
    }
    if (!ok || !m_scan->reserve(capacity + 1) || !m_sort->reserve(capacity)) return false;
    m_gaussianCapacity = capacity;
    return true;
}

bool SplatRenderer::ensurePixelBuffers(uint32_t width, uint32_t height) {
    if (width == m_pixelWidth && height == m_pixelHeight && m_color) return true;
    m_ctx->waitIdle();
    const VkDeviceSize pixels = VkDeviceSize(width) * height;
    const VkDeviceSize tiles = VkDeviceSize(divUp(width, kTileSize)) * divUp(height, kTileSize);
    m_color = deviceBuffer(*m_ctx, pixels * 3 * sizeof(float));
    m_finalT = deviceBuffer(*m_ctx, pixels * sizeof(float));
    m_nContrib = deviceBuffer(*m_ctx, pixels * sizeof(uint32_t));
    m_packed = deviceBuffer(*m_ctx, pixels * sizeof(uint32_t));
    m_tileRangesBuf = deviceBuffer(*m_ctx, tiles * sizeof(glm::uvec2));
    m_pixelWidth = width;
    m_pixelHeight = height;
    return m_color && m_finalT && m_nContrib && m_packed && m_tileRangesBuf;
}

bool SplatRenderer::ensureInstanceBuffers(uint32_t count) {
    if (count <= m_instanceCapacity) return true;
    m_ctx->waitIdle();
    // Grow with headroom so small camera motions do not reallocate every frame.
    uint64_t wanted = std::max<uint64_t>(uint64_t(count) + count / 4, 1u << 16);
    uint32_t capacity = static_cast<uint32_t>(std::min<uint64_t>(wanted, 0xFFFFFFFFu));
    const VkDeviceSize bytes = VkDeviceSize(capacity) * sizeof(uint32_t);
    for (int i = 0; i < 2; ++i) {
        m_tileKeys[i].reset();
        m_values[i].reset();
    }
    for (int i = 0; i < 2; ++i) {
        m_tileKeys[i] = deviceBuffer(*m_ctx, bytes);
        m_values[i] = deviceBuffer(*m_ctx, bytes);
        if (!m_tileKeys[i] || !m_values[i]) {
            m_instanceCapacity = 0;
            return false;
        }
    }
    if (!m_sort->reserve(capacity)) return false;
    m_instanceCapacity = capacity;
    return true;
}

void SplatRenderer::recordFrame(VkCommandBuffer cmd, uint32_t tilesX, uint32_t tilesY, const FrameTarget& target) {
    const uint32_t n = m_numGaussians;
    const uint32_t numTiles = tilesX * tilesY;
    const uint32_t gaussianGroups = divUp(n, kGroupSize);
    const VkDeviceAddress paramsAddr = m_frameParams->address();
    const VkDeviceAddress visibleCount = m_visibleScan->address() + VkDeviceSize(n) * sizeof(uint32_t);
    const VkDeviceAddress instanceCount = m_tileOffsets->address() + VkDeviceSize(n) * sizeof(uint32_t);
    auto mark = [&](const char* label) {
        if (m_profiling && m_timer) m_timer->mark(cmd, label);
    };

    if (m_profiling && m_timer) m_timer->reset(cmd);
    vkCmdFillBuffer(cmd, m_visibleScan->handle(), VkDeviceSize(n) * sizeof(uint32_t), sizeof(uint32_t), 0);
    vkCmdFillBuffer(cmd, m_tileRangesBuf->handle(), 0, VK_WHOLE_SIZE, 0);
    computeBarrier(cmd);

    // 1. Project every Gaussian.
    if (m_trainingInputs) {
        const TrainingParams& tp = m_trainingParams;
        PreprocessTrainPush ppc{paramsAddr,
                                tp.means,
                                tp.scaleLog,
                                tp.rotation,
                                tp.opacityLogit,
                                tp.shDc,
                                tp.shRest,
                                m_means2D->address(),
                                m_conicOpacity->address(),
                                m_colorDepth->address(),
                                m_radii->address(),
                                m_tilesTouched->address(),
                                m_visibleScan->address()};
        m_preprocessTrain->dispatch(cmd, ppc, gaussianGroups);
    } else {
        PreprocessPush ppc{paramsAddr,
                           m_positionOpacity->address(),
                           m_scales->address(),
                           m_rotations->address(),
                           m_sh->address(),
                           m_means2D->address(),
                           m_conicOpacity->address(),
                           m_colorDepth->address(),
                           m_radii->address(),
                           m_tilesTouched->address(),
                           m_visibleScan->address()};
        m_preprocess->dispatch(cmd, ppc, gaussianGroups);
    }
    computeBarrier(cmd);
    mark("preprocess");

    // 2-3. Compact the visible Gaussians and sort them by depth.
    m_scan->exclusiveScan(cmd, m_visibleScan->address(), n + 1);
    computeBarrier(cmd);
    CompactPush cpc{paramsAddr, m_visibleScan->address(), m_colorDepth->address(), m_depthKeys[0]->address(),
                    m_depthOrder[0]->address()};
    m_compact->dispatch(cmd, cpc, gaussianGroups);
    computeBarrier(cmd);
    mark("compact");
    GpuRadixSort::KeyValueArrays depthA{m_depthKeys[0]->address(), m_depthOrder[0]->address()};
    GpuRadixSort::KeyValueArrays depthB{m_depthKeys[1]->address(), m_depthOrder[1]->address()};
    m_depthSortedSet = m_sort->sort(cmd, depthA, depthB, visibleCount, n, 32) ? 1 : 0;
    const VkDeviceAddress sortedIndices = m_depthOrder[m_depthSortedSet]->address();
    mark("depth sort");

    // 4. Instance offsets in depth order.
    GatherTilesPush gpc{paramsAddr, visibleCount, sortedIndices, m_tilesTouched->address(),
                        m_tileOffsets->address()};
    m_gatherTiles->dispatch(cmd, gpc, divUp(n + 1, kGroupSize));
    computeBarrier(cmd);
    m_scan->exclusiveScan(cmd, m_tileOffsets->address(), n + 1);
    computeBarrier(cmd);
    mark("offsets");

    // 5. One instance per Gaussian/tile overlap.
    DuplicatePush dpc{paramsAddr,
                      visibleCount,
                      sortedIndices,
                      m_tileOffsets->address(),
                      m_means2D->address(),
                      m_conicOpacity->address(),
                      m_radii->address(),
                      m_tileKeys[0]->address(),
                      m_values[0]->address(),
                      m_instanceCapacity,
                      numTiles};
    m_duplicate->dispatch(cmd, dpc, gaussianGroups);
    computeBarrier(cmd);
    mark("duplicate");

    // 6. Stable sort by tile id (numTiles itself is the padding key).
    GpuRadixSort::KeyValueArrays tileA{m_tileKeys[0]->address(), m_values[0]->address()};
    GpuRadixSort::KeyValueArrays tileB{m_tileKeys[1]->address(), m_values[1]->address()};
    m_sortedSet = m_sort->sort(cmd, tileA, tileB, instanceCount, m_instanceCapacity, bitWidth(numTiles)) ? 1 : 0;
    mark("tile sort");

    // 7. Per-tile ranges.
    TileRangesPush tpc{m_tileKeys[m_sortedSet]->address(), m_tileRangesBuf->address(), instanceCount,
                       m_instanceCapacity, numTiles};
    m_tileRanges->dispatchLinear(cmd, tpc, divUp(m_instanceCapacity, kGroupSize));
    computeBarrier(cmd);
    mark("tile ranges");

    // 8-9. Composite and pack.
    RenderPush rpc{paramsAddr,
                   m_tileRangesBuf->address(),
                   m_values[m_sortedSet]->address(),
                   m_means2D->address(),
                   m_conicOpacity->address(),
                   m_colorDepth->address(),
                   target.color,
                   m_finalT->address(),
                   m_nContrib->address()};
    m_render->dispatch(cmd, rpc, tilesX, tilesY);
    computeBarrier(cmd);
    mark("render");

    if (target.pack) {
        PackPush pkc{m_packed->address(), target.color, m_pixelWidth, m_pixelHeight,
                     target.swapRedBlue ? 1u : 0u};
        m_pack->dispatchLinear(cmd, pkc, divUp(m_pixelWidth * m_pixelHeight, kGroupSize));
        computeBarrier(cmd);
        mark("pack");
    }
    if (target.radiiOut) {
        VkBufferCopy region{0, 0, VkDeviceSize(n) * sizeof(int32_t)};
        vkCmdCopyBuffer(cmd, m_radii->handle(), target.radiiOut->handle(), 1, &region);
    }

    VkBufferCopy copies[2] = {
        {VkDeviceSize(n) * sizeof(uint32_t), 0, sizeof(uint32_t)},
        {VkDeviceSize(n) * sizeof(uint32_t), sizeof(uint32_t), sizeof(uint32_t)},
    };
    vkCmdCopyBuffer(cmd, m_visibleScan->handle(), m_countReadback->handle(), 1, &copies[0]);
    vkCmdCopyBuffer(cmd, m_tileOffsets->handle(), m_countReadback->handle(), 1, &copies[1]);
    computeBarrier(cmd);
}

bool SplatRenderer::render(const Camera& camera, uint32_t width, uint32_t height, const RenderSettings& settings,
                           bool swapRedBlue) {
    if (!m_cloud || width == 0 || height == 0) return false;
    m_trainingInputs = false;
    m_backwardReady = false;
    m_numGaussians = static_cast<uint32_t>(m_cloud->size());

    CameraIntrinsics k = camera.intrinsics(width, height);
    glm::mat4 model = m_cloud->worldMatrix();
    ViewParams view;
    view.worldToCamera = camera.viewMatrixCV() * model;
    view.cameraPosition = glm::vec3(glm::inverse(model) * glm::vec4(camera.worldPosition(), 1.0f));
    view.fx = k.fx;
    view.fy = k.fy;
    view.cx = k.cx;
    view.cy = k.cy;
    view.width = width;
    view.height = height;
    uint32_t shDegree = settings.shDegree < 0 ? m_sceneShDegree
                                              : std::min<uint32_t>(settings.shDegree, m_sceneShDegree);
    FrameTarget target;
    target.pack = true;
    target.swapRedBlue = swapRedBlue;
    return renderView(view, settings, shDegree, target);
}

bool SplatRenderer::renderView(const ViewParams& view, const RenderSettings& settings, uint32_t shDegree,
                               const FrameTarget& targetIn) {
    const uint32_t width = view.width, height = view.height;
    if (width == 0 || height == 0 || view.fx <= 0.0f || view.fy <= 0.0f) return false;
    if (!ensurePixelBuffers(width, height)) return false;
    FrameTarget target = targetIn;
    if (!target.color) target.color = m_color->address();

    const uint32_t tilesX = divUp(width, kTileSize);
    const uint32_t tilesY = divUp(height, kTileSize);
    const uint32_t n = m_numGaussians;

    // --- frame constants -------------------------------------------------
    FrameParams params{};
    params.modelView = view.worldToCamera;
    params.camPosObj = glm::vec4(view.cameraPosition, 1.0f);
    // Shift the principal point so integer pixel coordinates are pixel centres,
    // matching the reference rasterizer.
    params.intrinsics = glm::vec4(view.fx, view.fy, view.cx - 0.5f, view.cy - 0.5f);
    const float tanFovX = 0.5f * static_cast<float>(width) / view.fx;
    const float tanFovY = 0.5f * static_cast<float>(height) / view.fy;
    params.frustum = glm::vec4(tanFovX, tanFovY, settings.scaleModifier, settings.nearCull);
    params.background = glm::vec4(settings.background, 1.0f);
    params.dims = glm::uvec4(width, height, tilesX, tilesY);
    params.misc = glm::uvec4(n, shDegree, settings.tightTiles ? 1u : 0u, 0);
    m_frameParams->upload(&params, sizeof(params));

    // Everything runs in one submission; instance counts are only known on
    // the GPU. If they overflowed the instance buffers, grow them and render
    // again (rare: capacity keeps 25% headroom over the largest frame so far).
    uint32_t counts[2] = {0, 0};
    for (int attempt = 0; attempt < 2; ++attempt) {
        m_ctx->submitAndWait([&](VkCommandBuffer cmd) { recordFrame(cmd, tilesX, tilesY, target); });
        std::memcpy(counts, m_countReadback->mapped(), sizeof(counts));
        if (counts[1] <= m_instanceCapacity) break;
        if (attempt == 1 || !ensureInstanceBuffers(counts[1])) {
            std::cerr << "SplatRenderer: cannot allocate " << counts[1] << " tile instances\n";
            return false;
        }
    }

    m_lastTilesX = tilesX;
    m_lastTilesY = tilesY;
    m_stats.numGaussians = n;
    m_stats.numVisible = counts[0];
    m_stats.numInstances = counts[1];
    m_stats.width = width;
    m_stats.height = height;
    return true;
}

bool SplatRenderer::renderTraining(const TrainingParams& scene, const ViewParams& view,
                                   const RenderSettings& settings, const Buffer& outImage, const Buffer* outRadii) {
    m_backwardReady = false;
    if (scene.count == 0 || scene.count >= (1u << 31)) {
        std::cerr << "SplatRenderer: invalid Gaussian count " << scene.count << "\n";
        return false;
    }
    const VkDeviceSize imageBytes = VkDeviceSize(3) * view.width * view.height * sizeof(float);
    if (outImage.size() < imageBytes || (outRadii && outRadii->size() < VkDeviceSize(scene.count) * 4)) {
        std::cerr << "SplatRenderer: training output buffers are too small\n";
        return false;
    }
    m_trainingInputs = true;
    m_trainingParams = scene;
    m_numGaussians = scene.count;
    if (!ensureGaussianBuffers(scene.count) ||
        !ensureInstanceBuffers(std::max<uint32_t>(2 * scene.count, 1u << 20))) {
        return false;
    }
    FrameTarget target;
    target.color = outImage.address();
    target.radiiOut = outRadii;
    m_backwardReady = renderView(view, settings, std::min<uint32_t>(scene.shDegree, 3), target);
    return m_backwardReady;
}

bool SplatRenderer::backward(const TrainingParams& scene, const TrainingGrads& grads, const Buffer& gradImage) {
    if (!m_renderBackward) {
        std::cerr << "SplatRenderer: this device does not support the backward pass\n";
        return false;
    }
    const TrainingParams& tp = m_trainingParams;
    if (!m_backwardReady || scene.count != tp.count || scene.means != tp.means || scene.scaleLog != tp.scaleLog ||
        scene.rotation != tp.rotation || scene.opacityLogit != tp.opacityLogit || scene.shDc != tp.shDc ||
        scene.shRest != tp.shRest) {
        std::cerr << "SplatRenderer: backward() must follow renderTraining() of the same parameters\n";
        return false;
    }
    if (gradImage.size() < VkDeviceSize(3) * m_pixelWidth * m_pixelHeight * sizeof(float)) {
        std::cerr << "SplatRenderer: gradient image is too small\n";
        return false;
    }
    if (!m_gradAccum) {
        std::cerr << "SplatRenderer: missing gradient accumulator\n";
        return false;
    }

    const uint32_t n = tp.count;
    m_ctx->submitAndWait([&](VkCommandBuffer cmd) {
        vkCmdFillBuffer(cmd, m_gradAccum->handle(), 0, VkDeviceSize(n) * kGradAccumFloats * sizeof(float), 0);
        computeBarrier(cmd);
        RenderBackwardPush rpc{m_frameParams->address(),
                               m_tileRangesBuf->address(),
                               m_values[m_sortedSet]->address(),
                               m_means2D->address(),
                               m_conicOpacity->address(),
                               m_colorDepth->address(),
                               m_finalT->address(),
                               m_nContrib->address(),
                               gradImage.address(),
                               m_gradAccum->address()};
        m_renderBackward->dispatch(cmd, rpc, m_lastTilesX, m_lastTilesY);
        computeBarrier(cmd);
        PreprocessBackwardPush ppc{m_frameParams->address(),
                                   tp.means,
                                   tp.scaleLog,
                                   tp.rotation,
                                   tp.opacityLogit,
                                   tp.shDc,
                                   tp.shRest,
                                   m_radii->address(),
                                   m_gradAccum->address(),
                                   grads.means,
                                   grads.scaleLog,
                                   grads.rotation,
                                   grads.opacityLogit,
                                   grads.shDc,
                                   grads.shRest,
                                   grads.means2D};
        m_preprocessBackward->dispatch(cmd, ppc, divUp(n, kGroupSize));
        computeBarrier(cmd);
    });
    return true;
}

std::vector<std::pair<std::string, double>> SplatRenderer::stageTimings() const {
    return m_profiling && m_timer ? m_timer->results() : std::vector<std::pair<std::string, double>>{};
}

void SplatRenderer::download(const Buffer& src, void* dst, VkDeviceSize bytes) {
    std::unique_ptr<Buffer> staging = Buffer::create(*m_ctx, bytes, Buffer::Memory::Readback);
    if (!staging) throw std::runtime_error("SplatRenderer::download: staging allocation failed");
    m_ctx->submitAndWait([&](VkCommandBuffer cmd) {
        VkBufferCopy region{0, 0, bytes};
        vkCmdCopyBuffer(cmd, src.handle(), staging->handle(), 1, &region);
        computeBarrier(cmd);
    });
    std::memcpy(dst, staging->mapped(), static_cast<size_t>(bytes));
}
