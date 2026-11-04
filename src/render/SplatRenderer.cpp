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
#include "shaders/render.spv.h"
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
    const VkDeviceSize uintBytes = n * sizeof(uint32_t);
    m_positionOpacity = deviceBuffer(*m_ctx, vec4Bytes);
    m_scales = deviceBuffer(*m_ctx, vec4Bytes);
    m_rotations = deviceBuffer(*m_ctx, vec4Bytes);
    m_sh = deviceBuffer(*m_ctx, cloud.sh.size() * sizeof(float));
    m_means2D = deviceBuffer(*m_ctx, n * sizeof(glm::vec2));
    m_conicOpacity = deviceBuffer(*m_ctx, vec4Bytes);
    m_colorDepth = deviceBuffer(*m_ctx, vec4Bytes);
    m_radii = deviceBuffer(*m_ctx, uintBytes);
    m_tilesTouched = deviceBuffer(*m_ctx, uintBytes);
    m_visibleScan = deviceBuffer(*m_ctx, uintBytes + sizeof(uint32_t));
    m_tileOffsets = deviceBuffer(*m_ctx, uintBytes + sizeof(uint32_t));
    bool ok = m_positionOpacity && m_scales && m_rotations && m_sh && m_means2D && m_conicOpacity &&
              m_colorDepth && m_radii && m_tilesTouched && m_visibleScan && m_tileOffsets;
    for (int i = 0; i < 2 && ok; ++i) {
        m_depthKeys[i] = deviceBuffer(*m_ctx, uintBytes);
        m_depthOrder[i] = deviceBuffer(*m_ctx, uintBytes);
        ok = m_depthKeys[i] && m_depthOrder[i];
    }
    if (!ok) return false;

    m_positionOpacity->upload(positionOpacity.data(), vec4Bytes);
    m_scales->upload(scales.data(), vec4Bytes);
    m_rotations->upload(rotations.data(), vec4Bytes);
    m_sh->upload(cloud.sh.data(), cloud.sh.size() * sizeof(float));

    m_cloud = &cloud;
    m_numGaussians = static_cast<uint32_t>(n);
    m_sceneShDegree = cloud.shDegree;
    // Start with room for two tiles per Gaussian; render() grows it on demand.
    return m_scan->reserve(m_numGaussians + 1) && m_sort->reserve(m_numGaussians) &&
           ensureInstanceBuffers(std::max<uint32_t>(2 * m_numGaussians, 1u << 20));
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

void SplatRenderer::recordFrame(VkCommandBuffer cmd, uint32_t tilesX, uint32_t tilesY, bool swapRedBlue) {
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
                   m_color->address(),
                   m_finalT->address(),
                   m_nContrib->address()};
    m_render->dispatch(cmd, rpc, tilesX, tilesY);
    computeBarrier(cmd);
    mark("render");

    PackPush pkc{m_packed->address(), m_color->address(), m_pixelWidth, m_pixelHeight, swapRedBlue ? 1u : 0u};
    m_pack->dispatchLinear(cmd, pkc, divUp(m_pixelWidth * m_pixelHeight, kGroupSize));
    computeBarrier(cmd);
    mark("pack");

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
    if (!ensurePixelBuffers(width, height)) return false;

    const uint32_t tilesX = divUp(width, kTileSize);
    const uint32_t tilesY = divUp(height, kTileSize);
    const uint32_t n = m_numGaussians;

    // --- frame constants -------------------------------------------------
    CameraIntrinsics k = camera.intrinsics(width, height);
    glm::mat4 model = m_cloud->worldMatrix();
    glm::vec3 camPosObj = glm::vec3(glm::inverse(model) * glm::vec4(camera.worldPosition(), 1.0f));
    uint32_t shDegree = settings.shDegree < 0 ? m_sceneShDegree
                                              : std::min<uint32_t>(settings.shDegree, m_sceneShDegree);

    FrameParams params{};
    params.modelView = camera.viewMatrixCV() * model;
    params.camPosObj = glm::vec4(camPosObj, 1.0f);
    // Shift the principal point so integer pixel coordinates are pixel centres,
    // matching the reference rasterizer.
    params.intrinsics = glm::vec4(k.fx, k.fy, k.cx - 0.5f, k.cy - 0.5f);
    params.frustum = glm::vec4(k.tanFovX, k.tanFovY, settings.scaleModifier, settings.nearCull);
    params.background = glm::vec4(settings.background, 1.0f);
    params.dims = glm::uvec4(width, height, tilesX, tilesY);
    params.misc = glm::uvec4(n, shDegree, settings.tightTiles ? 1u : 0u, 0);
    m_frameParams->upload(&params, sizeof(params));

    // Everything runs in one submission; instance counts are only known on
    // the GPU. If they overflowed the instance buffers, grow them and render
    // again (rare: capacity keeps 25% headroom over the largest frame so far).
    uint32_t counts[2] = {0, 0};
    for (int attempt = 0; attempt < 2; ++attempt) {
        m_ctx->submitAndWait([&](VkCommandBuffer cmd) { recordFrame(cmd, tilesX, tilesY, swapRedBlue); });
        std::memcpy(counts, m_countReadback->mapped(), sizeof(counts));
        if (counts[1] <= m_instanceCapacity) break;
        if (attempt == 1 || !ensureInstanceBuffers(counts[1])) {
            std::cerr << "SplatRenderer: cannot allocate " << counts[1] << " tile instances\n";
            return false;
        }
    }

    m_stats.numGaussians = n;
    m_stats.numVisible = counts[0];
    m_stats.numInstances = counts[1];
    m_stats.width = width;
    m_stats.height = height;
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
