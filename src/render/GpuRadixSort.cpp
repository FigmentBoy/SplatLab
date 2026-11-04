#include "render/GpuRadixSort.hpp"

#include "render/GpuScan.hpp"
#include "vk/Buffer.hpp"
#include "vk/ComputePipeline.hpp"
#include "vk/VulkanContext.hpp"

#include "shaders/radix_histogram.spv.h"
#include "shaders/radix_scatter.spv.h"

#include <utility>

namespace {

constexpr uint32_t kBlockSize = 4096; // RADIX_BLOCK in shaders/radix_common.glsl
constexpr uint32_t kRadix = 256;

struct HistogramPush {
    VkDeviceAddress keys;
    VkDeviceAddress histogram;
    VkDeviceAddress countBuf;
    uint32_t maxCount;
    uint32_t shift;
    uint32_t numBlocks;
};

struct ScatterPush {
    VkDeviceAddress keysIn;
    VkDeviceAddress valuesIn;
    VkDeviceAddress keysOut;
    VkDeviceAddress valuesOut;
    VkDeviceAddress histogram;
    VkDeviceAddress countBuf;
    uint32_t maxCount;
    uint32_t shift;
    uint32_t numBlocks;
};

uint32_t blocksFor(uint32_t count) {
    return (count + kBlockSize - 1) / kBlockSize;
}

} // namespace

std::unique_ptr<GpuRadixSort> GpuRadixSort::create(VulkanContext& ctx, GpuScan& scan) {
    std::unique_ptr<GpuRadixSort> sort(new GpuRadixSort());
    if (!sort->init(ctx, scan)) return nullptr;
    return sort;
}

bool GpuRadixSort::init(VulkanContext& ctx, GpuScan& scan) {
    m_ctx = &ctx;
    m_scan = &scan;
    const std::vector<uint32_t> spec{ctx.subgroupSize()};
    m_histogram = ComputePipeline::create(ctx, kSpv_radix_histogram, kSpv_radix_histogram_size,
                                          sizeof(HistogramPush), spec);
    m_scatter = ComputePipeline::create(ctx, kSpv_radix_scatter, kSpv_radix_scatter_size, sizeof(ScatterPush),
                                        spec);
    return m_histogram && m_scatter;
}

GpuRadixSort::~GpuRadixSort() = default;

bool GpuRadixSort::reserve(uint32_t maxCount) {
    if (maxCount <= m_reserved) return true;
    uint32_t histogramSize = kRadix * blocksFor(maxCount);
    m_histogramBuffer = Buffer::create(*m_ctx, VkDeviceSize(histogramSize) * sizeof(uint32_t),
                                       Buffer::Memory::DeviceLocal);
    if (!m_histogramBuffer || !m_scan->reserve(histogramSize)) return false;
    m_reserved = maxCount;
    return true;
}

bool GpuRadixSort::sort(VkCommandBuffer cmd, const KeyValueArrays& primary, const KeyValueArrays& scratch,
                        VkDeviceAddress countAddress, uint32_t maxCount, uint32_t keyBits) const {
    if (maxCount == 0 || keyBits == 0) return false;
    if (maxCount > m_reserved) throw std::runtime_error("GpuRadixSort: count exceeds reserved size");

    // Blocks past the GPU-side count exit early, so the dispatch and the
    // histogram layout only depend on maxCount.
    const uint32_t numBlocks = blocksFor(maxCount);
    const VkDeviceAddress histogram = m_histogramBuffer->address();

    KeyValueArrays src = primary;
    KeyValueArrays dst = scratch;
    bool inScratch = false;
    for (uint32_t shift = 0; shift < keyBits; shift += 8) {
        HistogramPush hpc{src.keys, histogram, countAddress, maxCount, shift, numBlocks};
        m_histogram->dispatchLinear(cmd, hpc, numBlocks);
        computeBarrier(cmd);
        m_scan->exclusiveScan(cmd, histogram, kRadix * numBlocks);
        computeBarrier(cmd);
        ScatterPush spc{src.keys, src.values, dst.keys, dst.values, histogram, countAddress, maxCount, shift,
                        numBlocks};
        m_scatter->dispatchLinear(cmd, spc, numBlocks);
        computeBarrier(cmd);
        std::swap(src, dst);
        inScratch = !inScratch;
    }
    return inScratch;
}
