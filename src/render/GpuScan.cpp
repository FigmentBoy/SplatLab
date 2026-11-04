#include "render/GpuScan.hpp"

#include "vk/Buffer.hpp"
#include "vk/ComputePipeline.hpp"
#include "vk/VulkanContext.hpp"

#include "shaders/scan_add.spv.h"
#include "shaders/scan_block.spv.h"

namespace {

constexpr uint32_t kElementsPerBlock = 2048;

struct ScanBlockPush {
    VkDeviceAddress data;
    VkDeviceAddress blockSums;
    uint32_t count;
    uint32_t numBlocks;
    uint32_t writeBlockSums;
};

struct ScanAddPush {
    VkDeviceAddress data;
    VkDeviceAddress blockSums;
    uint32_t count;
    uint32_t numBlocks;
};

uint32_t blocksFor(uint32_t count) {
    return (count + kElementsPerBlock - 1) / kElementsPerBlock;
}

} // namespace

std::unique_ptr<GpuScan> GpuScan::create(VulkanContext& ctx) {
    std::unique_ptr<GpuScan> scan(new GpuScan());
    if (!scan->init(ctx)) return nullptr;
    return scan;
}

bool GpuScan::init(VulkanContext& ctx) {
    m_ctx = &ctx;
    m_scanBlock = ComputePipeline::create(ctx, kSpv_scan_block, kSpv_scan_block_size, sizeof(ScanBlockPush),
                                          {ctx.subgroupSize()});
    m_scanAdd = ComputePipeline::create(ctx, kSpv_scan_add, kSpv_scan_add_size, sizeof(ScanAddPush));
    return m_scanBlock && m_scanAdd;
}

GpuScan::~GpuScan() = default;

bool GpuScan::reserve(uint32_t count) {
    if (count <= m_reserved) return true;
    size_t level = 0;
    for (uint32_t n = count; blocksFor(n) > 1; n = blocksFor(n), ++level) {
        VkDeviceSize bytes = VkDeviceSize(blocksFor(n)) * sizeof(uint32_t);
        if (level >= m_levelSums.size()) m_levelSums.emplace_back();
        if (!m_levelSums[level] || m_levelSums[level]->size() < bytes) {
            m_levelSums[level] = Buffer::create(*m_ctx, bytes, Buffer::Memory::DeviceLocal);
            if (!m_levelSums[level]) return false;
        }
    }
    m_reserved = count;
    return true;
}

void GpuScan::exclusiveScan(VkCommandBuffer cmd, VkDeviceAddress data, uint32_t count) const {
    if (count == 0) return;
    if (count > m_reserved) throw std::runtime_error("GpuScan: count exceeds reserved size");
    scanLevel(cmd, data, count, 0);
}

void GpuScan::scanLevel(VkCommandBuffer cmd, VkDeviceAddress data, uint32_t count, size_t level) const {
    uint32_t blocks = blocksFor(count);
    if (blocks == 1) {
        ScanBlockPush pc{data, 0, count, 1, 0};
        m_scanBlock->dispatchLinear(cmd, pc, 1);
        return;
    }
    VkDeviceAddress sums = m_levelSums[level]->address();
    ScanBlockPush pc{data, sums, count, blocks, 1};
    m_scanBlock->dispatchLinear(cmd, pc, blocks);
    computeBarrier(cmd);
    scanLevel(cmd, sums, blocks, level + 1);
    computeBarrier(cmd);
    ScanAddPush addPc{data, sums, count, blocks};
    m_scanAdd->dispatchLinear(cmd, addPc, blocks);
}
