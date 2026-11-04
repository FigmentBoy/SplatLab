#pragma once

#include "vk/VkCommon.hpp"

#include <memory>
#include <vector>

class Buffer;
class ComputePipeline;
class VulkanContext;

// In-place exclusive prefix sum of a uint32 array on the GPU (multi-level
// block scan with subgroup operations, 2048 elements per block).
class GpuScan {
public:
    static std::unique_ptr<GpuScan> create(VulkanContext& ctx);
    bool init(VulkanContext& ctx);
    ~GpuScan();

    GpuScan(const GpuScan&) = delete;
    GpuScan& operator=(const GpuScan&) = delete;

    // Allocates scratch space for scans of up to `count` elements. Must be
    // called outside of command recording.
    bool reserve(uint32_t count);

    // Records the scan. `count` must not exceed the reserved size.
    void exclusiveScan(VkCommandBuffer cmd, VkDeviceAddress data, uint32_t count) const;

private:
    GpuScan() = default;

    void scanLevel(VkCommandBuffer cmd, VkDeviceAddress data, uint32_t count, size_t level) const;

    VulkanContext* m_ctx = nullptr;
    std::unique_ptr<ComputePipeline> m_scanBlock;
    std::unique_ptr<ComputePipeline> m_scanAdd;
    std::vector<std::unique_ptr<Buffer>> m_levelSums;
    uint32_t m_reserved = 0;
};
