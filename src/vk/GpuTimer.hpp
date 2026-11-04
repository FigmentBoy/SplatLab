#pragma once

#include "vk/VkCommon.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

class VulkanContext;

// Timestamp queries for profiling GPU work. Call reset() at the start of a
// command buffer, mark() after each stage, and read results() once the
// command buffer has finished executing.
class GpuTimer {
public:
    static std::unique_ptr<GpuTimer> create(VulkanContext& ctx, uint32_t maxMarks = 64);
    bool init(VulkanContext& ctx, uint32_t maxMarks);
    ~GpuTimer();

    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void reset(VkCommandBuffer cmd);
    // Records a timestamp ending the stage `label` (the stage started at the
    // previous mark, or at reset()).
    void mark(VkCommandBuffer cmd, const char* label);

    // (label, milliseconds) for each stage of the last recorded command buffer.
    std::vector<std::pair<std::string, double>> results() const;

private:
    GpuTimer() = default;

    VulkanContext* m_ctx = nullptr;
    VkQueryPool m_pool = VK_NULL_HANDLE;
    uint32_t m_maxMarks = 0;
    double m_periodNs = 1.0;
    std::vector<std::string> m_labels; // label i ends at query i + 1
};
