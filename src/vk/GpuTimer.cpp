#include "vk/GpuTimer.hpp"

#include "vk/VulkanContext.hpp"

std::unique_ptr<GpuTimer> GpuTimer::create(VulkanContext& ctx, uint32_t maxMarks) {
    std::unique_ptr<GpuTimer> timer(new GpuTimer());
    if (!timer->init(ctx, maxMarks)) return nullptr;
    return timer;
}

bool GpuTimer::init(VulkanContext& ctx, uint32_t maxMarks) {
    m_ctx = &ctx;
    m_maxMarks = maxMarks;
    m_periodNs = ctx.properties().limits.timestampPeriod;
    if (m_periodNs <= 0.0) return false;
    VkQueryPoolCreateInfo ci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    ci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    ci.queryCount = maxMarks + 1;
    return vkCreateQueryPool(ctx.device(), &ci, nullptr, &m_pool) == VK_SUCCESS;
}

GpuTimer::~GpuTimer() {
    if (m_pool) vkDestroyQueryPool(m_ctx->device(), m_pool, nullptr);
}

void GpuTimer::reset(VkCommandBuffer cmd) {
    m_labels.clear();
    vkCmdResetQueryPool(cmd, m_pool, 0, m_maxMarks + 1);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, m_pool, 0);
}

void GpuTimer::mark(VkCommandBuffer cmd, const char* label) {
    if (m_labels.size() >= m_maxMarks) return;
    m_labels.emplace_back(label);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, m_pool, static_cast<uint32_t>(m_labels.size()));
}

std::vector<std::pair<std::string, double>> GpuTimer::results() const {
    std::vector<std::pair<std::string, double>> out;
    if (m_labels.empty()) return out;
    std::vector<uint64_t> ticks(m_labels.size() + 1);
    VkResult r = vkGetQueryPoolResults(m_ctx->device(), m_pool, 0, static_cast<uint32_t>(ticks.size()),
                                       ticks.size() * sizeof(uint64_t), ticks.data(), sizeof(uint64_t),
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (r != VK_SUCCESS) return out;
    for (size_t i = 0; i < m_labels.size(); ++i) {
        out.emplace_back(m_labels[i], double(ticks[i + 1] - ticks[i]) * m_periodNs * 1e-6);
    }
    return out;
}
