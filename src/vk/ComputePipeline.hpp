#pragma once

#include "vk/VkCommon.hpp"

#include <cstddef>
#include <memory>
#include <vector>

class VulkanContext;

// A compute pipeline whose only interface is a push-constant block. Buffers are
// passed as device addresses inside the push constants. specConstants[i] sets
// the uint specialization constant with constant_id i.
class ComputePipeline {
public:
    static std::unique_ptr<ComputePipeline> create(VulkanContext& ctx, const unsigned char* spirv,
                                                   size_t spirvSize, uint32_t pushConstantSize,
                                                   const std::vector<uint32_t>& specConstants = {});
    bool init(VulkanContext& ctx, const unsigned char* spirv, size_t spirvSize, uint32_t pushConstantSize,
              const std::vector<uint32_t>& specConstants);
    ~ComputePipeline();

    ComputePipeline(const ComputePipeline&) = delete;
    ComputePipeline& operator=(const ComputePipeline&) = delete;

    void dispatch(VkCommandBuffer cmd, const void* pushConstants, uint32_t groupsX, uint32_t groupsY = 1,
                  uint32_t groupsZ = 1) const;

    // Dispatches `groups` workgroups, folding them into x/y when they exceed
    // the device limit. Shaders use linearGroupId() and must bounds-check.
    void dispatchLinear(VkCommandBuffer cmd, const void* pushConstants, uint32_t groups) const;

    template <typename T>
    void dispatch(VkCommandBuffer cmd, const T& pc, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1) const {
        checkSize(sizeof(T));
        dispatch(cmd, static_cast<const void*>(&pc), gx, gy, gz);
    }

    template <typename T>
    void dispatchLinear(VkCommandBuffer cmd, const T& pc, uint32_t groups) const {
        checkSize(sizeof(T));
        dispatchLinear(cmd, static_cast<const void*>(&pc), groups);
    }

private:
    ComputePipeline() = default;

    void checkSize(size_t size) const;

    VulkanContext* m_ctx = nullptr;
    VkPipelineLayout m_layout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    uint32_t m_pushConstantSize = 0;
    uint32_t m_maxGroupsX = 65535;
};

// Full compute/transfer -> compute/transfer/host memory barrier between passes.
void computeBarrier(VkCommandBuffer cmd);
