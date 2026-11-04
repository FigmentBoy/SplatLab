#include "vk/ComputePipeline.hpp"

#include "vk/VulkanContext.hpp"

#include <algorithm>
#include <iostream>

std::unique_ptr<ComputePipeline> ComputePipeline::create(VulkanContext& ctx, const unsigned char* spirv,
                                                         size_t spirvSize, uint32_t pushConstantSize,
                                                         const std::vector<uint32_t>& specConstants) {
    std::unique_ptr<ComputePipeline> pipeline(new ComputePipeline());
    if (!pipeline->init(ctx, spirv, spirvSize, pushConstantSize, specConstants)) return nullptr;
    return pipeline;
}

bool ComputePipeline::init(VulkanContext& ctx, const unsigned char* spirv, size_t spirvSize,
                           uint32_t pushConstantSize, const std::vector<uint32_t>& specConstants) {
    m_ctx = &ctx;
    m_pushConstantSize = pushConstantSize;
    m_maxGroupsX = ctx.properties().limits.maxComputeWorkGroupCount[0];
    VkDevice device = ctx.device();

    if (pushConstantSize > ctx.properties().limits.maxPushConstantsSize) {
        std::cerr << "Push constant block (" << pushConstantSize << " bytes) exceeds device limit\n";
        return false;
    }

    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spirvSize;
    smci.pCode = reinterpret_cast<const uint32_t*>(spirv);
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &smci, nullptr, &module) != VK_SUCCESS) {
        std::cerr << "vkCreateShaderModule failed\n";
        return false;
    }

    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstantSize};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.pushConstantRangeCount = pushConstantSize > 0 ? 1 : 0;
    plci.pPushConstantRanges = &range;
    VkResult result = vkCreatePipelineLayout(device, &plci, nullptr, &m_layout);

    std::vector<VkSpecializationMapEntry> entries;
    for (uint32_t i = 0; i < specConstants.size(); ++i) {
        entries.push_back({i, uint32_t(i * sizeof(uint32_t)), sizeof(uint32_t)});
    }
    VkSpecializationInfo spec{};
    spec.mapEntryCount = static_cast<uint32_t>(entries.size());
    spec.pMapEntries = entries.data();
    spec.dataSize = specConstants.size() * sizeof(uint32_t);
    spec.pData = specConstants.data();

    if (result == VK_SUCCESS) {
        VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = module;
        cpci.stage.pName = "main";
        cpci.stage.pSpecializationInfo = specConstants.empty() ? nullptr : &spec;
        cpci.layout = m_layout;
        result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &m_pipeline);
    }
    vkDestroyShaderModule(device, module, nullptr);
    if (result != VK_SUCCESS) {
        std::cerr << "Compute pipeline creation failed: " << vkutil::resultString(result) << "\n";
        return false;
    }
    return true;
}

ComputePipeline::~ComputePipeline() {
    if (!m_ctx) return;
    vkDestroyPipeline(m_ctx->device(), m_pipeline, nullptr);
    vkDestroyPipelineLayout(m_ctx->device(), m_layout, nullptr);
}

void ComputePipeline::checkSize(size_t size) const {
    if (size != m_pushConstantSize) {
        throw std::runtime_error("Push constant struct size does not match pipeline layout");
    }
}

void ComputePipeline::dispatch(VkCommandBuffer cmd, const void* pushConstants, uint32_t groupsX,
                               uint32_t groupsY, uint32_t groupsZ) const {
    if (groupsX == 0 || groupsY == 0 || groupsZ == 0) return;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    if (m_pushConstantSize > 0) {
        vkCmdPushConstants(cmd, m_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, m_pushConstantSize, pushConstants);
    }
    vkCmdDispatch(cmd, groupsX, groupsY, groupsZ);
}

void ComputePipeline::dispatchLinear(VkCommandBuffer cmd, const void* pushConstants, uint32_t groups) const {
    if (groups == 0) return;
    uint32_t gx = std::min(groups, m_maxGroupsX);
    uint32_t gy = (groups + gx - 1) / gx;
    dispatch(cmd, pushConstants, gx, gy, 1);
}

void computeBarrier(VkCommandBuffer cmd) {
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                            VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                            VK_ACCESS_HOST_READ_BIT;
    VkPipelineStageFlags stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    vkCmdPipelineBarrier(cmd, stages, stages | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0,
                         nullptr);
}
