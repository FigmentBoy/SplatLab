#pragma once

#include "vk/VkCommon.hpp"

#include <memory>

class Buffer;
class ComputePipeline;
class GpuScan;
class VulkanContext;

// Stable LSD radix sort (8 bits per pass) of uint32 keys with a uint32
// payload. Keys are ranked with subgroup ballots, 4096 per workgroup. The
// element count is read on the GPU, so it can come from an earlier pass of the
// same command buffer without a host round trip.
class GpuRadixSort {
public:
    struct KeyValueArrays {
        VkDeviceAddress keys = 0;
        VkDeviceAddress values = 0;
    };

    static std::unique_ptr<GpuRadixSort> create(VulkanContext& ctx, GpuScan& scan);
    bool init(VulkanContext& ctx, GpuScan& scan);
    ~GpuRadixSort();

    GpuRadixSort(const GpuRadixSort&) = delete;
    GpuRadixSort& operator=(const GpuRadixSort&) = delete;

    // Allocates scratch for sorts of up to `maxCount` elements (also reserves
    // the scan). Must be called outside of command recording.
    bool reserve(uint32_t maxCount);

    // Sorts min(*countAddress, maxCount) elements of `primary` by the low
    // `keyBits` bits of their keys, using `scratch` as ping-pong storage.
    // Returns true if the sorted result ends up in `scratch`.
    bool sort(VkCommandBuffer cmd, const KeyValueArrays& primary, const KeyValueArrays& scratch,
              VkDeviceAddress countAddress, uint32_t maxCount, uint32_t keyBits) const;

private:
    GpuRadixSort() = default;

    VulkanContext* m_ctx = nullptr;
    GpuScan* m_scan = nullptr;
    std::unique_ptr<ComputePipeline> m_histogram;
    std::unique_ptr<ComputePipeline> m_scatter;
    std::unique_ptr<Buffer> m_histogramBuffer;
    uint32_t m_reserved = 0;
};
