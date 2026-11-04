#pragma once

#include <cstdint>
#include <memory>
#include <string>

class Buffer;
class CudaInterop;
class SplatRenderer;
class VulkanContext;

// Everything the trainer's Python side talks to: a headless Vulkan context on
// the same GPU as a CUDA device (matched by UUID), the splat renderer, and
// the CUDA interop used to share buffers with PyTorch.
class TrainingContext {
public:
    static std::unique_ptr<TrainingContext> create(int cudaDevice, bool validation);
    bool init(int cudaDevice, bool validation);
    ~TrainingContext();

    TrainingContext(const TrainingContext&) = delete;
    TrainingContext& operator=(const TrainingContext&) = delete;

    VulkanContext& vulkan() { return *m_vulkan; }
    SplatRenderer& renderer() { return *m_renderer; }
    CudaInterop& cuda() { return *m_cuda; }

private:
    TrainingContext() = default;

    // Destroyed in reverse order: the renderer before the Vulkan context.
    std::unique_ptr<CudaInterop> m_cuda;
    std::unique_ptr<VulkanContext> m_vulkan;
    std::unique_ptr<SplatRenderer> m_renderer;
};

// A Vulkan buffer whose memory is also mapped into CUDA, so PyTorch sees it as
// a CUDA tensor (see splat_trainer/vk.py). Keeps its context alive.
class SharedBuffer {
public:
    static std::unique_ptr<SharedBuffer> create(std::shared_ptr<TrainingContext> ctx, uint64_t bytes);
    bool init(std::shared_ptr<TrainingContext> ctx, uint64_t bytes);
    ~SharedBuffer();

    SharedBuffer(const SharedBuffer&) = delete;
    SharedBuffer& operator=(const SharedBuffer&) = delete;

    const Buffer& buffer() const { return *m_buffer; }
    uint64_t devicePtr() const { return m_devicePtr; }
    uint64_t size() const { return m_size; }
    uint64_t address() const; // Vulkan device address

private:
    SharedBuffer() = default;

    std::shared_ptr<TrainingContext> m_ctx;
    std::unique_ptr<Buffer> m_buffer;
    uint64_t m_devicePtr = 0;
    void* m_externalMemory = nullptr;
    uint64_t m_size = 0;
};
