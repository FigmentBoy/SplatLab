#include "TrainingContext.hpp"

#include "CudaInterop.hpp"

#include "render/SplatRenderer.hpp"
#include "vk/Buffer.hpp"
#include "vk/VulkanContext.hpp"

#include <iostream>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

std::unique_ptr<TrainingContext> TrainingContext::create(int cudaDevice, bool validation) {
    std::unique_ptr<TrainingContext> ctx(new TrainingContext());
    if (!ctx->init(cudaDevice, validation)) return nullptr;
    return ctx;
}

bool TrainingContext::init(int cudaDevice, bool validation) {
    m_cuda = CudaInterop::create(cudaDevice);
    if (!m_cuda) return false;

    VulkanContext::CreateInfo info;
    info.enableValidation = validation;
    info.externalMemory = true;
    info.deviceUuid = m_cuda->deviceUuid();
    m_vulkan = VulkanContext::create(info);
    if (!m_vulkan) {
        std::cerr << "TrainingContext: no Vulkan device shares memory with CUDA device " << cudaDevice << "\n";
        return false;
    }
    m_renderer = SplatRenderer::create(*m_vulkan);
    if (!m_renderer) return false;
    if (!m_renderer->supportsBackward()) {
        std::cerr << "TrainingContext: " << m_vulkan->deviceName()
                  << " lacks float atomics (VK_EXT_shader_atomic_float) or subgroup votes\n";
        return false;
    }
    return true;
}

TrainingContext::~TrainingContext() {
    m_renderer.reset();
    m_vulkan.reset();
    m_cuda.reset();
}

std::unique_ptr<SharedBuffer> SharedBuffer::create(std::shared_ptr<TrainingContext> ctx, uint64_t bytes) {
    std::unique_ptr<SharedBuffer> buffer(new SharedBuffer());
    if (!buffer->init(std::move(ctx), bytes)) return nullptr;
    return buffer;
}

bool SharedBuffer::init(std::shared_ptr<TrainingContext> ctx, uint64_t bytes) {
    m_ctx = std::move(ctx);
    m_size = bytes;
    m_buffer = Buffer::create(m_ctx->vulkan(), bytes, Buffer::Memory::DeviceLocal, true);
    if (!m_buffer) return false;

    void* handle = nullptr;
    int fd = -1;
    if (!m_buffer->exportHandle(handle, fd)) {
        std::cerr << "SharedBuffer: exporting the Vulkan allocation failed\n";
        return false;
    }
    CudaInterop::Mapping mapping;
    std::string error;
    bool ok = m_ctx->cuda().importMemory(handle, fd, m_buffer->allocationSize(), mapping, error);
#ifdef _WIN32
    CloseHandle(reinterpret_cast<HANDLE>(handle)); // CUDA holds its own reference
#else
    if (!ok && fd >= 0) close(fd); // on success CUDA owns the fd
#endif
    if (!ok) {
        std::cerr << "SharedBuffer: " << error << "\n";
        return false;
    }
    m_devicePtr = mapping.devicePtr;
    m_externalMemory = mapping.externalMemory;
    return true;
}

SharedBuffer::~SharedBuffer() {
    if (m_externalMemory) {
        CudaInterop::Mapping mapping{m_devicePtr, m_externalMemory};
        m_ctx->cuda().releaseMemory(mapping);
    }
    if (m_buffer) {
        m_ctx->vulkan().waitIdle();
        m_buffer.reset();
    }
}

uint64_t SharedBuffer::address() const {
    return m_buffer->address();
}
