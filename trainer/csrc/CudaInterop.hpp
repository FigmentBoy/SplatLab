#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>

// Imports Vulkan allocations into CUDA so PyTorch can use them as ordinary
// CUDA tensors (zero copy). Uses the CUDA driver API, loaded at runtime from
// nvcuda.dll / libcuda.so.1 (installed with the NVIDIA driver), so no CUDA
// toolkit is needed to build. Memory is imported into the device's primary
// context, which is the context PyTorch uses.
class CudaInterop {
public:
    static std::unique_ptr<CudaInterop> create(int deviceOrdinal);
    bool init(int deviceOrdinal);
    ~CudaInterop();

    CudaInterop(const CudaInterop&) = delete;
    CudaInterop& operator=(const CudaInterop&) = delete;

    // UUID of the CUDA device; the Vulkan device must have the same one.
    const std::array<uint8_t, 16>& deviceUuid() const { return m_uuid; }

    struct Mapping {
        uint64_t devicePtr = 0;
        void* externalMemory = nullptr;
    };
    // Imports a dedicated allocation from its exported OS handle (Win32 NT
    // handle or POSIX fd). On Windows the handle stays owned by the caller; on
    // POSIX a successful import takes ownership of the fd.
    bool importMemory(void* win32Handle, int fd, uint64_t allocationSize, Mapping& out, std::string& error);
    void releaseMemory(const Mapping& mapping);

    // Waits for all work in the primary context.
    void synchronize();

    struct Api; // driver entry points (CudaInterop.cpp)

private:
    CudaInterop() = default;

    std::unique_ptr<Api> m_api;
    void* m_library = nullptr;
    int m_device = 0;
    void* m_context = nullptr;
    std::array<uint8_t, 16> m_uuid{};
};
