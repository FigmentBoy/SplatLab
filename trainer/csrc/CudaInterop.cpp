#include "CudaInterop.hpp"

#include <cstring>
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
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace {

// Minimal declarations of the CUDA driver API (cuda.h), so the CUDA toolkit
// is not needed at build time. Layouts match cuda.h.
using CUresult = int;
using CUdevice = int;
using CUcontext = void*;
using CUdeviceptr = unsigned long long;
using CUexternalMemory = void*;
struct CUuuid {
    char bytes[16];
};

[[maybe_unused]] constexpr int CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = 1;
[[maybe_unused]] constexpr int CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 = 2;
constexpr unsigned CUDA_EXTERNAL_MEMORY_DEDICATED = 0x1;

struct CudaExternalMemoryHandleDesc {
    int type;
    union {
        int fd;
        struct {
            void* handle;
            const void* name;
        } win32;
        const void* nvSciBufObject;
    } handle;
    unsigned long long size;
    unsigned int flags;
    unsigned int reserved[16];
};

struct CudaExternalMemoryBufferDesc {
    unsigned long long offset;
    unsigned long long size;
    unsigned int flags;
    unsigned int reserved[16];
};

static_assert(sizeof(CudaExternalMemoryHandleDesc) == 104, "CUDA_EXTERNAL_MEMORY_HANDLE_DESC layout");
static_assert(sizeof(CudaExternalMemoryBufferDesc) == 88, "CUDA_EXTERNAL_MEMORY_BUFFER_DESC layout");

void* loadLibrary() {
#ifdef _WIN32
    return reinterpret_cast<void*>(LoadLibraryA("nvcuda.dll"));
#else
    void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    return lib ? lib : dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
#endif
}

void* loadSymbol(void* lib, const char* name) {
#ifdef _WIN32
    return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(lib), name));
#else
    return dlsym(lib, name);
#endif
}

} // namespace

struct CudaInterop::Api {
    CUresult (*cuInit)(unsigned);
    CUresult (*cuDeviceGet)(CUdevice*, int);
    CUresult (*cuDeviceGetUuid)(CUuuid*, CUdevice);
    CUresult (*cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice);
    CUresult (*cuDevicePrimaryCtxRelease)(CUdevice);
    CUresult (*cuCtxPushCurrent)(CUcontext);
    CUresult (*cuCtxPopCurrent)(CUcontext*);
    CUresult (*cuCtxSynchronize)();
    CUresult (*cuImportExternalMemory)(CUexternalMemory*, const CudaExternalMemoryHandleDesc*);
    CUresult (*cuExternalMemoryGetMappedBuffer)(CUdeviceptr*, CUexternalMemory, const CudaExternalMemoryBufferDesc*);
    CUresult (*cuDestroyExternalMemory)(CUexternalMemory);
    CUresult (*cuMemFree)(CUdeviceptr);
    CUresult (*cuGetErrorString)(CUresult, const char**);
};

namespace {

// Makes the primary context current for a scope.
class ContextScope {
public:
    ContextScope(const CudaInterop::Api& api, void* ctx) : m_api(api) { m_ok = api.cuCtxPushCurrent(ctx) == 0; }
    ~ContextScope() {
        CUcontext popped = nullptr;
        if (m_ok) m_api.cuCtxPopCurrent(&popped);
    }
    bool ok() const { return m_ok; }

private:
    const CudaInterop::Api& m_api;
    bool m_ok = false;
};

} // namespace

std::unique_ptr<CudaInterop> CudaInterop::create(int deviceOrdinal) {
    std::unique_ptr<CudaInterop> interop(new CudaInterop());
    if (!interop->init(deviceOrdinal)) return nullptr;
    return interop;
}

bool CudaInterop::init(int deviceOrdinal) {
    m_library = loadLibrary();
    if (!m_library) {
        std::cerr << "CudaInterop: the CUDA driver library was not found (NVIDIA driver required)\n";
        return false;
    }
    m_api = std::make_unique<Api>();
    bool ok = true;
    auto load = [&](auto& fn, const char* name) {
        fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(loadSymbol(m_library, name));
        if (!fn) {
            std::cerr << "CudaInterop: missing driver symbol " << name << "\n";
            ok = false;
        }
    };
    load(m_api->cuInit, "cuInit");
    load(m_api->cuDeviceGet, "cuDeviceGet");
    load(m_api->cuDeviceGetUuid, "cuDeviceGetUuid");
    load(m_api->cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
    load(m_api->cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2");
    load(m_api->cuCtxPushCurrent, "cuCtxPushCurrent_v2");
    load(m_api->cuCtxPopCurrent, "cuCtxPopCurrent_v2");
    load(m_api->cuCtxSynchronize, "cuCtxSynchronize");
    load(m_api->cuImportExternalMemory, "cuImportExternalMemory");
    load(m_api->cuExternalMemoryGetMappedBuffer, "cuExternalMemoryGetMappedBuffer");
    load(m_api->cuDestroyExternalMemory, "cuDestroyExternalMemory");
    load(m_api->cuMemFree, "cuMemFree_v2");
    load(m_api->cuGetErrorString, "cuGetErrorString");
    if (!ok) return false;

    if (m_api->cuInit(0) != 0 || m_api->cuDeviceGet(&m_device, deviceOrdinal) != 0) {
        std::cerr << "CudaInterop: CUDA device " << deviceOrdinal << " is not available\n";
        return false;
    }
    CUuuid uuid{};
    if (m_api->cuDeviceGetUuid(&uuid, m_device) != 0) return false;
    std::memcpy(m_uuid.data(), uuid.bytes, 16);
    if (m_api->cuDevicePrimaryCtxRetain(&m_context, m_device) != 0) {
        m_context = nullptr;
        return false;
    }
    return true;
}

CudaInterop::~CudaInterop() {
    if (m_api && m_context) m_api->cuDevicePrimaryCtxRelease(m_device);
    // The driver library stays loaded (PyTorch uses it as well).
}

bool CudaInterop::importMemory(void* win32Handle, int fd, uint64_t allocationSize, Mapping& out,
                               std::string& error) {
    ContextScope scope(*m_api, m_context);
    if (!scope.ok()) {
        error = "cannot make the CUDA context current";
        return false;
    }
    CudaExternalMemoryHandleDesc desc{};
#ifdef _WIN32
    (void)fd;
    desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32;
    desc.handle.win32.handle = win32Handle;
#else
    (void)win32Handle;
    desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
    desc.handle.fd = fd;
#endif
    desc.size = allocationSize;
    desc.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;

    auto describe = [&](CUresult r) {
        const char* msg = nullptr;
        m_api->cuGetErrorString(r, &msg);
        return std::string(msg ? msg : "unknown CUDA error") + " (" + std::to_string(r) + ")";
    };

    CUexternalMemory ext = nullptr;
    CUresult r = m_api->cuImportExternalMemory(&ext, &desc);
    if (r != 0) {
        error = "cuImportExternalMemory: " + describe(r);
        return false;
    }
    CudaExternalMemoryBufferDesc bufDesc{};
    bufDesc.offset = 0;
    bufDesc.size = allocationSize;
    CUdeviceptr ptr = 0;
    r = m_api->cuExternalMemoryGetMappedBuffer(&ptr, ext, &bufDesc);
    if (r != 0) {
        m_api->cuDestroyExternalMemory(ext);
        error = "cuExternalMemoryGetMappedBuffer: " + describe(r);
        return false;
    }
    out.devicePtr = ptr;
    out.externalMemory = ext;
    return true;
}

void CudaInterop::releaseMemory(const Mapping& mapping) {
    if (!mapping.externalMemory) return;
    ContextScope scope(*m_api, m_context);
    m_api->cuCtxSynchronize();
    m_api->cuMemFree(mapping.devicePtr);
    m_api->cuDestroyExternalMemory(mapping.externalMemory);
}

void CudaInterop::synchronize() {
    ContextScope scope(*m_api, m_context);
    m_api->cuCtxSynchronize();
}
