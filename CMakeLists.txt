cmake_minimum_required(VERSION 3.20)

project(splat_lab LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(EXTERNALS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/externals")

# The viewer needs a window system (GLFW, ImGui, native file dialogs). Turn it
# off for headless machines (cloud GPUs), which only need the trainer module.
option(SPLAT_BUILD_VIEWER "Build the splat_lab viewer / offline renderer" ON)
option(SPLAT_BUILD_PYTHON "Build the splat_vk Python module used by trainer/" ON)

# The core library is linked into a Python extension module too.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT EXISTS "${EXTERNALS_DIR}/glm/glm/glm.hpp")
    message(FATAL_ERROR "Submodules are missing. Run: git submodule update --init --recursive")
endif()

# ---------------------------------------------------------------------------
# External dependencies (git submodules in externals/)
# ---------------------------------------------------------------------------

# Vulkan headers + volk (the Vulkan loader is resolved at runtime, so no SDK is
# needed to link).
add_library(vulkan_headers INTERFACE)
target_include_directories(vulkan_headers INTERFACE "${EXTERNALS_DIR}/Vulkan-Headers/include")

add_library(volk STATIC "${EXTERNALS_DIR}/volk/volk.c")
target_include_directories(volk PUBLIC "${EXTERNALS_DIR}/volk")
target_link_libraries(volk PUBLIC vulkan_headers)
if(WIN32)
    target_compile_definitions(volk PUBLIC VK_USE_PLATFORM_WIN32_KHR WIN32_LEAN_AND_MEAN NOMINMAX)
endif()

if(SPLAT_BUILD_VIEWER)
# GLFW (windowing)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory("${EXTERNALS_DIR}/glfw" EXCLUDE_FROM_ALL)

# Dear ImGui (core + GLFW and Vulkan backends, Vulkan functions through volk)
set(IMGUI_DIR "${EXTERNALS_DIR}/imgui")
add_library(imgui STATIC
    "${IMGUI_DIR}/imgui.cpp"
    "${IMGUI_DIR}/imgui_demo.cpp"
    "${IMGUI_DIR}/imgui_draw.cpp"
    "${IMGUI_DIR}/imgui_tables.cpp"
    "${IMGUI_DIR}/imgui_widgets.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp"
)
target_include_directories(imgui PUBLIC "${IMGUI_DIR}" "${IMGUI_DIR}/backends")
target_compile_definitions(imgui PUBLIC IMGUI_IMPL_VULKAN_USE_VOLK GLFW_INCLUDE_NONE)
target_link_libraries(imgui PUBLIC volk glfw)

# Native file dialogs
set(NFD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NFD_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory("${EXTERNALS_DIR}/nativefiledialog-extended" EXCLUDE_FROM_ALL)
endif()

# Header-only libraries
add_library(glm_headers INTERFACE)
target_include_directories(glm_headers INTERFACE "${EXTERNALS_DIR}/glm")
target_compile_definitions(glm_headers INTERFACE GLM_FORCE_DEPTH_ZERO_TO_ONE GLM_ENABLE_EXPERIMENTAL)

add_library(stb_headers INTERFACE)
target_include_directories(stb_headers INTERFACE "${EXTERNALS_DIR}/stb")

add_library(json_headers INTERFACE)
target_include_directories(json_headers INTERFACE "${EXTERNALS_DIR}/json/single_include")

# ---------------------------------------------------------------------------
# Shader compiler: prefer the Vulkan SDK's glslangValidator, otherwise build
# glslang from the submodule (host tool only, not linked into the renderer).
# ---------------------------------------------------------------------------
find_program(GLSLANG_VALIDATOR NAMES glslangValidator glslang
    HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin" NO_CACHE)

if(GLSLANG_VALIDATOR)
    message(STATUS "Using shader compiler: ${GLSLANG_VALIDATOR}")
    set(SHADER_COMPILER "${GLSLANG_VALIDATOR}")
    set(SHADER_COMPILER_DEP "")
else()
    message(STATUS "No glslangValidator found; building glslang from externals/glslang")
    set(ENABLE_OPT OFF CACHE BOOL "" FORCE)
    set(ENABLE_HLSL OFF CACHE BOOL "" FORCE)
    set(ENABLE_SPVREMAPPER OFF CACHE BOOL "" FORCE)
    set(ENABLE_GLSLANG_BINARIES ON CACHE BOOL "" FORCE)
    set(GLSLANG_TESTS OFF CACHE BOOL "" FORCE)
    set(GLSLANG_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BUILD_EXTERNAL OFF CACHE BOOL "" FORCE)
    add_subdirectory("${EXTERNALS_DIR}/glslang" EXCLUDE_FROM_ALL)
    set(SHADER_COMPILER "$<TARGET_FILE:glslang-standalone>")
    set(SHADER_COMPILER_DEP glslang-standalone)
endif()

# ---------------------------------------------------------------------------
# Compute shaders -> SPIR-V -> embedded C++ headers
# ---------------------------------------------------------------------------
set(SHADER_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(SHADER_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(GLOB SHADER_SOURCES CONFIGURE_DEPENDS "${SHADER_SRC_DIR}/*.comp")
file(GLOB SHADER_INCLUDES CONFIGURE_DEPENDS "${SHADER_SRC_DIR}/*.glsl")

set(SHADER_HEADERS "")
foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    set(SPV "${SHADER_GEN_DIR}/shaders/${SHADER_NAME}.spv")
    set(HDR "${SHADER_GEN_DIR}/shaders/${SHADER_NAME}.spv.h")
    add_custom_command(
        OUTPUT ${HDR}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SHADER_GEN_DIR}/shaders"
        COMMAND ${SHADER_COMPILER} -V --target-env vulkan1.2 -I${SHADER_SRC_DIR} -o ${SPV} ${SHADER}
        COMMAND ${CMAKE_COMMAND} -DINPUT=${SPV} -DOUTPUT=${HDR} -DNAME=kSpv_${SHADER_NAME}
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedSpirv.cmake"
        DEPENDS ${SHADER} ${SHADER_INCLUDES} ${SHADER_COMPILER_DEP}
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedSpirv.cmake"
        COMMENT "Compiling shader ${SHADER_NAME}.comp"
        VERBATIM)
    list(APPEND SHADER_HEADERS ${HDR})
endforeach()
add_custom_target(shaders DEPENDS ${SHADER_HEADERS} SOURCES ${SHADER_SOURCES} ${SHADER_INCLUDES})

# ---------------------------------------------------------------------------
# Core library: Vulkan context, renderer (forward + backward), scene and I/O.
# Shared by the viewer executable and the Python module.
# ---------------------------------------------------------------------------
function(splat_warnings target)
    if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(${target} PRIVATE /W4 /permissive- /utf-8 /bigobj)
    else()
        # `VkFooInfo info{VK_STRUCTURE_TYPE_FOO}` intentionally zero-fills the rest.
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
    endif()
endfunction()

file(GLOB_RECURSE CORE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vk/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/vk/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/render/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/render/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/scene/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/scene/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/io/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/io/*.hpp"
)
add_library(splat_core STATIC ${CORE_SOURCES})
add_dependencies(splat_core shaders)
target_include_directories(splat_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src" "${SHADER_GEN_DIR}")
target_link_libraries(splat_core PUBLIC volk glm_headers stb_headers json_headers)
if(UNIX)
    find_package(Threads REQUIRED)
    target_link_libraries(splat_core PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
endif()
splat_warnings(splat_core)

# ---------------------------------------------------------------------------
# Viewer / offline renderer executable
# ---------------------------------------------------------------------------
if(SPLAT_BUILD_VIEWER)
    file(GLOB_RECURSE APP_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/app/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/app/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/gui/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/gui/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp"
    )
    add_executable(splat_lab ${APP_SOURCES})
    target_link_libraries(splat_lab PRIVATE splat_core glfw imgui nfd)
    target_compile_definitions(splat_lab PRIVATE GLFW_INCLUDE_NONE)
    splat_warnings(splat_lab)
    if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT splat_lab)
    endif()
endif()

# ---------------------------------------------------------------------------
# Python module for the trainer (trainer/): the renderer's forward and
# backward passes on buffers shared with PyTorch through CUDA external memory.
# The CUDA driver is loaded at runtime, so no CUDA toolkit is needed.
# ---------------------------------------------------------------------------
if(SPLAT_BUILD_PYTHON)
    # Build against the trainer's virtual environment when it exists.
    if(NOT DEFINED Python_EXECUTABLE)
        foreach(candidate "${CMAKE_CURRENT_SOURCE_DIR}/trainer/.venv/Scripts/python.exe"
                          "${CMAKE_CURRENT_SOURCE_DIR}/trainer/.venv/bin/python")
            if(EXISTS "${candidate}")
                set(Python_EXECUTABLE "${candidate}")
                break()
            endif()
        endforeach()
    endif()
    find_package(Python 3.9 COMPONENTS Interpreter Development.Module)
    if(Python_FOUND)
        set(PYBIND11_FINDPYTHON ON)
        add_subdirectory("${EXTERNALS_DIR}/pybind11" EXCLUDE_FROM_ALL)
        file(GLOB TRAINER_CSRC CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/trainer/csrc/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/trainer/csrc/*.hpp")
        pybind11_add_module(splat_vk ${TRAINER_CSRC})
        target_link_libraries(splat_vk PRIVATE splat_core)
        splat_warnings(splat_vk)
        message(STATUS "Building the splat_vk Python module for ${Python_EXECUTABLE}")
    else()
        message(WARNING "Python development files not found; the trainer module (splat_vk) is not built")
    endif()
endif()
