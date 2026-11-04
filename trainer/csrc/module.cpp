// splat_vk: Python bindings of the Vulkan splat renderer for the trainer.
//
// Buffers are allocated by Vulkan, exported, and imported into CUDA, so the
// trainer wraps them as PyTorch CUDA tensors without copies (see
// trainer/splat_trainer/vk.py). Rendering and the backward pass block until
// the GPU is done; the caller synchronizes its CUDA stream before each call.
#include "TrainingContext.hpp"

#include "render/SplatRenderer.hpp"
#include "vk/VulkanContext.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <stdexcept>

namespace py = pybind11;

namespace {

using ParamBuffers = std::array<std::shared_ptr<SharedBuffer>, 6>; // means, scaleLog, rotation, opacity, dc, rest
using GradBuffers = std::array<std::shared_ptr<SharedBuffer>, 7>;  // the same + means2D

TrainingParams toParams(const ParamBuffers& p, uint32_t count, uint32_t shDegree) {
    for (const auto& b : p) {
        if (!b) throw std::invalid_argument("parameter buffer is None");
    }
    auto need = [&](const std::shared_ptr<SharedBuffer>& b, uint64_t floatsPer, const char* name) {
        if (b->size() < uint64_t(count) * floatsPer * 4) {
            throw std::invalid_argument(std::string(name) + " buffer is smaller than count");
        }
    };
    need(p[0], 3, "means");
    need(p[1], 3, "scales");
    need(p[2], 4, "rotations");
    need(p[3], 1, "opacities");
    need(p[4], 3, "sh_dc");
    need(p[5], 45, "sh_rest");
    TrainingParams t;
    t.means = p[0]->address();
    t.scaleLog = p[1]->address();
    t.rotation = p[2]->address();
    t.opacityLogit = p[3]->address();
    t.shDc = p[4]->address();
    t.shRest = p[5]->address();
    t.count = count;
    t.shDegree = shDegree;
    return t;
}

ViewParams toView(const std::array<float, 16>& worldToCamRowMajor, const std::array<float, 3>& campos, float fx,
                  float fy, float cx, float cy, uint32_t width, uint32_t height) {
    ViewParams v;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) v.worldToCamera[c][r] = worldToCamRowMajor[r * 4 + c]; // glm is column-major
    }
    v.cameraPosition = glm::vec3(campos[0], campos[1], campos[2]);
    v.fx = fx;
    v.fy = fy;
    v.cx = cx;
    v.cy = cy;
    v.width = width;
    v.height = height;
    return v;
}

} // namespace

PYBIND11_MODULE(splat_vk, m) {
    m.doc() = "Vulkan Gaussian splat rasterizer (forward + backward) sharing memory with CUDA";

    py::class_<SharedBuffer, std::shared_ptr<SharedBuffer>>(m, "SharedBuffer")
        .def_property_readonly("ptr", &SharedBuffer::devicePtr, "CUDA device pointer")
        .def_property_readonly("nbytes", &SharedBuffer::size);

    py::class_<TrainingContext, std::shared_ptr<TrainingContext>>(m, "Context")
        .def(py::init([](int device, bool validation) {
                 std::unique_ptr<TrainingContext> ctx = TrainingContext::create(device, validation);
                 if (!ctx) throw std::runtime_error("failed to create the Vulkan/CUDA training context");
                 return std::shared_ptr<TrainingContext>(std::move(ctx));
             }),
             py::arg("device") = 0, py::arg("validation") = false)
        .def_property_readonly("device_name", [](TrainingContext& c) { return c.vulkan().deviceName(); })
        .def(
            "alloc",
            [](std::shared_ptr<TrainingContext> self, uint64_t nbytes) {
                std::unique_ptr<SharedBuffer> b = SharedBuffer::create(self, nbytes);
                if (!b) throw std::runtime_error("shared buffer allocation failed (" + std::to_string(nbytes) + " bytes)");
                return std::shared_ptr<SharedBuffer>(std::move(b));
            },
            py::arg("nbytes"), "Allocates a Vulkan buffer that is also mapped into CUDA.")
        .def(
            "render",
            [](TrainingContext& self, const ParamBuffers& params, uint32_t count, uint32_t shDegree,
               const std::array<float, 16>& worldToCam, const std::array<float, 3>& campos, float fx, float fy,
               float cx, float cy, uint32_t width, uint32_t height, const std::array<float, 3>& background,
               float nearCull, std::shared_ptr<SharedBuffer> outImage, std::shared_ptr<SharedBuffer> outRadii) {
                TrainingParams scene = toParams(params, count, shDegree);
                ViewParams view = toView(worldToCam, campos, fx, fy, cx, cy, width, height);
                RenderSettings settings;
                settings.background = glm::vec3(background[0], background[1], background[2]);
                settings.nearCull = nearCull;
                bool ok = false;
                {
                    py::gil_scoped_release release;
                    ok = self.renderer().renderTraining(scene, view, settings, outImage->buffer(),
                                                        outRadii ? &outRadii->buffer() : nullptr);
                }
                if (!ok) throw std::runtime_error("render failed");
                const RenderStats& st = self.renderer().stats();
                return py::make_tuple(st.numVisible, st.numInstances);
            },
            py::arg("params"), py::arg("count"), py::arg("sh_degree"), py::arg("world_to_camera"),
            py::arg("camera_position"), py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"),
            py::arg("width"), py::arg("height"), py::arg("background"), py::arg("near"), py::arg("out_image"),
            py::arg("out_radii") = nullptr,
            "Renders raw parameters into out_image (planar CHW float32). Returns (visible, instances).")
        .def(
            "backward",
            [](TrainingContext& self, const ParamBuffers& params, uint32_t count, uint32_t shDegree,
               const GradBuffers& grads, std::shared_ptr<SharedBuffer> gradImage) {
                TrainingParams scene = toParams(params, count, shDegree);
                for (const auto& g : grads) {
                    if (!g) throw std::invalid_argument("gradient buffer is None");
                }
                TrainingGrads out;
                out.means = grads[0]->address();
                out.scaleLog = grads[1]->address();
                out.rotation = grads[2]->address();
                out.opacityLogit = grads[3]->address();
                out.shDc = grads[4]->address();
                out.shRest = grads[5]->address();
                out.means2D = grads[6]->address();
                bool ok = false;
                {
                    py::gil_scoped_release release;
                    ok = self.renderer().backward(scene, out, gradImage->buffer());
                }
                if (!ok) throw std::runtime_error("backward failed");
            },
            py::arg("params"), py::arg("count"), py::arg("sh_degree"), py::arg("grads"), py::arg("grad_image"),
            "Backward pass of the last render() of the same parameters.");
}
