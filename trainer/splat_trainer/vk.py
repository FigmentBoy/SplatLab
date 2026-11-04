"""PyTorch bridge to the Vulkan splat rasterizer (the splat_vk C++ module).

The Gaussian parameters, their gradients and the rendered image all live in
Vulkan buffers that are also mapped into CUDA, so torch reads and writes them
in place: the forward pass reads the parameters the optimizer just updated,
and the backward pass writes the gradients the optimizer will read next.
Nothing is copied between the two APIs. The two only synchronize through the
host: torch's stream is drained before each Vulkan submission, and Vulkan
calls return once the GPU is done.
"""
from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# (name, floats per Gaussian), in the order the C++ module expects.
FIELDS = (("means", 3), ("scales", 3), ("rotations", 4), ("opacities", 1), ("sh_dc", 3), ("sh_rest", 45))
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module(module_dir: str | None = None):
    """Imports splat_vk, looking in module_dir, $SPLAT_VK_PATH and build*/ folders."""
    if "splat_vk" in sys.modules:
        return sys.modules["splat_vk"]
    pattern = "splat_vk*.pyd" if os.name == "nt" else "splat_vk*.so"
    dirs = [d for d in (module_dir, os.environ.get("SPLAT_VK_PATH")) if d]
    found = [Path(d) for d in dirs if any(Path(d).glob(pattern))]
    if not found:
        # Newest build of the module under the repository's build directories.
        builds = sorted(REPO_ROOT.glob("build*/" + pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        builds += sorted(REPO_ROOT.glob("build*/*/" + pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        found = [b.parent for b in builds[:1]]
    for d in found:
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    try:
        return importlib.import_module("splat_vk")
    except ImportError as e:
        raise ImportError(
            "The splat_vk module was not found. Build it with CMake (see README, 'Trainer') "
            "or point --module-dir / SPLAT_VK_PATH at the folder containing it."
        ) from e


class _CudaArray:
    """Exposes a shared buffer through __cuda_array_interface__ (zero copy)."""

    def __init__(self, buf, shape, typestr):
        self._buf = buf  # keeps the Vulkan memory alive as long as the tensor
        self.__cuda_array_interface__ = {
            "shape": tuple(int(s) for s in shape),
            "typestr": typestr,
            "data": (int(buf.ptr), False),
            "version": 3,
        }


def wrap(buf, shape, dtype=torch.float32, device="cuda") -> torch.Tensor:
    typestr = {torch.float32: "<f4", torch.int32: "<i4"}[dtype]
    t = torch.as_tensor(_CudaArray(buf, shape, typestr), device=device)
    assert t.data_ptr() == buf.ptr, "tensor does not alias the shared buffer"
    return t


@dataclass
class Camera:
    """A training/test view: pinhole intrinsics and a world->camera transform (CV convention)."""

    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    world_to_camera: np.ndarray  # 4x4 float64
    image: torch.Tensor | None = None  # uint8 HxWx3 on the GPU (ground truth)
    file_name: str = ""                # image file in the dataset's images/ folder

    @property
    def position(self) -> np.ndarray:
        r = self.world_to_camera[:3, :3]
        t = self.world_to_camera[:3, 3]
        return -r.T @ t


class VulkanRasterizer:
    """Owns the Vulkan/CUDA context and the shared buffers.

    params[name] / grads[name] are (capacity, dim) float32 tensors aliasing
    Vulkan memory; the first `count` rows are the live Gaussians.
    """

    def __init__(self, device: int = 0, module_dir: str | None = None, near: float = 0.2):
        self.vk = load_module(module_dir)
        self.device = torch.device("cuda", device)
        torch.cuda.init()
        torch.cuda.set_device(self.device)
        self.ctx = self.vk.Context(device)
        self.near = near
        self.capacity = 0
        self.params: dict[str, torch.Tensor] = {}
        self.grads: dict[str, torch.Tensor] = {}
        self._param_bufs = []
        self._grad_bufs = []
        self._image_shape = None

    @property
    def device_name(self) -> str:
        return self.ctx.device_name

    def _alloc(self, shape, dtype=torch.float32):
        nbytes = int(np.prod(shape)) * 4
        buf = self.ctx.alloc(max(nbytes, 16))
        return buf, wrap(buf, shape, dtype, self.device)

    def ensure_capacity(self, count: int) -> bool:
        """Grows the parameter/gradient buffers to hold `count` Gaussians.

        Returns True if they were reallocated: tensors handed out earlier no
        longer alias the buffers the renderer uses, and their contents were
        NOT copied (the caller writes the new values).
        """
        if count <= self.capacity:
            return False
        capacity = max(count + count // 4, 1024)
        torch.cuda.synchronize(self.device)
        self.params.clear()
        self.grads.clear()
        self._param_bufs, self._grad_bufs = [], []
        for name, dim in FIELDS:
            b, t = self._alloc((capacity, dim))
            self._param_bufs.append(b)
            self.params[name] = t
            b, t = self._alloc((capacity, dim))
            self._grad_bufs.append(b)
            self.grads[name] = t
        b, self.grad_means2d = self._alloc((capacity, 2))
        self._grad_bufs.append(b)
        self._radii_buf, self.radii = self._alloc((capacity,), torch.int32)
        self.capacity = capacity
        return True

    def _ensure_image(self, width: int, height: int):
        if self._image_shape == (height, width):
            return
        torch.cuda.synchronize(self.device)
        self._image_buf, self.image = self._alloc((3, height, width))
        self._grad_image_buf, self.grad_image = self._alloc((3, height, width))
        self._image_shape = (height, width)

    def render_raw(self, count: int, sh_degree: int, cam: Camera, background=(0.0, 0.0, 0.0)):
        """Renders the first `count` Gaussians; returns the shared (3, H, W) image tensor."""
        self._ensure_image(cam.width, cam.height)
        torch.cuda.current_stream(self.device).synchronize()  # parameters written by torch
        w2c = np.asarray(cam.world_to_camera, dtype=np.float64)
        visible, instances = self.ctx.render(
            self._param_bufs, count, sh_degree, w2c.astype(np.float32).reshape(-1).tolist(),
            cam.position.astype(np.float32).tolist(), cam.fx, cam.fy, cam.cx, cam.cy, cam.width, cam.height,
            list(background), self.near, self._image_buf, self._radii_buf)
        self.last_stats = (visible, instances)
        return self.image

    def backward_raw(self, count: int, sh_degree: int, grad_image: torch.Tensor):
        """Backward of the last render_raw(); fills self.grads / grad_means2d."""
        self.grad_image.copy_(grad_image)
        torch.cuda.current_stream(self.device).synchronize()
        self.ctx.backward(self._param_bufs, count, sh_degree, self._grad_bufs, self._grad_image_buf)


class _RenderFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rast, count, sh_degree, cam, background, set_grads, *params):
        image = rast.render_raw(count, sh_degree, cam, background)
        ctx.rast, ctx.count, ctx.sh_degree, ctx.set_grads = rast, count, sh_degree, set_grads
        ctx.params = params
        # The shared image buffer is reused by the next render.
        return image.clone()

    @staticmethod
    def backward(ctx, grad_image):
        rast, n = ctx.rast, ctx.count
        rast.backward_raw(n, ctx.sh_degree, grad_image.contiguous())
        grads = [rast.grads[name][:n] for name, _ in FIELDS]
        if ctx.set_grads:
            # Zero-copy: the parameters' .grad alias the renderer's gradient
            # buffers (valid until the next backward). Autograd gets None.
            for p, g in zip(ctx.params, grads):
                if p.grad is None:
                    p.grad = g
                else:
                    p.grad.add_(g)
            return (None,) * (6 + len(grads))
        return (None,) * 6 + tuple(g.clone() for g in grads)


def render(rast: VulkanRasterizer, params, count: int, sh_degree: int, cam: Camera,
           background=(0.0, 0.0, 0.0), set_grads: bool = True) -> torch.Tensor:
    """Differentiable render of `params` (tensors in FIELDS order aliasing rast.params).

    With set_grads, the backward pass assigns the parameters' .grad directly
    (no copies) instead of returning gradients through autograd.
    """
    return _RenderFunction.apply(rast, count, sh_degree, cam, tuple(background), set_grads, *params)
