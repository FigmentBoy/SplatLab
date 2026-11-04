"""Validates the Vulkan forward and backward passes.

1. Forward: the Vulkan image vs the dense PyTorch reference (float64).
2. Backward: the Vulkan gradients of every parameter vs autograd through the
   reference, for a random dL/d(image).
3. Finite differences of the Vulkan forward itself for a few parameters.

Run: python trainer/tests/test_gradients.py [--module-dir build]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splat_trainer import vk  # noqa: E402
from splat_trainer.reference import render_reference  # noqa: E402


def look_at(eye, target, up=(0.0, -1.0, 0.0)):
    """World->camera (CV convention: x right, y down, z forward)."""
    eye, target, up = map(lambda v: np.asarray(v, dtype=np.float64), (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(-up, z)  # right
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    m = np.eye(4)
    m[:3, :3] = R
    m[:3, 3] = -R @ eye
    return m


def random_scene(n, sh_degree, gen):
    def r(*shape, lo=-1.0, hi=1.0):
        return lo + (hi - lo) * torch.rand(*shape, generator=gen, dtype=torch.float64)

    p = {
        "means": r(n, 3) * torch.tensor([1.2, 0.9, 0.8], dtype=torch.float64),
        "scales": r(n, 3, lo=-3.2, hi=-1.6),
        "rotations": r(n, 4),
        "opacities": r(n, 1, lo=-2.0, hi=2.0),
        "sh_dc": r(n, 3, lo=-1.0, hi=1.5),
        "sh_rest": r(n, 45, lo=-0.3, hi=0.3),
    }
    if sh_degree < 3:
        p["sh_rest"].view(n, 15, 3)[:, (sh_degree + 1) ** 2 - 1:] = 0
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-dir")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rast = vk.VulkanRasterizer(module_dir=args.module_dir)
    print("GPU:", rast.device_name)
    dev = rast.device
    gen = torch.Generator().manual_seed(args.seed)
    W, H = 72, 52  # several tiles, not multiples of 16
    cam = vk.Camera("test", W, H, fx=60.0, fy=62.0, cx=W / 2 + 1.3, cy=H / 2 - 0.7,
                    world_to_camera=look_at((0.3, -0.4, -3.5), (0.0, 0.0, 0.0)))
    bg = (0.2, 0.4, 0.1)
    ok = True

    for sh_degree in (0, 1, 3):
        p64 = random_scene(args.n, sh_degree, gen)
        rast.ensure_capacity(args.n)
        names = [n for n, _ in vk.FIELDS]
        for name in names:
            rast.params[name][: args.n].copy_(p64[name].float().to(dev))
        live = [rast.params[name][: args.n] for name in names]

        # --- forward ------------------------------------------------------
        img_vk = vk.render(rast, live, args.n, sh_degree, cam, bg, set_grads=False)
        ref_in = [p64[n].to(dev).requires_grad_(True) for n in names]
        img_ref = render_reference(*ref_in, sh_degree, cam, bg)
        f_err = (img_vk.double() - img_ref).abs().max().item()
        print(f"[deg {sh_degree}] forward max |vk - ref| = {f_err:.2e}  "
              f"(visible {rast.last_stats[0]}, instances {rast.last_stats[1]})")
        ok &= f_err < 1e-4

        # --- backward -------------------------------------------------------
        g_img = torch.randn(3, H, W, generator=gen, dtype=torch.float64).to(dev)
        (img_ref * g_img).sum().backward()
        params_req = [t.requires_grad_(True) for t in live]
        img = vk.render(rast, params_req, args.n, sh_degree, cam, bg, set_grads=False)
        grads_vk = torch.autograd.grad((img.double() * g_img).sum(), params_req)
        for name, gv, pr in zip(names, grads_vk, ref_in):
            gr = pr.grad
            scale = gr.abs().max().item()
            err = (gv.double() - gr).abs().max().item()
            rel = err / max(scale, 1e-12)
            cos = torch.nn.functional.cosine_similarity(gv.double().flatten(), gr.flatten(), dim=0).item() \
                if scale > 0 else 1.0
            status = "ok" if (rel < 2e-3 or scale == 0) else "MISMATCH"
            ok &= status == "ok"
            print(f"    d/d{name:<10} max|ref| {scale:9.3e}  max err {err:9.2e}  rel {rel:8.1e}  cos {cos:.6f}  {status}")

    # --- finite differences of the Vulkan forward ---------------------------
    n = args.n
    sh_degree = 3
    g_img = torch.randn(3, H, W, generator=gen).to(dev)
    params_req = [rast.params[name][:n].requires_grad_(True) for name, _ in vk.FIELDS]
    img = vk.render(rast, params_req, n, sh_degree, cam, bg, set_grads=False)
    grads = torch.autograd.grad((img * g_img).sum(), params_req)
    for p in params_req:
        p.requires_grad_(False)

    def loss():
        with torch.no_grad():
            return (rast.render_raw(n, sh_degree, cam, bg).double() * g_img.double()).sum().item()

    print("finite differences (central, eps = 1e-3):")
    fd_ok = 0
    fd_total = 0
    for fi, (name, dim) in enumerate(vk.FIELDS):
        g = grads[fi]
        # the 3 entries with the largest analytic gradient
        flat = g.abs().flatten()
        for idx in torch.topk(flat, 3).indices.tolist():
            row, col = divmod(idx, dim)
            buf = rast.params[name]
            old = buf[row, col].item()
            eps = 1e-3
            buf[row, col] = old + eps
            lp = loss()
            buf[row, col] = old - eps
            lm = loss()
            buf[row, col] = old
            fd = (lp - lm) / (2 * eps)
            an = g[row, col].item()
            rel = abs(fd - an) / max(abs(an), 1e-6)
            fd_total += 1
            fd_ok += rel < 0.05
            print(f"    {name}[{row},{col}]  analytic {an:+.5e}  fd {fd:+.5e}  rel {rel:.1e}")
    print(f"finite differences within 5%: {fd_ok}/{fd_total}")
    ok &= fd_ok >= math.ceil(0.85 * fd_total)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
