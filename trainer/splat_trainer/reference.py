"""Dense PyTorch implementation of the same rasterizer (forward only, autograd
gives the gradients). It evaluates every Gaussian at every pixel, so it is
only usable on small scenes and images. It mirrors the Vulkan shaders: the same
projection with frustum clamping and the 0.3 low-pass, the reference's
3-sigma tile squares, the 1/255 alpha cutoff, the 0.99 alpha clamp and early
termination at T < 1e-4. It is used to validate the Vulkan backward pass,
and is the "differentiable PyTorch renderer" fallback of the proposal.
"""
from __future__ import annotations

import math

import torch

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
         -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def eval_sh(sh: torch.Tensor, degree: int, d: torch.Tensor) -> torch.Tensor:
    """sh: (N, 16, 3), d: (N, 3) normalized -> (N, 3) before the +0.5 offset."""
    x, y, z = d[:, 0:1], d[:, 1:2], d[:, 2:3]
    r = SH_C0 * sh[:, 0]
    if degree > 0:
        r = r - SH_C1 * y * sh[:, 1] + SH_C1 * z * sh[:, 2] - SH_C1 * x * sh[:, 3]
    if degree > 1:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        r = (r + SH_C2[0] * xy * sh[:, 4] + SH_C2[1] * yz * sh[:, 5] + SH_C2[2] * (2 * zz - xx - yy) * sh[:, 6]
             + SH_C2[3] * xz * sh[:, 7] + SH_C2[4] * (xx - yy) * sh[:, 8])
    if degree > 2:
        r = (r + SH_C3[0] * y * (3 * xx - yy) * sh[:, 9] + SH_C3[1] * xy * z * sh[:, 10]
             + SH_C3[2] * y * (4 * zz - xx - yy) * sh[:, 11] + SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[:, 12]
             + SH_C3[4] * x * (4 * zz - xx - yy) * sh[:, 13] + SH_C3[5] * z * (xx - yy) * sh[:, 14]
             + SH_C3[6] * x * (xx - 3 * yy) * sh[:, 15])
    return r


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    r, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(-1, 3, 3)


def render_reference(means, scales, rotations, opacities, sh_dc, sh_rest, sh_degree, cam,
                     background=(0.0, 0.0, 0.0), near=0.2):
    """Raw parameters as in vk.FIELDS -> (3, H, W) image."""
    dev, dt = means.device, means.dtype
    w2c = torch.as_tensor(cam.world_to_camera, dtype=dt, device=dev)
    Wr, Wt = w2c[:3, :3], w2c[:3, 3]
    t = means @ Wr.T + Wt
    tz = t[:, 2]
    in_front = tz > near

    s = torch.exp(scales)
    q = rotations / rotations.norm(dim=-1, keepdim=True)
    R = quat_to_mat(q)
    M = R * s[:, None, :]
    sigma = M @ M.transpose(1, 2)

    fx, fy, W, H = cam.fx, cam.fy, cam.width, cam.height
    limx, limy = 1.3 * 0.5 * W / fx, 1.3 * 0.5 * H / fy
    tzs = torch.where(in_front, tz, torch.ones_like(tz))
    tcx = (t[:, 0] / tzs).clamp(-limx, limx) * tzs
    tcy = (t[:, 1] / tzs).clamp(-limy, limy) * tzs
    J = torch.zeros(means.shape[0], 3, 3, dtype=dt, device=dev)
    J[:, 0, 0] = fx / tzs
    J[:, 0, 2] = -fx * tcx / tzs ** 2
    J[:, 1, 1] = fy / tzs
    J[:, 1, 2] = -fy * tcy / tzs ** 2
    T = J @ Wr
    cov = T @ sigma @ T.transpose(1, 2)
    a, b, c = cov[:, 0, 0] + 0.3, cov[:, 0, 1], cov[:, 1, 1] + 0.3
    det = a * c - b * b
    conic = torch.stack([c / det, -b / det, a / det], -1)
    mean2d = torch.stack([fx * t[:, 0] / tzs + cam.cx - 0.5, fy * t[:, 1] / tzs + cam.cy - 0.5], -1)

    sh = torch.cat([sh_dc[:, None, :], sh_rest.reshape(-1, 15, 3)], 1)
    campos = torch.as_tensor(cam.position, dtype=dt, device=dev)
    d = means - campos
    color = (eval_sh(sh, sh_degree, d / d.norm(dim=-1, keepdim=True)) + 0.5).clamp_min(0.0)
    opac = torch.sigmoid(opacities.reshape(-1))

    # Reference tile binning: the 3-sigma square in tile units.
    with torch.no_grad():
        mid = 0.5 * (a + c)
        lam = mid + torch.sqrt(torch.clamp(mid * mid - det, min=0.1))
        radius = torch.ceil(3.0 * torch.sqrt(lam))
        gx, gy = (W + 15) // 16, (H + 15) // 16
        rmin_x = torch.clamp(((mean2d[:, 0] - radius) / 16).trunc(), 0, gx)
        rmax_x = torch.clamp(((mean2d[:, 0] + radius + 15) / 16).trunc(), 0, gx)
        rmin_y = torch.clamp(((mean2d[:, 1] - radius) / 16).trunc(), 0, gy)
        rmax_y = torch.clamp(((mean2d[:, 1] + radius + 15) / 16).trunc(), 0, gy)
        keep = in_front & (det != 0)
        order = torch.argsort(torch.where(keep, tz, torch.full_like(tz, math.inf)), stable=True)

    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=dt), torch.arange(W, device=dev, dtype=dt),
                            indexing="ij")
    pix = torch.stack([xs.reshape(-1), ys.reshape(-1)], -1)  # (P, 2)
    o = order
    dxy = mean2d[o][None, :, :] - pix[:, None, :]
    co = conic[o]
    power = -0.5 * (co[None, :, 0] * dxy[..., 0] ** 2 + co[None, :, 2] * dxy[..., 1] ** 2) \
        - co[None, :, 1] * dxy[..., 0] * dxy[..., 1]
    alpha = torch.clamp(opac[o][None, :] * torch.exp(power), max=0.99)
    with torch.no_grad():
        tx, ty = (pix[:, 0] // 16)[:, None], (pix[:, 1] // 16)[:, None]
        in_tile = (tx >= rmin_x[o][None]) & (tx < rmax_x[o][None]) & (ty >= rmin_y[o][None]) & (ty < rmax_y[o][None])
        mask = keep[o][None] & in_tile & (power <= 0) & (alpha >= 1.0 / 255.0)
        a_m = torch.where(mask, alpha, torch.zeros_like(alpha))
        t_excl = torch.cumprod(torch.cat([torch.ones_like(a_m[:, :1]), 1 - a_m[:, :-1]], 1), 1)
        test = mask & (t_excl * (1 - a_m) < 1e-4)
        stopped = torch.cumsum(test.to(torch.int32), 1) > 0
        mask = mask & ~stopped
    a_f = torch.where(mask, alpha, torch.zeros_like(alpha))
    T_excl = torch.cumprod(torch.cat([torch.ones_like(a_f[:, :1]), 1 - a_f[:, :-1]], 1), 1)
    wgt = a_f * T_excl
    rgb = wgt @ color[o]
    T_final = T_excl[:, -1] * (1 - a_f[:, -1])
    bg = torch.as_tensor(background, dtype=dt, device=dev)
    img = rgb + T_final[:, None] * bg
    return img.T.reshape(3, H, W)
