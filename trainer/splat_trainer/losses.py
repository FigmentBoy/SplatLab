"""Image losses: L1, SSIM (Wang et al. 2004, 11x11 Gaussian window, sigma 1.5,
as in the reference implementation) and PSNR.

SSIM uses a separable window and filters the five maps it needs (x, y, x^2,
y^2, xy) in one grouped convolution per axis.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_WINDOWS: dict = {}


def _window(device, size=11, sigma=1.5):
    key = (device, size)
    if key not in _WINDOWS:
        x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
        g = torch.exp(-x * x / (2 * sigma * sigma))
        _WINDOWS[key] = g / g.sum()
    return _WINDOWS[key]


def ssim(img: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """img, gt: (3, H, W) in [0, 1]. Mean SSIM (zero padding, like the reference)."""
    g = _window(img.device)
    k = g.numel()
    c = img.shape[0]
    stack = torch.cat([img, gt, img * img, gt * gt, img * gt], 0)[None]  # (1, 5C, H, W)
    ch = stack.shape[1]
    wx = g.view(1, 1, 1, k).expand(ch, 1, 1, k)
    wy = g.view(1, 1, k, 1).expand(ch, 1, k, 1)
    f = F.conv2d(F.conv2d(stack, wx, padding=(0, k // 2), groups=ch), wy, padding=(k // 2, 0), groups=ch)[0]
    mu1, mu2, s11, s22, s12 = f[:c], f[c:2 * c], f[2 * c:3 * c], f[3 * c:4 * c], f[4 * c:]
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq, sigma2_sq, sigma12 = s11 - mu1_sq, s22 - mu2_sq, s12 - mu12
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def l1(img: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return (img - gt).abs().mean()


def psnr(img: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    mse = ((img.clamp(0, 1) - gt) ** 2).mean()
    return -10.0 * torch.log10(mse.clamp_min(1e-12))
