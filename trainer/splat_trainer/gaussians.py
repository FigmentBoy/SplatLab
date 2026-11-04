"""The optimized Gaussians: parameters, Adam, and adaptive density control.

The parameters are views of the Vulkan-shared buffers (VulkanRasterizer.params),
so the renderer reads exactly what Adam wrote. Whenever the number of
Gaussians changes (densification, pruning) the new values are computed in
ordinary torch memory and copied back into the shared buffers, and the Adam
moments are carried over (or zero-initialized for new Gaussians), as in the
reference implementation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .reference import quat_to_mat
from .vk import FIELDS, VulkanRasterizer

C0 = 0.28209479177387814


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


@dataclass
class LearningRates:
    means_init: float = 1.6e-4   # x scene extent, decays exponentially
    means_final: float = 1.6e-6
    sh_dc: float = 2.5e-3
    sh_rest: float = 2.5e-3 / 20.0
    opacities: float = 2.5e-2
    scales: float = 5e-3
    rotations: float = 1e-3


def expon_lr(step, lr_init, lr_final, max_steps):
    t = min(max(step / max_steps, 0.0), 1.0)
    return math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)


def knn_mean_sq_dist(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Mean squared distance to the k nearest neighbours (brute force on the GPU, chunked)."""
    n = x.shape[0]
    out = torch.empty(n, device=x.device)
    chunk = max(1, int(4e7 // max(n, 1)))
    for i in range(0, n, chunk):
        d = torch.cdist(x[i:i + chunk], x)
        kk = min(k + 1, n)
        vals = torch.topk(d, kk, dim=1, largest=False).values[:, 1:]  # drop self
        out[i:i + chunk] = (vals ** 2).mean(1) if vals.numel() else 1.0
    return out


class GaussianModel:
    def __init__(self, rast: VulkanRasterizer, max_sh_degree: int = 3):
        self.rast = rast
        self.max_sh_degree = max_sh_degree
        self.active_sh_degree = 0
        self.count = 0
        self.params: dict[str, torch.nn.Parameter] = {}
        self.optimizer: torch.optim.Optimizer | None = None
        self.extent = 1.0
        self.lr = LearningRates()

    # --- setup ---------------------------------------------------------------
    def init_from_points(self, xyz: np.ndarray, rgb: np.ndarray, extent: float):
        dev = self.rast.device
        self.extent = extent
        pts = torch.tensor(xyz, dtype=torch.float32, device=dev)
        n = pts.shape[0]
        colors = torch.tensor(rgb, dtype=torch.float32, device=dev) / 255.0
        dist2 = knn_mean_sq_dist(pts).clamp_min(1e-7)
        values = {
            "means": pts,
            "scales": torch.log(torch.sqrt(dist2))[:, None].repeat(1, 3),
            "rotations": torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).repeat(n, 1),
            "opacities": inverse_sigmoid(torch.full((n, 1), 0.1, device=dev)),
            "sh_dc": (colors - 0.5) / C0,
            "sh_rest": torch.zeros(n, 45, device=dev),
        }
        self._set(values, states=None)

    def setup_optimizer(self):
        groups = [{"params": [self.params[name]], "lr": 0.0, "name": name} for name, _ in FIELDS]
        try:
            self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15, fused=True)
        except (RuntimeError, TypeError):
            self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        self.reset_stats()

    def set_learning_rates(self, step: int, max_steps: int, freeze_geometry: bool = False):
        lr = self.lr
        rates = {
            "means": 0.0 if freeze_geometry else expon_lr(step, lr.means_init * self.extent,
                                                          lr.means_final * self.extent, max_steps),
            "scales": 0.0 if freeze_geometry else lr.scales,
            "rotations": 0.0 if freeze_geometry else lr.rotations,
            "opacities": lr.opacities,
            "sh_dc": lr.sh_dc,
            "sh_rest": lr.sh_rest,
        }
        for g in self.optimizer.param_groups:
            g["lr"] = rates[g["name"]]

    def param_list(self):
        return [self.params[name] for name, _ in FIELDS]

    def reset_stats(self):
        dev = self.rast.device
        self.grad_accum = torch.zeros(self.count, device=dev)
        self.denom = torch.zeros(self.count, device=dev)
        self.max_radii = torch.zeros(self.count, device=dev)

    # --- storage -------------------------------------------------------------
    def _set(self, values: dict, states: dict | None):
        """Makes `values` (name -> (n, dim) tensors) the live parameters.

        states: name -> (exp_avg, exp_avg_sq) for the new rows, or None to
        start the optimizer from scratch.
        """
        n = values["means"].shape[0]
        old_state = {}
        if self.optimizer is not None:
            for g in self.optimizer.param_groups:
                p = g["params"][0]
                old_state[g["name"]] = self.optimizer.state.pop(p, None)
        self.rast.ensure_capacity(n)
        for name, dim in FIELDS:
            view = self.rast.params[name][:n]
            view.copy_(values[name].reshape(n, dim))
            self.params[name] = torch.nn.Parameter(view)
        self.count = n
        if self.optimizer is None:
            return
        for g in self.optimizer.param_groups:
            name = g["name"]
            p = self.params[name]
            g["params"][0] = p
            st = old_state.get(name)
            if st and states is not None and name in states:
                avg, avg_sq = states[name]
                self.optimizer.state[p] = {"step": st["step"], "exp_avg": avg.contiguous(),
                                           "exp_avg_sq": avg_sq.contiguous()}

    def _state(self, name):
        st = self.optimizer.state.get(self.params[name])
        if not st:
            z = torch.zeros_like(self.params[name])
            return z, z.clone()
        return st["exp_avg"], st["exp_avg_sq"]

    # --- densification statistics ---------------------------------------------
    @torch.no_grad()
    def add_stats(self, width: int, height: int):
        n = self.count
        radii = self.rast.radii[:n]
        vis = radii > 0
        # Screen-space gradient in NDC units, as the reference measures it.
        g = self.rast.grad_means2d[:n] * torch.tensor([0.5 * width, 0.5 * height], device=radii.device)
        self.grad_accum += torch.where(vis, g.norm(dim=1), torch.zeros(1, device=g.device))
        self.denom += vis.float()
        self.max_radii = torch.where(vis, torch.maximum(self.max_radii, radii.float()), self.max_radii)

    # --- adaptive density control (Kerbl et al. 2023, Sec. 5) ------------------
    @torch.no_grad()
    def densify_and_prune(self, grad_threshold: float, min_opacity: float, max_screen_size: float | None,
                          percent_dense: float = 0.01, max_gaussians: int = 0):
        n = self.count
        grads = self.grad_accum / self.denom
        grads[~torch.isfinite(grads)] = 0.0
        scales = torch.exp(self.params["scales"])
        max_scale = scales.max(dim=1).values
        big = max_scale > percent_dense * self.extent
        candidates = grads >= grad_threshold
        clone = candidates & ~big
        split = candidates & big
        n_new = int(clone.sum()) + 2 * int(split.sum()) - int(split.sum())
        if max_gaussians and n + n_new > max_gaussians:
            # Over budget: densify only the strongest candidates.
            budget = max(0, max_gaussians - n)
            if budget == 0:
                clone[:] = False
                split[:] = False
            else:
                score = torch.where(candidates, grads, torch.full_like(grads, -1.0))
                keep = torch.topk(score, min(budget, int(candidates.sum()))).indices
                chosen = torch.zeros_like(candidates)
                chosen[keep] = True
                clone &= chosen
                split &= chosen

        cur = {name: self.params[name].detach() for name, _ in FIELDS}
        st = {name: self._state(name) for name, _ in FIELDS}

        # Clones: identical copies.
        new_parts = {name: [cur[name], cur[name][clone]] for name in cur}
        # Splits: 2 samples from the Gaussian, scales / (0.8 * 2).
        sel = split.nonzero().squeeze(1)
        if sel.numel():
            stds = scales[sel].repeat(2, 1)
            samples = torch.normal(torch.zeros_like(stds), stds)
            q = self.params["rotations"][sel]
            R = quat_to_mat(q / q.norm(dim=1, keepdim=True)).repeat(2, 1, 1)
            new_parts["means"].append(torch.bmm(R, samples[..., None]).squeeze(-1) + cur["means"][sel].repeat(2, 1))
            new_parts["scales"].append(torch.log(scales[sel].repeat(2, 1) / (0.8 * 2)))
            for name in ("rotations", "opacities", "sh_dc", "sh_rest"):
                new_parts[name].append(cur[name][sel].repeat(2, 1))
        values = {name: torch.cat(parts, 0) for name, parts in new_parts.items()}
        total = values["means"].shape[0]
        added = total - n
        states = {}
        for name in cur:
            avg, sq = st[name]
            pad = torch.zeros(added, avg.shape[1], device=avg.device)
            states[name] = (torch.cat([avg, pad]), torch.cat([sq, pad]))
        max_radii = torch.cat([self.max_radii, torch.zeros(added, device=self.max_radii.device)])

        # Prune: split originals, transparent, and (after the first opacity
        # reset) too large in screen or world space.
        prune = torch.cat([split, torch.zeros(added, dtype=torch.bool, device=split.device)])
        prune |= torch.sigmoid(values["opacities"]).squeeze(1) < min_opacity
        if max_screen_size:
            prune |= max_radii > max_screen_size
            prune |= torch.exp(values["scales"]).max(dim=1).values > 0.1 * self.extent
        keep = ~prune
        values = {k: v[keep] for k, v in values.items()}
        states = {k: (a[keep], b[keep]) for k, (a, b) in states.items()}
        self._set(values, states)
        self.reset_stats()
        return int(clone.sum()), int(split.sum()), int(prune.sum())

    @torch.no_grad()
    def reset_opacity(self, value: float = 0.01):
        p = self.params["opacities"]
        p.copy_(torch.minimum(p, inverse_sigmoid(torch.tensor(value, device=p.device))))
        st = self.optimizer.state.get(p)
        if st:
            st["exp_avg"].zero_()
            st["exp_avg_sq"].zero_()

    # --- export --------------------------------------------------------------
    @torch.no_grad()
    def snapshot(self) -> dict:
        """CPU copy of the parameters (numpy), e.g. for writing a PLY."""
        return {name: self.params[name].detach().cpu().numpy() for name, _ in FIELDS}
