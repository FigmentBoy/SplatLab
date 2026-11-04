"""The optimization loop (Kerbl et al. 2023, with the reference hyperparameters).

Each iteration renders a random training view with the Vulkan renderer,
computes (1 - lambda) L1 + lambda (1 - SSIM) in PyTorch, backpropagates the
image gradient through the Vulkan backward pass and takes an Adam step.

Phases (reported to the viewer, with a snapshot at the end of each):
  appearance  - the first `warmup` iterations: colours and opacities only, the
                geometry (positions, scales, rotations) is frozen
  densify     - all parameters, with adaptive density control every 100
                iterations until `densify_until`
  refine      - all parameters, fixed set of Gaussians
"""
from __future__ import annotations

import queue
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from . import export, losses, vk
from .gaussians import GaussianModel


@dataclass
class TrainConfig:
    iterations: int = 30_000
    sh_degree: int = 3
    lambda_dssim: float = 0.2
    warmup: int = 300
    densify_from: int = 500
    densify_until: int = 15_000
    densify_interval: int = 100
    densify_grad_threshold: float = 2e-4
    opacity_reset_interval: int = 3000
    percent_dense: float = 0.01
    max_gaussians: int = 4_000_000
    snapshot_every: int = 1000
    preview_every: int = 0       # render a preview image every N iterations (0: off)
    preview_camera: int = 0      # index into all cameras (0 is a held-out view when evaluating)
    save_iterations: tuple = (7000,)
    log_every: int = 50
    seed: int = 0


class SnapshotWriter:
    """Writes PLY files on a background thread so training does not wait for the disk."""

    def __init__(self, progress):
        self.progress = progress
        self.jobs: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.recent: list[Path] = []

    def submit(self, path: Path, params: dict, label: str, event: bool, keep_recent: int = 0):
        self.jobs.put((path, params, label, event, keep_recent))

    def _run(self):
        while True:
            job = self.jobs.get()
            if job is None:
                return
            path, p, label, event, keep_recent = job
            try:
                export.write_splat_ply(path, p["means"], p["scales"], p["rotations"], p["opacities"],
                                       p["sh_dc"], p["sh_rest"])
                if event:
                    self.progress.emit("ply", path=str(path.resolve()), kind="gaussians", label=label)
                if keep_recent:
                    # Progress snapshots: keep only the newest few (the viewer
                    # may still be reading the previous one).
                    self.recent.append(path)
                    while len(self.recent) > keep_recent:
                        old = self.recent.pop(0)
                        try:
                            old.unlink()
                        except OSError:
                            pass
            except Exception as e:  # noqa: BLE001
                self.progress.log(f"Writing {path} failed: {e}")
            finally:
                self.jobs.task_done()

    def close(self):
        self.jobs.join()
        self.jobs.put(None)
        self.thread.join()


def load_images(cams, image_dir: Path, device, progress):
    """Loads the ground-truth images onto the GPU as uint8 (H, W, 3)."""
    for i, cam in enumerate(cams):
        bgr = cv2.imread(str(image_dir / cam.file_name), cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"cannot read {image_dir / cam.file_name}")
        if bgr.shape[:2] != (cam.height, cam.width):
            bgr = cv2.resize(bgr, (cam.width, cam.height), interpolation=cv2.INTER_AREA)
        cam.image = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).to(device, non_blocking=True)
        if i % 10 == 0 or i == len(cams) - 1:
            progress.stage("load", "Loading images to the GPU", (i + 1) / len(cams))


def gt_image(cam) -> torch.Tensor:
    return cam.image.permute(2, 0, 1).float().div_(255.0)


@torch.no_grad()
def evaluate(model: GaussianModel, cams, iteration: int, progress):
    if not cams:
        return None
    psnrs, ssims = [], []
    for cam in cams:
        img = model.rast.render_raw(model.count, model.active_sh_degree, cam).clamp(0, 1)
        gt = gt_image(cam)
        psnrs.append(losses.psnr(img, gt).item())
        ssims.append(losses.ssim(img, gt).item())
    p, s = float(np.mean(psnrs)), float(np.mean(ssims))
    progress.log(f"[{iteration}] test views: PSNR {p:.2f} dB, SSIM {s:.4f} ({len(cams)} views)")
    progress.emit("eval", iteration=iteration, psnr=p, ssim=s, views=len(cams))
    return p, s


@torch.no_grad()
def write_preview(model: GaussianModel, cam, iteration: int, path: Path, progress, held_out: bool):
    """Saves the current render at `cam` next to the real image, labelled."""
    img = model.rast.render_raw(model.count, model.active_sh_degree, cam).clamp(0, 1)
    gt = gt_image(cam)
    psnr = losses.psnr(img, gt).item()
    pair = torch.cat([img, gt], dim=2).permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
    bar = np.full((40, pair.shape[1], 3), 24, np.uint8)
    canvas = cv2.cvtColor(np.concatenate([bar, pair]), cv2.COLOR_RGB2BGR)
    kind = "held-out view" if held_out else "training view"
    text = f"Iteration {iteration}: {model.count:,} Gaussians, PSNR {psnr:.2f} dB"
    cv2.putText(canvas, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"{cam.name} ({kind})", (cam.width + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    progress.emit("preview", path=str(path.resolve()), iteration=iteration, psnr=psnr)


def train(model: GaussianModel, train_cams, test_cams, cfg: TrainConfig, out_dir: Path, progress,
          all_cams=None) -> int:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True  # fixed image size: autotune the SSIM convolutions
    writer = SnapshotWriter(progress)
    snap_dir = out_dir / "progress"
    snap_dir.mkdir(parents=True, exist_ok=True)

    def snapshot(it: int, label: str):
        writer.submit(snap_dir / f"iteration_{it:06d}.ply", model.snapshot(), label, event=True, keep_recent=3)

    def save(it: int):
        path = out_dir / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"
        writer.submit(path, model.snapshot(), f"Saved iteration {it}", event=False)
        progress.log(f"[{it}] saving {path}")

    model.setup_optimizer()
    cams_for_preview = all_cams or (list(test_cams) + list(train_cams))
    preview_cam = cams_for_preview[min(cfg.preview_camera, len(cams_for_preview) - 1)] \
        if cfg.preview_every and cams_for_preview else None
    snapshot(0, "Initial Gaussians")
    phase_names = {"appearance": "Training: appearance warm-up (colour, opacity)",
                   "densify": "Training: optimization with densification",
                   "refine": "Training: refinement"}

    def phase_of(it):
        if it <= cfg.warmup:
            return "appearance"
        return "densify" if it <= cfg.densify_until else "refine"

    stack: list = []
    ema_loss, ema_psnr = None, None
    t0 = time.time()
    last_log_time, last_log_it = t0, 0
    phase = None
    for it in range(1, cfg.iterations + 1):
        new_phase = phase_of(it)
        if new_phase != phase:
            if phase is not None:
                snapshot(it - 1, f"End of {phase} phase")
            phase = new_phase
            progress.stage_start(phase, phase_names[phase])
        model.set_learning_rates(it, cfg.iterations, freeze_geometry=phase == "appearance")
        if it % 1000 == 0 and model.active_sh_degree < model.max_sh_degree:
            model.active_sh_degree += 1

        if not stack:
            stack = list(train_cams)
            random.shuffle(stack)
        cam = stack.pop()

        img = vk.render(model.rast, model.param_list(), model.count, model.active_sh_degree, cam)
        gt = gt_image(cam)
        l1 = losses.l1(img, gt)
        ssim = losses.ssim(img, gt)
        loss = (1.0 - cfg.lambda_dssim) * l1 + cfg.lambda_dssim * (1.0 - ssim)
        loss.backward()

        # Periodic snapshot before this iteration's densification / opacity
        # reset, so it never shows the transparent state right after a reset.
        if cfg.snapshot_every and it % cfg.snapshot_every == 0 and it != cfg.iterations:
            snapshot(it, f"Iteration {it}")

        with torch.no_grad():
            if it < cfg.densify_until:  # as the reference (no reset at the last iteration)
                model.add_stats(cam.width, cam.height)
                if it > cfg.densify_from and it % cfg.densify_interval == 0:
                    size_limit = 20 if it > cfg.opacity_reset_interval else None
                    model.densify_and_prune(cfg.densify_grad_threshold, 0.005, size_limit, cfg.percent_dense,
                                            cfg.max_gaussians)
                    # The Vulkan buffers live outside PyTorch's caching allocator:
                    # hand back the blocks densification freed so both fit.
                    torch.cuda.empty_cache()
                if it % cfg.opacity_reset_interval == 0:
                    model.reset_opacity()
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)

            if it % 10 == 0:
                l_val = loss.item()
                p_val = losses.psnr(img.detach(), gt).item()
                ema_loss = l_val if ema_loss is None else 0.6 * ema_loss + 0.4 * l_val
                ema_psnr = p_val if ema_psnr is None else 0.6 * ema_psnr + 0.4 * p_val
            if it % cfg.log_every == 0 or it == cfg.iterations:
                now = time.time()
                its = (it - last_log_it) / max(now - last_log_time, 1e-6)
                last_log_time, last_log_it = now, it
                progress.emit("train", iteration=it, total=cfg.iterations, phase=phase, loss=ema_loss,
                              psnr=ema_psnr, gaussians=model.count, its=its)
                progress.stage(phase, phase_names[phase], it / cfg.iterations)
                if it % (cfg.log_every * 10) == 0 or it == cfg.iterations:
                    progress.log(f"[{it}/{cfg.iterations}] loss {ema_loss:.4f}  PSNR {ema_psnr:.2f} dB  "
                                 f"{model.count:,} Gaussians  {its:.1f} it/s")
                    free, total = torch.cuda.mem_get_info()
                    progress.log(f"    GPU memory {(total - free) / 2**30:.1f}/{total / 2**30:.1f} GiB used "
                                 f"(torch {torch.cuda.memory_allocated() / 2**30:.1f} allocated, "
                                 f"{torch.cuda.memory_reserved() / 2**30:.1f} reserved), "
                                 f"{model.rast.last_stats[1]:,} tile instances")

        if cfg.preview_every and (it % cfg.preview_every == 0 or it == cfg.iterations) and preview_cam is not None:
            write_preview(model, preview_cam, it, out_dir / "previews" / f"iter_{it:06d}.jpg", progress,
                          any(c is preview_cam for c in test_cams))
        if it in cfg.save_iterations and it != cfg.iterations:
            save(it)
            evaluate(model, test_cams, it, progress)

    elapsed = time.time() - t0
    progress.log(f"Training finished: {cfg.iterations} iterations in {elapsed / 60:.1f} min "
                 f"({cfg.iterations / elapsed:.1f} it/s), {model.count:,} Gaussians")
    save(cfg.iterations)
    evaluate(model, test_cams, cfg.iterations, progress)
    writer.close()
    return cfg.iterations
