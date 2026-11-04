"""Video -> 3D Gaussian Splat.

    python trainer/train.py <video or image folder> -o <output folder> [options]

Pipeline: frame extraction (OpenCV) -> Structure from Motion (pycolmap) ->
Gaussian initialization -> optimization with the Vulkan renderer's forward
and backward passes (PyTorch computes the loss and runs Adam) -> export.

The output folder has the reference 3DGS layout (point_cloud/iteration_N/
point_cloud.ply, cameras.json, test_cameras.json, cfg_args, input.ply) and can
be opened in splat_lab or any 3DGS viewer. The COLMAP dataset is kept in
<output>/source; re-running on the same output folder reuses it unless
--redo-sfm is given.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from splat_trainer import export  # noqa: E402
from splat_trainer.frames import extract_frames  # noqa: E402
from splat_trainer.gaussians import GaussianModel, knn_mean_sq_dist  # noqa: E402
from splat_trainer.progress import Progress  # noqa: E402
from splat_trainer.sfm import camera_extent, load_scene, run_sfm  # noqa: E402
from splat_trainer.training import TrainConfig, load_images, train  # noqa: E402
from splat_trainer.vk import VulkanRasterizer  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path,
                    help="video file, a folder of images, or a COLMAP dataset (a folder with sparse/ and images/)")
    ap.add_argument("-o", "--output", type=Path, required=True, help="output model folder")
    g = ap.add_argument_group("frames and SfM")
    g.add_argument("--frames", type=int, default=200, help="number of frames to extract (default 200)")
    g.add_argument("--max-size", type=int, default=1600, help="longer image side in pixels (default 1600)")
    g.add_argument("--matcher", choices=("sequential", "exhaustive"), default="sequential")
    g.add_argument("--images", default="images",
                   help="image folder of a COLMAP dataset input, e.g. images_4 (default images)")
    g.add_argument("--redo-sfm", action="store_true", help="redo frames and SfM even if <output>/source exists")
    g = ap.add_argument_group("training")
    g.add_argument("--iterations", type=int, default=30_000)
    g.add_argument("--sh-degree", type=int, default=3, choices=(0, 1, 2, 3))
    g.add_argument("--warmup", type=int, default=300, help="appearance-only iterations at the start")
    g.add_argument("--densify-until", type=int, default=None, help="default: half of --iterations, max 15000")
    g.add_argument("--max-gaussians", type=int, default=4_000_000)
    g.add_argument("--no-eval", action="store_true", help="train on every frame (no held-out test views)")
    g.add_argument("--snapshot-every", type=int, default=1000, help="progress snapshot interval (0: off)")
    g.add_argument("--preview-every", type=int, default=0,
                   help="save <output>/previews/iter_N.jpg (render next to the photo) every N iterations (0: off)")
    g.add_argument("--preview-camera", type=int, default=0,
                   help="camera index (in cameras.json) for previews; 0 is a held-out view (default 0)")
    g.add_argument("--seed", type=int, default=0)
    g = ap.add_argument_group("system")
    g.add_argument("--device", type=int, default=0, help="CUDA device index")
    g.add_argument("--module-dir", help="folder containing the splat_vk module (default: build*/)")
    g.add_argument("--progress-json", action="store_true", help="print machine-readable progress events")
    return ap.parse_args(argv)


def write_sparse_splats(path: Path, xyz: np.ndarray, rgb: np.ndarray, device):
    """The SfM points as small opaque splats, so any splat viewer shows the point cloud."""
    pts = torch.tensor(xyz, dtype=torch.float32, device=device)
    radius = (0.3 * torch.sqrt(knn_mean_sq_dist(pts).clamp_min(1e-10))).clamp_max(0.05).cpu().numpy()
    n = xyz.shape[0]
    rest = np.zeros((n, 45), np.float32)
    dc = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.28209479177387814
    export.write_splat_ply(path, xyz.astype(np.float32), np.log(np.repeat(radius[:, None], 3, 1)),
                           np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
                           np.full((n, 1), 3.0, np.float32), dc, rest)


def main(argv=None) -> int:
    args = parse_args(argv)
    progress = Progress(json_events=args.progress_json)
    out = args.output.resolve()
    try:
        if not args.input.exists():
            raise FileNotFoundError(f"input not found: {args.input}")
        out.mkdir(parents=True, exist_ok=True)
        source, work = out / "source", out / "work"
        dataset_input = args.input.is_dir() and (args.input / "sparse").is_dir()

        progress.stage_start("init", "Initializing the Vulkan renderer")
        rast = VulkanRasterizer(device=args.device, module_dir=args.module_dir)
        progress.log(f"GPU: {rast.device_name} (Vulkan, sharing memory with CUDA device {args.device})")

        if dataset_input:
            source = args.input.resolve()
            progress.log(f"Using the COLMAP dataset {source} ({args.images})")
        elif args.redo_sfm or not (source / "sparse").exists():
            progress.stage_start("frames", "Extracting frames")
            t = time.time()
            n = extract_frames(args.input, work / "frames", args.frames, args.max_size, progress)
            progress.log(f"{n} frames extracted in {time.time() - t:.1f} s")
            t = time.time()
            run_sfm(work / "frames", source, work, progress, matcher=args.matcher)
            progress.log(f"SfM finished in {time.time() - t:.1f} s")
            shutil.rmtree(work, ignore_errors=True)  # frames and feature database
        else:
            progress.log(f"Reusing the SfM dataset in {source}")

        scene = load_scene(source, args.images if dataset_input else "images")
        cams = scene.cameras
        test_ids = set() if args.no_eval else {i for i in range(len(cams)) if i % 8 == 0}
        train_cams = [c for i, c in enumerate(cams) if i not in test_ids]
        test_cams = [c for i, c in enumerate(cams) if i in test_ids]
        extent = camera_extent(cams)
        progress.log(f"{len(cams)} cameras ({len(train_cams)} train, {len(test_cams)} test), "
                     f"{len(scene.points):,} SfM points, {cams[0].width}x{cams[0].height}, extent {extent:.2f}")

        export.write_cameras(out / "cameras.json", cams)
        export.write_cameras(out / "test_cameras.json", test_cams if test_cams else cams[::8])
        export.write_cfg_args(out / "cfg_args", source_path=source, model_path=out, sh_degree=args.sh_degree,
                              eval_split=not args.no_eval)
        export.write_points_ply(out / "input.ply", scene.points, scene.colors)
        points_ply = out / "progress" / "sfm_points.ply"
        write_sparse_splats(points_ply, scene.points, scene.colors, rast.device)
        progress.emit("cameras", path=str(out / "cameras.json"))
        progress.emit("ply", path=str(points_ply), kind="points", label="SfM point cloud")

        progress.stage_start("load", "Loading images to the GPU")
        load_images(cams, scene.image_dir, rast.device, progress)

        cfg = TrainConfig(iterations=args.iterations, sh_degree=args.sh_degree, warmup=args.warmup,
                          max_gaussians=args.max_gaussians, snapshot_every=args.snapshot_every, seed=args.seed,
                          preview_every=args.preview_every, preview_camera=args.preview_camera)
        cfg.densify_until = args.densify_until if args.densify_until is not None \
            else min(15_000, args.iterations // 2)
        cfg.save_iterations = tuple(i for i in (7000,) if i < args.iterations)
        model = GaussianModel(rast, max_sh_degree=args.sh_degree)
        model.init_from_points(scene.points, scene.colors, extent)
        progress.log(f"Initialized {model.count:,} Gaussians")
        train(model, train_cams, test_cams, cfg, out, progress, all_cams=cams)
        for snap in (out / "progress").glob("iteration_*.ply"):
            try:
                snap.unlink()  # intermediate snapshots; the final model is in point_cloud/
            except OSError:
                pass
        progress.emit("done", output=str(out))
        progress.log(f"Done: {out}")
        return 0
    except KeyboardInterrupt:
        progress.emit("error", message="cancelled")
        return 130
    except Exception as e:  # noqa: BLE001
        traceback.print_exc(file=sys.stdout)
        progress.emit("error", message=f"{type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
