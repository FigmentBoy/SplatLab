"""Structure from Motion with pycolmap.

SIFT features, sequential matching (video frames are ordered, so each frame
is matched with its neighbours plus quadratically spaced ones), incremental
mapping, and undistortion to a pinhole camera. Falls back to exhaustive
matching if too few frames register. The result is written as a COLMAP
dataset (<source>/images + <source>/sparse), the reference 3DGS input format.

The scene is then normalized (similarity transform): centred on the camera
centres and scaled so the median depth of the SfM points is TARGET_DEPTH. That
makes the renderer's fixed 0.2 near plane and the viewer's default speeds
reasonable whatever scale SfM picked.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .vk import Camera

TARGET_DEPTH = 4.0


@dataclass
class SceneData:
    cameras: list          # vk.Camera, sorted by name (images not loaded)
    points: np.ndarray     # (M, 3) float64
    colors: np.ndarray     # (M, 3) uint8
    image_dir: Path


def _pose(image) -> np.ndarray:
    cam_from_world = image.cam_from_world
    if callable(cam_from_world):
        cam_from_world = cam_from_world()
    m = np.eye(4)
    m[:3, :4] = np.asarray(cam_from_world.matrix())
    return m


def _map(pycolmap, database: Path, image_dir: Path, sparse_dir: Path, num_frames: int, progress):
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True)
    registered = [0]

    def on_next_image():
        registered[0] += 1
        progress.stage("mapping", f"Reconstructing cameras ({registered[0]}/{num_frames} registered)",
                       min(registered[0] / num_frames, 1.0))

    def on_initial_pair():
        registered[0] = 2
        progress.stage("mapping", "Reconstructing cameras (initial pair)", 2 / num_frames)

    options = pycolmap.IncrementalPipelineOptions()
    options.extract_colors = True
    recs = pycolmap.incremental_mapping(str(database), str(image_dir), str(sparse_dir), options=options,
                                        initial_image_pair_callback=on_initial_pair,
                                        next_image_callback=on_next_image)
    if not recs:
        return None, None
    idx, rec = max(recs.items(), key=lambda kv: kv[1].num_reg_images())
    return idx, rec


def run_sfm(frames_dir: Path, source_dir: Path, work_dir: Path, progress, matcher: str = "sequential",
            camera_model: str = "SIMPLE_RADIAL") -> None:
    """Runs SfM on frames_dir and writes the undistorted dataset into source_dir."""
    import pycolmap

    # COLMAP logs every image at INFO level; warnings are enough here (the
    # stage callbacks report progress).
    try:
        pycolmap.logging.minloglevel = 1
    except AttributeError:
        pass
    work_dir.mkdir(parents=True, exist_ok=True)
    database = work_dir / "database.db"
    if database.exists():
        database.unlink()
    num_frames = len(list(frames_dir.glob("*.jpg")))

    progress.stage_start("features", f"Extracting SIFT features from {num_frames} frames")
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = camera_model
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.sift.max_num_features = 8192
    pycolmap.extract_features(str(database), str(frames_dir), camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=extraction)

    def match(kind: str):
        progress.stage_start("matching", f"Matching features ({kind})")
        if kind == "exhaustive":
            pycolmap.match_exhaustive(str(database))
        else:
            pairing = pycolmap.SequentialPairingOptions()
            pairing.overlap = 15
            pairing.quadratic_overlap = True
            pycolmap.match_sequential(str(database), pairing_options=pairing)

    match(matcher)
    progress.stage_start("mapping", "Reconstructing cameras")
    idx, rec = _map(pycolmap, database, frames_dir, work_dir / "sparse", num_frames, progress)
    registered = rec.num_reg_images() if rec else 0
    if matcher != "exhaustive" and registered < 0.6 * num_frames:
        progress.log(f"Only {registered}/{num_frames} frames registered; retrying with exhaustive matching")
        match("exhaustive")
        idx, rec = _map(pycolmap, database, frames_dir, work_dir / "sparse", num_frames, progress)
        registered = rec.num_reg_images() if rec else 0
    if rec is None or registered < 3:
        raise RuntimeError("Structure from Motion failed: the frames could not be registered. "
                           "Use a video with more camera motion, texture and overlap.")
    progress.log(f"SfM: {registered}/{num_frames} frames registered, {rec.num_points3D()} points, "
                 f"mean reprojection error {rec.compute_mean_reprojection_error():.3f} px")

    progress.stage_start("undistort", "Undistorting images")
    if source_dir.exists():
        shutil.rmtree(source_dir)
    pycolmap.undistort_images(str(source_dir), str(work_dir / "sparse" / str(idx)), str(frames_dir))
    # undistort_images writes sparse/ directly; the reference layout uses sparse/0.
    # Only images/ and sparse/ are needed (not the MVS scaffolding).
    shutil.rmtree(source_dir / "stereo", ignore_errors=True)
    for script in source_dir.glob("run-colmap-*.sh"):
        script.unlink()
    sparse = source_dir / "sparse"
    if (sparse / "cameras.bin").exists() or (sparse / "cameras.txt").exists():
        (sparse / "0").mkdir(exist_ok=True)
        for f in list(sparse.iterdir()):
            if f.is_file():
                f.replace(sparse / "0" / f.name)


def load_scene(source_dir: Path, images: str = "images") -> SceneData:
    """Reads an undistorted COLMAP dataset and normalizes its scale.

    `images` is the image folder inside the dataset; it may hold downscaled
    copies (e.g. Mip-NeRF 360's images_4), in which case the intrinsics are
    rescaled to the actual image size.
    """
    import cv2
    import pycolmap

    sparse = source_dir / "sparse" / "0"
    if not sparse.exists():
        sparse = source_dir / "sparse"
    rec = pycolmap.Reconstruction(str(sparse))

    cams = []
    for image in sorted(rec.images.values(), key=lambda im: im.name):
        if hasattr(image, "has_pose") and not image.has_pose:
            continue
        c = rec.cameras[image.camera_id]
        if c.model_name not in ("PINHOLE", "SIMPLE_PINHOLE"):
            raise ValueError(f"expected undistorted pinhole cameras, got {c.model_name}")
        cams.append(Camera(name=Path(image.name).stem, width=c.width, height=c.height,
                           fx=c.focal_length_x, fy=c.focal_length_y,
                           cx=c.principal_point_x, cy=c.principal_point_y,
                           world_to_camera=_pose(image), file_name=image.name))
    image_dir = source_dir / images
    first = cv2.imread(str(image_dir / cams[0].file_name), cv2.IMREAD_UNCHANGED) if cams else None
    if first is None:
        raise IOError(f"cannot read {image_dir / cams[0].file_name}")
    h, w = first.shape[:2]
    for c in cams:
        if (c.width, c.height) != (w, h):
            sx, sy = w / c.width, h / c.height
            c.fx, c.cx, c.fy, c.cy = c.fx * sx, c.cx * sx, c.fy * sy, c.cy * sy
            c.width, c.height = w, h
    pts = rec.points3D
    xyz = np.array([p.xyz for p in pts.values()], dtype=np.float64).reshape(-1, 3)
    rgb = np.array([p.color for p in pts.values()], dtype=np.uint8).reshape(-1, 3)

    # --- normalization -----------------------------------------------------
    centres = np.stack([c.position for c in cams])
    centre = centres.mean(axis=0)
    depths = []
    for c in cams[:: max(1, len(cams) // 50)]:
        z = (xyz @ c.world_to_camera[2, :3]) + c.world_to_camera[2, 3]
        depths.append(z[z > 0])
    depths = np.concatenate(depths) if depths else np.zeros(0)
    median = float(np.median(depths)) if depths.size else 1.0
    scale = TARGET_DEPTH / max(median, 1e-9)
    xyz = (xyz - centre) * scale
    for c in cams:
        R = c.world_to_camera[:3, :3]
        t = c.world_to_camera[:3, 3]
        c.world_to_camera = c.world_to_camera.copy()
        c.world_to_camera[:3, 3] = scale * (t + R @ centre)
    return SceneData(cameras=cams, points=xyz, colors=rgb, image_dir=image_dir)


def camera_extent(cams) -> float:
    """The reference's getNerfppNorm radius: 1.1 x the largest distance from the camera centroid."""
    centres = np.stack([c.position for c in cams])
    return float(np.linalg.norm(centres - centres.mean(0), axis=1).max() * 1.1) or 1.0
