"""Output files in the reference 3DGS layout:

  <out>/cfg_args                  Namespace(...) of the run
  <out>/cameras.json              every camera (train and test)
  <out>/test_cameras.json         the held-out test cameras
  <out>/input.ply                 the SfM points (x y z nx ny nz red green blue)
  <out>/point_cloud/iteration_<N>/point_cloud.ply
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def splat_ply_header(n: int, rest_coeffs: int = 45) -> str:
    props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    props += [f"f_rest_{i}" for i in range(rest_coeffs)]
    props += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    lines = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    lines += [f"property float {p}" for p in props]
    lines.append("end_header")
    return "\n".join(lines) + "\n"


def write_splat_ply(path: Path, means, scales, rotations, opacities, sh_dc, sh_rest):
    """Raw parameters (numpy, float32) -> reference PLY. sh_rest is (N, 15, 3) or (N, 45)."""
    n = means.shape[0]
    rest = np.asarray(sh_rest, dtype=np.float32).reshape(n, 15, 3).transpose(0, 2, 1).reshape(n, 45)
    data = np.concatenate([
        np.asarray(means, np.float32), np.zeros((n, 3), np.float32), np.asarray(sh_dc, np.float32).reshape(n, 3),
        rest, np.asarray(opacities, np.float32).reshape(n, 1), np.asarray(scales, np.float32).reshape(n, 3),
        np.asarray(rotations, np.float32).reshape(n, 4),
    ], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        f.write(splat_ply_header(n).encode("ascii"))
        f.write(np.ascontiguousarray(data, dtype="<f4").tobytes())
    os.replace(tmp, path)  # readers never see a partial file


def write_points_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    """Plain coloured point cloud (the reference's input.ply)."""
    n = xyz.shape[0]
    dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    arr = np.zeros(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header += [f"property float {c}" for c in ("x", "y", "z", "nx", "ny", "nz")]
    header += ["property uchar red", "property uchar green", "property uchar blue", "end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(arr.tobytes())


def camera_entry(idx: int, cam) -> dict:
    w2c = np.asarray(cam.world_to_camera, dtype=np.float64)
    c2w = np.linalg.inv(w2c)
    return {
        "id": idx,
        "img_name": cam.name,
        "width": int(cam.width),
        "height": int(cam.height),
        "position": c2w[:3, 3].tolist(),
        "rotation": [row.tolist() for row in c2w[:3, :3]],
        "fy": float(cam.fy),
        "fx": float(cam.fx),
    }


def write_cameras(path: Path, cams):
    with open(path, "w") as f:
        json.dump([camera_entry(i, c) for i, c in enumerate(cams)], f, indent=1)


def write_cfg_args(path: Path, *, source_path: Path, model_path: Path, sh_degree: int, eval_split: bool,
                   white_background: bool = False):
    text = (f"Namespace(data_device='cuda', eval={eval_split}, images='images', model_path='{model_path.as_posix()}', "
            f"resolution=-1, sh_degree={sh_degree}, source_path='{source_path.as_posix()}', "
            f"white_background={white_background})")
    path.write_text(text)
