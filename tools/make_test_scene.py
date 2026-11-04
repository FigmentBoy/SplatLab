"""Generates a small synthetic 3DGS scene (PLY + cameras.json) for testing.

The PLY uses the reference 3DGS layout (raw parameters: log-scales, logit
opacity, unnormalized quaternions, SH coefficients) and the cameras use the
reference cameras.json format (camera-to-world, x right / y down / z forward).

Usage: python tools/make_test_scene.py <out_dir> [--count N] [--sh-degree D] [--seed S]
"""
import argparse
import json
import os

import numpy as np


def write_ply(path, xyz, f_dc, f_rest, opacity, scale, rot):
    n = xyz.shape[0]
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{i}" for i in range(3)]
    names += [f"f_rest_{i}" for i in range(f_rest.shape[1])]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.concatenate(
        [xyz, np.zeros((n, 3)), f_dc, f_rest, opacity[:, None], scale, rot], axis=1
    ).astype("<f4")
    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {n}\n"
    header += "".join(f"property float {name}\n" for name in names)
    header += "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def look_at_c2w(eye, target, up):
    """Camera-to-world rotation in the CV convention (z forward, y down)."""
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)  # `up` is the world up direction, e.g. (0,-1,0) for y-down scenes
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    n = args.count

    # Gaussians on a noisy sphere shell plus a few larger blobs in the middle.
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    xyz = dirs * rng.uniform(0.8, 1.0, size=(n, 1))
    xyz[: n // 10] *= 0.3

    colors = 0.5 + 0.5 * dirs  # colour by direction
    sh_c0 = 0.28209479177387814
    f_dc = (colors - 0.5) / sh_c0
    num_rest = [0, 3, 8, 15][args.sh_degree]
    f_rest = rng.normal(scale=0.1, size=(n, 3 * num_rest))

    opacity = rng.uniform(-1.0, 3.0, size=n)  # logits
    scale = np.log(rng.uniform(0.01, 0.08, size=(n, 3)))
    rot = rng.normal(size=(n, 4))

    os.makedirs(args.out_dir, exist_ok=True)
    write_ply(os.path.join(args.out_dir, "point_cloud.ply"), xyz, f_dc, f_rest, opacity, scale, rot)

    cameras = []
    width, height = 640, 480
    fy = fx = 0.5 * height / np.tan(np.radians(50) / 2)
    for i, angle in enumerate(np.linspace(0, 2 * np.pi, 8, endpoint=False)):
        eye = np.array([3.0 * np.sin(angle), -0.8, -3.0 * np.cos(angle)])
        rot_c2w = look_at_c2w(eye, np.zeros(3), np.array([0.0, -1.0, 0.0]))
        cameras.append({
            "id": i, "img_name": f"view_{i:02d}", "width": width, "height": height,
            "position": eye.tolist(), "rotation": rot_c2w.tolist(), "fy": float(fy), "fx": float(fx),
        })
    with open(os.path.join(args.out_dir, "cameras.json"), "w") as f:
        json.dump(cameras, f, indent=1)
    print(f"Wrote {n} Gaussians and {len(cameras)} cameras to {args.out_dir}")


if __name__ == "__main__":
    main()
