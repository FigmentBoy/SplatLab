"""Independent NumPy implementation of the official 3DGS forward pass
(diff-gaussian-rasterization, forward.cu), evaluated on a window of 16x16 tiles
of a scene and compared against the same window of one of our renders.

Usage: python tools/reference_crop.py <model_dir> <img_name> <our_png> tx0 ty0 tx1 ty1 [out_prefix] [--bg r,g,b]
  <model_dir> holds point_cloud/iteration_30000/point_cloud.ply and cameras.json.
  The tile window is [tx0, tx1) x [ty0, ty1).
"""
import json, sys
import numpy as np
from PIL import Image

# All arithmetic runs in float32, like the official CUDA rasterizer (and ours).
F = np.float32

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
SH_C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
         -0.4570457994644658, 1.445305721320277, -0.5900435899266435]

def load_ply(path):
    with open(path, "rb") as f:
        names = []
        while True:
            line = f.readline().decode().strip()
            if line.startswith("element vertex"): n = int(line.split()[-1])
            if line.startswith("property"): names.append(line.split()[-1])
            if line == "end_header": break
        offset = f.tell()
    data = np.memmap(path, dtype="<f4", mode="r", offset=offset, shape=(n, len(names)))
    return data, {nm: i for i, nm in enumerate(names)}, n

def check_window(model, img_name, our_png, tx0, ty0, tx1, ty1, prefix=None, background=(0.0, 0.0, 0.0)):
    data, col, n = load_ply(f"{model}/point_cloud/iteration_30000/point_cloud.ply")
    cam = next(c for c in json.load(open(f"{model}/cameras.json")) if c["img_name"] == img_name)
    # cameras.json stores full-resolution intrinsics; render at the size of our image
    # (the official code rescales via the field of view, which is equivalent).
    W, H = Image.open(our_png).size
    fx = F(cam["fx"] * W / cam["width"])
    fy = F(cam["fy"] * H / cam["height"])
    R = np.array(cam["rotation"], dtype=F); C = np.array(cam["position"], dtype=F)
    Rw2c = R.T; tw2c = -R.T @ C
    tanx, tany = F(W) / (F(2) * fx), F(H) / (F(2) * fy)
    gx, gy = (W + 15) // 16, (H + 15) // 16

    sel_idx, sel = [], {}
    keys = ["mean", "conic", "opac", "depth", "rmin", "rmax"]
    for k in keys: sel[k] = []
    for s in range(0, n, 1_000_000):
        d = np.asarray(data[s:s + 1_000_000], dtype=F)
        m = d.shape[0]
        xyz = d[:, [col["x"], col["y"], col["z"]]]
        t = xyz @ Rw2c.T + tw2c
        tz = t[:, 2]
        valid = tz > 0.2
        scale = np.exp(d[:, [col["scale_0"], col["scale_1"], col["scale_2"]]])
        q = d[:, [col["rot_0"], col["rot_1"], col["rot_2"], col["rot_3"]]]
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        r, x, y, z = q.T
        Rm = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
                       2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
                       2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)], 1).reshape(m, 3, 3)
        M = Rm * scale[:, None, :]
        Sigma = M @ M.transpose(0, 2, 1)
        with np.errstate(all="ignore"):
            txc = np.clip(t[:, 0] / tz, -1.3 * tanx, 1.3 * tanx) * tz
            tyc = np.clip(t[:, 1] / tz, -1.3 * tany, 1.3 * tany) * tz
            J = np.zeros((m, 2, 3), dtype=F)
            J[:, 0, 0] = fx / tz; J[:, 0, 2] = -fx * txc / tz**2
            J[:, 1, 1] = fy / tz; J[:, 1, 2] = -fy * tyc / tz**2
            T = J @ Rw2c
            cov = T @ Sigma @ T.transpose(0, 2, 1)
            a = cov[:, 0, 0] + 0.3; b = cov[:, 0, 1]; c = cov[:, 1, 1] + 0.3
            det = a * c - b * b
            valid &= det != 0
            conic = np.stack([c / det, -b / det, a / det], 1)
            mid = 0.5 * (a + c)
            radius = np.ceil(3 * np.sqrt(mid + np.sqrt(np.maximum(0.1, mid * mid - det))))
            mean = np.stack([fx * t[:, 0] / tz + W / 2 - 0.5, fy * t[:, 1] / tz + H / 2 - 0.5], 1)
            rmin = np.stack([np.clip(np.trunc((mean[:, 0] - radius) / 16), 0, gx), np.clip(np.trunc((mean[:, 1] - radius) / 16), 0, gy)], 1)
            rmax = np.stack([np.clip(np.trunc((mean[:, 0] + radius + 15) / 16), 0, gx), np.clip(np.trunc((mean[:, 1] + radius + 15) / 16), 0, gy)], 1)
        valid &= (rmax[:, 0] - rmin[:, 0]) * (rmax[:, 1] - rmin[:, 1]) > 0
        # keep only Gaussians overlapping the tile window
        valid &= (rmin[:, 0] < tx1) & (rmax[:, 0] > tx0) & (rmin[:, 1] < ty1) & (rmax[:, 1] > ty0)
        idx = np.nonzero(valid)[0]
        sel_idx.append(idx + s)
        sel["mean"].append(mean[idx]); sel["conic"].append(conic[idx]); sel["depth"].append(tz[idx])
        sel["opac"].append(1 / (1 + np.exp(-d[idx, col["opacity"]])))
        sel["rmin"].append(rmin[idx]); sel["rmax"].append(rmax[idx])
    gid = np.concatenate(sel_idx)
    for k in keys: sel[k] = np.concatenate(sel[k])
    print(f"{n} Gaussians, {len(gid)} overlap the window", flush=True)

    # colours for the selected Gaussians
    d = np.asarray(data[gid], dtype=F)
    xyz = d[:, [col["x"], col["y"], col["z"]]]
    nrest = sum(1 for k in col if k.startswith("f_rest_")) // 3
    sh = np.zeros((len(gid), 16, 3), dtype=F)
    for ch in range(3):
        sh[:, 0, ch] = d[:, col[f"f_dc_{ch}"]]
        for k in range(nrest): sh[:, k + 1, ch] = d[:, col[f"f_rest_{ch * nrest + k}"]]
    dirs = xyz - C; dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    dx, dy, dz = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
    xx, yy, zz, xy, yz, xz = dx*dx, dy*dy, dz*dz, dx*dy, dy*dz, dx*dz
    colr = SH_C0 * sh[:, 0] - SH_C1 * dy * sh[:, 1] + SH_C1 * dz * sh[:, 2] - SH_C1 * dx * sh[:, 3]
    colr += SH_C2[0]*xy*sh[:,4] + SH_C2[1]*yz*sh[:,5] + SH_C2[2]*(2*zz-xx-yy)*sh[:,6] + SH_C2[3]*xz*sh[:,7] + SH_C2[4]*(xx-yy)*sh[:,8]
    colr += SH_C3[0]*dy*(3*xx-yy)*sh[:,9] + SH_C3[1]*xy*dz*sh[:,10] + SH_C3[2]*dy*(4*zz-xx-yy)*sh[:,11] \
        + SH_C3[3]*dz*(2*zz-3*xx-3*yy)*sh[:,12] + SH_C3[4]*dx*(4*zz-xx-yy)*sh[:,13] + SH_C3[5]*dz*(xx-yy)*sh[:,14] + SH_C3[6]*dx*(xx-3*yy)*sh[:,15]
    colr = np.maximum(colr + 0.5, 0)

    # Sort by depth, ties broken by Gaussian index (what a stable sort of
    # index-ordered keys gives, as in both renderers).
    order = np.lexsort((gid, sel["depth"]))
    mean, conic, opac, rmin, rmax = sel["mean"], sel["conic"], sel["opac"], sel["rmin"], sel["rmax"]
    x0, y0 = tx0 * 16, ty0 * 16
    x1, y1 = min(tx1 * 16, W), min(ty1 * 16, H)
    img = np.zeros((y1 - y0, x1 - x0, 3), dtype=F)
    for ty in range(ty0, ty1):
        for tx in range(tx0, tx1):
            o = order[(rmin[order, 0] <= tx) & (tx < rmax[order, 0]) & (rmin[order, 1] <= ty) & (ty < rmax[order, 1])]
            ys, xs = np.mgrid[ty*16:min(ty*16+16, H), tx*16:min(tx*16+16, W)]
            px = np.stack([xs.ravel(), ys.ravel()], 1).astype(F)
            Tr = np.ones(len(px), dtype=F); Cc = np.zeros((len(px), 3), dtype=F); done = np.zeros(len(px), bool)
            for g in o:
                dd = mean[g] - px
                power = -0.5 * (conic[g, 0] * dd[:, 0]**2 + conic[g, 2] * dd[:, 1]**2) - conic[g, 1] * dd[:, 0] * dd[:, 1]
                alpha = np.minimum(F(0.99), opac[g] * np.exp(power))
                ok = (~done) & (power <= 0) & (alpha >= F(1.0) / F(255.0))
                testT = Tr * (1 - alpha)
                stop = ok & (testT < F(0.0001))
                done |= stop; ok &= ~stop
                Cc[ok] += colr[g] * (alpha[ok] * Tr[ok])[:, None]
                Tr = np.where(ok, testT, Tr)
                if done.all(): break
            img[ys.ravel() - y0, xs.ravel() - x0] = Cc + Tr[:, None] * np.asarray(background, dtype=F)[None, :]
    ref = np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)
    ours = np.asarray(Image.open(our_png).convert("RGB"))[y0:y1, x0:x1]
    diff = np.abs(ref.astype(int) - ours.astype(int))
    result = {"window": [x0, x1, y0, y1], "gaussians_in_window": int(len(gid)), "max_abs": int(diff.max()),
              "mean_abs": float(diff.mean()), "pixels_gt_1": int((diff.max(2) > 1).sum()),
              "pixels": int(diff.shape[0] * diff.shape[1])}
    print(f"window px [{x0},{x1})x[{y0},{y1}): max|d|={result['max_abs']} mean|d|={result['mean_abs']:.4f} "
          f"pixels with |d|>1: {result['pixels_gt_1']} / {result['pixels']}")
    if prefix:
        Image.fromarray(ref).save(prefix + "_ref.png")
        Image.fromarray(ours).save(prefix + "_ours.png")
        Image.fromarray(np.clip(diff * 20, 0, 255).astype(np.uint8)).save(prefix + "_diff.png")
    return result, ours, ref, diff


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    bg = (0.0, 0.0, 0.0)
    if "--bg" in sys.argv:
        bg = tuple(float(v) for v in sys.argv[sys.argv.index("--bg") + 1].split(","))
        args.remove(sys.argv[sys.argv.index("--bg") + 1])
    check_window(args[0], args[1], args[2], *map(int, args[3:7]), args[7] if len(args) > 7 else None, bg)
