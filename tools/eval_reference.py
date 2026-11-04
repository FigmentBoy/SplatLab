"""Evaluates our renderer on the official pretrained 3DGS scenes.

For each scene under data/models/<scene> (pretrained model) with matching
official evaluation images under data/eval/<scene>/test/ours_30000:
  1. renders the test views at the official resolution and background,
  2. computes PSNR of ours vs ground truth, official vs ground truth and ours vs
     official, plus the global sub-pixel shift between ours and official,
  3. checks a 128x128 window pixel by pixel against the NumPy reference of the
     official forward pass (tools/reference_crop.py),
  4. writes a labelled comparison image to data/report/<scene>.png.
Results go to data/report/summary.json and data/report/summary.md.

Usage: python tools/eval_reference.py [--renderer build/splat_lab.exe] [--no-crop] [scene ...]
"""
import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference_crop  # noqa: E402


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))


def global_shift(a, b):
    """Sub-pixel translation between two images (phase correlation)."""
    a = a.mean(axis=2); b = b.mean(axis=2)
    R = np.fft.fft2(a - a.mean()) * np.conj(np.fft.fft2(b - b.mean()))
    r = np.real(np.fft.ifft2(R / (np.abs(R) + 1e-12)))
    y, x = np.unravel_index(np.argmax(r), r.shape)
    H, W = r.shape

    def sub(m, c, p):
        den = m - 2 * c + p
        return 0.5 * (m - p) / den if den != 0 else 0.0

    dy = sub(r[(y - 1) % H, x], r[y, x], r[(y + 1) % H, x])
    dx = sub(r[y, (x - 1) % W], r[y, x], r[y, (x + 1) % W])
    y = y - H if y > H // 2 else y
    x = x - W if x > W // 2 else x
    return float(x + dx), float(y + dy)


def labelled_row(images, labels, width):
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    ims = []
    for im in images:
        h = round(im.height * width / im.width)
        ims.append(im.convert("RGB").resize((width, h), Image.NEAREST if im.width < width else Image.LANCZOS))
    gap, label_h = 8, 30
    canvas = Image.new("RGB", (len(ims) * width + gap * (len(ims) - 1), max(i.height for i in ims) + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for k, (im, lab) in enumerate(zip(ims, labels)):
        x = k * (width + gap)
        canvas.paste(im, (x, label_h))
        draw.text((x + 4, 4), lab, fill="black", font=font)
    return canvas


def stack(rows, title):
    try:
        font = ImageFont.truetype("arialbd.ttf", 26)
    except OSError:
        font = ImageFont.load_default()
    W = max(r.width for r in rows)
    H = sum(r.height for r in rows) + 44 + 10 * len(rows)
    out = Image.new("RGB", (W, H), "white")
    ImageDraw.Draw(out).text((4, 6), title, fill="black", font=font)
    y = 44
    for r in rows:
        out.paste(r, (0, y))
        y += r.height + 10
    return out


def evaluate(scene, args):
    model = os.path.join(args.data, "models", scene)
    evald = os.path.join(args.data, "eval", scene, "test", "ours_30000")
    ply = os.path.join(model, "point_cloud", "iteration_30000", "point_cloud.ply")
    if not (os.path.exists(ply) and os.path.isdir(os.path.join(evald, "renders"))):
        print(f"[{scene}] missing model or evaluation images, skipped")
        return None

    names = sorted(n for n in os.listdir(os.path.join(evald, "renders")) if n.endswith(".png"))
    cams = json.load(open(os.path.join(model, "cameras.json")))
    by_name = {c["img_name"]: c for c in cams}
    test = [by_name[n[:-4]] for n in names if n[:-4] in by_name]
    if len(test) != len(names):
        print(f"[{scene}] only {len(test)}/{len(names)} test views found in cameras.json")
    names = [c["img_name"] + ".png" for c in test]
    test_json = os.path.join(model, "test_cameras.json")
    json.dump(test, open(test_json, "w"), indent=1)

    W, H = Image.open(os.path.join(evald, "renders", names[0])).size
    cfg = open(os.path.join(model, "cfg_args")).read()
    white = "white_background=True" in cfg
    bg = (1.0, 1.0, 1.0) if white else (0.0, 0.0, 0.0)

    out_dir = os.path.join(args.data, "ours", scene)
    cmd = [args.renderer, ply, "--cameras", test_json, "--width", str(W), "--height", str(H),
           "--out", out_dir, "--bg", ",".join(str(v) for v in bg)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout, proc.stderr)
        raise RuntimeError(f"renderer failed on {scene}")
    times = [float(m) for m in re.findall(r"([\d.]+) ms", proc.stdout)]
    gaussians = int(re.search(r"\((\d+) Gaussians", proc.stdout).group(1))

    rows = []
    for n in names:
        ours = load(os.path.join(out_dir, n))
        off = load(os.path.join(evald, "renders", n))
        gt = load(os.path.join(evald, "gt", n))
        rows.append({"image": n, "ours_gt": psnr(ours, gt), "official_gt": psnr(off, gt), "ours_official": psnr(ours, off)})
    mid = names[len(names) // 2]
    shift = global_shift(load(os.path.join(out_dir, mid)), load(os.path.join(evald, "renders", mid)))

    result = {
        "scene": scene, "views": len(names), "width": W, "height": H, "gaussians": gaussians,
        "white_background": white,
        # Skip the first render (includes buffer allocation) when averaging.
        "ms_per_view": float(np.mean(times[1:] if len(times) > 1 else times)),
        "psnr_ours_gt": float(np.mean([r["ours_gt"] for r in rows])),
        "psnr_official_gt": float(np.mean([r["official_gt"] for r in rows])),
        "psnr_ours_official": float(np.mean([r["ours_official"] for r in rows])),
        "shift_px": shift,
        "per_view": rows,
    }
    print(f"[{scene}] {gaussians:,} Gaussians, {len(names)} views at {W}x{H}, {result['ms_per_view']:.1f} ms/view | "
          f"PSNR vs GT: ours {result['psnr_ours_gt']:.2f}, official {result['psnr_official_gt']:.2f} | "
          f"ours vs official {result['psnr_ours_official']:.2f} dB, shift ({shift[0]:.2f}, {shift[1]:.2f}) px", flush=True)

    report_rows = [labelled_row(
        [Image.open(os.path.join(out_dir, mid)), Image.open(os.path.join(evald, "renders", mid)),
         Image.open(os.path.join(evald, "gt", mid))],
        ["Ours (Vulkan)", "Official 3DGS render", "Ground truth"], 560)]

    if args.crop:
        tiles_x, tiles_y = (W + 15) // 16, (H + 15) // 16
        tx0, ty0 = tiles_x // 2 - 4, tiles_y // 2 - 4
        crop, ours_c, ref_c, diff = reference_crop.check_window(
            model, mid[:-4], os.path.join(out_dir, mid), tx0, ty0, tx0 + 8, ty0 + 8, None, bg)
        result["crop"] = crop
        diff_img = Image.fromarray(np.clip(diff * 20, 0, 255).astype(np.uint8))
        report_rows.append(labelled_row(
            [Image.fromarray(ours_c), Image.fromarray(ref_c), diff_img],
            ["Ours, centre 128x128", "NumPy reference of official math",
             f"|diff| x20 (max {crop['max_abs']}/255)"], 560))

    title = (f"{scene}  -  {gaussians:,} Gaussians, {W}x{H}, view {mid[:-4]}  -  PSNR vs GT: ours "
             f"{result['psnr_ours_gt']:.2f} dB, official {result['psnr_official_gt']:.2f} dB")
    os.makedirs(os.path.join(args.data, "report"), exist_ok=True)
    stack(report_rows, title).save(os.path.join(args.data, "report", f"{scene}.png"))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--data", default="data")
    ap.add_argument("--renderer", default=os.path.join("build", "splat_lab.exe" if os.name == "nt" else "splat_lab"))
    ap.add_argument("--no-crop", dest="crop", action="store_false")
    args = ap.parse_args()

    scenes = args.scenes or sorted(os.listdir(os.path.join(args.data, "models")))
    summary_path = os.path.join(args.data, "report", "summary.json")
    summary = {}
    if os.path.exists(summary_path):
        summary = {r["scene"]: r for r in json.load(open(summary_path))}
    for scene in scenes:
        r = evaluate(scene, args)
        if r:
            summary[scene] = r
            json.dump(list(summary.values()), open(summary_path, "w"), indent=1)

    lines = ["| Scene | Gaussians | Resolution | ms/view | PSNR ours vs GT | PSNR official vs GT | PSNR ours vs official | Shift (px) | Crop max err |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(summary.values(), key=lambda r: r["scene"]):
        crop = f"{r['crop']['max_abs']}/255" if "crop" in r else "-"
        lines.append(f"| {r['scene']} | {r['gaussians']:,} | {r['width']}x{r['height']} | {r['ms_per_view']:.1f} | "
                     f"{r['psnr_ours_gt']:.2f} | {r['psnr_official_gt']:.2f} | {r['psnr_ours_official']:.2f} | "
                     f"{max(abs(r['shift_px'][0]), abs(r['shift_px'][1])):.2f} | {crop} |")
    open(os.path.join(args.data, "report", "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
