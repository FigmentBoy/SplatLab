"""Compares a model trained by trainer/ with the official pretrained model.

Both are rendered with splat_lab at the official test views and resolution
and scored against the official ground-truth test images (PSNR, SSIM). The
official evaluation renders are scored too, for reference.

  trainer/.venv/Scripts/python tools/eval_trained.py data/ours_trained/bicycle --scene bicycle

Writes <out>/summary.md, per-view JSON, and a comparison figure.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "trainer"))
from splat_trainer import losses  # noqa: E402


def latest_ply(model: Path) -> Path:
    plys = sorted(model.glob("point_cloud/iteration_*/point_cloud.ply"),
                  key=lambda p: int(p.parent.name.split("_")[1]))
    return plys[-1]


def load(path: Path) -> torch.Tensor:
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).cuda()


def render(renderer: Path, ply: Path, cameras: Path, w: int, h: int, out: Path) -> float:
    out.mkdir(parents=True, exist_ok=True)
    p = subprocess.run([str(renderer), str(ply), "--cameras", str(cameras), "--width", str(w), "--height", str(h),
                        "--out", str(out)], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stdout + p.stderr)
    times = [float(t) for t in re.findall(r"([\d.]+) ms", p.stdout)]
    return float(np.median(times[1:] if len(times) > 1 else times))


def count_gaussians(ply: Path) -> int:
    with open(ply, "rb") as f:
        for line in f:
            if line.startswith(b"element vertex"):
                return int(line.split()[-1])
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path, help="model folder written by trainer/train.py")
    ap.add_argument("--scene", default="bicycle")
    ap.add_argument("--data", type=Path, default=REPO / "data")
    ap.add_argument("--renderer", type=Path, default=REPO / "build" / ("splat_lab.exe" if sys.platform == "win32"
                                                                       else "splat_lab"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--figure-views", type=int, default=3)
    args = ap.parse_args()

    evald = args.data / "eval" / args.scene / "test" / "ours_30000"
    gt_dir, official_dir = evald / "gt", evald / "renders"
    ref_model = args.data / "models" / args.scene
    out = args.out or args.model / "eval"
    names = sorted(p.name for p in gt_dir.glob("*.png"))
    w, h = Image.open(gt_dir / names[0]).size

    # Our test cameras, keyed by image name (our scene is normalized
    # differently, so the official cameras do not apply to our model).
    ours_cams = {c["img_name"]: c for c in json.load(open(args.model / "cameras.json"))}
    ref_cams = {c["img_name"]: c for c in json.load(open(ref_model / "cameras.json"))}
    missing = [n for n in names if n[:-4] not in ours_cams]
    if missing:
        raise SystemExit(f"test views missing from {args.model / 'cameras.json'}: {missing[:3]}")
    out.mkdir(parents=True, exist_ok=True)
    json.dump([ours_cams[n[:-4]] for n in names], open(out / "ours_test_cameras.json", "w"), indent=1)
    json.dump([ref_cams[n[:-4]] for n in names], open(out / "ref_test_cameras.json", "w"), indent=1)

    ours_ply, ref_ply = latest_ply(args.model), latest_ply(ref_model)
    ms_ours = render(args.renderer, ours_ply, out / "ours_test_cameras.json", w, h, out / "ours")
    ms_ref = render(args.renderer, ref_ply, out / "ref_test_cameras.json", w, h, out / "pretrained")

    rows = []
    with torch.no_grad():
        for n in names:
            gt = load(gt_dir / n)
            row = {"image": n}
            for key, d in (("ours", out / "ours"), ("pretrained", out / "pretrained"), ("official", official_dir)):
                img = load(d / n)
                row[f"psnr_{key}"] = losses.psnr(img, gt).item()
                row[f"ssim_{key}"] = losses.ssim(img, gt).item()
            rows.append(row)
    json.dump(rows, open(out / "per_view.json", "w"), indent=1)
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "image"}

    log = args.model.parent / f"{args.model.name}_train.log"
    train_info = ""
    if log.exists():
        m = re.search(r"Training finished: (\d+) iterations in ([\d.]+) min \(([\d.]+) it/s\)", log.read_text(errors="ignore"))
        if m:
            train_info = f"{m.group(1)} iterations in {m.group(2)} min ({m.group(3)} it/s)"

    lines = [
        f"# {args.scene}: trained by trainer/ vs official pretrained model",
        "",
        f"{len(names)} official test views at {w}x{h}, scored against the official ground truth.",
        "",
        "| Model | Gaussians | PSNR (dB) | SSIM | Render (ms/view) |",
        "|---|---|---|---|---|",
        f"| Ours ({ours_ply.parent.name}) | {count_gaussians(ours_ply):,} | {mean['psnr_ours']:.2f} | "
        f"{mean['ssim_ours']:.4f} | {ms_ours:.1f} |",
        f"| Official pretrained, our renderer ({ref_ply.parent.name}) | {count_gaussians(ref_ply):,} | "
        f"{mean['psnr_pretrained']:.2f} | {mean['ssim_pretrained']:.4f} | {ms_ref:.1f} |",
        f"| Official evaluation renders (paper) | | {mean['psnr_official']:.2f} | {mean['ssim_official']:.4f} | |",
        "",
    ]
    if train_info:
        lines += [f"Training: {train_info}.", ""]
    lines += ["| View | PSNR ours | PSNR pretrained | SSIM ours | SSIM pretrained |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['image']} | {r['psnr_ours']:.2f} | {r['psnr_pretrained']:.2f} | {r['ssim_ours']:.4f} | "
                     f"{r['ssim_pretrained']:.4f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:10]))

    # Figure: the views with the largest, median and smallest gap.
    order = sorted(rows, key=lambda r: r["psnr_pretrained"] - r["psnr_ours"])
    picks = [order[0], order[len(order) // 2], order[-1]][: args.figure_views]
    tw, th = w // 2, h // 2
    fig = Image.new("RGB", (3 * tw, len(picks) * (th + 30)), (20, 20, 20))
    draw = ImageDraw.Draw(fig)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for i, r in enumerate(picks):
        y = i * (th + 30)
        for j, (label, path) in enumerate((("Ground truth", gt_dir), (f"Ours {r['psnr_ours']:.2f} dB", out / "ours"),
                                           (f"Pretrained {r['psnr_pretrained']:.2f} dB", out / "pretrained"))):
            fig.paste(Image.open(path / r["image"]).convert("RGB").resize((tw, th)), (j * tw, y + 30))
            draw.text((j * tw + 8, y + 6), f"{label}  ({r['image'][:-4]})" if j == 0 else label,
                      fill=(255, 255, 255), font=font)
    fig.save(out / "comparison.jpg", quality=90)
    print(f"Wrote {out / 'summary.md'} and {out / 'comparison.jpg'}")


if __name__ == "__main__":
    main()
