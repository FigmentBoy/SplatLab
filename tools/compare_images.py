"""Compares two folders of images with matching file names (e.g. our renders vs
reference renders) and reports per-image and mean PSNR plus pixel statistics.
Optionally writes amplified absolute-difference images.

Usage: python tools/compare_images.py <ours_dir> <reference_dir> [--diff-dir DIR] [--amplify K]
"""
import argparse
import json
import os

import numpy as np
from PIL import Image


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ours")
    ap.add_argument("reference")
    ap.add_argument("--diff-dir")
    ap.add_argument("--amplify", type=float, default=10.0)
    ap.add_argument("--json", help="write per-image results to this file")
    args = ap.parse_args()

    names = sorted(n for n in os.listdir(args.reference) if n.lower().endswith(".png"))
    rows = []
    for name in names:
        ours_path = os.path.join(args.ours, name)
        if not os.path.exists(ours_path):
            print(f"missing: {name}")
            continue
        a, b = load(ours_path), load(os.path.join(args.reference, name))
        if a.shape != b.shape:
            print(f"size mismatch: {name} {a.shape} vs {b.shape}")
            continue
        diff = np.abs(a - b) * 255.0
        row = {
            "image": name,
            "psnr": psnr(a, b),
            "mean_abs": float(diff.mean()),
            "max_abs": float(diff.max()),
            "pct_gt_1": float((diff.max(axis=2) > 1.0 + 1e-9).mean() * 100),
            "pct_gt_8": float((diff.max(axis=2) > 8.0).mean() * 100),
        }
        rows.append(row)
        print(f"{name:24s} PSNR {row['psnr']:7.2f} dB  mean|d| {row['mean_abs']:6.3f}  max|d| {row['max_abs']:5.0f}"
              f"  >1: {row['pct_gt_1']:6.3f}%  >8: {row['pct_gt_8']:6.3f}%")
        if args.diff_dir:
            os.makedirs(args.diff_dir, exist_ok=True)
            out = np.clip(diff * args.amplify, 0, 255).astype(np.uint8)
            Image.fromarray(out).save(os.path.join(args.diff_dir, name))

    if rows:
        finite = [r["psnr"] for r in rows if np.isfinite(r["psnr"])]
        print(f"\n{len(rows)} images  mean PSNR {np.mean(finite) if finite else float('inf'):.2f} dB  "
              f"mean|d| {np.mean([r['mean_abs'] for r in rows]):.3f}  worst max|d| {max(r['max_abs'] for r in rows):.0f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
