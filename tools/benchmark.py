"""Benchmarks the renderer on the official test views of every scene under
data/models (as set up by tools/eval_reference.py), and checks the output.

For each scene it renders the test views at the official resolution with
--profile (per-stage GPU timings) and --check-binning (tight vs reference tile
binning, compared bit for bit), and, with --baseline <dir>, compares every
PNG with <dir>/<scene>/<view>.png pixel for pixel.

Usage: python tools/benchmark.py [--renderer build/splat_lab.exe]
           [--baseline data/ours] [--out <dir>] [--json results.json] [scene ...]
"""
import argparse
import json
import os
import re
import subprocess

import numpy as np
from PIL import Image


def scene_setup(data, scene):
    model = os.path.join(data, "models", scene)
    evald = os.path.join(data, "eval", scene, "test", "ours_30000", "renders")
    test_json = os.path.join(model, "test_cameras.json")
    ply = os.path.join(model, "point_cloud", "iteration_30000", "point_cloud.ply")
    if not (os.path.exists(test_json) and os.path.exists(ply) and os.path.isdir(evald)):
        return None
    first = sorted(n for n in os.listdir(evald) if n.endswith(".png"))[0]
    W, H = Image.open(os.path.join(evald, first)).size
    white = "white_background=True" in open(os.path.join(model, "cfg_args")).read()
    return ply, test_json, W, H, ("1,1,1" if white else "0,0,0")


def run_scene(args, scene):
    setup = scene_setup(args.data, scene)
    if setup is None:
        print(f"[{scene}] not set up, skipped")
        return None
    ply, test_json, W, H, bg = setup
    out_dir = os.path.join(args.out, scene)
    cmd = [args.renderer, ply, "--cameras", test_json, "--width", str(W), "--height", str(H), "--bg", bg,
           "--out", out_dir, "--profile"]
    if not args.no_check:
        cmd.append("--check-binning")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout, proc.stderr)
        raise RuntimeError(f"renderer failed on {scene}")
    out = proc.stdout

    stages = {m[0].strip(): float(m[1]) for m in re.findall(r"^  (\S[^\n]*?)\s+([\d.]+) ms$", out, re.M)}
    views = re.findall(r"^(.*?\.png)\s+\d+x\d+\s+[\d.]+ ms\s+\((\d+) Gaussians, (\d+) visible, (\d+) tile", out, re.M)
    checks = [(int(a), int(b), int(c)) for a, b, c in
              re.findall(r"binning check: (\d+) of \d+ floats differ; (\d+) tight vs (\d+) reference", out)]
    result = {
        "scene": scene,
        "resolution": f"{W}x{H}",
        "views": len(views),
        "gaussians": int(views[0][1]) if views else 0,
        "visible": float(np.mean([int(v[2]) for v in views])) if views else 0.0,
        "instances": float(np.mean([int(v[3]) for v in views])) if views else 0.0,
        "stages": stages,
    }
    if checks:
        result["binning_differing_floats"] = sum(c[0] for c in checks)
        result["reference_instances"] = float(np.mean([c[2] for c in checks]))

    if args.baseline:
        differing_pixels = 0
        max_diff = 0
        compared = 0
        for path, *_ in views:
            name = os.path.basename(path.replace("\\", "/"))
            base = os.path.join(args.baseline, scene, name)
            if not os.path.exists(base):
                continue
            a = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
            b = np.asarray(Image.open(base).convert("RGB"), dtype=np.int16)
            d = np.abs(a - b).max(axis=2)
            differing_pixels += int((d > 0).sum())
            max_diff = max(max_diff, int(d.max()))
            compared += 1
        result["baseline_views"] = compared
        result["baseline_differing_pixels"] = differing_pixels
        result["baseline_max_diff"] = max_diff

    line = (f"[{scene}] {result['views']} views {W}x{H}: GPU {stages.get('GPU total', 0):.2f} ms, "
            f"wall {stages.get('wall', 0):.2f} ms, {result['instances']:,.0f} instances")
    if checks:
        line += (f" ({100 * result['instances'] / max(result['reference_instances'], 1):.1f}% of reference), "
                 f"binning diffs {result['binning_differing_floats']}")
    if args.baseline:
        line += (f", vs baseline: {result['baseline_differing_pixels']} px differ "
                 f"(max {result['baseline_max_diff']}) over {result['baseline_views']} views")
    print(line, flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--data", default="data")
    ap.add_argument("--renderer", default=os.path.join("build", "splat_lab.exe" if os.name == "nt" else "splat_lab"))
    ap.add_argument("--baseline", help="directory of earlier renders to compare against (<dir>/<scene>/*.png)")
    ap.add_argument("--out", default=os.path.join("data", "bench"))
    ap.add_argument("--json", help="write the results to this file")
    ap.add_argument("--no-check", action="store_true", help="skip the tight vs reference binning check")
    args = ap.parse_args()

    scenes = args.scenes or sorted(os.listdir(os.path.join(args.data, "models")))
    results = [r for r in (run_scene(args, s) for s in scenes) if r]
    if args.json:
        json.dump(results, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
