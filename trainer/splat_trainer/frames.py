"""Frame extraction: picks evenly spaced, sharp frames from a video.

The video is split into `count` equal windows; a few candidate frames per
window are decoded and the sharpest (variance of the Laplacian) is kept, which
avoids motion-blurred frames while preserving viewpoint overlap. Frames are
downscaled so their longer side is at most `max_size` and saved as
00000.jpg, 00001.jpg, ... (the order sequential matching relies on).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale = 480.0 / max(gray.shape)
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def _resize(frame: np.ndarray, max_size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def _write(path: Path, frame: np.ndarray):
    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise IOError(f"cannot write {path}")


def extract_frames(source: Path, out_dir: Path, count: int, max_size: int, progress) -> int:
    """Fills out_dir with frames from a video file or an image folder; returns the frame count."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    if source.is_dir():
        images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not images:
            raise ValueError(f"no images in {source}")
        step = max(1.0, len(images) / count)
        picked = [images[int(i * step)] for i in range(min(count, len(images)))]
        for k, p in enumerate(picked):
            frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if frame is None:
                raise IOError(f"cannot read {p}")
            _write(out_dir / f"{k:05d}.jpg", _resize(frame, max_size))
            progress.stage("frames", "Loading images", (k + 1) / len(picked))
        return len(picked)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise IOError(f"cannot open video {source}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if total <= 0:
        # Some containers do not report a frame count: count by grabbing.
        while cap.grab():
            total += 1
        cap.release()
        cap = cv2.VideoCapture(str(source))
    if total <= 0:
        raise IOError(f"no frames in {source}")
    progress.log(f"Video: {total} frames at {fps:.2f} fps, "
                 f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    count = min(count, total)
    bounds = np.linspace(0, total, count + 1)
    candidates_per_window = 5
    window_of = {}
    for w in range(count):
        lo, hi = int(bounds[w]), max(int(bounds[w]) + 1, int(bounds[w + 1]))
        for idx in np.unique(np.linspace(lo, hi - 1, min(candidates_per_window, hi - lo)).astype(int)):
            window_of[int(idx)] = w

    last_of_window = {}
    for idx, w in window_of.items():
        last_of_window[w] = max(idx, last_of_window.get(w, -1))

    best = {}  # window -> (sharpness, frame)
    written = 0
    last_needed = max(window_of)
    for i in range(last_needed + 1):
        w = window_of.get(i)
        if w is None:
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        score = _sharpness(frame)
        if w not in best or score > best[w][0]:
            best[w] = (score, frame)
        # A window is complete once its last candidate has been read.
        if i == last_of_window[w]:
            _write(out_dir / f"{written:05d}.jpg", _resize(best.pop(w)[1], max_size))
            written += 1
            progress.stage("frames", "Extracting frames", written / count)
    cap.release()
    for w in sorted(best):  # windows cut short by a truncated video
        _write(out_dir / f"{written:05d}.jpg", _resize(best[w][1], max_size))
        written += 1
    if written < 3:
        raise ValueError(f"only {written} frames could be read from {source}")
    return written
