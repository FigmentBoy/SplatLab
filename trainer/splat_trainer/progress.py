"""Progress reporting.

Human-readable log lines go to stdout. With --progress-json, machine-readable
events are printed too, one per line, as "@@" followed by a JSON object; the
viewer (src/app/TrainingSession.cpp) parses them to show progress and load
intermediate results. Event types:

  stage    {stage, label, progress}   progress in 0..1, or -1 if unknown
  ply      {path, kind, label}        kind: "points" (SfM) or "gaussians"
  cameras  {path}                     cameras.json of the reconstruction
  train    {iteration, total, phase, loss, psnr, gaussians, its}
  eval     {iteration, psnr, ssim, views}
  preview  {path, iteration, psnr}    render-vs-photo image (--preview-every)
  done     {output}
  error    {message}
"""
from __future__ import annotations

import json
import sys
import threading
import time


class Progress:
    def __init__(self, json_events: bool = False):
        self.json_events = json_events
        self._lock = threading.Lock()
        self._start = time.time()

    def _write(self, line: str):
        with self._lock:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def emit(self, type_: str, **fields):
        if self.json_events:
            self._write("@@" + json.dumps({"type": type_, **fields}))

    def log(self, message: str):
        elapsed = time.time() - self._start
        self._write(f"[{int(elapsed) // 60:3d}:{int(elapsed) % 60:02d}] {message}")

    def stage(self, stage: str, label: str, progress: float = -1.0):
        self.emit("stage", stage=stage, label=label, progress=float(progress))

    def stage_start(self, stage: str, label: str):
        self.log(f"== {label}")
        self.stage(stage, label, -1.0)
