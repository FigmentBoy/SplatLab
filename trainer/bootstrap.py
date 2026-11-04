"""Prepares the Python environment, then runs train.py with the same arguments.

    python trainer/bootstrap.py <train.py arguments>

1. Uses trainer/.venv (creating it if needed), unless SPLAT_NO_VENV=1 or this
   interpreter already is a virtual environment / conda environment.
2. pip-installs whatever is missing: PyTorch with CUDA (from the PyTorch
   index), pycolmap, OpenCV, NumPy.
3. Runs train.py in that environment.

The viewer always starts training through this script. It only uses the
standard library, so it runs on any Python 3.9+.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv"
TORCH_INDEX = os.environ.get("SPLAT_TORCH_INDEX", "https://download.pytorch.org/whl/cu128")
# import name -> pip requirement
PACKAGES = {
    "numpy": "numpy",
    "cv2": "opencv-python-headless",
    "pycolmap": "pycolmap",
}


def emit(type_: str, **fields):
    if "--progress-json" in sys.argv:
        print("@@" + json.dumps({"type": type_, **fields}), flush=True)


def log(msg: str):
    print(msg, flush=True)


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def in_isolated_env() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix) or "CONDA_PREFIX" in os.environ


def run(cmd) -> int:
    log("$ " + " ".join(str(c) for c in cmd))
    return subprocess.call([str(c) for c in cmd])


def missing_packages(python: Path) -> list[str]:
    """Asks `python` which packages it lacks (torch counts as missing without CUDA)."""
    probe = (
        "import importlib.util, json\n"
        f"names = {list(PACKAGES)!r}\n"
        "missing = [n for n in names if importlib.util.find_spec(n) is None]\n"
        "try:\n"
        "    import torch\n"
        "    if not torch.cuda.is_available(): missing.append('torch')\n"
        "except Exception:\n"
        "    missing.append('torch')\n"
        "print(json.dumps(missing))\n"
    )
    out = subprocess.run([str(python), "-c", probe], capture_output=True, text=True)
    if out.returncode != 0:
        return list(PACKAGES) + ["torch"]
    return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> int:
    emit("hello", python=sys.executable, version=sys.version.split()[0])
    if sys.version_info < (3, 9):
        emit("error", message=f"Python 3.9 or newer is required (found {sys.version.split()[0]})")
        return 1

    python = Path(sys.executable)
    if os.environ.get("SPLAT_NO_VENV") != "1" and not in_isolated_env():
        if not venv_python().exists():
            emit("stage", stage="setup", label="Creating the Python environment (trainer/.venv)", progress=-1.0)
            if run([sys.executable, "-m", "venv", VENV]) != 0:
                emit("error", message="Creating the virtual environment failed")
                return 1
        python = venv_python()

    missing = missing_packages(python)
    if missing:
        emit("stage", stage="setup", label="Installing Python packages: " + ", ".join(missing), progress=-1.0)
        run([python, "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
        if "torch" in missing:
            if run([python, "-m", "pip", "install", "--upgrade", "torch", "--index-url", TORCH_INDEX]) != 0:
                emit("error", message="Installing PyTorch failed (see the log)")
                return 1
        others = [PACKAGES[m] for m in missing if m in PACKAGES]
        if others and run([python, "-m", "pip", "install", *others]) != 0:
            emit("error", message="Installing " + ", ".join(others) + " failed (see the log)")
            return 1
        still = missing_packages(python)
        if still:
            hint = " (PyTorch cannot use the GPU: an NVIDIA GPU and driver are required)" if "torch" in still else ""
            emit("error", message="Missing Python packages after install: " + ", ".join(still) + hint)
            return 1

    return subprocess.call([str(python), "-u", str(HERE / "train.py"), *sys.argv[1:]])


if __name__ == "__main__":
    sys.exit(main())
