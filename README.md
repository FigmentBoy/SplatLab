# Vulkan Gaussian Splat Renderer

3D Gaussian Splatting renderer and trainer (CS535 final project). Loads PLY
splats in the reference 3DGS layout, renders them from arbitrary camera poses
with Vulkan compute shaders, and writes images or shows them in an interactive
window. The renderer also has a backward pass, which the Python trainer in
`trainer/` uses (through PyTorch) to turn a video into a splat. See
[Trainer](#trainer).

## Building

```sh
git submodule update --init --recursive
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Requirements: CMake 3.20+, a C++17 compiler and a GPU driver with Vulkan 1.2 and
`bufferDeviceAddress`. The Vulkan SDK is **not** required. Volk loads the Vulkan
runtime when the program starts, and shaders are compiled with the SDK's
`glslangValidator` if `VULKAN_SDK` is set, otherwise with glslang built from
`externals/glslang`. The SPIR-V is embedded in the executable.

## Usage

```sh
# Interactive viewer (a scene can also be opened from File > Load PLY)
build/splat_lab [scene.ply] [--cameras cameras.json]

# Offline: one PNG per camera in cameras.json
build/splat_lab scene.ply --cameras cameras.json --out renders/ [--aux]

# Offline: a single arbitrary pose
build/splat_lab scene.ply --out renders/ --eye 0,-1,-4 --target 0,0,0 --fovy 50
```

`cameras.json` is the file the reference implementation writes next to
`point_cloud/` after training. Run with `--help` for all options.

Viewer controls: left-click the scene to capture the mouse, and press Esc to
release it. While the mouse is captured, moving it looks around, WASD moves in
the horizontal plane, Space/Shift move up/down, Ctrl moves faster and the
scroll wheel changes speed. "Up" is the mean up vector of the loaded dataset
cameras (or `--up` without them). `]`/`[` cycle through dataset cameras, R
resets the view, P saves a screenshot, and C prints the current pose as a
`cameras.json` entry.

File > Load Model lists every trained model under `data/models/` (searched from
the working directory upwards) and loads its latest `point_cloud.ply` together
with its `cameras.json`, starting at the first camera. File > Load PLY and Load
Cameras open single files. Files load in the background behind a progress
window. View > Cameras opens a list of the dataset cameras with previous/next
buttons, and it opens on its own once a scene's cameras are loaded. File > Exit
quits.

## Conventions

- Splat PLY: `x y z nx ny nz f_dc_0..2 f_rest_* opacity scale_0..2 rot_0..3`.
  Scales are log-space, opacity is a logit, and rotations are (w, x, y, z)
  quaternions. `f_rest` is channel-major.
- Camera space for rendering is COLMAP/3DGS: x right, y down, z forward. Scene
  graph nodes (`Node`, `Camera`, `GaussianCloud`) use the OpenGL convention
  (-Z forward, +Y up), and `Camera::viewMatrixCV()` converts between the two.
- Projection, culling, the EWA covariance (including the 0.3 low-pass and the
  1.3x frustum clamp), the tile size (16) and the compositing thresholds
  follow the reference CUDA rasterizer.

## Pipeline (all Vulkan compute, `shaders/`)

1. `preprocess`: projection, 2D conic, SH colour, radius, tiles touched
2. `scan_block`/`scan_add` + `compact`: the visible Gaussians as (depth, index) pairs
3. `radix_histogram`/`radix_scatter`: stable sort of the visible Gaussians by depth
4. `gather_tiles` + scan: tile counts in depth order, giving instance offsets
5. `duplicate`: one (tile, Gaussian) instance per overlap, emitted in depth order
6. `radix_histogram`/`radix_scatter`: stable sort of the instances by tile id
7. `tile_ranges`: per-tile ranges in the sorted list
8. `render`: front-to-back alpha compositing per 16x16 tile
9. `pack`: float CHW image to RGBA8

The whole frame is one submission: counts stay on the GPU, and the sort
kernels read them from there. Per-pixel final transmittance, the last
contributor and the sorted per-tile lists are kept on the GPU for the backward
pass.

Training uses `preprocess_train` instead of `preprocess`. It is the same
shader body (`preprocess_body.glsl`), but it reads the raw parameters (log
scales, unnormalized quaternions, opacity logits, SH split into DC and the
rest) and activates them itself. The backward pass is two more dispatches in
one submission:

10. `render_backward`: per pixel, the tile's list is walked back to front from
    the last contributor, reconstructing T from the final transmittance (as
    in the reference `backward.cu`). This gives dL/d(2D mean, conic, opacity,
    colour). All pixels of a workgroup handle the same Gaussian at once, so the
    gradients are summed with subgroup adds and one lane per subgroup does the
    float atomics. The walk also starts at the tile's largest last-contributor
    index instead of the end of the list.
11. `preprocess_backward`: dL/d(raw parameters) through the conic inverse, the
    EWA projection (with the frustum clamp), the covariance R S S^T R^T, the
    SH evaluation (including its view direction) and the activations.

The backward pass needs `VK_EXT_shader_atomic_float` (buffer float add) and
subgroup votes.

## Performance

The reference rasterizer ([Kerbl et al. 2023]) sorts one 64-bit `tile | depth`
key per Gaussian/tile overlap and bins each Gaussian into every tile of its
3-sigma square (the EWA footprint of [Zwicker et al. 2001]). This renderer
keeps that algorithm's output exactly but does less work:

- **Depth first, then tile.** The visible Gaussians are sorted by depth once
  (4 radix passes over Gaussians), duplicated in that order, and the instances
  are then sorted stably by tile id only (2 passes over 16-bit keys). Because
  both sorts are stable, each tile's list comes out in the same order as with
  the reference's single key sort, ties included.
- **Tight tile binning.** Compositing skips a Gaussian wherever
  `alpha = opacity * exp(-q/2) < 1/255`. So a Gaussian only matters inside the
  ellipse `q <= 2 ln(255 opacity)`, which for faint or elongated Gaussians is
  far smaller than the 3-sigma square. Each Gaussian is binned only into the
  tiles of the square that this ellipse touches. The minimum of `q` over a tile
  is found exactly, on the tile's edges, and a margin covers float rounding.
  The dropped pairs could never contribute, so colours and transmittance are
  bit-identical. Only the last-contributor index, which counts list entries, is
  relative to the shorter lists. `--reference-binning` turns this off, and
  `--check-binning` renders every view both ways and compares the float
  buffers.
- **Subgroup radix sort and scan.** Each workgroup ranks 4096 keys with
  subgroup ballots (warp-level multisplit) instead of 1-bit split passes. The
  prefix sums use subgroup scans.
- Smaller things: SH coefficients are read as `vec4`s, only for the bands in
  use, and Gaussian colours are cached in shared memory during compositing.

`python tools/benchmark.py [--baseline <dir>]` measures every scene at the
official test resolution. It reports warm GPU timings (median of 7 renders
after a 0.3 s warm-up, since laptop GPUs idle at low clocks between frames),
runs the binning check, and compares the PNGs with an earlier set of renders.
On an RTX 3080 Ti Laptop GPU:

| Scene | Before (ms) | Now (ms) | Instances vs reference binning | Pixels changed |
|---|---|---|---|---|
| bicycle | 65.1 | 11.9 | 33.8% | 0 |
| bonsai | 79.5 | 5.1 | 33.9% | 0 |
| counter | 87.7 | 6.1 | 31.2% | 0 |
| drjohnson | 77.3 | 5.9 | 23.8% | 0 |
| flowers | 69.5 | 6.6 | 45.5% | 0 |
| garden | 85.9 | 13.1 | 46.7% | 0 |
| kitchen | 81.9 | 9.5 | 40.6% | 0 |
| playroom | 60.9 | 4.7 | 27.0% | 0 |
| room | 77.0 | 5.5 | 25.5% | 0 |
| stump | 70.3 | 7.9 | 46.2% | 0 |
| train | 60.6 | 4.5 | 32.9% | 0 |
| treehill | 69.8 | 7.0 | 39.8% | 0 |
| truck | 59.7 | 6.0 | 37.7% | 0 |

"Before" is the earlier version's `render()` time from the validation table
below, which was not warmed up. Profiled after one warm-up render, the earlier
version took 43.5 ms of GPU time on bicycle: 29.0 ms sorting and 8.7 ms
compositing. On bicycle, the new
sort alone (with `--reference-binning`) gives 23.4 ms, and tight binning brings
that to 11.6 ms, mostly by cutting compositing from 12.7 ms to 4.3 ms. Every
test view of every scene renders pixel-identical to the earlier version, and
the binning check finds 0 differing floats.

[Kerbl et al. 2023]: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
[Zwicker et al. 2001]: https://doi.org/10.1109/VISUAL.2001.964490

## Trainer

`trainer/` turns a video into a splat, following the final deliverables of the
proposal:

1. **Frames** (`frames.py`, OpenCV). The video is split into N windows
   (default 200), and the sharpest of 5 candidate frames in each window is
   kept, resized to at most 1600 px.
2. **SfM** (`sfm.py`, pycolmap). SIFT, sequential matching (with an
   exhaustive fallback when under 60% of the frames register), incremental
   mapping, then undistortion to a pinhole camera. The scene is normalized so
   the median SfM point depth is 4.
3. **Initialization** (`gaussians.py`). One Gaussian per SfM point: colour
   from the point, scale from the mean distance to its 3 nearest neighbours,
   opacity 0.1.
4. **Optimization** (`training.py`). This uses the reference
   hyperparameters: (1-0.2) L1 + 0.2 (1-SSIM), Adam, an exponentially
   decaying position learning rate, one more SH band every 1000 iterations,
   and densification (clone, split, prune, opacity reset) up to iteration
   15000. The first 300 iterations optimize only colour and opacity. Each
   iteration runs the Vulkan forward pass, then the loss in PyTorch, then the
   Vulkan backward pass, then Adam.
5. **Export** (`export.py`). The output has the reference layout:
   `point_cloud/iteration_<N>/point_cloud.ply` (7000 and final),
   `cameras.json`, `test_cameras.json`, `cfg_args` and `input.ply`.

`cameras.json` lists every camera. `test_cameras.json` lists the held-out
views (every 8th frame), which are never trained on. The trainer reports PSNR
and SSIM on them, and `tools/eval_reference.py` / `benchmark.py` render them.
The COLMAP dataset stays in `<output>/source`, so re-running into the same
folder skips frame extraction and SfM (unless `--redo-sfm` is given).

**Vulkan and PyTorch share memory.** The `splat_vk` Python module
(`trainer/csrc`, pybind11) allocates the Gaussian parameters, their gradients
and the image buffers as exportable Vulkan memory
(`VK_KHR_external_memory_win32` or `_fd`). It imports that memory into CUDA
with the driver API (`cuImportExternalMemory`), which is loaded at run time,
so no CUDA toolkit is needed. PyTorch wraps the result as ordinary CUDA
tensors (`__cuda_array_interface__`). As a consequence:

- The `nn.Parameter`s are views of the buffers the shaders read.
- The backward pass writes straight into the tensors Adam reads as `.grad`.

Nothing is copied between the APIs. They synchronize on the host: torch's
stream is drained before each submission, and each submission is waited on.
The Vulkan device is the one whose UUID matches the CUDA device.

### Running it

From the viewer, open **Train > Train from Video...**, pick a video and press
**Start training**. Alternatively, run `splat_lab --train video.mp4
[--train-iterations N]`. The viewer shows the run as it goes:

- the SfM point cloud and the cameras as soon as they exist;
- the Gaussians at the start, at the end of each phase and every 1000
  iterations;
- the final model at the end.

The Train window shows the stage, a progress bar, the loss, PSNR, Gaussian
count, speed and a log. While training, loading other scenes is disabled, and
closing the window (X, Alt+F4 or File > Exit) asks for confirmation.
The viewer starts `trainer/bootstrap.py`, which creates `trainer/.venv` and
pip-installs anything missing before running `train.py`. That includes
PyTorch with CUDA, pycolmap and OpenCV. If Python itself is missing, the
viewer says so.

From the command line:

```sh
python trainer/bootstrap.py video.mp4 -o data/models/myscene      # sets up the venv, then trains
trainer/.venv/Scripts/python trainer/train.py video.mp4 -o out    # once set up
```

Useful options: `--iterations`, `--frames`, `--max-size`, `--no-eval`,
`--max-gaussians`, `--snapshot-every`, `--redo-sfm` and `--module-dir`.
`--preview-every N` saves `<output>/previews/iter_N.jpg` every N iterations:
the current render next to the real frame, with the iteration, Gaussian count
and PSNR. `--preview-camera i` picks the view; 0 is a held-out one. The
`splat_vk` module is found in `build*/` automatically. The CMake build
compiles it against `trainer/.venv` when that exists, otherwise against the
Python on the path, so build after creating the venv. The module must match
the Python version that runs the trainer.

### Cloud GPUs

The trainer is headless. On Linux, `-DSPLAT_BUILD_VIEWER=OFF` builds only the
core and the module, with no window system or GTK needed:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSPLAT_BUILD_VIEWER=OFF
cmake --build build --target splat_vk
python3 trainer/bootstrap.py video.mp4 -o out
```

You need an NVIDIA driver with Vulkan support (on Ubuntu, the
`libnvidia-gl-*` package provides the Vulkan ICD) plus `libvulkan1`. There is
also a Dockerfile; it needs the NVIDIA Container Toolkit on the host:

```sh
docker build -t splat-trainer .
docker run --rm --gpus all -v "$PWD/runs:/runs" splat-trainer /runs/video.mp4 -o /runs/out
```

Copy the output folder into `data/models/` to open it with
File > Load Scene.

**Free notebooks (Colab / Kaggle).** `trainer/colab.ipynb` runs the trainer
on a free T4 notebook GPU. It does the following:

1. Installs the Vulkan loader and checks that the runtime exposes NVIDIA's
   Vulkan driver; it stops if not.
2. Builds the module.
3. Takes an uploaded video, or a Kaggle dataset path.
4. Trains, showing a preview image every `PREVIEW_EVERY` iterations.
5. Zips the model for download.

Set `REPO_URL` (or upload a zip of the repository) in its first cell. The
previews, a timelapse GIF and the model zip are copied to a share folder as
they are produced: Google Drive on Colab (`SHARE_TO_DRIVE`), or the output
folder on Kaggle.

### Gradient validation

`python trainer/tests/test_gradients.py` checks the Vulkan passes against
`splat_trainer/reference.py`, a dense float64 PyTorch implementation of the
same rasterizer (the proposal's PyTorch fallback renderer). The test scene has
300 Gaussians at 72x52 with SH degrees 0, 1 and 3, a random dL/d(image) and
an off-centre principal point:

| Check | Result |
|---|---|
| Forward, max abs difference | 1.7e-6 |
| Gradients of all 6 parameter groups vs autograd, max relative error | 3.0e-6 (cosine 1.000000) |
| Central finite differences of the Vulkan forward, 18 largest gradients | 16 within 5% |

The two finite-difference outliers are discontinuities. In each, a Gaussian
crosses the 1/255 alpha cutoff or a tile edge when nudged by the step. Like
the reference, the analytic gradient ignores those events.

## Validation

`tools/make_test_scene.py` generates a synthetic scene and cameras. On it the
renderer matches an independent NumPy implementation of the reference forward
pass within 1/255 on every pixel, at up to 1.4M tile instances.

### Mip-NeRF 360 bicycle (official pretrained model)

`tools/remote_zip.py` pulls single scenes out of the official archives over
HTTP range requests, so the full 14 GB and 7 GB downloads aren't needed:

```sh
M=https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip
E=https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/evaluation/images.zip
python tools/remote_zip.py extract $M data/models bicycle/cameras.json bicycle/cfg_args bicycle/point_cloud/iteration_30000/
python tools/remote_zip.py extract $E data/eval bicycle/test/ours_30000/ bicycle/per_view.json
```

The scene has 6.13M Gaussians (SH degree 3). The 25 test views were rendered at
1237x822, the `images_4` training resolution (about 12 ms each now; see
[Performance](#performance)).

| Check | Result |
|---|---|
| Ours vs NumPy reference, two 128x128 windows (93k and 24k overlapping Gaussians) | max error 1/255, 0 pixels off by more than 1 |
| Ours vs ground-truth photos, mean PSNR | 25.19 dB (paper reports 25.25 dB) |
| Per-view PSNR vs the paper's `per_view.json` | mean -0.05 dB, all within +/-0.6 dB |
| Global image shift, ours vs official renders | 0.00 px |
| Ours vs official evaluation renders, mean PSNR | 29.3 dB |

The last row isn't a renderer error. The evaluation images were rendered from
the paper-era models, while `models.zip` holds models re-trained with the
release code (see the official README). The renders line up exactly and score
the same against ground truth, but the Gaussians themselves differ. An exact
pixel comparison against the official rasterizer would need it run on the
pretrained PLY, which requires CUDA and PyTorch.

`tools/compare_images.py <ours> <reference> --diff-dir <dir>` reports PSNR
and pixel statistics, and writes amplified difference images.

### All 13 official pretrained scenes

`python tools/download_data.py [scene ...]` fetches the models and evaluation
images for all 13 scenes (or the ones named) into `data/models/` and
`data/eval/`. It skips files that are already present, so re-running it
resumes an interrupted download.

`python tools/eval_reference.py` evaluates every scene under `data/models/`.
It renders the official test views at the official resolution, compares them
with the official renders and ground truth, and checks a 128x128 center crop
against a float32 NumPy implementation of the official forward pass
(`tools/reference_crop.py`). The script writes `data/report/<scene>.png` and
`data/report/summary.md`.

| Scene | Gaussians | Resolution | ms/view | PSNR ours vs GT | PSNR official vs GT | PSNR ours vs official | Shift (px) | Crop max err |
|---|---|---|---|---|---|---|---|---|
| bicycle | 6,131,954 | 1237x822 | 65.1 | 25.19 | 25.25 | 29.29 | 0.00 | 2/255 |
| bonsai | 1,244,819 | 1559x1039 | 79.5 | 32.05 | 31.98 | 36.47 | 0.00 | 1/255 |
| counter | 1,222,956 | 1558x1038 | 87.7 | 28.93 | 28.69 | 34.19 | 0.00 | 1/255 |
| drjohnson | 3,405,153 | 1332x876 | 77.3 | 28.97 | 28.77 | 32.28 | 0.01 | 1/255 |
| flowers | 3,636,448 | 1256x828 | 69.5 | 21.51 | 21.52 | 25.17 | 0.01 | 1/255 |
| garden | 5,834,784 | 1297x840 | 85.9 | 27.31 | 27.41 | 32.16 | 0.00 | 1/255 |
| kitchen | 1,852,335 | 1558x1039 | 81.9 | 30.84 | 30.32 | 34.88 | 0.00 | 1/255 |
| playroom | 2,546,116 | 1264x832 | 60.9 | 30.07 | 30.04 | 34.78 | 0.01 | 1/255 |
| room | 1,593,376 | 1557x1038 | 77.0 | 31.38 | 30.63 | 34.14 | 0.00 | 1/255 |
| stump | 4,961,797 | 1245x825 | 70.3 | 26.58 | 26.55 | 29.11 | 0.00 | 2/255 |
| train | 1,026,508 | 980x545 | 60.6 | 21.79 | 21.10 | 24.74 | 0.01 | 1/255 |
| treehill | 3,783,761 | 1267x832 | 69.8 | 22.31 | 22.49 | 25.30 | 0.01 | 1/255 |
| truck | 2,541,226 | 979x546 | 59.7 | 24.99 | 25.19 | 28.80 | 0.00 | 1/255 |

Across all scenes, 11 of the 13 center crops match exactly within 1/255. Bicycle
and stump each have a single pixel at 2/255, out of 16,384 per crop. The
"ours vs official" column compares against renders of the paper-era models,
which differ from the downloadable models (see above), so it isn't expected
to be exact. The ms/view figures are from the version before the performance
work (see [Performance](#performance)), measured without warm-up.
