# Headless trainer image for cloud GPUs (NVIDIA, Linux).
#
#   docker build -t splat-trainer .
#   docker run --rm --gpus all -v "$PWD/runs:/runs" splat-trainer /runs/video.mp4 -o /runs/out
#
# Requires the NVIDIA Container Toolkit on the host. The Vulkan driver comes
# from the host through the toolkit (graphics capability); the image only adds
# the Vulkan loader and the ICD manifest pointing at it.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git ca-certificates \
        python3 python3-dev python3-venv \
        libvulkan1 glslang-tools libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/share/vulkan/icd.d \
    && printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3"}}\n' \
        > /usr/share/vulkan/icd.d/nvidia_icd.json

# Python environment (bootstrap.py recognises it as a virtual environment and
# does not create another one).
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir numpy opencv-python-headless pycolmap

WORKDIR /splat
COPY . .
# Only the core renderer and the Python module; no window system needed.
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DSPLAT_BUILD_VIEWER=OFF -DPython_EXECUTABLE=/opt/venv/bin/python \
    && cmake --build build --target splat_vk

ENTRYPOINT ["python", "-u", "trainer/bootstrap.py"]
