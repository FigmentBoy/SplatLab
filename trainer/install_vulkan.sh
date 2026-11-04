#!/usr/bin/env bash
# Makes Vulkan usable on an NVIDIA Linux machine where only the CUDA side of
# the driver is present (typical of Colab, Kaggle and other GPU containers).
#
#   bash trainer/install_vulkan.sh
#
# 1. Installs the Vulkan loader and vulkaninfo if missing (apt).
# 2. If no NVIDIA Vulkan device is visible, downloads NVIDIA's installer for
#    exactly the running driver version, extracts it (no kernel module, nothing
#    is "installed"), and copies only the missing Vulkan userspace libraries and
#    the ICD manifest into place. Libraries already present are left alone.
# 3. Checks again with vulkaninfo; exits non-zero if Vulkan is still unusable.
set -euo pipefail

SUDO=""
if [ "$(id -u)" != "0" ]; then SUDO="sudo"; fi

have_nvidia_vulkan() {
    command -v vulkaninfo >/dev/null 2>&1 || return 1
    vulkaninfo --summary 2>/dev/null | grep -iE "deviceName.*(NVIDIA|Tesla|Quadro|GeForce|RTX|A100|H100|L4|T4)" >/dev/null
}

if ! command -v vulkaninfo >/dev/null 2>&1 || ! ldconfig -p | grep -q "libvulkan.so.1"; then
    echo "== Installing the Vulkan loader"
    $SUDO apt-get -qq update >/dev/null
    $SUDO apt-get -qq install -y libvulkan1 vulkan-tools >/dev/null
fi

if have_nvidia_vulkan; then
    echo "== NVIDIA Vulkan driver already available"
    vulkaninfo --summary 2>/dev/null | grep -iE "deviceName|apiVersion|driverVersion" | head -3
    exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "No NVIDIA driver (nvidia-smi not found): a GPU runtime is required." >&2
    exit 1
fi
VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')"
echo "== Installing the NVIDIA Vulkan userspace driver $VERSION"

LIBDIR=/usr/lib/x86_64-linux-gnu
[ -d "$LIBDIR" ] || LIBDIR=/usr/lib64
WORK="$(mktemp -d)"
RUN="$WORK/NVIDIA-Linux-x86_64-$VERSION.run"
for url in "https://us.download.nvidia.com/tesla/$VERSION/NVIDIA-Linux-x86_64-$VERSION.run" \
           "https://us.download.nvidia.com/XFree86/Linux-x86_64/$VERSION/NVIDIA-Linux-x86_64-$VERSION.run"; do
    if curl -fsSL -o "$RUN" "$url" || wget -q -O "$RUN" "$url"; then break; fi
    rm -f "$RUN"
done
if [ ! -s "$RUN" ]; then
    echo "Could not download the NVIDIA $VERSION installer." >&2
    exit 1
fi
sh "$RUN" --extract-only --target "$WORK/driver" >/dev/null
DRV="$WORK/driver"

# Userspace pieces the Vulkan ICD (libGLX_nvidia) needs; all must match the
# kernel driver version exactly, which is why they come from this installer.
copied=0
for pattern in "libGLX_nvidia.so.$VERSION" "libnvidia-glcore.so.$VERSION" "libnvidia-glvkspirv.so.$VERSION" \
               "libnvidia-gpucomp.so.$VERSION" "libnvidia-tls.so.$VERSION" "libnvidia-glsi.so.$VERSION" \
               "libnvidia-rtcore.so.$VERSION" "libnvidia-eglcore.so.$VERSION"; do
    src="$DRV/$pattern"
    [ -f "$src" ] || continue
    if [ ! -e "$LIBDIR/$pattern" ]; then
        $SUDO cp "$src" "$LIBDIR/" && copied=$((copied + 1))
    fi
done
# libGLX_nvidia.so.0 -> versioned file (the name the ICD manifest refers to).
if [ ! -e "$LIBDIR/libGLX_nvidia.so.0" ]; then
    $SUDO ln -sf "libGLX_nvidia.so.$VERSION" "$LIBDIR/libGLX_nvidia.so.0"
fi
$SUDO ldconfig

# ICD manifest: from the installer when it has one, otherwise a minimal one.
$SUDO mkdir -p /usr/share/vulkan/icd.d /etc/vulkan/icd.d
if [ -f "$DRV/nvidia_icd.json" ]; then
    $SUDO cp "$DRV/nvidia_icd.json" /usr/share/vulkan/icd.d/nvidia_icd.json
else
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3"}}' \
        | $SUDO tee /usr/share/vulkan/icd.d/nvidia_icd.json >/dev/null
fi
rm -rf "$WORK"
echo "   copied $copied libraries into $LIBDIR"

if have_nvidia_vulkan; then
    echo "== NVIDIA Vulkan driver installed"
    vulkaninfo --summary 2>/dev/null | grep -iE "deviceName|apiVersion|driverVersion" | head -3
    exit 0
fi
echo "Vulkan is still not usable on this machine:" >&2
vulkaninfo --summary 2>&1 | tail -15 >&2 || true
exit 1
