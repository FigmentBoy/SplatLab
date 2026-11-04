#include "app/Options.hpp"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <vector>

namespace {

bool parseVec3(const std::string& text, glm::vec3& out) {
    std::stringstream ss(text);
    std::string item;
    std::vector<float> values;
    while (std::getline(ss, item, ',')) {
        try {
            values.push_back(std::stof(item));
        } catch (...) {
            return false;
        }
    }
    if (values.size() != 3) return false;
    out = glm::vec3(values[0], values[1], values[2]);
    return true;
}

} // namespace

std::unique_ptr<Options> Options::create(int argc, char** argv) {
    std::unique_ptr<Options> options(new Options());
    if (!options->init(argc, argv)) return nullptr;
    return options;
}

void Options::printUsage(const char* program) {
    std::cout <<
        "Usage: " << program << " [scene.ply] [options]\n"
        "\n"
        "Renders a 3D Gaussian Splatting PLY (reference 3DGS layout).\n"
        "Without --out an interactive window opens (scenes can also be loaded from\n"
        "File > Load PLY); with --out images are written.\n"
        "\n"
        "Cameras:\n"
        "  --cameras <file.json>   cameras.json in the reference 3DGS format\n"
        "  --camera <i>            only use camera i\n"
        "  --eye x,y,z             custom camera position (world space)\n"
        "  --target x,y,z          custom camera look-at point (default 0,0,0)\n"
        "  --up x,y,z              custom camera up vector (default 0,-1,0)\n"
        "  --fovy <degrees>        vertical field of view for custom/free cameras (default 60)\n"
        "\n"
        "Output:\n"
        "  --out <dir>             render offline into <dir> (one PNG per camera)\n"
        "  --width <w> --height <h> output size (default: camera's native size)\n"
        "  --aux                   also write per-pixel final transmittance images\n"
        "  --profile               print per-stage GPU timings (each view is rendered for\n"
        "                          ~0.3 s to warm up, then the median of 7 renders is reported)\n"
        "  --check-binning         also render each view with the reference 3-sigma tile\n"
        "                          binning and compare colour/transmittance bit for bit\n"
        "\n"
        "Rendering:\n"
        "  --bg r,g,b              background colour in [0,1] (default 0,0,0)\n"
        "  --sh-degree <d>         limit the SH degree used (default: all stored)\n"
        "  --scale-modifier <s>    multiply all Gaussian scales (default 1)\n"
        "  --reference-binning     bin Gaussians into the reference 3-sigma tile squares\n"
        "                          (slower, same image) instead of only the tiles they reach\n"
        "  --vsync                 enable vsync in the viewer\n"
        "  --validation            enable Vulkan validation layers if installed\n"
        "\n"
        "Viewer controls:\n"
        "  Left click capture mouse | Esc release mouse\n"
        "  While captured: mouse look | WASD move | Space/Shift up/down | Ctrl fast | scroll speed\n"
        "  ] / [ next / previous dataset camera | R reset | P screenshot\n"
        "  C print current pose as cameras.json entry\n"
        "  File > Load Scene / Load Cameras, View > Cameras to browse dataset cameras\n";
}

bool Options::init(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](std::string& value) {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << arg << "\n";
                return false;
            }
            value = argv[++i];
            return true;
        };
        auto nextVec3 = [&](glm::vec3& out) {
            std::string value;
            if (!next(value)) return false;
            if (!parseVec3(value, out)) {
                std::cerr << "Expected x,y,z for " << arg << ", got '" << value << "'\n";
                return false;
            }
            return true;
        };
        auto nextNumber = [&](auto& out) {
            std::string value;
            if (!next(value)) return false;
            char* end = nullptr;
            double v = std::strtod(value.c_str(), &end);
            if (end == value.c_str() || *end != '\0') {
                std::cerr << "Expected a number for " << arg << ", got '" << value << "'\n";
                return false;
            }
            out = static_cast<std::remove_reference_t<decltype(out)>>(v);
            return true;
        };

        if (arg == "-h" || arg == "--help") {
            help = true;
            return true;
        } else if (arg == "--cameras") {
            if (!next(camerasPath)) return false;
        } else if (arg == "--camera") {
            if (!nextNumber(cameraIndex)) return false;
        } else if (arg == "--eye") {
            glm::vec3 e;
            if (!nextVec3(e)) return false;
            eye = e;
        } else if (arg == "--target") {
            if (!nextVec3(target)) return false;
        } else if (arg == "--up") {
            if (!nextVec3(up)) return false;
        } else if (arg == "--fovy") {
            if (!nextNumber(fovYDegrees)) return false;
        } else if (arg == "--out") {
            if (!next(outputDir)) return false;
        } else if (arg == "--width") {
            if (!nextNumber(width)) return false;
        } else if (arg == "--height") {
            if (!nextNumber(height)) return false;
        } else if (arg == "--aux") {
            saveAux = true;
        } else if (arg == "--profile") {
            profile = true;
        } else if (arg == "--check-binning") {
            checkBinning = true;
        } else if (arg == "--reference-binning") {
            referenceBinning = true;
        } else if (arg == "--bg") {
            if (!nextVec3(background)) return false;
        } else if (arg == "--sh-degree") {
            if (!nextNumber(shDegree)) return false;
        } else if (arg == "--scale-modifier") {
            if (!nextNumber(scaleModifier)) return false;
        } else if (arg == "--vsync") {
            vsync = true;
        } else if (arg == "--validation") {
            validation = true;
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "Unknown option " << arg << "\n";
            return false;
        } else if (scenePath.empty()) {
            scenePath = arg;
        } else {
            std::cerr << "Unexpected argument " << arg << "\n";
            return false;
        }
    }
    if (scenePath.empty() && !outputDir.empty()) {
        std::cerr << "Offline rendering (--out) needs a scene PLY\n";
        return false;
    }
    return true;
}
