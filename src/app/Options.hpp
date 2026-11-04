#pragma once

#include <glm/glm.hpp>

#include <memory>
#include <optional>
#include <string>

// Command-line options. See Options::usage() for the full list.
class Options {
public:
    // Returns nullptr on invalid arguments (after printing the problem).
    static std::unique_ptr<Options> create(int argc, char** argv);
    bool init(int argc, char** argv);

    static void printUsage(const char* program);

    std::string scenePath;
    std::string camerasPath;
    std::string outputDir; // non-empty: render offline into this directory, no window
    int cameraIndex = -1;  // -1: all cameras (offline) / first camera (viewer)

    std::optional<glm::vec3> eye;
    glm::vec3 target{0.0f};
    glm::vec3 up{0.0f, -1.0f, 0.0f}; // COLMAP scenes are usually y-down
    float fovYDegrees = 60.0f;

    uint32_t width = 0;  // 0: camera's native size (or 1280x720)
    uint32_t height = 0;

    glm::vec3 background{0.0f};
    int shDegree = -1;
    float scaleModifier = 1.0f;
    bool saveAux = false;
    bool profile = false;
    bool checkBinning = false;
    bool referenceBinning = false;
    bool validation = false;
    bool vsync = false;
    bool help = false;

    // Viewer: start training on this video right away (Train > Train from Video).
    std::string trainInput;
    std::string trainOutput; // default: data/models/<video name>
    int trainIterations = 30000;
    int trainFrames = 200;

private:
    Options() = default;
};
