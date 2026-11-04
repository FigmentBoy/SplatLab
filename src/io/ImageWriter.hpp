#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Stateless utility: it cannot be instantiated.
class ImageWriter {
public:
    ImageWriter() = delete;

    // Writes 8-bit RGBA pixels (row-major, top row first) as PNG.
    static void writePngRgba(const std::string& path, uint32_t width, uint32_t height, const uint8_t* rgba);

    // Writes a single-channel float image in [0,1] as an 8-bit grayscale PNG.
    static void writePngGray(const std::string& path, uint32_t width, uint32_t height, const float* values);
};
