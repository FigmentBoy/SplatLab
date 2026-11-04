#include "io/ImageWriter.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>
#include <stdexcept>

void ImageWriter::writePngRgba(const std::string& path, uint32_t width, uint32_t height, const uint8_t* rgba) {
    if (!stbi_write_png(path.c_str(), static_cast<int>(width), static_cast<int>(height), 4, rgba,
                        static_cast<int>(width * 4))) {
        throw std::runtime_error("Failed to write PNG '" + path + "'");
    }
}

void ImageWriter::writePngGray(const std::string& path, uint32_t width, uint32_t height, const float* values) {
    std::vector<uint8_t> bytes(static_cast<size_t>(width) * height);
    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<uint8_t>(std::clamp(values[i], 0.0f, 1.0f) * 255.0f + 0.5f);
    }
    if (!stbi_write_png(path.c_str(), static_cast<int>(width), static_cast<int>(height), 1, bytes.data(),
                        static_cast<int>(width))) {
        throw std::runtime_error("Failed to write PNG '" + path + "'");
    }
}
