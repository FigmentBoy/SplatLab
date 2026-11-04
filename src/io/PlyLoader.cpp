#include "io/PlyLoader.hpp"

#include "scene/GaussianCloud.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

struct Property {
    std::string name;
    std::string type;
    size_t offset = 0;
    size_t size = 0;
};

size_t typeSize(const std::string& type) {
    if (type == "char" || type == "uchar" || type == "int8" || type == "uint8") return 1;
    if (type == "short" || type == "ushort" || type == "int16" || type == "uint16") return 2;
    if (type == "int" || type == "uint" || type == "float" || type == "int32" || type == "uint32" ||
        type == "float32") return 4;
    if (type == "double" || type == "float64") return 8;
    throw std::runtime_error("PLY: unsupported property type '" + type + "'");
}

template <typename T>
T readAs(const char* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}

float readFloat(const char* row, const Property& prop) {
    const char* p = row + prop.offset;
    const std::string& t = prop.type;
    if (t == "float" || t == "float32") return readAs<float>(p);
    if (t == "double" || t == "float64") return static_cast<float>(readAs<double>(p));
    if (t == "uchar" || t == "uint8") return static_cast<float>(readAs<uint8_t>(p));
    if (t == "char" || t == "int8") return static_cast<float>(readAs<int8_t>(p));
    if (t == "ushort" || t == "uint16") return static_cast<float>(readAs<uint16_t>(p));
    if (t == "short" || t == "int16") return static_cast<float>(readAs<int16_t>(p));
    if (t == "uint" || t == "uint32") return static_cast<float>(readAs<uint32_t>(p));
    return static_cast<float>(readAs<int32_t>(p));
}

void loadOrThrow(const std::string& path, GaussianCloud& cloud, const PlyLoader::ProgressCallback& progress);

} // namespace

bool PlyLoader::load(const std::string& path, GaussianCloud& cloud, std::string& error,
                     const ProgressCallback& progress) {
    try {
        loadOrThrow(path, cloud, progress);
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
    return true;
}

namespace {

void loadOrThrow(const std::string& path, GaussianCloud& cloud, const PlyLoader::ProgressCallback& progress) {
    // Reading the file is the first half of the progress, parsing the second.
    auto report = [&](float fraction) {
        if (progress && !progress(fraction)) throw std::runtime_error("PLY: loading cancelled");
    };
    report(0.0f);

    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("PLY: cannot open '" + path + "'");

    std::string line;
    std::getline(file, line);
    if (line.rfind("ply", 0) != 0) throw std::runtime_error("PLY: missing magic in '" + path + "'");

    size_t vertexCount = 0;
    bool inVertexElement = false;
    std::vector<Property> props;
    size_t stride = 0;

    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream ss(line);
        std::string keyword;
        ss >> keyword;
        if (keyword == "format") {
            std::string fmt;
            ss >> fmt;
            if (fmt != "binary_little_endian") {
                throw std::runtime_error("PLY: only binary_little_endian is supported (got " + fmt + ")");
            }
        } else if (keyword == "element") {
            std::string elem;
            size_t count = 0;
            ss >> elem >> count;
            if (elem == "vertex") {
                inVertexElement = true;
                vertexCount = count;
            } else {
                if (vertexCount == 0) throw std::runtime_error("PLY: vertex element must come first");
                inVertexElement = false;
            }
        } else if (keyword == "property" && inVertexElement) {
            Property prop;
            ss >> prop.type;
            if (prop.type == "list") throw std::runtime_error("PLY: list properties in vertex element");
            ss >> prop.name;
            prop.size = typeSize(prop.type);
            prop.offset = stride;
            stride += prop.size;
            props.push_back(prop);
        } else if (keyword == "end_header") {
            break;
        }
    }
    if (vertexCount == 0 || stride == 0) throw std::runtime_error("PLY: no vertices in '" + path + "'");

    std::unordered_map<std::string, const Property*> byName;
    for (const auto& p : props) byName[p.name] = &p;
    auto require = [&](const std::string& name) -> const Property& {
        auto it = byName.find(name);
        if (it == byName.end()) throw std::runtime_error("PLY: missing property '" + name + "'");
        return *it->second;
    };

    const Property* pos[3] = {&require("x"), &require("y"), &require("z")};
    const Property* dc[3] = {&require("f_dc_0"), &require("f_dc_1"), &require("f_dc_2")};
    const Property* scale[3] = {&require("scale_0"), &require("scale_1"), &require("scale_2")};
    const Property* rot[4] = {&require("rot_0"), &require("rot_1"), &require("rot_2"), &require("rot_3")};
    const Property& opacity = require("opacity");

    std::vector<const Property*> rest;
    while (byName.count("f_rest_" + std::to_string(rest.size()))) {
        rest.push_back(byName["f_rest_" + std::to_string(rest.size())]);
    }
    uint32_t restPerChannel = static_cast<uint32_t>(rest.size() / 3);
    uint32_t degree = 0;
    switch (rest.size()) {
    case 0: degree = 0; break;
    case 9: degree = 1; break;
    case 24: degree = 2; break;
    case 45: degree = 3; break;
    default:
        throw std::runtime_error("PLY: unexpected number of f_rest properties (" + std::to_string(rest.size()) + ")");
    }

    std::vector<char> data(vertexCount * stride);
    constexpr size_t kChunk = size_t(32) << 20;
    for (size_t done = 0; done < data.size();) {
        size_t n = std::min(kChunk, data.size() - done);
        file.read(data.data() + done, static_cast<std::streamsize>(n));
        if (static_cast<size_t>(file.gcount()) != n) throw std::runtime_error("PLY: file truncated");
        done += n;
        report(0.5f * static_cast<float>(done) / static_cast<float>(data.size()));
    }

    cloud.resize(vertexCount);
    cloud.shDegree = degree;
    constexpr uint32_t K = GaussianCloud::kMaxShCoeffs;
    for (size_t i = 0; i < vertexCount; ++i) {
        if ((i & 0xFFFF) == 0) report(0.5f + 0.5f * static_cast<float>(i) / static_cast<float>(vertexCount));
        const char* row = data.data() + i * stride;
        cloud.position[i] = {readFloat(row, *pos[0]), readFloat(row, *pos[1]), readFloat(row, *pos[2])};
        cloud.scaleLog[i] = {readFloat(row, *scale[0]), readFloat(row, *scale[1]), readFloat(row, *scale[2])};
        cloud.rotation[i] = {readFloat(row, *rot[0]), readFloat(row, *rot[1]), readFloat(row, *rot[2]),
                             readFloat(row, *rot[3])};
        cloud.opacityLogit[i] = readFloat(row, opacity);

        float* sh = cloud.sh.data() + i * K * 3;
        for (int c = 0; c < 3; ++c) sh[c] = readFloat(row, *dc[c]);
        for (uint32_t c = 0; c < 3; ++c) {
            for (uint32_t k = 0; k < restPerChannel; ++k) {
                sh[(k + 1) * 3 + c] = readFloat(row, *rest[c * restPerChannel + k]);
            }
        }
    }
}

} // namespace
