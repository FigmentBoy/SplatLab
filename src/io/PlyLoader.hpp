#pragma once

#include <functional>
#include <string>

class GaussianCloud;

// Reads binary little-endian PLY files in the layout written by the reference
// 3DGS implementation:
//   x y z  nx ny nz  f_dc_0..2  f_rest_0..(3*K-1)  opacity  scale_0..2  rot_0..3
// f_rest is channel-major (all coefficients of R, then G, then B). The SH
// degree is inferred from the number of f_rest properties.
//
// Stateless utility: it cannot be instantiated.
class PlyLoader {
public:
    PlyLoader() = delete;

    // Called with the fraction done (0..1); returning false cancels the load.
    using ProgressCallback = std::function<bool(float fraction)>;

    // Returns false and fills `error` on malformed or unsupported files, or
    // if `progress` cancelled the load.
    static bool load(const std::string& path, GaussianCloud& cloud, std::string& error,
                     const ProgressCallback& progress = {});
};
