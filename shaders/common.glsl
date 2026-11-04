// Shared definitions for the Gaussian splat compute pipeline.
//
// All buffers are accessed through buffer device addresses passed in push
// constants, so no descriptor sets are needed anywhere in the renderer.
#ifndef SPLAT_COMMON_GLSL
#define SPLAT_COMMON_GLSL

#extension GL_EXT_buffer_reference : require

#define TILE_SIZE 16u
#define TILE_PIXELS 256u

// Mirrors FrameParams in src/render/SplatRenderer.hpp.
// Camera space follows the 3DGS / COLMAP convention: +x right, +y down, +z forward.
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer FrameParamsRef {
    mat4 modelView;   // object space -> camera space
    vec4 camPosObj;   // camera centre in object space (xyz)
    vec4 intrinsics;  // fx, fy, cx, cy (pixels, cx/cy already shifted to pixel-centre convention)
    vec4 frustum;     // tanFovX, tanFovY, scaleModifier, nearCull
    vec4 background;  // rgb
    uvec4 dims;       // width, height, tilesX, tilesY
    uvec4 misc;       // numGaussians, shDegree, tightTiles (0/1), 0
};

layout(buffer_reference, std430, buffer_reference_align = 4) buffer UintBuf { uint v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer IntBuf { int v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuf { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) buffer Vec2Buf { vec2 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) buffer UVec2Buf { uvec2 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) buffer Vec4Buf { vec4 v[]; };

// ---------------------------------------------------------------------------
// Tile binning
//
// The reference rasterizer bins a Gaussian into every tile of the square
// [mean - r, mean + r] with r = ceil(3 sqrt(lambda_max)), the 3-sigma EWA
// footprint. Compositing then skips a Gaussian at a pixel when
// alpha = opacity * exp(-q/2) < 1/255, q being the conic's quadratic form of
// the pixel offset. So a Gaussian only affects the pixels of the ellipse
//     q(d) <= 2 ln(255 opacity),
// usually much smaller than the 3-sigma square, especially for faint or
// elongated Gaussians. "Tight" binning keeps only the tiles of the reference
// square that this ellipse touches. The test is conservative (a margin covers
// float rounding in the compositing shader), so every dropped (Gaussian, tile)
// pair is one that compositing would have skipped at every pixel: colours and
// transmittance are bit-identical to reference binning.

// Bounds of the tiles a Gaussian is binned into, [rectMin, rectMax). Returns
// false when it cannot contribute to any pixel. limit < 0 disables the
// per-tile test (reference binning).
bool tileBounds(vec2 mean, vec4 conicOpacity, int radius, ivec2 grid, bool tight,
                out ivec2 rectMin, out ivec2 rectMax, out float limit) {
    float r = float(radius);
    rectMin = clamp(ivec2((mean - r) / float(TILE_SIZE)), ivec2(0), grid);
    rectMax = clamp(ivec2((mean + r + float(TILE_SIZE - 1u)) / float(TILE_SIZE)), ivec2(0), grid);
    limit = -1.0;
    if (!tight) return true;

    float opacity = conicOpacity.w;
    // alpha <= opacity everywhere, and compositing skips alpha < 1/255.
    if (opacity < 1.0 / 255.0) return false;
    float A = conicOpacity.x, B = conicOpacity.y, C = conicOpacity.z;
    float detConic = A * C - B * B;
    if (!(detConic > 0.0)) return true; // degenerate: keep reference binning

    // q <= t where alpha >= 1/255. The margin grows with the size of the
    // terms of q (bounded by 4 t A C / det on the ellipse), since the
    // compositing shader's rounding error in q does too.
    float t = 2.0 * log(255.0 * opacity);
    limit = (t + 0.01) * (1.0 + 4e-5 * (A * C / detConic));
    // Axis extents of the ellipse q <= limit: sqrt(limit * Sigma_ii), Sigma = conic^-1.
    vec2 ext = sqrt(limit * vec2(C, A) / detConic) + 0.01;
    ivec2 tightMin = ivec2(floor((mean - ext) / float(TILE_SIZE)));
    ivec2 tightMax = ivec2(floor((mean + ext) / float(TILE_SIZE))) + 1;
    rectMin = max(rectMin, tightMin);
    rectMax = min(rectMax, tightMax);
    return true;
}

// Whether the minimum of q over the pixel centres of `tile` is <= limit.
// q is convex with its minimum at d = 0, so if 0 is outside the tile's box of
// offsets, the minimum lies on one of its edges, where it is a 1D quadratic.
bool tileOverlaps(vec2 mean, vec3 conic, float limit, ivec2 tile) {
    if (limit < 0.0) return true;
    vec2 lo = vec2(tile) * float(TILE_SIZE);
    vec2 dMin = mean - (lo + float(TILE_SIZE - 1u)); // offsets d = mean - pixel
    vec2 dMax = mean - lo;
    if (dMin.x <= 0.0 && dMax.x >= 0.0 && dMin.y <= 0.0 && dMax.y >= 0.0) return true;

    float A = conic.x, B = conic.y, C = conic.z;
    float best = 3.4e38;
    for (int e = 0; e < 2; ++e) {
        float dx = e == 0 ? dMin.x : dMax.x;
        float dy = clamp(-B * dx / C, dMin.y, dMax.y);
        best = min(best, A * dx * dx + 2.0 * B * dx * dy + C * dy * dy);
        dy = e == 0 ? dMin.y : dMax.y;
        dx = clamp(-B * dy / A, dMin.x, dMax.x);
        best = min(best, A * dx * dx + 2.0 * B * dx * dy + C * dy * dy);
    }
    return best <= limit;
}

// Large 1D dispatches are split over x/y to respect maxComputeWorkGroupCount.
uint linearGroupId() {
    return gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x;
}

#endif
