// Per-Gaussian preprocessing: projection, 2D covariance (EWA), SH colour,
// screen-space radius and number of tiles touched.
// Follows the reference 3DGS rasterizer (diff-gaussian-rasterization, forward.cu).
//
// Included by preprocess.comp (activated parameters uploaded by the viewer)
// and preprocess_train.comp (TRAINING_INPUTS: the raw parameters the trainer
// optimizes, activated here), so both share the exact same projection math.
#include "common.glsl"
#include "splat_math.glsl"

layout(local_size_x = 256) in;

#ifdef TRAINING_INPUTS
layout(push_constant) uniform PushConstants {
    FrameParamsRef params;
    FloatBuf means;          // xyz per Gaussian
    FloatBuf scaleLog;       // log scale xyz (activation: exp)
    FloatBuf rotation;       // raw quaternion (w, x, y, z) (activation: normalize)
    FloatBuf opacityLogit;   // (activation: sigmoid)
    FloatBuf shDc;           // 3 per Gaussian
    FloatBuf shRest;         // 45 per Gaussian: coefficients 1..15 x rgb
    Vec2Buf means2D;
    Vec4Buf conicOpacity;
    Vec4Buf colorDepth;
    IntBuf radii;
    UintBuf tilesTouched;
    UintBuf visible;
} pc;

void loadGaussian(uint idx, out vec3 pos, out float opacity, out vec3 scale, out vec4 rot) {
    uint i3 = 3u * idx, i4 = 4u * idx;
    pos = vec3(pc.means.v[i3], pc.means.v[i3 + 1u], pc.means.v[i3 + 2u]);
    opacity = sigmoid(pc.opacityLogit.v[idx]);
    scale = exp(vec3(pc.scaleLog.v[i3], pc.scaleLog.v[i3 + 1u], pc.scaleLog.v[i3 + 2u]));
    rot = normalizeQuat(vec4(pc.rotation.v[i4], pc.rotation.v[i4 + 1u], pc.rotation.v[i4 + 2u],
                             pc.rotation.v[i4 + 3u]));
}

void loadSH(uint idx, uint degree, out vec3 sh[16]) {
    uint i3 = 3u * idx;
    sh[0] = vec3(pc.shDc.v[i3], pc.shDc.v[i3 + 1u], pc.shDc.v[i3 + 2u]);
    uint count = (degree + 1u) * (degree + 1u);
    uint base = 45u * idx;
    for (uint k = 1u; k < count; ++k) {
        uint o = base + 3u * (k - 1u);
        sh[k] = vec3(pc.shRest.v[o], pc.shRest.v[o + 1u], pc.shRest.v[o + 2u]);
    }
}
#else
layout(push_constant) uniform PushConstants {
    FrameParamsRef params;
    Vec4Buf positionOpacity; // xyz position (object space), w = activated opacity
    Vec4Buf scales;          // xyz activated scale
    Vec4Buf rotations;       // normalized quaternion (w, x, y, z)
    Vec4Buf sh;              // 16 coefficients x 3 channels per Gaussian (12 vec4)
    Vec2Buf means2D;         // out: pixel-space mean
    Vec4Buf conicOpacity;    // out: inverse 2D covariance (xx, xy, yy), opacity
    Vec4Buf colorDepth;      // out: rgb, view-space depth
    IntBuf radii;            // out: screen radius in pixels (0 = culled)
    UintBuf tilesTouched;    // out: number of tiles the Gaussian is binned into
    UintBuf visible;         // out: 1 if binned into any tile, else 0
} pc;

void loadGaussian(uint idx, out vec3 pos, out float opacity, out vec3 scale, out vec4 rot) {
    vec4 po = pc.positionOpacity.v[idx];
    pos = po.xyz;
    opacity = po.w;
    scale = pc.scales.v[idx].xyz;
    rot = pc.rotations.v[idx];
}

// SH coefficients are 48 consecutive floats per Gaussian (coefficient-major,
// rgb), read as 12 vec4s; only the bands in use are loaded.
void loadSH(uint idx, uint degree, out vec3 sh[16]) {
    uint base = idx * 12u;
    vec4 q[12];
    q[0] = pc.sh.v[base];
    if (degree > 0u) {
        q[1] = pc.sh.v[base + 1u];
        q[2] = pc.sh.v[base + 2u];
    }
    if (degree > 1u) {
        q[3] = pc.sh.v[base + 3u];
        q[4] = pc.sh.v[base + 4u];
        q[5] = pc.sh.v[base + 5u];
        q[6] = pc.sh.v[base + 6u];
    }
    if (degree > 2u) {
        for (uint i = 7u; i < 12u; ++i) q[i] = pc.sh.v[base + i];
    }
    uint count = (degree + 1u) * (degree + 1u);
    for (uint k = 0u; k < count; ++k) {
        uint f = 3u * k;
        sh[k] = vec3(q[f / 4u][f % 4u], q[(f + 1u) / 4u][(f + 1u) % 4u], q[(f + 2u) / 4u][(f + 2u) % 4u]);
    }
}
#endif

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint numGaussians = pc.params.misc.x;
    if (idx >= numGaussians) return;

    pc.radii.v[idx] = 0;
    pc.tilesTouched.v[idx] = 0u;
    pc.visible.v[idx] = 0u;

    vec3 pos, scale;
    float opacity;
    vec4 rot;
    loadGaussian(idx, pos, opacity, scale, rot);

    mat4 modelView = pc.params.modelView;
    vec3 t = (modelView * vec4(pos, 1.0)).xyz;
    if (t.z <= pc.params.frustum.w) return; // near-plane cull

    // 3D covariance in object space: Sigma = R S S^T R^T
    vec3 s = scale * pc.params.frustum.z;
    mat3 M = quatToMat(rot) * mat3(s.x, 0.0, 0.0, 0.0, s.y, 0.0, 0.0, 0.0, s.z);
    mat3 sigma = M * transpose(M);

    // EWA projection with the reference implementation's frustum clamping.
    float fx = pc.params.intrinsics.x, fy = pc.params.intrinsics.y;
    float limX = 1.3 * pc.params.frustum.x;
    float limY = 1.3 * pc.params.frustum.y;
    vec3 tc = vec3(clamp(t.x / t.z, -limX, limX) * t.z, clamp(t.y / t.z, -limY, limY) * t.z, t.z);
    mat3 J = mat3(fx / tc.z, 0.0, 0.0,
                  0.0, fy / tc.z, 0.0,
                  -fx * tc.x / (tc.z * tc.z), -fy * tc.y / (tc.z * tc.z), 0.0);
    mat3 T = J * mat3(modelView);
    mat3 cov = T * sigma * transpose(T);

    float a = cov[0][0] + 0.3;
    float b = cov[0][1];
    float c = cov[1][1] + 0.3;
    float det = a * c - b * b;
    if (det == 0.0) return;
    float invDet = 1.0 / det;
    vec3 conic = vec3(c * invDet, -b * invDet, a * invDet);

    float mid = 0.5 * (a + c);
    float lambda1 = mid + sqrt(max(0.1, mid * mid - det));
    float radius = ceil(3.0 * sqrt(lambda1));

    vec2 mean = vec2(fx * t.x / t.z + pc.params.intrinsics.z,
                     fy * t.y / t.z + pc.params.intrinsics.w);

    ivec2 grid = ivec2(pc.params.dims.zw);
    vec4 conicOpacity = vec4(conic, opacity);
    ivec2 rectMin, rectMax;
    float limit;
    if (!tileBounds(mean, conicOpacity, int(radius), grid, pc.params.misc.z != 0u, rectMin, rectMax, limit)) {
        return;
    }
    uint tiles = 0u;
    if (limit < 0.0) {
        tiles = uint(max(rectMax.x - rectMin.x, 0) * max(rectMax.y - rectMin.y, 0));
    } else {
        for (int y = rectMin.y; y < rectMax.y; ++y) {
            for (int x = rectMin.x; x < rectMax.x; ++x) {
                if (tileOverlaps(mean, conic, limit, ivec2(x, y))) ++tiles;
            }
        }
    }
    if (tiles == 0u) return;

    vec3 dir = normalize(pos - pc.params.camPosObj.xyz);
    uint degree = pc.params.misc.y;
    vec3 sh[16];
    loadSH(idx, degree, sh);
    vec3 rgb = max(evalSH(sh, degree, dir) + 0.5, vec3(0.0));

    pc.means2D.v[idx] = mean;
    pc.conicOpacity.v[idx] = conicOpacity;
    pc.colorDepth.v[idx] = vec4(rgb, t.z);
    pc.radii.v[idx] = int(radius);
    pc.tilesTouched.v[idx] = tiles;
    pc.visible.v[idx] = 1u;
}
