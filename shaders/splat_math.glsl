// Gaussian math shared by the forward preprocessing (rendering and training
// variants) and the backward pass: SH evaluation and quaternion rotation.
#ifndef SPLAT_MATH_GLSL
#define SPLAT_MATH_GLSL

const float SH_C0 = 0.28209479177387814;
const float SH_C1 = 0.4886025119029199;
const float SH_C2[5] = float[](1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
                               -1.0925484305920792, 0.5462742152960396);
const float SH_C3[7] = float[](-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
                               0.3731763325901154, -0.4570457994644658, 1.445305721320277,
                               -0.5900435899266435);

// View-dependent colour before the +0.5 offset and clamping. `sh` only needs
// the coefficients of the bands up to `degree`; `dir` is normalized.
vec3 evalSH(vec3 sh[16], uint degree, vec3 dir) {
    vec3 result = SH_C0 * sh[0];
    if (degree > 0u) {
        float x = dir.x, y = dir.y, z = dir.z;
        result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];
        if (degree > 1u) {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            result = result + SH_C2[0] * xy * sh[4]
                            + SH_C2[1] * yz * sh[5]
                            + SH_C2[2] * (2.0 * zz - xx - yy) * sh[6]
                            + SH_C2[3] * xz * sh[7]
                            + SH_C2[4] * (xx - yy) * sh[8];
            if (degree > 2u) {
                result = result + SH_C3[0] * y * (3.0 * xx - yy) * sh[9]
                                + SH_C3[1] * xy * z * sh[10]
                                + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh[11]
                                + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[12]
                                + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh[13]
                                + SH_C3[5] * z * (xx - yy) * sh[14]
                                + SH_C3[6] * x * (xx - 3.0 * yy) * sh[15];
            }
        }
    }
    return result;
}

// Rotation matrix of a normalized quaternion (w, x, y, z).
mat3 quatToMat(vec4 q) {
    float r = q.x, x = q.y, y = q.z, z = q.w;
    // GLSL constructors are column-major, so pass the transposed rows.
    return transpose(mat3(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - r * z),       2.0 * (x * z + r * y),
        2.0 * (x * y + r * z),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - r * x),
        2.0 * (x * z - r * y),       2.0 * (y * z + r * x),       1.0 - 2.0 * (x * x + y * y)));
}

// Activation of a raw (unnormalized) quaternion, as in the reference.
vec4 normalizeQuat(vec4 q) {
    float len = length(q);
    return len > 0.0 ? q / len : vec4(1.0, 0.0, 0.0, 0.0);
}

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

#endif
