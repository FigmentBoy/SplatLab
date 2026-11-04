// Shared definitions for the 8-bit LSD radix sort (radix_histogram /
// radix_scatter). A block of 256 invocations handles RADIX_BLOCK keys in
// RADIX_CHUNKS consecutive chunks of 256. Within a chunk, element k is owned
// by the invocation with linear subgroup index k (subgroup id, then lane), so
// the subgroup ballots below rank keys in element order and the sort is stable.
#ifndef SPLAT_RADIX_COMMON_GLSL
#define SPLAT_RADIX_COMMON_GLSL

#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_ballot : require

#define RADIX_CHUNKS 16u
#define RADIX_BLOCK (RADIX_CHUNKS * 256u)

layout(constant_id = 0) const uint SUBGROUP_SIZE = 32u;
const uint NUM_SUBGROUPS = 256u / SUBGROUP_SIZE;

uint chunkLane() {
    return gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
}

// Lanes of this subgroup that hold a valid key with the same 8-bit digit
// (warp-level multisplit: one ballot per digit bit).
uvec4 digitPeers(uint digit, bool valid) {
    uvec4 peers = subgroupBallot(valid);
    for (uint b = 0u; b < 8u; ++b) {
        bool set = ((digit >> b) & 1u) != 0u;
        uvec4 vote = subgroupBallot(set);
        peers &= set ? vote : ~vote;
    }
    return valid ? peers : uvec4(0u);
}

#endif
