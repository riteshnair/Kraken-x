// TurboQuant 3-bit KV cache (arxiv 2504.19874v1, Zandieh et al., Google).
//
// WHAT THE PAPER DOES, and which part is here.
//
// TurboQuant_mse quantizes a d-dimensional vector at b bits per coordinate in
// three steps: (1) multiply by a random rotation Pi, which makes the coordinate
// distribution dependent only on d (Beta, converging to N(0, 1/d) in high d);
// (2) quantize each coordinate independently with the Lloyd-Max codebook for
// that distribution; (3) store the indices, and a per-vector L2 norm because
// real keys are not unit vectors. The distortion is 0.03 for b=3, versus a
// Shannon lower bound of 2^-6 = 0.0156 -- within the paper's ~1.9x factor.
//
// The paper's *inner-product* variant spends one of the bits on a QJL residual
// sketch so that E[<y, x_hat>] = <y, x> exactly. That is not here yet: this is
// the MSE-optimal form at 3 bits, which is biased for inner products and is the
// honest first step. `kTurboBits` is a parameter so the two-stage form can land
// on top without a second layout.
//
// WHAT IS NOT THE PAPER, deliberately.
//
// The rotation is a randomized Fast Walsh-Hadamard transform (random +-1 signs,
// then H/sqrt(d)) instead of a dense Haar-random Pi. A dense Pi costs d^2 floats
// per rotation -- 256 KiB per head at d=256 -- and O(d^2) per vector, which no
// KV cache on a CPU can pay. H/sqrt(d) is orthogonal (so inner products are
// preserved exactly), its entries are +-1/sqrt(d) (sub-Gaussian, which is what
// the concentration argument needs), and it costs O(d log d) with no storage.
// This is the same substitution QuaRot/SpinQuant make for the same reason.
//
// The rotation is seeded per (layer, kv head), not per vector: every key of one
// head shares it, so a query rotates once and the cache stores only the rotated
// coefficients. Since H is orthogonal and symmetric, <q, k> = <Hq, Hk> -- the
// keys never need rotating back, and a value vector's weighted sum can be
// accumulated in the rotated basis and rotated once per output.
//
// Layout of one cache row (d = head_dim of ONE kv head):
//   [ ceil(d/8)*3 bytes of packed 3-bit codes ][ f32 scale ]
// d=256 -> 100 bytes against f16's 512, i.e. 5.1x compression; d=128 -> 52 vs
// 256 (4.9x). The scale is sigma = ||x|| / sqrt(d), the standard deviation of a
// rotated coordinate, so the codebook itself is the unit-variance one.

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>

#include "krk/common.hpp"

namespace krk {

// Lloyd-Max centroids for N(0,1) at 3 bits (8 levels), and their decision
// boundaries. These are the textbook values; solve_lloyd() reproduces them and
// `turbo_kv_selftest` checks the reproduction rather than trusting the copy.
inline constexpr i32 kTurboBits = 3;
inline constexpr i32 kTurboLevels = 1 << kTurboBits;
inline constexpr f32 kTurboCentroid[kTurboLevels] = {
    -2.15194566f, -1.34390926f, -0.75600525f, -0.24509420f,
     0.24509420f,  0.75600525f,  1.34390926f,  2.15194566f};
inline constexpr f32 kTurboBoundary[kTurboLevels - 1] = {
    -1.74793313f, -1.05003882f, -0.50062380f,
     0.0f,
     0.50062380f, 1.05003882f, 1.74793313f};

inline size_t turbo_row_bytes(i64 d) {
    const size_t n = static_cast<size_t>(d <= 0 ? 0 : d);
    return ((n + 7) / 8) * 3 + sizeof(f32);
}

// One 64-bit mix, so the sign pattern is deterministic across runs and machines
// (a rotation that differed between two processes would make the cache
// unreadable across a restart, and the suite would go flaky rather than red).
inline uint64_t turbo_mix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

inline uint32_t turbo_seed(i32 layer, i32 kv_head) {
    uint64_t s = 0x243F6A8885A308D3ull;
    s = turbo_mix(s ^ static_cast<uint64_t>(static_cast<uint32_t>(layer) + 1));
    s = turbo_mix(s ^ static_cast<uint64_t>(static_cast<uint32_t>(kv_head) + 1));
    return static_cast<uint32_t>(s);
}

// In-place orthogonal Hadamard over the LARGEST POWER-OF-TWO CHUNKS of the
// vector, each normalised by 1/sqrt(chunk).
//
// The transform only has to be orthogonal -- that is what makes <q,k> =
// <Hq,Hk> -- and a block-diagonal Hadamard is, for ANY d. The previous version
// assumed d was a power of two and, at d=96 (Phi-3's head width) or d=576,
// ran a final pass that wrote past the end of the scratch: a 32-float overflow
// on the default KV format. Chunking costs some global mixing (96 = 64+32) and
// nothing else.
//
// Normalising per chunk is not optional: blocks of different sizes need
// different factors, so a single 1/sqrt(d) would leave the transform
// non-orthogonal and silently re-scale the scores. For a power-of-two d the
// chunk IS the vector, so this is bit-identical to the old form.
inline void turbo_hadamard(f32 *v, i64 d) {
    i64 off = 0;
    while (off < d) {
        i64 len = 1;
        while (off + len * 2 <= d) len <<= 1;
        for (i64 h = 1; h < len; h <<= 1) {
            for (i64 i = 0; i < len; i += h << 1) {
                for (i64 j = i; j < i + h; j++) {
                    const f32 a = v[off + j], b = v[off + j + h];
                    v[off + j] = a + b;
                    v[off + j + h] = a - b;
                }
            }
        }
        const f32 inv = 1.0f / std::sqrt(static_cast<f32>(len));
        for (i64 j = 0; j < len; j++) v[off + j] *= inv;
        off += len;
    }
}

// y = (H/ sqrt(d)) * diag(signs(seed)) * x. Orthogonal, so the inverse is
// turbo_unrotate() below -- R^-1 = S H / sqrt(d), which is NOT the same
// operation: H and S do not commute, so applying this twice is not the
// identity. (An earlier version assumed it was, and the reconstruction came
// back scrambled with MSE 2.0 instead of 0.03.)
inline void turbo_rotate(const f32 *x, f32 *y, i64 d, uint32_t seed) {
    std::memcpy(y, x, static_cast<size_t>(d) * sizeof(f32));
    uint64_t s = turbo_mix(seed);
    for (i64 j = 0; j < d; j++) {
        if ((j & 63) == 0) s = turbo_mix(s);
        const f32 sign = ((s >> (j & 63)) & 1ull) ? -1.0f : 1.0f;
        y[j] *= sign;
    }
    turbo_hadamard(y, d);
    // turbo_hadamard is orthogonal on its own (per-chunk normalisation), so
    // there is no global scale here: adding 1/sqrt(d) would double-normalise a
    // full-size chunk and mis-scale a chunked vector.
}

// x = R^-1 y = S H y / sqrt(d): Hadamard FIRST, then the same sign flips.
inline void turbo_unrotate(const f32 *y, f32 *x, i64 d, uint32_t seed) {
    std::memcpy(x, y, static_cast<size_t>(d) * sizeof(f32));
    turbo_hadamard(x, d);
    uint64_t s = turbo_mix(seed);
    for (i64 j = 0; j < d; j++) {
        if ((j & 63) == 0) s = turbo_mix(s);
        const f32 sign = ((s >> (j & 63)) & 1ull) ? -1.0f : 1.0f;
        x[j] *= sign;
    }
}

inline i32 turbo_nearest(f32 t) {
    // 8 levels is a few compares; a binary search on 7 boundaries would not be
    // faster and would branch just as unpredictably.
    int idx = 0;
    while (idx < kTurboLevels - 1 && t > kTurboBoundary[idx]) idx++;
    return idx;
}

// Quantizes ONE head-dim vector. `rot` is scratch of `d` floats.
inline void turbo_quant_row(const f32 *x, i64 d, uint32_t seed, u8 *row,
                            f32 *rot) {
    turbo_rotate(x, rot, d, seed);
    f64 n2 = 0.0;
    for (i64 j = 0; j < d; j++) n2 += static_cast<f64>(rot[j]) * rot[j];
    const f32 norm = static_cast<f32>(std::sqrt(n2));
    const f32 sigma = d > 0 ? norm / std::sqrt(static_cast<f32>(d)) : 0.0f;
    const f32 inv = sigma > 0.0f ? 1.0f / sigma : 0.0f;
    const size_t nbytes = (static_cast<size_t>(d) + 7) / 8 * 3;
    std::memset(row, 0, nbytes);
    uint32_t acc = 0;
    i32 in_group = 0;
    size_t out = 0;
    for (i64 j = 0; j < d; j++) {
        const i32 idx = turbo_nearest(rot[j] * inv);
        acc |= static_cast<uint32_t>(idx) << (3 * in_group);
        if (++in_group == 8) {
            row[out++] = static_cast<u8>(acc & 0xffu);
            row[out++] = static_cast<u8>((acc >> 8) & 0xffu);
            row[out++] = static_cast<u8>((acc >> 16) & 0xffu);
            acc = 0;
            in_group = 0;
        }
    }
    if (in_group) {
        row[out++] = static_cast<u8>(acc & 0xffu);
        row[out++] = static_cast<u8>((acc >> 8) & 0xffu);
        row[out++] = static_cast<u8>((acc >> 16) & 0xffu);
    }
    f32 sc = sigma;
    std::memcpy(row + nbytes, &sc, sizeof(f32));
}

inline f32 turbo_row_scale(const u8 *row, i64 d) {
    f32 s = 0.0f;
    std::memcpy(&s, row + (static_cast<size_t>(d) + 7) / 8 * 3, sizeof(f32));
    return s;
}

// ---------------------------------------------------------------------------
// rot4: Q4_0 block layout in the rotated basis.//
// The rotation does the job Bonsai's per-model mean-centring bias does, and
// more: after a Walsh-Hadamard rotation every coordinate has the same variance,
// so a 32-value block's max is set by the block's own spread and not by one
// outlier lane. That is what makes a per-32 scale safe to use here, and it is
// why this is the conservative default: block scales bound the damage of any
// one bad estimate, and a 16-level uniform grid needs no distributional
// assumption at all.
//
// Block = 16 bytes of nibbles + one f16 scale (18 B), 32 values, exactly the
// ggml Q4_0 geometry: d = max / -8, q = min(15, (int)(x/d + 8.5)), value =
// (q - 8) * d. d = 256 -> 8 blocks -> 144 B against f16's 512 (3.56x).
// ---------------------------------------------------------------------------

// IEEE half round-trip. Written out rather than pulled from a header because
// the engine has no f16 scalar helper and the block scale is the one place a
// silent conversion bug would read as "the quantizer is bad".
inline u16 f32_to_f16(f32 v) {
    u32 x;
    std::memcpy(&x, &v, 4);
    const u32 sign = (x >> 16) & 0x8000u;
    const i32 exp = static_cast<i32>((x >> 23) & 0xffu);
    u32 mant = x & 0x7fffffu;
    if (exp == 0xff) return static_cast<u16>(sign | 0x7c00u | (mant ? 0x200u : 0u));
    i32 e = exp - 127 + 15;
    if (e >= 0x1f) return static_cast<u16>(sign | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return static_cast<u16>(sign);
        mant |= 0x800000u;
        const i32 shift = 14 - e;
        u32 m = mant >> shift;
        const u32 rem = mant & ((1u << shift) - 1u);
        const u32 half = 1u << (shift - 1);
        if (rem > half || (rem == half && (m & 1u))) m++;
        return static_cast<u16>(sign | m);
    }
    u32 m = mant >> 13;
    const u32 rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (m & 1u))) {
        m++;
        if (m == 0x400u) {
            m = 0;
            if (++e >= 0x1f) return static_cast<u16>(sign | 0x7c00u);
        }
    }
    return static_cast<u16>(sign | (static_cast<u32>(e) << 10) | m);
}

inline f32 f16_to_f32(u16 h) {
    const u32 sign = static_cast<u32>(h & 0x8000u) << 16;
    const u32 exp = (h >> 10) & 0x1fu;
    u32 mant = h & 0x3ffu;
    u32 bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            i32 e = -1;
            do { e++; mant <<= 1; } while ((mant & 0x400u) == 0);
            mant &= 0x3ffu;
            bits = sign | (static_cast<u32>(127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    f32 v;
    std::memcpy(&v, &bits, 4);
    return v;
}

// turbo3's readers live after rot4 in this file (they are the older pair and
// the pk_* dispatch names both), so declare them before rot4's diagnostic arm
// reuses them.
inline void turbo_decode_row(const u8 *row, i64 d, f32 *y);
inline f32 turbo_dot(const f32 *q_rot, const u8 *row, i64 d);

// Row = one f16 scale + 12 code bytes (32 values x 3 bits) per 32-value block,
// i.e. 14 B/block, 112 B at d=256 against f16's 512 -> 4.57x.
//
// WHY 3 BITS AND NOT 4. This started as Q4_0's uniform 16-level grid at a
// per-32 max-derived scale, which measured *better* MSE than turbo3 (0.0089 vs
// 0.033) and still broke the model at the second token. The A/B that settled it
// kept this row layout and swapped in turbo3's codec: the text came back
// correct, so the layout, the stride and the rotation were never the problem --
// the uniform grid was. `step = max/7` spreads 15 levels over +-max, so the
// core of the distribution (where the mass and the attention energy are) lands
// on ~5 coarse levels while the tails get fine ones. Lloyd-Max puts its levels
// where the density is, and at 3 bits it costs only a megabyte per 32K tokens
// more than the 4-bit grid would.
inline size_t rot4_row_bytes(i64 d) {
    return static_cast<size_t>(d / 32) * 14u;
}

// Quantizes one head-dim vector as per-32-block Lloyd-Max in the rotated basis.
inline void rot4_quant_row(const f32 *x, i64 d, uint32_t seed, u8 *row,
                           f32 *rot) {
    turbo_rotate(x, rot, d, seed);
    size_t out = 0;
    for (i64 o = 0; o < d; o += 32) {
        f64 m2 = 0.0;
        for (i64 i = 0; i < 32; i++)
            m2 += static_cast<f64>(rot[o + i]) * rot[o + i];
        // Block RMS, not the block max: the codebook is Lloyd-Max for N(0,1),
        // so the scale it wants is the block's coordinate standard deviation.
        // A max-derived scale is inflated by E[max|N|]/sigma ~ 2.35 and coarsens
        // every level -- the defect that made the uniform grid unusable.
        const f32 s = static_cast<f32>(std::sqrt(m2 / 32.0));
        const f32 inv = s > 0.0f ? 1.0f / s : 0.0f;
        const u16 sh = f32_to_f16(s);
        std::memcpy(row + out, &sh, 2);
        out += 2;
        for (i64 g = 0; g < 4; g++) {
            u32 acc = 0;
            for (i64 b = 0; b < 8; b++)
                acc |= static_cast<u32>(turbo_nearest(
                           rot[o + g * 8 + b] * inv))
                       << (3 * b);
            row[out++] = static_cast<u8>(acc & 0xffu);
            row[out++] = static_cast<u8>((acc >> 8) & 0xffu);
            row[out++] = static_cast<u8>((acc >> 16) & 0xffu);
        }
    }
}

inline void rot4_decode_row(const u8 *row, i64 d, f32 *y) {
    size_t in = 0;
    for (i64 o = 0; o < d; o += 32) {
        u16 sh;
        std::memcpy(&sh, row + in, 2);
        in += 2;
        const f32 s = f16_to_f32(sh);
        for (i64 g = 0; g < 4; g++) {
            const u32 acc = static_cast<u32>(row[in]) |
                            (static_cast<u32>(row[in + 1]) << 8) |
                            (static_cast<u32>(row[in + 2]) << 16);
            in += 3;
            for (i64 b = 0; b < 8; b++)
                y[o + g * 8 + b] =
                    s * kTurboCentroid[(acc >> (3 * b)) & 7u];
        }
    }
}

// <q, stored> with q already rotated; see turbo_dot for why that is the score.
inline f32 rot4_dot(const f32 *q_rot, const u8 *row, i64 d) {
    f32 acc = 0.0f;
    size_t in = 0;
    for (i64 o = 0; o < d; o += 32) {
        u16 sh;
        std::memcpy(&sh, row + in, 2);
        in += 2;
        const f32 s = f16_to_f32(sh);
        for (i64 g = 0; g < 4; g++) {
            const u32 bits = static_cast<u32>(row[in]) |
                             (static_cast<u32>(row[in + 1]) << 8) |
                             (static_cast<u32>(row[in + 2]) << 16);
            in += 3;
            for (i64 b = 0; b < 8; b++)
                acc += q_rot[o + g * 8 + b] *
                       (s * kTurboCentroid[(bits >> (3 * b)) & 7u]);
        }
    }
    return acc;
}

// The row's values in the ROTATED basis (sigma * centroids). This is what both
// the score dot and the value accumulation consume; neither needs the vector
// rotated back.
inline void turbo_decode_row(const u8 *row, i64 d, f32 *y) {
    const f32 sigma = turbo_row_scale(row, d);
    size_t in = 0;
    i64 j = 0;
    while (j < d) {
        const uint32_t acc = static_cast<uint32_t>(row[in]) |
                             (static_cast<uint32_t>(row[in + 1]) << 8) |
                             (static_cast<uint32_t>(row[in + 2]) << 16);
        in += 3;
        for (i32 b = 0; b < 8 && j < d; b++, j++)
            y[j] = sigma * kTurboCentroid[(acc >> (3 * b)) & 7u];
    }
}

// <q, stored> where q is ALREADY rotated. Equal to <W^T q, x> = <q, x> for the
// orthogonal W, i.e. the true score, up to the quantizer's own error.
inline f32 turbo_dot(const f32 *q_rot, const u8 *row, i64 d) {
    const f32 sigma = turbo_row_scale(row, d);
    f32 acc = 0.0f;
    size_t in = 0;
    i64 j = 0;
    while (j < d) {
        const uint32_t bits = static_cast<uint32_t>(row[in]) |
                              (static_cast<uint32_t>(row[in + 1]) << 8) |
                              (static_cast<uint32_t>(row[in + 2]) << 16);
        in += 3;
        for (i32 b = 0; b < 8 && j < d; b++, j++)
            acc += q_rot[j] * (sigma * kTurboCentroid[(bits >> (3 * b)) & 7u]);
    }
    return acc;
}

// One switch for the two packed formats, so attention and the append have one
// body each instead of one per format. Declared last because it names both
// quantizers.
enum class KvPack : i32 { Turbo3 = 0, Rot4 = 1 };

inline size_t pk_row_bytes(i64 d, KvPack k) {
    return k == KvPack::Rot4 ? rot4_row_bytes(d) : turbo_row_bytes(d);
}

inline void pk_quant_row(const f32 *x, i64 d, uint32_t seed, KvPack k, u8 *row,
                         f32 *rot) {
    if (k == KvPack::Rot4) rot4_quant_row(x, d, seed, row, rot);
    else turbo_quant_row(x, d, seed, row, rot);
}

inline f32 pk_dot(const f32 *q_rot, const u8 *row, i64 d, KvPack k) {
    return k == KvPack::Rot4 ? rot4_dot(q_rot, row, d)
                             : turbo_dot(q_rot, row, d);
}

inline void pk_decode_row(const u8 *row, i64 d, KvPack k, f32 *y) {
    if (k == KvPack::Rot4) rot4_decode_row(row, d, y);
    else turbo_decode_row(row, d, y);
}

} // namespace krk
