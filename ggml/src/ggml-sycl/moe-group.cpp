//
// MIT license
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "moe-group.hpp"
#include "iq4nl.hpp"
#include "dequantize.hpp"

#if defined(GGML_SYCL_MOE_DEVICE_ROUTING) && \
    (!defined(GGML_SYCL_XE_FAMILY_AOT) ||    \
     !(defined(GGML_SYCL_XE_FAMILY_XE_LP) || defined(GGML_SYCL_XE_FAMILY_XE_LPG) || \
       defined(GGML_SYCL_XE_FAMILY_XE_LPGPLUS) || defined(GGML_SYCL_XE_FAMILY_XE_HPG)))
#define GGML_SYCL_MOE_GROUP_XMX
#endif

#ifdef GGML_SYCL_MOE_GROUP_XMX

namespace mx = sycl::ext::oneapi::experimental::matrix;

static constexpr int MOE_BT = 32;
static constexpr int MOE_BN = 64;
static constexpr int MOE_KB = 2;

// Q4_0 expert in the per-expert reorder layout: qs[N][K/2] then d[N][K/32].
struct moe_dq_q4_0 {
    static constexpr int k_align = QK4_0 * MOE_KB;

    static __dpct_inline__ void block(const uint8_t * slice, int N, int K, int n, int kb, sycl::half2 lo[8],
                                      sycl::half2 hi[8]) {
        const int          nb = K / QK4_0;
        const sycl::uint4  v4 = *(const sycl::uint4 *) (slice + (size_t) n * (K / 2) + (size_t) kb * (QK4_0 / 2));
        const float        d  = *((const sycl::half *) (slice + (size_t) N * (K / 2)) + (size_t) n * nb + kb);
        const uint32_t     w[4] = { v4.x(), v4.y(), v4.z(), v4.w() };
#pragma unroll
        for (int jp = 0; jp < 8; ++jp) {
            const uint32_t v = (w[jp / 2] >> (16 * (jp % 2))) & 0xFFFF;
            lo[jp] = sycl::half2(d * ((int) (v & 0xF) - 8), d * ((int) ((v >> 8) & 0xF) - 8));
            hi[jp] = sycl::half2(d * ((int) ((v >> 4) & 0xF) - 8), d * ((int) (v >> 12) - 8));
        }
    }
};

// 32 bytes at p (16-byte aligned) as eight words.
static __dpct_inline__ void moe_load32(const uint8_t * p, uint32_t w[8]) {
    const sycl::uint4 a = *(const sycl::uint4 *) p;
    const sycl::uint4 b = *(const sycl::uint4 *) (p + 16);
    w[0] = a.x(); w[1] = a.y(); w[2] = a.z(); w[3] = a.w();
    w[4] = b.x(); w[5] = b.y(); w[6] = b.z(); w[7] = b.w();
}

static __dpct_inline__ uint32_t moe_byte(const uint32_t w[8], int j) {
    return (w[j / 4] >> (8 * (j % 4))) & 0xFF;
}

static __dpct_inline__ void moe_pack(const float v[32], sycl::half2 lo[8], sycl::half2 hi[8]) {
#pragma unroll
    for (int jp = 0; jp < 8; ++jp) {
        lo[jp] = sycl::half2(v[2 * jp], v[2 * jp + 1]);
        hi[jp] = sycl::half2(v[16 + 2 * jp], v[16 + 2 * jp + 1]);
    }
}

// Q8_0 expert in the per-expert reorder layout: qs[N][K] then d[N][K/32].
struct moe_dq_q8_0 {
    static constexpr int k_align = QK8_0 * MOE_KB;

    static __dpct_inline__ void block(const uint8_t * slice, int N, int K, int n, int kb, sycl::half2 lo[8],
                                      sycl::half2 hi[8]) {
        uint32_t w[8];
        moe_load32(slice + (size_t) n * K + (size_t) kb * QK8_0, w);
        const float d = *((const sycl::half *) (slice + (size_t) N * K) + (size_t) n * (K / QK8_0) + kb);
        float       v[32];
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            v[j] = d * (int8_t) moe_byte(w, j);
        }
        moe_pack(v, lo, hi);
    }
};

// IQ4_NL expert in the per-expert reorder layout (the Q4_0 one): qs[N][K/2] then d[N][K/32].
struct moe_dq_iq4_nl {
    static constexpr int k_align = QK4_NL * MOE_KB;

    static __dpct_inline__ void block(const uint8_t * slice, int N, int K, int n, int kb, sycl::half2 lo[8],
                                      sycl::half2 hi[8]) {
        const int         nb = K / QK4_NL;
        const sycl::uint4 v4 = *(const sycl::uint4 *) (slice + (size_t) n * (K / 2) + (size_t) kb * (QK4_NL / 2));
        const float       d  = *((const sycl::half *) (slice + (size_t) N * (K / 2)) + (size_t) n * nb + kb);
        const uint32_t    w[4] = { v4.x(), v4.y(), v4.z(), v4.w() };
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t l = iq4nl_lookup4(w[i] & 0x0F0F0F0F);
            const uint32_t h = iq4nl_lookup4((w[i] >> 4) & 0x0F0F0F0F);
            lo[2 * i + 0] = sycl::half2(d * (int8_t) (l >> 0), d * (int8_t) (l >> 8));
            lo[2 * i + 1] = sycl::half2(d * (int8_t) (l >> 16), d * (int8_t) (l >> 24));
            hi[2 * i + 0] = sycl::half2(d * (int8_t) (h >> 0), d * (int8_t) (h >> 8));
            hi[2 * i + 1] = sycl::half2(d * (int8_t) (h >> 16), d * (int8_t) (h >> 24));
        }
    }
};

// IQ3_S expert in the per-expert reorder layout: qs[nb][64], qh[nb][8], signs and scales[nb][36], d[nb].
struct moe_dq_iq3_s {
    static constexpr int k_align = QK_K;

    static __dpct_inline__ void block(const uint8_t * slice, int N, int K, int n, int kb, sycl::half2 lo[8],
                                      sycl::half2 hi[8]) {
        constexpr int   ss_size = QK_K / 8 + QK_K / 64;
        const int       nbk     = K / QK_K;
        const int       nblocks = N * nbk;
        const int       b       = n * nbk + kb / (QK_K / 32);
        const int       ib      = kb % (QK_K / 32);
        const sycl::uint2 q8    = *(const sycl::uint2 *) (slice + (size_t) b * (QK_K / 4) + 8 * ib);
        const uint32_t  qh      = slice[(size_t) nblocks * (QK_K / 4) + (size_t) b * (QK_K / 32) + ib];
        const uint8_t * ss      = slice + (size_t) nblocks * (QK_K / 4 + QK_K / 32) + (size_t) b * ss_size;
        const uint32_t  signs   = *(const uint32_t *) (ss + 4 * ib);
        const float     dall    = *((const sycl::half *) (slice + (size_t) nblocks * (QK_K / 4 + QK_K / 32 + ss_size)) + b);
        const float     d       = dall * (1 + 2 * ((ss[QK_K / 8 + ib / 2] >> 4 * (ib % 2)) & 0xf));
        const uint32_t  qs[2]   = { q8.x(), q8.y() };
        float           v[32];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t i0 = (qs[l / 2] >> (16 * (l % 2))) & 0xFF;
            const uint32_t i1 = (qs[l / 2] >> (16 * (l % 2) + 8)) & 0xFF;
            const uint32_t g1 = iq3s_grid[i0 | ((qh << (8 - 2 * l)) & 256)];
            const uint32_t g2 = iq3s_grid[i1 | ((qh << (7 - 2 * l)) & 256)];
            const uint32_t sg = signs >> (8 * l);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                v[8 * l + j]     = d * (float) ((g1 >> (8 * j)) & 0xFF) * ((sg >> j) & 1 ? -1.f : 1.f);
                v[8 * l + 4 + j] = d * (float) ((g2 >> (8 * j)) & 0xFF) * ((sg >> (4 + j)) & 1 ? -1.f : 1.f);
            }
        }
#pragma unroll
        for (int jp = 0; jp < 8; ++jp) {
            lo[jp] = sycl::half2(v[2 * jp], v[2 * jp + 1]);
            hi[jp] = sycl::half2(v[16 + 2 * jp], v[16 + 2 * jp + 1]);
        }
    }
};

// Q4_K expert, per-expert reorder layout qs[nb][128], scales[nb][12], dm[nb]. Group g of a super-block is the
// low (even g) or high (odd g) nibbles of its 32-byte quarter g / 2, with one scale and min.
struct moe_dq_q4_K {
    static constexpr int k_align = QK_K;

    static __dpct_inline__ void block(const uint8_t * slice, int N, int K, int n, int kb, sycl::half2 lo[8],
                                      sycl::half2 hi[8]) {
        const int nblocks = N * (K / QK_K);
        const int b       = n * (K / QK_K) + kb / 8;
        const int g       = kb % 8;
        uint32_t  w[8];
        moe_load32(slice + (size_t) b * (QK_K / 2) + 32 * (g / 2), w);
        const ggml_half2 dm = *(const ggml_half2 *) (slice + (size_t) nblocks * (QK_K / 2 + K_SCALE_SIZE) + (size_t) b * 4);
        uint8_t sc, mn;
        get_scale_min_k4(g, slice + (size_t) nblocks * (QK_K / 2) + (size_t) b * K_SCALE_SIZE, sc, mn);
        const float d1 = (float) dm.x() * sc;
        const float m1 = (float) dm.y() * mn;
        const int   sh = 4 * (g % 2);
        float       v[32];
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            v[j] = d1 * ((moe_byte(w, j) >> sh) & 0xF) - m1;
        }
        moe_pack(v, lo, hi);
    }
};

// Q5_K expert, per-expert reorder layout qs[nb][128], qh[nb][32], scales[nb][12], dm[nb].
struct moe_dq_q5_K {
    static constexpr int k_align = QK_K;

    static __dpct_inline__ void block(const uint8_t * slice, int N, int K, int n, int kb, sycl::half2 lo[8],
                                      sycl::half2 hi[8]) {
        const int nblocks = N * (K / QK_K);
        const int b       = n * (K / QK_K) + kb / 8;
        const int g       = kb % 8;
        uint32_t  w[8], h[8];
        moe_load32(slice + (size_t) b * (QK_K / 2) + 32 * (g / 2), w);
        moe_load32(slice + (size_t) nblocks * (QK_K / 2) + (size_t) b * (QK_K / 8), h);
        const uint8_t *  base_sc = slice + (size_t) nblocks * (QK_K / 2 + QK_K / 8);
        const ggml_half2 dm = *(const ggml_half2 *) (base_sc + (size_t) nblocks * K_SCALE_SIZE + (size_t) b * 4);
        uint8_t sc, mn;
        get_scale_min_k4(g, base_sc + (size_t) b * K_SCALE_SIZE, sc, mn);
        const float d1 = (float) dm.x() * sc;
        const float m1 = (float) dm.y() * mn;
        const int   sh = 4 * (g % 2);
        float       v[32];
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            v[j] = d1 * (((moe_byte(w, j) >> sh) & 0xF) + ((moe_byte(h, j) >> g) & 1 ? 16 : 0)) - m1;
        }
        moe_pack(v, lo, hi);
    }
};

// Q6_K expert, per-expert reorder layout ql[nb][128], qh[nb][64], scales[nb][16], d[nb]. Group g is quarter
// g % 4 of half g / 4, with a new scale every 16 weights.
struct moe_dq_q6_K {
    static constexpr int k_align = QK_K;

    static __dpct_inline__ void block(const uint8_t * slice, int N, int K, int n, int kb, sycl::half2 lo[8],
                                      sycl::half2 hi[8]) {
        const int nblocks = N * (K / QK_K);
        const int b       = n * (K / QK_K) + kb / 8;
        const int ip      = (kb % 8) / 4;
        const int q4      = kb % 4;
        uint32_t  w[8], h[8];
        moe_load32(slice + (size_t) b * (QK_K / 2) + 64 * ip + 32 * (q4 & 1), w);
        moe_load32(slice + (size_t) nblocks * (QK_K / 2) + (size_t) b * (QK_K / 4) + 32 * ip, h);
        const int8_t * sc = (const int8_t *) (slice + (size_t) nblocks * (QK_K / 2 + QK_K / 4) + (size_t) b * (QK_K / 16) +
                                              8 * ip + 2 * q4);
        const float    d  = *((const sycl::half *) (slice + (size_t) nblocks * (QK_K / 2 + QK_K / 4 + QK_K / 16)) + b);
        const float    s0 = d * sc[0];
        const float    s1 = d * sc[1];
        const int      sh = 4 * (q4 >> 1);
        float          v[32];
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            const int q = (int) (((moe_byte(w, j) >> sh) & 0xF) | (((moe_byte(h, j) >> (2 * q4)) & 3) << 4)) - 32;
            v[j]        = (j < 16 ? s0 : s1) * q;
        }
        moe_pack(v, lo, hi);
    }
};

// Q2_K expert, per-expert reorder layout qs[nb][64], scales[nb][16], dm[nb]. Group g uses bits 2 * (g % 4) of
// the 32-byte half g / 4, with a new scale and min every 16 weights.
struct moe_dq_q2_K {
    static constexpr int k_align = QK_K;

    static __dpct_inline__ void block(const uint8_t * slice, int N, int K, int n, int kb, sycl::half2 lo[8],
                                      sycl::half2 hi[8]) {
        const int nblocks = N * (K / QK_K);
        const int b       = n * (K / QK_K) + kb / 8;
        const int hh      = (kb % 8) / 4;
        const int j4      = kb % 4;
        uint32_t  w[8];
        moe_load32(slice + (size_t) b * (QK_K / 4) + 32 * hh, w);
        const uint8_t *  scales = slice + (size_t) nblocks * (QK_K / 4) + (size_t) b * (QK_K / 16) + 8 * hh + 2 * j4;
        const ggml_half2 dm     = *(const ggml_half2 *) (slice + (size_t) nblocks * (QK_K / 4 + QK_K / 16) + (size_t) b * 4);
        const float      dall   = dm.x();
        const float      dmin   = dm.y();
        const float      d0 = dall * (scales[0] & 0xF), m0 = dmin * (scales[0] >> 4);
        const float      d1 = dall * (scales[1] & 0xF), m1 = dmin * (scales[1] >> 4);
        float            v[32];
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            const int q = (moe_byte(w, j) >> (2 * j4)) & 3;
            v[j]        = j < 16 ? d0 * q - m0 : d1 * q - m1;
        }
        moe_pack(v, lo, hi);
    }
};

// Q3_K expert, per-expert reorder layout qs[nb][64], hmask[nb][32], scales[nb][12], d[nb].
struct moe_dq_q3_K {
    static constexpr int k_align = QK_K;

    static __dpct_inline__ int scale(const uint8_t * scales, int is) {
        const int us = is < 4 ? (scales[is - 0] & 0xF) | (((scales[is + 8] >> 0) & 3) << 4) :
                       is < 8 ? (scales[is - 0] & 0xF) | (((scales[is + 4] >> 2) & 3) << 4) :
                       is < 12 ? (scales[is - 8] >> 4) | (((scales[is + 0] >> 4) & 3) << 4) :
                                 (scales[is - 8] >> 4) | (((scales[is - 4] >> 6) & 3) << 4);
        return us - 32;
    }

    static __dpct_inline__ void block(const uint8_t * slice, int N, int K, int n, int kb, sycl::half2 lo[8],
                                      sycl::half2 hi[8]) {
        const int nblocks = N * (K / QK_K);
        const int b       = n * (K / QK_K) + kb / 8;
        const int nn      = (kb % 8) / 4;
        const int j4      = kb % 4;
        uint32_t  w[8], h[8];
        moe_load32(slice + (size_t) b * (QK_K / 4) + 32 * nn, w);
        moe_load32(slice + (size_t) nblocks * (QK_K / 4) + (size_t) b * (QK_K / 8), h);
        const uint8_t * scales = slice + (size_t) nblocks * (QK_K / 4 + QK_K / 8) + (size_t) b * 12;
        const float     d_all  = *((const sycl::half *) (slice + (size_t) nblocks * (QK_K / 4 + QK_K / 8 + 12)) + b);
        const int       is     = 8 * nn + 2 * j4;
        const float     s0     = d_all * scale(scales, is);
        const float     s1     = d_all * scale(scales, is + 1);
        const int       mk     = 4 * nn + j4;
        float           v[32];
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            const int q = (int) ((moe_byte(w, j) >> (2 * j4)) & 3) - ((moe_byte(h, j) >> mk) & 1 ? 0 : 4);
            v[j]        = (j < 16 ? s0 : s1) * q;
        }
        moe_pack(v, lo, hi);
    }
};

// IQ4_XS expert in the standard layout, one 32-element sub-block per call.
struct moe_dq_iq4_xs {
    static constexpr int k_align = QK_K;

    static __dpct_inline__ void block(const uint8_t * slice, int /* N */, int K, int n, int kb, sycl::half2 lo[8],
                                      sycl::half2 hi[8]) {
        const int            ib = kb % (QK_K / 32);
        const block_iq4_xs * x  = (const block_iq4_xs *) slice + (size_t) n * (K / QK_K) + kb / (QK_K / 32);
        const float d = (float) x->d * ((((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4)) - 32);
        const sycl::uint2 * q = (const sycl::uint2 *) (x->qs + 16 * ib);
        const sycl::uint2   a = q[0];
        const sycl::uint2   b = q[1];
        const uint32_t      w[4] = { a.x(), a.y(), b.x(), b.y() };
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t l = iq4nl_lookup4(w[i] & 0x0F0F0F0F);
            const uint32_t h = iq4nl_lookup4((w[i] >> 4) & 0x0F0F0F0F);
            lo[2 * i + 0] = sycl::half2(d * (int8_t) (l >> 0), d * (int8_t) (l >> 8));
            lo[2 * i + 1] = sycl::half2(d * (int8_t) (l >> 16), d * (int8_t) (l >> 24));
            hi[2 * i + 0] = sycl::half2(d * (int8_t) (h >> 0), d * (int8_t) (h >> 8));
            hi[2 * i + 1] = sycl::half2(d * (int8_t) (h >> 16), d * (int8_t) (h >> 24));
        }
    }
};

// One work-group computes MOE_BT routed rows (one tile of one expert) x MOE_BN outputs. Each step dequantizes
// MOE_KB 32-wide K blocks of the expert's weights into SLM as VNNI-packed fp16 and multiplies them with the
// gathered fp16 rows on XMX. Tiles past *n_tiles exit, so the grid can be sized from host-known bounds only.
template <typename dq>
static void moe_grouped_gemm(const uint8_t * W, size_t expert_stride, const sycl::half * Xh, const int * tile_expert,
                             const int * tile_row0, const int * tile_rows, const int * n_tiles, const int * sorted_src,
                             float * dst, int N, int K, int max_tiles, int n_used, size_t dst_slot_stride,
                             size_t dst_token_stride, dpct::queue_ptr stream) {
    constexpr int TM = 8, TN = 16, TK = 16;
    constexpr int SG_N     = MOE_BN / TN;
    constexpr int SG_T     = 2;
    constexpr int T_PER_SG = MOE_BT / SG_T;
    constexpr int MT       = T_PER_SG / TM;
    constexpr int WG       = SG_N * SG_T * WARP_SIZE;
    constexpr int KS       = 32 * MOE_KB;
    static_assert(WARP_SIZE == 16, "the grouped MoE kernel uses 16-wide XMX");

    const int nb32       = K / 32;
    const int row_groups = (N + MOE_BN - 1) / MOE_BN;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<sycl::half2, 1> b_tile(sycl::range<1>(KS / 2 * MOE_BN), cgh);
        sycl::local_accessor<float, 1>       c_tile(sycl::range<1>(MOE_BT * MOE_BN), cgh);
        cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(max_tiles, row_groups * WG), sycl::range<2>(1, WG)),
                         [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const int tile = it.get_group(0);
            if (tile >= *n_tiles) {
                return;
            }
            const int       e     = tile_expert[tile];
            const int       r0    = tile_row0[tile];
            const int       nrows = tile_rows[tile];
            const int       n0    = it.get_group(1) * MOE_BN;
            const auto      sg    = it.get_sub_group();
            const int       lid   = it.get_local_id(1);
            const int       sgid  = lid / WARP_SIZE;
            const int       sg_n  = sgid % SG_N;
            const int       sg_t  = sgid / SG_N;
            const uint8_t * slice = W + (size_t) e * expert_stride;

            mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> acc[MT];
            for (int m = 0; m < MT; ++m) {
                mx::joint_matrix_fill(sg, acc[m], 0.0f);
            }
            auto b_ptr = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                (sycl::half *) &b_tile[0]);

            for (int kb0 = 0; kb0 < nb32; kb0 += MOE_KB) {
                for (int i = lid; i < MOE_BN * MOE_KB; i += WG) {
                    const int   nl  = i % MOE_BN;
                    const int   kbl = i / MOE_BN;
                    sycl::half2 lo[8], hi[8];
                    dq::block(slice, N, K, sycl::min(n0 + nl, N - 1), kb0 + kbl, lo, hi);
#pragma unroll
                    for (int jp = 0; jp < 8; ++jp) {
                        b_tile[(kbl * 16 + jp) * MOE_BN + nl]     = lo[jp];
                        b_tile[(kbl * 16 + 8 + jp) * MOE_BN + nl] = hi[jp];
                    }
                }
                sycl::group_barrier(it.get_group());
#pragma unroll
                for (int ks = 0; ks < KS / TK; ++ks) {
                    mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, TK, TN, mx::layout::ext_intel_packed> mb;
                    mx::joint_matrix_load(sg, mb, b_ptr + (ks * TK / 2) * (2 * MOE_BN) + 2 * (sg_n * TN), 2 * MOE_BN);
#pragma unroll
                    for (int m = 0; m < MT; ++m) {
                        mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, TM, TK, mx::layout::row_major> ma;
                        const sycl::half * ap = Xh + (size_t) (r0 + sg_t * T_PER_SG + m * TM) * K + kb0 * 32 + ks * TK;
                        mx::joint_matrix_load(
                            sg, ma,
                            sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(ap),
                            K);
                        mx::joint_matrix_mad(sg, acc[m], ma, mb, acc[m]);
                    }
                }
                sycl::group_barrier(it.get_group());
            }
            auto c_ptr = c_tile.template get_multi_ptr<sycl::access::decorated::no>();
            for (int m = 0; m < MT; ++m) {
                mx::joint_matrix_store(sg, acc[m], c_ptr + (sg_t * T_PER_SG + m * TM) * MOE_BN + sg_n * TN, MOE_BN,
                                       mx::layout::row_major);
            }
            sycl::group_barrier(it.get_group());
            for (int i = lid; i < MOE_BT * MOE_BN; i += WG) {
                const int t  = i / MOE_BN;
                const int nl = i % MOE_BN;
                if (t < nrows && n0 + nl < N) {
                    const int src = sorted_src[r0 + t];
                    dst[(size_t) (src / n_used) * dst_token_stride + (size_t) (src % n_used) * dst_slot_stride + n0 + nl] =
                        c_tile[i];
                }
            }
        });
    });
}

#endif

bool ggml_sycl_moe_grouped_supported(int device, const ggml_tensor * src0, const ggml_tensor * src1) {
#ifdef GGML_SYCL_MOE_GROUP_XMX
    const sycl_xe_family family = ggml_sycl_info().devices[device].hw_info.xe_family;
    if (get_xe_family_caps(family).dpas_n != 16 || !is_xe_family_compiled(family)) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || !ggml_is_contiguous(src1) || src0->ne[3] != 1) {
        return false;
    }
    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            return src0->ne[0] % moe_dq_q4_0::k_align == 0 && src0->nb[2] % 16 == 0;
        case GGML_TYPE_IQ4_XS:
            return src0->ne[0] % moe_dq_iq4_xs::k_align == 0 && src0->nb[2] % 8 == 0;
        case GGML_TYPE_IQ4_NL:
            return src0->ne[0] % moe_dq_iq4_nl::k_align == 0 && src0->nb[2] % 16 == 0;
        case GGML_TYPE_Q8_0:
            return src0->ne[0] % moe_dq_q8_0::k_align == 0 && src0->nb[2] % 16 == 0;
        case GGML_TYPE_IQ3_S:
            return src0->ne[0] % moe_dq_iq3_s::k_align == 0 && src0->nb[2] % 8 == 0;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return src0->ne[0] % QK_K == 0 && src0->nb[2] % 16 == 0;
        default:
            return false;
    }
#else
    GGML_UNUSED(device);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    return false;
#endif
}

bool ggml_sycl_moe_grouped(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                           const ggml_tensor * ids, ggml_tensor * dst) {
#ifdef GGML_SYCL_MOE_GROUP_XMX
    const int64_t n_as     = src0->ne[2];
    const int     N        = (int) src0->ne[1];
    const int     K        = (int) src0->ne[0];
    const int     n_used   = (int) ids->ne[0];
    const int     n_tokens = (int) ids->ne[1];
    const int64_t ne11     = src1->ne[1];
    const int     R        = n_tokens * n_used;

    int scan_wg = 1;
    while (scan_wg < n_as) {
        scan_wg *= 2;
    }
    if (scan_wg > (int) ggml_sycl_info().max_work_group_sizes[ctx.device] || ids->nb[0] != sizeof(int32_t) ||
        (ne11 != 1 && ne11 != n_used) || src1->ne[2] != n_tokens || dst->type != GGML_TYPE_F32) {
        return false;
    }

    const queue_ptr stream    = ctx.stream();
    const int       max_tiles = (R + MOE_BT - 1) / MOE_BT + (int) n_as;

    ggml_sycl_pool_alloc<int>        counts(ctx.pool(), n_as);
    ggml_sycl_pool_alloc<int>        cursor(ctx.pool(), n_as);
    ggml_sycl_pool_alloc<int>        tiles(ctx.pool(), (size_t) 3 * max_tiles + 1);
    ggml_sycl_pool_alloc<int>        sorted_src(ctx.pool(), R);
    ggml_sycl_pool_alloc<sycl::half> xh(ctx.pool(), (size_t) (R + MOE_BT) * K);

    int *        cnt       = counts.get();
    int *        cur       = cursor.get();
    int *        t_expert  = tiles.get();
    int *        t_row0    = t_expert + max_tiles;
    int *        t_rows    = t_row0 + max_tiles;
    int *        n_tiles   = t_rows + max_tiles;
    int *        srt       = sorted_src.get();
    sycl::half * x16       = xh.get();
    const char * ids_d     = (const char *) ids->data;
    const size_t ids_nb1   = ids->nb[1];
    const int    n_as_i    = (int) n_as;

    stream->memset(cnt, 0, n_as * sizeof(int));
    stream->parallel_for(sycl::range<1>(R), [=](sycl::id<1> i) {
        const int e = *(const int32_t *) (ids_d + (i / n_used) * ids_nb1 + (i % n_used) * sizeof(int32_t));
        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> c(cnt[e]);
        c.fetch_add(1);
    });
    stream->parallel_for(sycl::nd_range<1>(scan_wg, scan_wg), [=](sycl::nd_item<1> it) {
        const int e    = it.get_local_id(0);
        const int c    = e < n_as_i ? cnt[e] : 0;
        const int nt   = (c + MOE_BT - 1) / MOE_BT;
        const int off  = sycl::exclusive_scan_over_group(it.get_group(), c, sycl::plus<int>());
        const int toff = sycl::exclusive_scan_over_group(it.get_group(), nt, sycl::plus<int>());
        if (e < n_as_i) {
            cur[e] = off;
            for (int j = 0; j < nt; ++j) {
                t_expert[toff + j] = e;
                t_row0[toff + j]   = off + j * MOE_BT;
                t_rows[toff + j]   = sycl::min(MOE_BT, c - j * MOE_BT);
            }
        }
        if (e == scan_wg - 1) {
            *n_tiles = toff + nt;
        }
    });
    stream->parallel_for(sycl::range<1>(R), [=](sycl::id<1> i) {
        const int e = *(const int32_t *) (ids_d + (i / n_used) * ids_nb1 + (i % n_used) * sizeof(int32_t));
        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> c(cur[e]);
        srt[c.fetch_add(1)] = (int) i;
    });

    const char * src1_d   = (const char *) src1->data;
    const size_t nb11     = ne11 == 1 ? 0 : src1->nb[1];
    const size_t nb12     = src1->nb[2];
    const int    k8       = K / 8;
    stream->parallel_for(sycl::range<1>((size_t) R * k8), [=](sycl::id<1> idx) {
        const int    row = idx / k8;
        const int    c   = idx % k8;
        const int    src = srt[row];
        const float * xr = (const float *) (src1_d + (size_t) (src / n_used) * nb12 + (size_t) (src % n_used) * nb11) + c * 8;
        const sycl::vec<float, 8> v = *(const sycl::vec<float, 8> *) xr;
        *(sycl::vec<sycl::half, 8> *) (x16 + (size_t) row * K + c * 8) = v.convert<sycl::half, sycl::rounding_mode::rte>();
    });

    const uint8_t * W          = (const uint8_t *) src0->data;
    const size_t    slot_st    = dst->nb[1] / sizeof(float);
    const size_t    token_st   = dst->nb[2] / sizeof(float);
    float *         dst_d      = (float *) dst->data;
    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            moe_grouped_gemm<moe_dq_q4_0>(W, src0->nb[2], x16, t_expert, t_row0, t_rows, n_tiles, srt, dst_d, N, K,
                                          max_tiles, n_used, slot_st, token_st, stream);
            break;
        case GGML_TYPE_Q8_0:
            moe_grouped_gemm<moe_dq_q8_0>(W, src0->nb[2], x16, t_expert, t_row0, t_rows, n_tiles, srt, dst_d, N, K,
                                          max_tiles, n_used, slot_st, token_st, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            moe_grouped_gemm<moe_dq_iq4_nl>(W, src0->nb[2], x16, t_expert, t_row0, t_rows, n_tiles, srt, dst_d, N, K,
                                            max_tiles, n_used, slot_st, token_st, stream);
            break;
        case GGML_TYPE_IQ3_S:
            moe_grouped_gemm<moe_dq_iq3_s>(W, src0->nb[2], x16, t_expert, t_row0, t_rows, n_tiles, srt, dst_d, N, K,
                                           max_tiles, n_used, slot_st, token_st, stream);
            break;
        case GGML_TYPE_Q2_K:
            moe_grouped_gemm<moe_dq_q2_K>(W, src0->nb[2], x16, t_expert, t_row0, t_rows, n_tiles, srt,
                                                         dst_d, N, K, max_tiles, n_used, slot_st, token_st, stream);
            break;
        case GGML_TYPE_Q3_K:
            moe_grouped_gemm<moe_dq_q3_K>(W, src0->nb[2], x16, t_expert, t_row0, t_rows, n_tiles, srt,
                                                         dst_d, N, K, max_tiles, n_used, slot_st, token_st, stream);
            break;
        case GGML_TYPE_Q4_K:
            moe_grouped_gemm<moe_dq_q4_K>(W, src0->nb[2], x16, t_expert, t_row0, t_rows, n_tiles, srt,
                                                         dst_d, N, K, max_tiles, n_used, slot_st, token_st, stream);
            break;
        case GGML_TYPE_Q5_K:
            moe_grouped_gemm<moe_dq_q5_K>(W, src0->nb[2], x16, t_expert, t_row0, t_rows, n_tiles, srt,
                                                         dst_d, N, K, max_tiles, n_used, slot_st, token_st, stream);
            break;
        case GGML_TYPE_Q6_K:
            moe_grouped_gemm<moe_dq_q6_K>(W, src0->nb[2], x16, t_expert, t_row0, t_rows, n_tiles, srt,
                                                         dst_d, N, K, max_tiles, n_used, slot_st, token_st, stream);
            break;
        default:
            moe_grouped_gemm<moe_dq_iq4_xs>(W, src0->nb[2], x16, t_expert, t_row0, t_rows, n_tiles, srt, dst_d, N, K,
                                            max_tiles, n_used, slot_st, token_st, stream);
            break;
    }
    return true;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(ids);
    GGML_UNUSED(dst);
    return false;
#endif
}
