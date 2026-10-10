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

#ifndef GGML_SYCL_WDECOMP_LANES_HPP
#define GGML_SYCL_WDECOMP_LANES_HPP

#include "dequantize.hpp"

// Per-type decoders shared by the weight-decompression conversion and the grouped MoE kernel. Each one reads the
// 8 weights [8l, 8l + 8) of an nrows x ncols tensor holding k = nrows * ncols weights, in the reorder layout for
// the K-quants, IQ1-IQ3 types and IQ4_NL, and in the standard layout for IQ4_XS.
// The 8 weights [8l, 8l + 8) of a lane: weight m is s * q[m] + b.
struct wdecomp_lane {
    float s;
    float b;
    int   q[8];
};

#if QK_K == 256

static __dpct_inline__ wdecomp_lane wdecomp_q2_K(const uint8_t * base, int64_t l, int64_t k) {
    const int64_t n_blocks = k / QK_K;
    const int64_t i        = l / (QK_K / 8);
    const int     p        = 8 * (l % (QK_K / 8));
    const int     h        = p / 128;
    const int     j        = (p % 128) / 32;
    const int     s0       = p % 32;

    const uint32_t *   qs     = (const uint32_t *) (base + i * (QK_K / 4) + 32 * h + s0);
    const uint8_t *    scales = base + n_blocks * (QK_K / 4) + i * (QK_K / 16);
    const ggml_half2 * dm     = reinterpret_cast<const ggml_half2 *>(base + n_blocks * (QK_K / 4) + n_blocks * (QK_K / 16) + i * sizeof(ggml_half2));

    const uint8_t sc = scales[8 * h + 2 * j + s0 / 16];
    const float   dall = (*dm)[0];
    const float   dmin = (*dm)[1];

    wdecomp_lane r;
    r.s = dall * (sc & 0xF);
    r.b = -(dmin * (sc >> 4));
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        r.q[m] = (byte8(qs[0], qs[1], m) >> (2 * j)) & 3;
    }
    return r;
}

static __dpct_inline__ wdecomp_lane wdecomp_q3_K(const uint8_t * base, int64_t l, int64_t k) {
    const int64_t n_blocks = k / QK_K;
    const int64_t i        = l / (QK_K / 8);
    const int     p        = 8 * (l % (QK_K / 8));
    const int     n        = p / 128;
    const int     j        = (p % 128) / 32;
    const int     l0       = p % 32;

    const uint32_t * q      = (const uint32_t *) (base + i * (QK_K / 4) + 32 * n + l0);
    const uint32_t * hm     = (const uint32_t *) (base + n_blocks * (QK_K / 4) + i * (QK_K / 8) + l0);
    const uint8_t *  scales = base + n_blocks * (QK_K / 4) + n_blocks * (QK_K / 8) + i * 12;
    const float      d_all  = static_cast<float>(*reinterpret_cast<const ggml_half *>(
        base + n_blocks * (QK_K / 4) + n_blocks * (QK_K / 8) + n_blocks * 12 + i * sizeof(ggml_half)));

    const int     is = 8 * n + 2 * j + l0 / 16;
    const uint8_t mk = 1 << (4 * n + j);
    const uint8_t us = is < 4
        ? (scales[is - 0] & 0xF) | (((scales[is + 8] >> 0) & 3) << 4)
        : is < 8
            ? (scales[is - 0] & 0xF) | (((scales[is + 4] >> 2) & 3) << 4)
            : is < 12
                ? (scales[is - 8] >> 4) | (((scales[is + 0] >> 4) & 3) << 4)
                : (scales[is - 8] >> 4) | (((scales[is - 4] >> 6) & 3) << 4);

    wdecomp_lane r;
    r.s = d_all * (us - 32);
    r.b = 0.0f;
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        r.q[m] = (int) ((byte8(q[0], q[1], m) >> (2 * j)) & 3) - ((byte8(hm[0], hm[1], m) & mk) ? 0 : 4);
    }
    return r;
}

static __dpct_inline__ wdecomp_lane wdecomp_q4_K(const uint8_t * base, int64_t l, int64_t k) {
    const int64_t nb   = k / QK_K;
    const int64_t i    = l / (QK_K / 8);
    const int     p    = 8 * (l % (QK_K / 8));
    const int     il   = p / 64;
    const int     half = (p % 64) / 32;
    const int     pos  = p % 32;

    const uint32_t * qs         = (const uint32_t *) (base + i * (QK_K / 2) + 32 * il + pos);
    const uint8_t *  scales_ptr = base + nb * (QK_K / 2) + i * K_SCALE_SIZE;
    const ggml_half2 dm         = *reinterpret_cast<const ggml_half2 *>(base + nb * (QK_K / 2) + nb * K_SCALE_SIZE + i * sizeof(ggml_half2));

    uint8_t sc, mn;
    get_scale_min_k4(2 * il + half, scales_ptr, sc, mn);

    wdecomp_lane r;
    r.s = (float) dm.x() * sc;
    r.b = -((float) dm.y() * mn);
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        r.q[m] = byte8(qs[0] >> (4 * half), qs[1] >> (4 * half), m) & 0xF;
    }
    return r;
}

static __dpct_inline__ wdecomp_lane wdecomp_q5_K(const uint8_t * base, int64_t l, int64_t k) {
    const int64_t n_blocks = k / QK_K;
    const int64_t ib       = l / (QK_K / 8);
    const int     p        = 8 * (l % (QK_K / 8));
    const int     il       = p / 64;
    const int     half     = (p % 64) / 32;
    const int     pos      = p % 32;

    const uint32_t * ql         = (const uint32_t *) (base + ib * (QK_K / 2) + 32 * il + pos);
    const uint32_t * qh         = (const uint32_t *) (base + n_blocks * (QK_K / 2) + ib * (QK_K / 8) + pos);
    const uint8_t *  scales_ptr = base + n_blocks * (QK_K / 2) + n_blocks * (QK_K / 8) + ib * K_SCALE_SIZE;
    const ggml_half2 dm         = *reinterpret_cast<const ggml_half2 *>(
        base + n_blocks * (QK_K / 2) + n_blocks * (QK_K / 8) + n_blocks * K_SCALE_SIZE + ib * sizeof(ggml_half2));

    uint8_t sc, mn;
    get_scale_min_k4(2 * il + half, scales_ptr, sc, mn);
    const uint8_t hmk = 1 << (2 * il + half);

    wdecomp_lane r;
    r.s = (float) dm.x() * sc;
    r.b = -((float) dm.y() * mn);
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        r.q[m] = (byte8(ql[0] >> (4 * half), ql[1] >> (4 * half), m) & 0xF) + (byte8(qh[0], qh[1], m) & hmk ? 16 : 0);
    }
    return r;
}

static __dpct_inline__ wdecomp_lane wdecomp_q6_K(const uint8_t * base, int64_t l, int64_t k) {
    const int64_t n_blocks = k / QK_K;
    const int64_t ib       = l / (QK_K / 8);
    const int     p        = 8 * (l % (QK_K / 8));
    const int     ip       = p / 128;
    const int     q4       = (p % 128) / 32;
    const int     il       = p % 32;

    const uint32_t *  ql = (const uint32_t *) (base + ib * (QK_K / 2) + 64 * ip + il + 32 * (q4 & 1));
    const uint32_t *  qh = (const uint32_t *) (base + (QK_K / 2) * n_blocks + (QK_K / 4) * ib + 32 * ip + il);
    const int8_t *    sc = reinterpret_cast<const int8_t *>(base + (QK_K / 2) * n_blocks + (QK_K / 4) * n_blocks +
                                                            (QK_K / 16) * ib + 8 * ip + il / 16 + 2 * q4);
    const ggml_half * d  = (const ggml_half *) (base + ((QK_K / 2) + (QK_K / 4) + (QK_K / 16)) * n_blocks) + ib;

    const uint32_t la = ql[0] >> (4 * (q4 >> 1));
    const uint32_t lb = ql[1] >> (4 * (q4 >> 1));
    const uint32_t ha = qh[0] >> (2 * q4);
    const uint32_t hb = qh[1] >> (2 * q4);

    wdecomp_lane r;
    r.s = (float) *d * sc[0];
    r.b = 0.0f;
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        r.q[m] = (int) ((byte8(la, lb, m) & 0xF) | ((byte8(ha, hb, m) & 3) << 4)) - 32;
    }
    return r;
}

static __dpct_inline__ void wdecomp_signed_grid(wdecomp_lane & r, const uint8_t * grid, const uint8_t signs) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        r.q[j] = signs & kmask_iq2xs[j] ? -(int) grid[j] : (int) grid[j];
    }
}

static __dpct_inline__ wdecomp_lane wdecomp_iq2_xxs(const uint8_t * base, int64_t l, int64_t k) {
    const int64_t nb = k / QK_K;
    const int64_t i  = l / (QK_K / 8);
    const int     ib = (l % (QK_K / 8)) / 4;
    const int     il = l % 4;

    const uint8_t * q2    = base + i * (QK_K / 4) + 8 * ib;
    const uint32_t  aux32 = *reinterpret_cast<const uint32_t *>(q2 + 4);
    const float     dall  = *reinterpret_cast<const ggml_half *>(base + nb * (QK_K / 4) + i * sizeof(ggml_half));

    wdecomp_lane r;
    r.s = dall * (0.5f + (aux32 >> 28)) * 0.25f;
    r.b = 0.0f;
    wdecomp_signed_grid(r, (const uint8_t *) (iq2xxs_grid + q2[il]), ksigns_iq2xs[(aux32 >> 7*il) & 127]);
    return r;
}

static __dpct_inline__ wdecomp_lane wdecomp_iq2_xs(const uint8_t * base, int64_t l, int64_t k) {
    const int64_t nb = k / QK_K;
    const int64_t i  = l / (QK_K / 8);
    const int     ib = (l % (QK_K / 8)) / 4;
    const int     il = l % 4;

    const uint16_t q2     = *reinterpret_cast<const uint16_t *>(base + i * (QK_K / 4) + 2 * (4 * ib + il));
    const uint8_t  scales = base[nb * (QK_K / 4) + i * (QK_K / 32) + ib];
    const float    dall   = *reinterpret_cast<const ggml_half *>(base + nb * (QK_K / 4 + QK_K / 32) + i * sizeof(ggml_half));

    wdecomp_lane r;
    r.s = dall * (0.5f + ((scales >> 4*(il/2)) & 0xf)) * 0.25f;
    r.b = 0.0f;
    wdecomp_signed_grid(r, (const uint8_t *) (iq2xs_grid + (q2 & 511)), ksigns_iq2xs[q2 >> 9]);
    return r;
}

static __dpct_inline__ wdecomp_lane wdecomp_iq2_s(const uint8_t * base, int64_t l, int64_t k) {
    const int64_t nb = k / QK_K;
    const int64_t i  = l / (QK_K / 8);
    const int     ib = (l % (QK_K / 8)) / 4;
    const int     il = l % 4;

    const uint8_t   qs    = base[i * (QK_K / 8) + 4 * ib + il];
    const uint8_t   signs = base[nb * (QK_K / 8) + i * (QK_K / 8) + 4 * ib + il];
    const uint8_t * hs    = base + nb * (QK_K / 4) + i * (QK_K / 16);
    const float     dall  = *reinterpret_cast<const ggml_half *>(base + nb * (QK_K / 4 + QK_K / 16) + i * sizeof(ggml_half));

    wdecomp_lane r;
    r.s = dall * (0.5f + ((hs[QK_K / 32 + ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    r.b = 0.0f;
    wdecomp_signed_grid(r, (const uint8_t *) (iq2s_grid + (qs | ((hs[ib] << (8-2*il)) & 0x300))), signs);
    return r;
}

static __dpct_inline__ wdecomp_lane wdecomp_iq3_xxs(const uint8_t * base, int64_t l, int64_t k) {
    const int64_t nb = k / QK_K;
    const int64_t i  = l / (QK_K / 8);
    const int     ib = (l % (QK_K / 8)) / 4;
    const int     il = l % 4;

    const uint8_t * q3    = base + i * (QK_K / 4) + 8 * ib + 2 * il;
    const uint32_t  aux32 = *reinterpret_cast<const uint32_t *>(base + nb * (QK_K / 4) + i * (QK_K / 8) + 4 * ib);
    const float     dall  = *reinterpret_cast<const ggml_half *>(base + nb * (QK_K / 4 + QK_K / 8) + i * sizeof(ggml_half));

    const uint32_t g[2] = { iq3xxs_grid[q3[0]], iq3xxs_grid[q3[1]] };

    wdecomp_lane r;
    r.s = dall * (0.5f + (aux32 >> 28)) * 0.5f;
    r.b = 0.0f;
    wdecomp_signed_grid(r, (const uint8_t *) g, ksigns_iq2xs[(aux32 >> 7*il) & 127]);
    return r;
}

static __dpct_inline__ wdecomp_lane wdecomp_iq3_s(const uint8_t * base, int64_t l, int64_t k) {
    constexpr int ss_size = QK_K / 8 + QK_K / 64;

    const int64_t nb = k / QK_K;
    const int64_t i  = l / (QK_K / 8);
    const int     ib = (l % (QK_K / 8)) / 4;
    const int     il = l % 4;

    const uint8_t * qs   = base + i * (QK_K / 4) + 8 * ib + 2 * il;
    const uint8_t   qh   = base[nb * (QK_K / 4) + i * (QK_K / 32) + ib];
    const uint8_t * ss   = base + nb * (QK_K / 4 + QK_K / 32) + i * ss_size;
    const float     dall = *reinterpret_cast<const ggml_half *>(base + nb * (QK_K / 4 + QK_K / 32 + ss_size) + i * sizeof(ggml_half));

    const uint32_t g[2] = { iq3s_grid[qs[0] | ((qh << (8-2*il)) & 256)], iq3s_grid[qs[1] | ((qh << (7-2*il)) & 256)] };

    wdecomp_lane r;
    r.s = dall * (1 + 2*((ss[QK_K / 8 + ib/2] >> 4*(ib%2)) & 0xf));
    r.b = 0.0f;
    wdecomp_signed_grid(r, (const uint8_t *) g, ss[4*ib + il]);
    return r;
}

static __dpct_inline__ void wdecomp_iq1_grid(wdecomp_lane & r, const uint32_t g, const bool neg_delta) {
    const int c = neg_delta ? -9 : -7;
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        r.q[m + 0] = 8 * (int) ((g >> (8 * m)) & 0xF) + c;
        r.q[m + 4] = 8 * (int) ((g >> (8 * m + 4)) & 0xF) + c;
    }
}

static __dpct_inline__ wdecomp_lane wdecomp_iq1_s(const uint8_t * base, int64_t l, int64_t k) {
    static_assert(IQ1S_DELTA == 0.125f, "IQ1_S integer weights assume IQ1S_DELTA == 1/8");
    const int64_t nb = k / QK_K;
    const int64_t i  = l / (QK_K / 8);
    const int     ib = (l % (QK_K / 8)) / 4;
    const int     il = l % 4;

    const uint8_t  qs   = base[i * (QK_K / 8) + 4 * ib + il];
    const uint16_t qh   = *reinterpret_cast<const uint16_t *>(base + nb * (QK_K / 8) + i * (QK_K / 16) + 2 * ib);
    const float    dall = *reinterpret_cast<const ggml_half *>(base + nb * (QK_K / 8 + QK_K / 16) + i * sizeof(ggml_half));

    wdecomp_lane r;
    r.s = dall * (2*((qh >> 12) & 7) + 1) * 0.125f;
    r.b = 0.0f;
    wdecomp_iq1_grid(r, iq1s_grid_gpu[qs | (((qh >> 3*il) & 7) << 8)], qh & 0x8000);
    return r;
}

static __dpct_inline__ wdecomp_lane wdecomp_iq1_m(const uint8_t * base, int64_t l, int64_t k) {
    static_assert(IQ1M_DELTA == 0.125f, "IQ1_M integer weights assume IQ1M_DELTA == 1/8");
    const int64_t nb = k / QK_K;
    const int64_t i  = l / (QK_K / 8);
    const int     ib = (l % (QK_K / 8)) / 4;
    const int     il = l % 4;

    const uint8_t    qs = base[i * (QK_K / 8) + 4 * ib + il];
    const uint8_t    qh = base[nb * (QK_K / 8) + i * (QK_K / 16) + 2 * ib + il / 2];
    const uint16_t * sc = reinterpret_cast<const uint16_t *>(base + nb * (QK_K / 8 + QK_K / 16) + i * (QK_K / 32));

    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int ib16 = 2*ib + il/2;

    wdecomp_lane r;
    r.s = (float) scale.f16 * (2*((sc[ib16/4] >> 3*(ib16%4)) & 0x7) + 1) * 0.125f;
    r.b = 0.0f;
    wdecomp_iq1_grid(r, iq1s_grid_gpu[qs | (((qh >> 4*(il%2)) & 7) << 8)], qh & (0x08 << 4*(il%2)));
    return r;
}

#endif

static __dpct_inline__ void wdecomp_iq4_codes(wdecomp_lane & r, const uint32_t q0, const uint32_t q1) {
    const uint32_t lo = iq4nl_lookup4(q0 & 0x0F0F0F0F);
    const uint32_t hi = iq4nl_lookup4(q1 & 0x0F0F0F0F);
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        r.q[m + 0] = (int8_t) (lo >> (8 * m));
        r.q[m + 4] = (int8_t) (hi >> (8 * m));
    }
}

static __dpct_inline__ wdecomp_lane wdecomp_iq4_nl(const uint8_t * base, int64_t l, int64_t k) {
    const int64_t ib    = l / 4;
    const int     j     = 8 * (l % 4);
    const int     shift = j < QK4_NL / 2 ? 0 : 4;

    const uint32_t * q = (const uint32_t *) (base + ib * (QK4_NL / 2) + j % (QK4_NL / 2));

    wdecomp_lane r;
    r.s = *((const sycl::half *) (base + k / 2) + ib);
    r.b = 0.0f;
    wdecomp_iq4_codes(r, q[0] >> shift, q[1] >> shift);
    return r;
}

static __dpct_inline__ wdecomp_lane wdecomp_iq4_xs(const uint8_t * base, int64_t l, int64_t /* k */) {
    const int64_t i     = l / (QK_K / 8);
    const int     ib    = (l % (QK_K / 8)) / 4;
    const int     j     = 8 * (l % 4);
    const int     shift = j < 16 ? 0 : 4;

    const block_iq4_xs * x = (const block_iq4_xs *) base + i;
    const uint32_t *     q = (const uint32_t *) (x->qs + 16*ib + j%16);

    wdecomp_lane r;
    r.s = (float) x->d * ((((x->scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x->scales_h >> 2*ib) & 3) << 4)) - 32);
    r.b = 0.0f;
    wdecomp_iq4_codes(r, q[0] >> shift, q[1] >> shift);
    return r;
}

typedef wdecomp_lane (*wdecomp_decoder_t)(const uint8_t *, int64_t, int64_t);

#endif  // GGML_SYCL_WDECOMP_LANES_HPP
