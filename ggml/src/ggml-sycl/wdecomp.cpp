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

#include "wdecomp.hpp"
#include "wdecomp-lanes.hpp"


template <wdecomp_decoder_t decode, ggml_sycl_wdecomp_int wt, int group, bool has_bias>
static void dequantize_to_int_sycl(const void * vx, void * w, sycl::half * scales, sycl::half * bias, int64_t nrows,
                                   int64_t ncols, dpct::queue_ptr stream) {
    constexpr int chunks      = wt == GGML_SYCL_WDECOMP_S8 ? 2 : 4;
    constexpr int lane_elems  = 8 * chunks;
    constexpr int tile_rows   = 16;
    constexpr int tile_lanes  = WARP_SIZE;
    constexpr int tile_groups = tile_lanes * lane_elems / group;
    const int64_t k             = nrows * ncols;
    const int64_t lanes_per_row = ncols / lane_elems;
    const int64_t ngroups       = ncols / group;
    const sycl::range<2> global(ceil_div(nrows, tile_rows) * tile_rows, ceil_div(lanes_per_row, tile_lanes) * tile_lanes);
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<sycl::half, 2> s_tile(sycl::range<2>(tile_rows, tile_groups + 1), cgh);
        sycl::local_accessor<sycl::half, 2> b_tile(sycl::range<2>(has_bias ? tile_rows : 1, tile_groups + 1), cgh);
        cgh.parallel_for(sycl::nd_range<2>(global, sycl::range<2>(tile_rows, tile_lanes)),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const int     tr  = item.get_local_id(0);
            const int     tl  = item.get_local_id(1);
            const int64_t row = item.get_global_id(0);
            const int64_t lc  = item.get_global_id(1);
            if (row < nrows && lc < lanes_per_row) {
                const uint8_t * base = static_cast<const uint8_t *>(vx);
                const int64_t   lane = row * lanes_per_row + lc;
                uint32_t        v[4];
#pragma unroll
                for (int c = 0; c < chunks; ++c) {
                    const wdecomp_lane r = decode(base, chunks * lane + c, k);
                    if constexpr (wt == GGML_SYCL_WDECOMP_S8) {
                        v[2 * c + 0] = 0;
                        v[2 * c + 1] = 0;
#pragma unroll
                        for (int m = 0; m < 4; ++m) {
                            v[2 * c + 0] |= (uint32_t) (r.q[m + 0] & 0xFF) << (8 * m);
                            v[2 * c + 1] |= (uint32_t) (r.q[m + 4] & 0xFF) << (8 * m);
                        }
                    } else {
                        v[c] = 0;
#pragma unroll
                        for (int m = 0; m < 8; ++m) {
                            v[c] |= (uint32_t) (r.q[m] & 0xF) << (4 * m);
                        }
                    }
                    if ((8 * c) % group == 0) {
                        const int gi = (tl * lane_elems + 8 * c) / group;
                        s_tile[tr][gi] = sycl::half(r.s);
                        if constexpr (has_bias) {
                            b_tile[tr][gi] = sycl::half(r.b);
                        }
                    }
                }
                static_cast<sycl::uint4 *>(w)[lane] = sycl::uint4(v[0], v[1], v[2], v[3]);
            }
            sycl::group_barrier(item.get_group());
            const int64_t row0 = item.get_group(0) * tile_rows;
            const int64_t kg0  = item.get_group(1) * tile_groups;
#pragma unroll
            for (int gi = tr; gi < tile_groups; gi += tile_rows) {
                if (row0 + tl < nrows && kg0 + gi < ngroups) {
                    scales[(kg0 + gi) * nrows + row0 + tl] = s_tile[tl][gi];
                    if constexpr (has_bias) {
                        bias[(kg0 + gi) * nrows + row0 + tl] = b_tile[tl][gi];
                    }
                }
            }
        });
    });
}

bool ggml_sycl_wdecomp_format(ggml_type type, bool reordered, ggml_sycl_wdecomp_fmt & fmt) {
#if QK_K == 256
    if (type == GGML_TYPE_IQ4_XS) {
        fmt = { GGML_SYCL_WDECOMP_S8, 32, false };
        return !reordered;
    }
    if (!reordered) {
        return false;
    }
    switch (type) {
        case GGML_TYPE_Q2_K:    fmt = { GGML_SYCL_WDECOMP_U4, 16, true  }; return true;
        case GGML_TYPE_Q3_K:    fmt = { GGML_SYCL_WDECOMP_S4, 16, false }; return true;
        case GGML_TYPE_Q4_K:    fmt = { GGML_SYCL_WDECOMP_U4, 32, true  }; return true;
        case GGML_TYPE_Q5_K:    fmt = { GGML_SYCL_WDECOMP_S8, 32, true  }; return true;
        case GGML_TYPE_Q6_K:    fmt = { GGML_SYCL_WDECOMP_S8, 16, false }; return true;
        case GGML_TYPE_IQ1_S:   fmt = { GGML_SYCL_WDECOMP_S8, 32, false }; return true;
        case GGML_TYPE_IQ1_M:   fmt = { GGML_SYCL_WDECOMP_S8, 16, false }; return true;
        case GGML_TYPE_IQ2_XXS: fmt = { GGML_SYCL_WDECOMP_S8, 32, false }; return true;
        case GGML_TYPE_IQ2_XS:  fmt = { GGML_SYCL_WDECOMP_S8, 16, false }; return true;
        case GGML_TYPE_IQ2_S:   fmt = { GGML_SYCL_WDECOMP_S8, 16, false }; return true;
        case GGML_TYPE_IQ3_XXS: fmt = { GGML_SYCL_WDECOMP_S8, 32, false }; return true;
        case GGML_TYPE_IQ3_S:   fmt = { GGML_SYCL_WDECOMP_S8, 32, false }; return true;
        case GGML_TYPE_IQ4_NL:  fmt = { GGML_SYCL_WDECOMP_S8, 32, false }; return true;
        default:                return false;
    }
#else
    GGML_UNUSED(type); GGML_UNUSED(reordered); GGML_UNUSED(fmt);
    return false;
#endif
}

bool ggml_sycl_wdecomp_pays(const ggml_sycl_wdecomp_fmt & fmt, int64_t M, int64_t K) {
    return fmt.has_bias ? M * 256 <= K * fmt.group : 2 * M < K;
}

void ggml_sycl_dequantize_to_int(ggml_type type, bool reordered, const void * vx, void * w, sycl::half * scales,
                                 sycl::half * bias, int64_t nrows, int64_t ncols, dpct::queue_ptr stream) {
    GGML_UNUSED(reordered);
#if QK_K == 256
    switch (type) {
        case GGML_TYPE_Q2_K:    dequantize_to_int_sycl<wdecomp_q2_K,    GGML_SYCL_WDECOMP_U4, 16, true >(vx, w, scales, bias, nrows, ncols, stream); break;
        case GGML_TYPE_Q3_K:    dequantize_to_int_sycl<wdecomp_q3_K,    GGML_SYCL_WDECOMP_S4, 16, false>(vx, w, scales, bias, nrows, ncols, stream); break;
        case GGML_TYPE_Q4_K:    dequantize_to_int_sycl<wdecomp_q4_K,    GGML_SYCL_WDECOMP_U4, 32, true >(vx, w, scales, bias, nrows, ncols, stream); break;
        case GGML_TYPE_Q5_K:    dequantize_to_int_sycl<wdecomp_q5_K,    GGML_SYCL_WDECOMP_S8, 32, true >(vx, w, scales, bias, nrows, ncols, stream); break;
        case GGML_TYPE_Q6_K:    dequantize_to_int_sycl<wdecomp_q6_K,    GGML_SYCL_WDECOMP_S8, 16, false>(vx, w, scales, bias, nrows, ncols, stream); break;
        case GGML_TYPE_IQ1_S:   dequantize_to_int_sycl<wdecomp_iq1_s,   GGML_SYCL_WDECOMP_S8, 32, false>(vx, w, scales, bias, nrows, ncols, stream); break;
        case GGML_TYPE_IQ1_M:   dequantize_to_int_sycl<wdecomp_iq1_m,   GGML_SYCL_WDECOMP_S8, 16, false>(vx, w, scales, bias, nrows, ncols, stream); break;
        case GGML_TYPE_IQ2_XXS: dequantize_to_int_sycl<wdecomp_iq2_xxs, GGML_SYCL_WDECOMP_S8, 32, false>(vx, w, scales, bias, nrows, ncols, stream); break;
        case GGML_TYPE_IQ2_XS:  dequantize_to_int_sycl<wdecomp_iq2_xs,  GGML_SYCL_WDECOMP_S8, 16, false>(vx, w, scales, bias, nrows, ncols, stream); break;
        case GGML_TYPE_IQ2_S:   dequantize_to_int_sycl<wdecomp_iq2_s,   GGML_SYCL_WDECOMP_S8, 16, false>(vx, w, scales, bias, nrows, ncols, stream); break;
        case GGML_TYPE_IQ3_XXS: dequantize_to_int_sycl<wdecomp_iq3_xxs, GGML_SYCL_WDECOMP_S8, 32, false>(vx, w, scales, bias, nrows, ncols, stream); break;
        case GGML_TYPE_IQ3_S:   dequantize_to_int_sycl<wdecomp_iq3_s,   GGML_SYCL_WDECOMP_S8, 32, false>(vx, w, scales, bias, nrows, ncols, stream); break;
        case GGML_TYPE_IQ4_NL:  dequantize_to_int_sycl<wdecomp_iq4_nl,  GGML_SYCL_WDECOMP_S8, 32, false>(vx, w, scales, bias, nrows, ncols, stream); break;
        case GGML_TYPE_IQ4_XS:  dequantize_to_int_sycl<wdecomp_iq4_xs,  GGML_SYCL_WDECOMP_S8, 32, false>(vx, w, scales, bias, nrows, ncols, stream); break;
        default:                GGML_ABORT("%s: unsupported type %s", __func__, ggml_type_name(type));
    }
#else
    GGML_UNUSED(type); GGML_UNUSED(vx); GGML_UNUSED(w); GGML_UNUSED(scales); GGML_UNUSED(bias);
    GGML_UNUSED(nrows); GGML_UNUSED(ncols); GGML_UNUSED(stream);
    GGML_ABORT("%s: requires QK_K == 256", __func__);
#endif
}

template <int group>
static void f32_to_f16_gsum_sycl(const float * x, sycl::half * y, sycl::half * gsum, int64_t k, dpct::queue_ptr stream) {
    constexpr int wg_size = 256;
    constexpr int lanes   = group / 8;
    const int64_t n_lanes = k / 8;
    const int64_t n_wg    = (n_lanes + wg_size - 1) / wg_size;
    stream->parallel_for(sycl::nd_range<1>(n_wg * wg_size, wg_size),
                         [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        const int64_t l     = item.get_global_id(0);
        const bool    valid = l < n_lanes;
        float         sum   = 0.0f;
        if (valid) {
            const sycl::vec<float, 8> f = reinterpret_cast<const sycl::vec<float, 8> *>(x)[l];
            reinterpret_cast<sycl::vec<sycl::half, 8> *>(y)[l] = f.convert<sycl::half, sycl::rounding_mode::rte>();
#pragma unroll
            for (int m = 0; m < 8; ++m) {
                sum += f[m];
            }
        }
        const auto sg = item.get_sub_group();
#pragma unroll
        for (int o = 1; o < lanes; o <<= 1) {
            sum += sycl::permute_group_by_xor(sg, sum, o);
        }
        if (valid && l % lanes == 0) {
            gsum[l / lanes] = sycl::half(sum);
        }
    });
}

void ggml_sycl_f32_to_f16_gsum(const float * x, sycl::half * y, sycl::half * gsum, int64_t k, int group,
                               dpct::queue_ptr stream) {
    GGML_ASSERT(k % group == 0);
    switch (group) {
        case 16: f32_to_f16_gsum_sycl<16>(x, y, gsum, k, stream); break;
        case 32: f32_to_f16_gsum_sycl<32>(x, y, gsum, k, stream); break;
        default: GGML_ABORT("%s: unsupported group %d", __func__, group);
    }
}
