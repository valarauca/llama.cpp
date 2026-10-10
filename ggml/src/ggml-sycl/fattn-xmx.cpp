#include "fattn-xmx.hpp"
#include "fattn.hpp"
#include "fattn-tile.hpp"

#include <array>
#include <cmath>
#include <optional>

#ifdef GGML_SYCL_FA_XMX

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/experimental/prefetch.hpp>

namespace jm     = sycl::ext::oneapi::experimental::matrix;
namespace jmi    = sycl::ext::intel::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;

// DPAS on SIMD16 XMX: an M x 16 f32 accumulator from an M x 16 f16 A and a 16 x 16 f16 B, M = 8 or 16.
static constexpr int XMX_SG  = 16;
static constexpr int XMX_TN  = 16;
static constexpr int XMX_TK  = 16;
static constexpr int XMX_NSG = 8;
// 2D block prefetch tiles are at most 64 bytes wide.
static constexpr int XMX_PC  = 32;
// Below this much attention work (nq * nkv * (DK + DV)) the KV-bound pass costs more than the
// fully masked blocks it lets the kernel skip.
static constexpr int64_t XMX_KV_END_MIN = (int64_t) 1 << 29;
// MLA: K head_dim 576, V head_dim 512 (a view of the first 512 K dims in llama.cpp).
static constexpr int XMX_MLA_DK  = 576;
static constexpr int XMX_MLA_DV  = 512;
static constexpr int XMX_MLA_NSG = 4;

struct fattn_xmx_params {
    const sycl::half * Q;
    const float *      Qf;
    int64_t            q_s1, q_s2, q_s3;
    const sycl::half * K;
    const sycl::half * V;
    const sycl::half * mask;
    const int *        kv_end;
    float *            dst;
    int64_t            k_s1, k_s2, k_s3;
    int64_t            v_s1, v_s2, v_s3;
    int64_t            mask_s1;
    int                nq, nkv, H, Hkv, mb;
    float              scale;
};

template <typename T>
static auto fattn_xmx_global(const T * p) {
    return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(const_cast<T *>(p));
}

// One sub-group owns TM query rows of one head and streams the KV sequence in blocks of BC keys
// with an online softmax, stopping at the per-block KV bound when one is given. Per-row state is
// indexed by accumulator element, which relies on element i of a TM x 16 accumulator (and of the
// TM x 16 A operand) holding row i, column lane; fattn_xmx_layout_ok() checks that on the device.
//
// QREG keeps Q in registers for the whole KV loop, otherwise every Q tile is re-loaded (from L1)
// where it is used, which frees the registers head_dim 256 needs for its output accumulators.
// QF32 loads strided f32 Q as an accumulator tile and converts it to the A operand in registers,
// so no separate f16 copy of Q is needed.
//
// With 8-row tiles, full KV blocks take a branch-free scale (and mask add) path: per-element bounds
// checks in the softmax were a quarter of the stall samples in an EU stall profile, as divergent
// cmp/sel/goto. The 16-row tile has no register headroom for the extra paths: IGC retries it with a
// conservative schedule that runs 5x slower.
//
// While a block is processed, each sub-group prefetches its slice of the next K/V block into L1.
// Staging K/V through SLM was measured slower on Xe2: the 2D block loads already share the tiles
// across sub-groups through L1, and SLM adds a copy and two barriers per block. Two 8-row tiles
// per sub-group were measured 3-4x slower than one, and a native 16-row tile only fits (and only
// wins) at head_dim 64.
template <int DK, int DV, int TM, int BC, bool QREG, bool QF32>
struct fattn_xmx_kernel {
    static constexpr int DKT = DK / XMX_TK;
    static constexpr int DVT = DV / XMX_TN;
    static constexpr int CT  = BC / XMX_TN;
    static constexpr int PR  = BC / XMX_NSG;

    using a_t = jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, TM, XMX_TK, jm::layout::row_major>;
    using c_t = jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, TM, XMX_TN>;

    fattn_xmx_params p;

    auto get(syclex::properties_tag) const {
        return syclex::properties{ sycl::ext::intel::experimental::grf_size<256>, syclex::sub_group_size<XMX_SG> };
    }

    void load_q(sycl::sub_group sg, a_t & qa, const sycl::half * Qh, const float * Qfh, int q0, int t) const {
        if constexpr (QF32) {
            c_t qf;
            jmi::joint_matrix_load_checked(sg, qf, fattn_xmx_global(Qfh), p.q_s1, jm::layout::row_major, p.nq, DK, q0,
                                           t * XMX_TK);
            jm::joint_matrix_copy(sg, qf, qa);
        } else {
            jmi::joint_matrix_load_checked(sg, qa, fattn_xmx_global(Qh), DK, p.nq, DK, q0, t * XMX_TK);
        }
    }

    void operator()(sycl::nd_item<1> it) const {
        using namespace sycl;
        auto      sg   = it.get_sub_group();
        const int sgid = sg.get_group_linear_id();
        const int lane = sg.get_local_linear_id();

        const int nqb = (p.nq + TM * XMX_NSG - 1) / (TM * XMX_NSG);
        const int g   = it.get_group(0);
        const int qb  = g % nqb;
        const int h   = (g / nqb) % p.H;
        const int b   = g / (nqb * p.H);
        const int q0  = (qb * XMX_NSG + sgid) * TM;
        if (q0 >= p.nq) {
            return;
        }
        const int hk     = h / (p.H / p.Hkv);
        const int kv_end = p.kv_end ? p.kv_end[q0 / TM] : p.nkv;

        const half *  Qh  = QF32 ? nullptr : p.Q + ((size_t) b * p.H + h) * p.nq * DK;
        const float * Qfh = QF32 ? p.Qf + b * p.q_s3 + h * p.q_s2 : nullptr;
        const half *  Kh  = p.K + b * p.k_s3 + hk * p.k_s2;
        const half *  Vh  = p.V + b * p.v_s3 + hk * p.v_s2;

        const float log2e   = 1.4426950408889634f;
        const float s_scale = p.scale * log2e;

        a_t qa[QREG ? DKT : 1];
        if constexpr (QREG) {
#pragma unroll
            for (int t = 0; t < DKT; ++t) {
                load_q(sg, qa[t], Qh, Qfh, q0, t);
            }
        }

        c_t o[DVT];
#pragma unroll
        for (int t = 0; t < DVT; ++t) {
            jm::joint_matrix_fill(sg, o[t], 0.0f);
        }

        float m[TM];
        float l[TM];
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            m[i] = -INFINITY;
            l[i] = 0.0f;
        }

        for (int kb = 0; kb < kv_end; kb += BC) {
            if (kb + 2 * BC <= kv_end) {
                const int64_t r = kb + BC + sgid * PR;
#pragma unroll
                for (int cc = 0; cc < DK; cc += XMX_PC) {
                    jm::joint_matrix_prefetch<PR, XMX_PC>(sg, const_cast<half *>(Kh) + r * p.k_s1 + cc, p.k_s1,
                                                          jm::layout::row_major, syclex::properties{ syclex::prefetch_hint_L1 });
                }
#pragma unroll
                for (int cc = 0; cc < DV; cc += XMX_PC) {
                    jm::joint_matrix_prefetch<PR, XMX_PC>(sg, const_cast<half *>(Vh) + r * p.v_s1 + cc, p.v_s1,
                                                          jm::layout::row_major, syclex::properties{ syclex::prefetch_hint_L1 });
                }
            }

            c_t s[CT];
#pragma unroll
            for (int c = 0; c < CT; ++c) {
                jm::joint_matrix_fill(sg, s[c], 0.0f);
#pragma unroll
                for (int t = 0; t < DKT; ++t) {
                    jm::joint_matrix<sub_group, half, jm::use::b, XMX_TK, XMX_TN, jm::layout::col_major> kt;
                    jmi::joint_matrix_load_checked(sg, kt, fattn_xmx_global(Kh), p.k_s1, p.nkv, DK, kb + c * XMX_TN, t * XMX_TK);
                    if constexpr (QREG) {
                        jm::joint_matrix_mad(sg, s[c], qa[t], kt, s[c]);
                    } else {
                        a_t qt;
                        load_q(sg, qt, Qh, Qfh, q0, t);
                        jm::joint_matrix_mad(sg, s[c], qt, kt, s[c]);
                    }
                }
            }

            float bmax[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                bmax[i] = -INFINITY;
            }
            const bool full_block = kb + BC <= p.nkv;
            bool       scaled     = false;
            if (TM == 8 && full_block && !p.mask) {
#pragma unroll
                for (int c = 0; c < CT; ++c) {
                    int i = 0;
                    jm::joint_matrix_apply(sg, s[c], [&](float & x) {
                        x       = x * s_scale;
                        bmax[i] = sycl::fmax(bmax[i], x);
                        ++i;
                    });
                }
                scaled = true;
            }
            if constexpr (TM == 8) {
                if (!scaled && full_block && q0 + TM <= p.nq) {
#pragma unroll
                    for (int c = 0; c < CT; ++c) {
                        const half * mc = p.mask + (int64_t) q0 * p.mask_s1 + kb + c * XMX_TN + lane;
                        int          i  = 0;
                        jm::joint_matrix_apply(sg, s[c], [&](float & x) {
                            x       = x * s_scale + (float) mc[(int64_t) i * p.mask_s1] * log2e;
                            bmax[i] = sycl::fmax(bmax[i], x);
                            ++i;
                        });
                    }
                    scaled = true;
                }
            }
            if (!scaled) {
#pragma unroll
                for (int c = 0; c < CT; ++c) {
                    const int key = kb + c * XMX_TN + lane;
                    int       i   = 0;
                    jm::joint_matrix_apply(sg, s[c], [&](float & x) {
                        float v = x * s_scale;
                        if (p.mask && q0 + i < p.nq && key < p.nkv) {
                            v += (float) p.mask[(int64_t) (q0 + i) * p.mask_s1 + key] * log2e;
                        }
                        if (key >= p.nkv) {
                            v = -INFINITY;
                        }
                        x       = v;
                        bmax[i] = sycl::fmax(bmax[i], v);
                        ++i;
                    });
                }
            }

            float alpha[TM];
            float mref[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const float mn = sycl::fmax(m[i], reduce_over_group(sg, bmax[i], maximum<float>()));
                mref[i]        = mn == -INFINITY ? 0.0f : mn;
                alpha[i]       = sycl::exp2(m[i] - mref[i]);
                m[i]           = mn;
                l[i] *= alpha[i];
            }

            a_t pa[CT];
#pragma unroll
            for (int c = 0; c < CT; ++c) {
                int i = 0;
                jm::joint_matrix_apply(sg, s[c], [&](float & x) {
                    x = sycl::exp2(x - mref[i]);
                    l[i] += x;
                    ++i;
                });
                jm::joint_matrix_copy(sg, s[c], pa[c]);
            }

#pragma unroll
            for (int t = 0; t < DVT; ++t) {
                int i = 0;
                jm::joint_matrix_apply(sg, o[t], [&](float & x) {
                    x *= alpha[i];
                    ++i;
                });
#pragma unroll
                for (int c = 0; c < CT; ++c) {
                    jm::joint_matrix<sub_group, half, jm::use::b, XMX_TK, XMX_TN, jm::layout::row_major> vb;
                    jmi::joint_matrix_load_checked(sg, vb, fattn_xmx_global(Vh), p.v_s1, p.nkv, DV, kb + c * XMX_TK, t * XMX_TN);
                    jm::joint_matrix_mad(sg, o[t], pa[c], vb, o[t]);
                }
            }
        }

        float inv[TM];
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            const float ls = reduce_over_group(sg, l[i], plus<float>());
            inv[i]         = ls > 0.0f ? 1.0f / ls : 0.0f;
        }

        float * dh = p.dst + ((size_t) b * p.nq * p.H + h) * DV;
#pragma unroll
        for (int t = 0; t < DVT; ++t) {
            int i = 0;
            jm::joint_matrix_apply(sg, o[t], [&](float & x) {
                x *= inv[i];
                ++i;
            });
            jmi::joint_matrix_store_checked(sg, o[t], fattn_xmx_global(dh), (size_t) p.H * DV, jm::layout::row_major, p.nq, DV,
                                            q0, t * XMX_TN);
        }
    }
};

template <int DK, int DV, int TM, int BC, bool QREG, bool QF32>
static void fattn_xmx_launch(dpct::queue_ptr stream, const fattn_xmx_params & p) {
    const int    nqb    = (p.nq + TM * XMX_NSG - 1) / (TM * XMX_NSG);
    const size_t groups = (size_t) nqb * p.H * p.mb;
    stream->parallel_for(sycl::nd_range<1>(groups * XMX_NSG * XMX_SG, XMX_NSG * XMX_SG),
                         fattn_xmx_kernel<DK, DV, TM, BC, QREG, QF32>{ p });
}

// MLA (K head_dim 576, V head_dim 512) does not fit one sub-group: the 512-wide output alone
// would take all 256 registers. Since all heads of a KV group share K/V, a work-group instead
// takes 8 heads of one query as its rows and NSG sub-groups split the work: each computes partial
// scores over its DK / NSG slice of K, the partials are summed through double-buffered SLM (one
// barrier per KV block), every sub-group runs the same online softmax, and each accumulates its
// DV / NSG slice of the output. Heads as rows also fills the tile during decode.
template <int DK, int DV, int BC, bool QF32>
struct fattn_xmx_mla_kernel {
    static constexpr int NSG = XMX_MLA_NSG;
    static constexpr int TM  = 8;
    static constexpr int KT  = DK / NSG / XMX_TK;
    static constexpr int VT  = DV / NSG / XMX_TN;
    static constexpr int CT  = BC / XMX_TN;

    using a_t = jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, TM, XMX_TK, jm::layout::row_major>;
    using c_t = jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, TM, XMX_TN>;

    fattn_xmx_params               p;
    sycl::local_accessor<float, 1> part;

    auto get(syclex::properties_tag) const {
        return syclex::properties{ sycl::ext::intel::experimental::grf_size<256>, syclex::sub_group_size<XMX_SG> };
    }

    void operator()(sycl::nd_item<1> it) const {
        using namespace sycl;
        auto      sg   = it.get_sub_group();
        const int sgid = sg.get_group_linear_id();
        const int lane = sg.get_local_linear_id();

        const int nhb = (p.H + TM - 1) / TM;
        const int g   = it.get_group(0);
        const int h0  = (g % nhb) * TM;
        const int q   = (g / nhb) % p.nq;
        const int b   = g / (nhb * p.nq);
        const int hk  = h0 / (p.H / p.Hkv);
        const int k0  = sgid * (DK / NSG);
        const int v0  = sgid * (DV / NSG);

        const int kv_end = p.kv_end ? p.kv_end[q] : p.nkv;

        const half * Kh = p.K + b * p.k_s3 + hk * p.k_s2;
        const half * Vh = p.V + b * p.v_s3 + hk * p.v_s2;

        const float log2e   = 1.4426950408889634f;
        const float s_scale = p.scale * log2e;

        a_t qa[KT];
#pragma unroll
        for (int t = 0; t < KT; ++t) {
            if constexpr (QF32) {
                c_t qf;
                jmi::joint_matrix_load_checked(sg, qf, fattn_xmx_global(p.Qf + b * p.q_s3 + q * p.q_s1), p.q_s2,
                                               jm::layout::row_major, p.H, DK, h0, k0 + t * XMX_TK);
                jm::joint_matrix_copy(sg, qf, qa[t]);
            } else {
                jmi::joint_matrix_load_checked(sg, qa[t], fattn_xmx_global(p.Q + ((size_t) b * p.H * p.nq + q) * DK),
                                               (size_t) p.nq * DK, p.H, DK, h0, k0 + t * XMX_TK);
            }
        }

        c_t o[VT];
#pragma unroll
        for (int t = 0; t < VT; ++t) {
            jm::joint_matrix_fill(sg, o[t], 0.0f);
        }
        float m[TM];
        float l[TM];
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            m[i] = -INFINITY;
            l[i] = 0.0f;
        }

        float * slm = part.template get_multi_ptr<access::decorated::no>().get_raw();
        int     buf = 0;

        for (int kb = 0; kb < kv_end; kb += BC) {
            float * pb = slm + buf * (NSG * CT * TM * XMX_TN);
#pragma unroll
            for (int c = 0; c < CT; ++c) {
                c_t sc;
                jm::joint_matrix_fill(sg, sc, 0.0f);
#pragma unroll
                for (int t = 0; t < KT; ++t) {
                    jm::joint_matrix<sub_group, half, jm::use::b, XMX_TK, XMX_TN, jm::layout::col_major> kt;
                    jmi::joint_matrix_load_checked(sg, kt, fattn_xmx_global(Kh), p.k_s1, p.nkv, DK, kb + c * XMX_TN, k0 + t * XMX_TK);
                    jm::joint_matrix_mad(sg, sc, qa[t], kt, sc);
                }
                jm::joint_matrix_store(sg, sc,
                                       address_space_cast<access::address_space::local_space, access::decorated::no>(
                                           pb + (sgid * CT + c) * TM * XMX_TN),
                                       XMX_TN, jm::layout::row_major);
            }
            group_barrier(it.get_group());

            c_t   s[CT];
            float bmax[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                bmax[i] = -INFINITY;
            }
#pragma unroll
            for (int c = 0; c < CT; ++c) {
                jm::joint_matrix_load(sg, s[c],
                                      address_space_cast<access::address_space::local_space, access::decorated::no>(pb + c * TM * XMX_TN),
                                      XMX_TN, jm::layout::row_major);
#pragma unroll
                for (int j = 1; j < NSG; ++j) {
                    c_t pj;
                    jm::joint_matrix_load(sg, pj,
                                          address_space_cast<access::address_space::local_space, access::decorated::no>(
                                              pb + (j * CT + c) * TM * XMX_TN),
                                          XMX_TN, jm::layout::row_major);
                    jm::joint_matrix_apply(sg, s[c], pj, [](float & x, const float & y) { x += y; });
                }
                const int key = kb + c * XMX_TN + lane;
                float     mv  = 0.0f;
                if (p.mask && key < p.nkv) {
                    mv = (float) p.mask[(int64_t) q * p.mask_s1 + key] * log2e;
                }
                int i = 0;
                jm::joint_matrix_apply(sg, s[c], [&](float & x) {
                    float v = x * s_scale + mv;
                    if (key >= p.nkv) {
                        v = -INFINITY;
                    }
                    x       = v;
                    bmax[i] = sycl::fmax(bmax[i], v);
                    ++i;
                });
            }

            float alpha[TM];
            float mref[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const float mn = sycl::fmax(m[i], reduce_over_group(sg, bmax[i], maximum<float>()));
                mref[i]        = mn == -INFINITY ? 0.0f : mn;
                alpha[i]       = sycl::exp2(m[i] - mref[i]);
                m[i]           = mn;
                l[i] *= alpha[i];
            }

            a_t pa[CT];
#pragma unroll
            for (int c = 0; c < CT; ++c) {
                int i = 0;
                jm::joint_matrix_apply(sg, s[c], [&](float & x) {
                    x = sycl::exp2(x - mref[i]);
                    l[i] += x;
                    ++i;
                });
                jm::joint_matrix_copy(sg, s[c], pa[c]);
            }

#pragma unroll
            for (int t = 0; t < VT; ++t) {
                int i = 0;
                jm::joint_matrix_apply(sg, o[t], [&](float & x) {
                    x *= alpha[i];
                    ++i;
                });
#pragma unroll
                for (int c = 0; c < CT; ++c) {
                    jm::joint_matrix<sub_group, half, jm::use::b, XMX_TK, XMX_TN, jm::layout::row_major> vb;
                    jmi::joint_matrix_load_checked(sg, vb, fattn_xmx_global(Vh), p.v_s1, p.nkv, DV, kb + c * XMX_TK, v0 + t * XMX_TN);
                    jm::joint_matrix_mad(sg, o[t], pa[c], vb, o[t]);
                }
            }
            buf ^= 1;
        }

        float inv[TM];
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            const float ls = reduce_over_group(sg, l[i], plus<float>());
            inv[i]         = ls > 0.0f ? 1.0f / ls : 0.0f;
        }
        float * dq = p.dst + ((size_t) b * p.nq + q) * p.H * DV;
#pragma unroll
        for (int t = 0; t < VT; ++t) {
            int i = 0;
            jm::joint_matrix_apply(sg, o[t], [&](float & x) {
                x *= inv[i];
                ++i;
            });
            jmi::joint_matrix_store_checked(sg, o[t], fattn_xmx_global(dq), (size_t) DV, jm::layout::row_major, p.H, DV, h0,
                                            v0 + t * XMX_TN);
        }
    }
};

template <int DK, int DV, int BC, bool QF32>
static void fattn_xmx_mla_launch(dpct::queue_ptr stream, const fattn_xmx_params & p) {
    constexpr int NSG    = XMX_MLA_NSG;
    const int     nhb    = (p.H + 7) / 8;
    const size_t  groups = (size_t) p.mb * p.nq * nhb;
    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> part(sycl::range<1>(2 * NSG * (BC / XMX_TN) * 8 * XMX_TN), h);
        h.parallel_for(sycl::nd_range<1>(groups * NSG * XMX_SG, NSG * XMX_SG), fattn_xmx_mla_kernel<DK, DV, BC, QF32>{ p, part });
    });
}

// Strided f32 Q -> dense f16 [mb][H][nq][D], the row-major A operand layout the kernel loads.
static void fattn_xmx_q_to_f16(const char * src, sycl::half * dst, int64_t D, int64_t nq, int64_t H, int64_t mb,
                               size_t nb1, size_t nb2, size_t nb3, dpct::queue_ptr stream) {
    const int64_t n = D * nq * H * mb;
    stream->parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
        int64_t       i = id[0];
        const int64_t d = i % D;
        i /= D;
        const int64_t q = i % nq;
        i /= nq;
        const int64_t h = i % H;
        const int64_t b = i / H;
        dst[id[0]]      = (sycl::half) *(const float *) (src + d * sizeof(float) + q * nb1 + h * nb2 + b * nb3);
    });
}

// For each block of `rows` query rows, one past the last key any row of the block can attend to
// (0 if the block is fully masked), so the main kernel can stop before fully masked KV blocks.
// Mirrors the CUDA backend's flash_attn_mask_to_KV_max: one sub-group scans the mask backward
// 64 keys at a time.
static void fattn_xmx_kv_end(const sycl::half * mask, int * out, int nq, int nkv, int rows, int64_t mask_s1,
                             dpct::queue_ptr stream) {
    const int nblk = (nq + rows - 1) / rows;
    stream->parallel_for(sycl::nd_range<1>((size_t) nblk * XMX_SG, XMX_SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(XMX_SG)]] {
        auto      sg   = it.get_sub_group();
        const int r0   = it.get_group(0) * rows;
        const int lane = sg.get_local_linear_id();
        int       end  = 0;
        for (int k0 = ((nkv - 1) / 64) * 64; k0 >= 0; k0 -= 64) {
            bool any = false;
            for (int r = 0; r < rows && r0 + r < nq; ++r) {
                const sycl::half * mr = mask + (int64_t) (r0 + r) * mask_s1;
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int key = k0 + e * XMX_SG + lane;
                    if (key < nkv && (float) mr[key] != -INFINITY) {
                        any = true;
                    }
                }
            }
            if (sycl::any_of_group(sg, any)) {
                end = sycl::min(nkv, k0 + 64);
                break;
            }
        }
        if (lane == 0) {
            out[it.get_group(0)] = end;
        }
    });
}

// Checks that element i of the 8x16 and 16x16 accumulators and A operands is (row i, column lane).
// It shares the large-GRF property with fattn_xmx_kernel so both land in the same device image,
// which keeps joint_matrix code out of the image every other SYCL kernel is JIT-compiled from.
struct fattn_xmx_layout_probe {
    int * bad;

    auto get(syclex::properties_tag) const {
        return syclex::properties{ sycl::ext::intel::experimental::grf_size<256>, syclex::sub_group_size<XMX_SG> };
    }

    template <int TM>
    void check(sycl::sub_group sg, size_t lane) const {
        jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, TM, XMX_TN>                            c;
        jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, TM, XMX_TK, jm::layout::row_major> a;
        jm::joint_matrix_fill(sg, c, 0.0f);
        jm::joint_matrix_fill(sg, a, 0.0f);
        size_t i = 0;
        jmi::joint_matrix_apply(sg, c, [&](float &, size_t r, size_t col) {
            if (r != i || col != lane) {
                *bad = 1;
            }
            ++i;
        });
        i = 0;
        jmi::joint_matrix_apply(sg, a, [&](sycl::half &, size_t r, size_t col) {
            if (r != i || col != lane) {
                *bad = 1;
            }
            ++i;
        });
    }

    void operator()(sycl::nd_item<1> it) const {
        auto         sg   = it.get_sub_group();
        const size_t lane = sg.get_local_linear_id();
        check<8>(sg, lane);
        check<16>(sg, lane);
    }
};

// The kernel indexes per-row softmax state by accumulator element, so verify the element layout
// once per device before first use.
static bool fattn_xmx_layout_ok(int device, dpct::queue_ptr stream) {
    static int8_t checked[GGML_SYCL_MAX_DEVICES] = {};
    if (checked[device] != 0) {
        return checked[device] > 0;
    }

    sycl::queue q(stream->get_context(), stream->get_device(), sycl::property::queue::in_order{});
    int *       bad = sycl::malloc_device<int>(1, q);
    q.memset(bad, 0, sizeof(int));
    q.parallel_for(sycl::nd_range<1>(XMX_SG, XMX_SG), fattn_xmx_layout_probe{ bad });
    int host_bad = 1;
    q.memcpy(&host_bad, bad, sizeof(int)).wait();
    sycl::free(bad, q);

    checked[device] = host_bad ? -1 : 1;
    if (host_bad) {
        GGML_LOG_WARN("%s: unexpected joint_matrix element layout on device %d, XMX flash attention disabled\n",
                      __func__, device);
    }
    return !host_bad;
}

// True when Q can be loaded as f32 straight from the tensor: head_dim 256 re-loads Q for every
// tile, where converting it each time costs more than one separate f16 copy, and the 2D block
// load needs a 64-byte aligned base and a 16-byte aligned pitch of at least 64 bytes. The MLA
// kernel's rows are heads, so its pitch is the head stride and the query stride moves the base.
static bool fattn_xmx_q_fused(const ggml_tensor * Q) {
    if ((uintptr_t) Q->data % 64 != 0 || Q->nb[3] % 64 != 0) {
        return false;
    }
    if (Q->ne[0] == XMX_MLA_DK) {
        return Q->nb[2] % 16 == 0 && Q->nb[2] >= 64 && Q->nb[1] % 64 == 0;
    }
    return Q->ne[0] != 256 && Q->nb[1] % 16 == 0 && Q->nb[1] >= 64 && Q->nb[2] % 64 == 0;
}

static bool fattn_xmx_kv_type_ok(ggml_type type) {
    return type == GGML_TYPE_F16 || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0;
}

static bool fattn_xmx_v_is_k_view(const ggml_tensor * K, const ggml_tensor * V) {
    return V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));
}

// Byte strides of K or V as the kernel reads them: the tensor itself for f16, otherwise the f16
// copy fattn_xmx_stage() makes, which keeps the source layout when the source is contiguously
// allocated (a KV cache view) and is dense otherwise.
static std::array<size_t, 3> fattn_xmx_f16_strides(const ggml_tensor * t) {
    if (t->type == GGML_TYPE_F16) {
        return { t->nb[1], t->nb[2], t->nb[3] };
    }
    if (ggml_is_contiguously_allocated(t)) {
        const size_t bs = ggml_blck_size(t->type);
        const size_t ts = ggml_type_size(t->type);
        return { t->nb[1] / ts * bs * sizeof(sycl::half), t->nb[2] / ts * bs * sizeof(sycl::half),
                 t->nb[3] / ts * bs * sizeof(sycl::half) };
    }
    const size_t s1 = t->ne[0] * sizeof(sycl::half);
    return { s1, s1 * t->ne[1], s1 * t->ne[1] * t->ne[2] };
}

// Quantized K or V -> f16 in `buf` (scratch reserved with the op), or in the pool when nothing was
// reserved, laid out as fattn_xmx_f16_strides() describes. f16 tensors are used in place.
static const sycl::half * fattn_xmx_stage(ggml_backend_sycl_context & ctx, ggml_tensor * dst, const ggml_tensor * t,
                                          sycl::half * buf, std::optional<ggml_sycl_pool_alloc<sycl::half>> & pool_buf) {
    if (t->type == GGML_TYPE_F16) {
        return (const sycl::half *) t->data;
    }
    if (!buf) {
        pool_buf.emplace(ctx.pool(), (size_t) ggml_nelements(t));
        buf = pool_buf->get();
    }
    if (ggml_is_contiguously_allocated(t)) {
        ggml_get_to_fp16_sycl(t->type, dst)(t->data, buf, ggml_nelements(t), ctx.stream());
    } else {
        const size_t ts = ggml_type_size(t->type);
        ggml_get_to_fp16_nc_sycl(t->type)(t->data, buf, t->ne[0], t->ne[1], t->ne[2], t->ne[3], t->nb[1] / ts,
                                          t->nb[2] / ts, t->nb[3] / ts, ctx.stream());
    }
    return buf;
}

#endif // GGML_SYCL_FA_XMX

bool ggml_sycl_flash_attn_ext_xmx_supported(const ggml_tensor * dst) {
#ifndef GGML_SYCL_FA_XMX
    GGML_UNUSED(dst);
    return false;
#else
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    const sycl_xe_family_caps & caps = get_xe_family_caps(ggml_sycl_info().devices[ggml_sycl_get_device()].hw_info.xe_family);
    if (caps.dpas_n != XMX_SG || !caps.block_2d_io) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || !fattn_xmx_kv_type_ok(K->type) || !fattn_xmx_kv_type_ok(V->type) || sinks) {
        return false;
    }
    const int64_t D   = K->ne[0];
    const bool    mla = D == XMX_MLA_DK && V->ne[0] == XMX_MLA_DV;
    if (mla) {
        if (Q->ne[0] != D || K->ne[2] == 0 || (Q->ne[2] / K->ne[2]) % 8 != 0) {
            return false;
        }
    } else if ((D != 64 && D != 128 && D != 256) || Q->ne[0] != D || V->ne[0] != D) {
        return false;
    }
    float max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[3] != 1 || mask->ne[0] < K->ne[1] ||
                 mask->ne[1] < Q->ne[1] || mask->nb[0] != sizeof(sycl::half))) {
        return false;
    }
    if (K->ne[2] == 0 || K->ne[2] != V->ne[2] || Q->ne[2] % K->ne[2] != 0 || K->ne[1] != V->ne[1]) {
        return false;
    }
    for (const ggml_tensor * t : { K, V }) {
        if (t->ne[3] != 1 && t->ne[3] != Q->ne[3]) {
            return false;
        }
        const std::array<size_t, 3> s = fattn_xmx_f16_strides(t);
        if (t->nb[0] != ggml_type_size(t->type) || (t->type == GGML_TYPE_F16 && (uintptr_t) t->data % 64 != 0) ||
            s[0] % 16 != 0 || s[0] < 64 || s[1] % 64 != 0 || s[2] % 64 != 0) {
            return false;
        }
    }
    return Q->nb[0] == sizeof(float) && ggml_is_contiguous(dst);
#endif
}

bool ggml_sycl_flash_attn_ext_xmx_needs_q_f16(const ggml_tensor * dst) {
#ifndef GGML_SYCL_FA_XMX
    GGML_UNUSED(dst);
    return false;
#else
    return ggml_sycl_flash_attn_ext_xmx_supported(dst) && !fattn_xmx_q_fused(dst->src[0]);
#endif
}

void ggml_sycl_flash_attn_ext_xmx(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#ifndef GGML_SYCL_FA_XMX
    ggml_sycl_flash_attn_ext_tile(ctx, dst);
#else
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    dpct::queue_ptr stream = ctx.stream();
    if (!fattn_xmx_layout_ok(ctx.device, stream)) {
        ggml_sycl_flash_attn_ext_tile(ctx, dst);
        return;
    }

    const int64_t D   = K->ne[0];
    const int64_t DV  = V->ne[0];
    const int64_t nq  = Q->ne[1];
    const int64_t nkv = K->ne[1];
    const int64_t H   = Q->ne[2];
    const int64_t mb  = Q->ne[3];
    const bool    mla = D == XMX_MLA_DK;
    const int     TM  = mla ? 1 : D == 64 ? 16 : 8;

    const ggml_sycl_fattn_extra                     extra  = ggml_sycl_fattn_get_extra(dst);
    const bool                                      v_on_k = K->type != GGML_TYPE_F16 && fattn_xmx_v_is_k_view(K, V);
    std::optional<ggml_sycl_pool_alloc<sycl::half>> K_pool;
    std::optional<ggml_sycl_pool_alloc<sycl::half>> V_pool;
    const sycl::half * Kh = fattn_xmx_stage(ctx, dst, K, (sycl::half *) extra.K_buffer_ptr, K_pool);
    const sycl::half * Vh = v_on_k ? Kh : fattn_xmx_stage(ctx, dst, V, (sycl::half *) extra.V_buffer_ptr, V_pool);
    const std::array<size_t, 3> ks = fattn_xmx_f16_strides(K);
    const std::array<size_t, 3> vs = v_on_k ? ks : fattn_xmx_f16_strides(V);

    fattn_xmx_params p{};
    p.K       = Kh;
    p.V       = Vh;
    p.mask    = mask ? (const sycl::half *) mask->data : nullptr;
    p.dst     = (float *) dst->data;
    p.k_s1    = ks[0] / sizeof(sycl::half);
    p.k_s2    = ks[1] / sizeof(sycl::half);
    p.k_s3    = K->ne[3] == 1 ? 0 : ks[2] / sizeof(sycl::half);
    p.v_s1    = vs[0] / sizeof(sycl::half);
    p.v_s2    = vs[1] / sizeof(sycl::half);
    p.v_s3    = V->ne[3] == 1 ? 0 : vs[2] / sizeof(sycl::half);
    p.mask_s1 = mask ? mask->nb[1] / sizeof(sycl::half) : 0;
    p.nq      = (int) nq;
    p.nkv     = (int) nkv;
    p.H       = (int) H;
    p.Hkv     = (int) K->ne[2];
    p.mb      = (int) mb;
    memcpy(&p.scale, (const float *) dst->op_params + 0, sizeof(float));

    const bool q_fused = fattn_xmx_q_fused(Q);
    std::optional<ggml_sycl_pool_alloc<sycl::half>> Qh_pool;
    if (q_fused) {
        p.Qf   = (const float *) Q->data;
        p.q_s1 = Q->nb[1] / sizeof(float);
        p.q_s2 = Q->nb[2] / sizeof(float);
        p.q_s3 = Q->nb[3] / sizeof(float);
    } else {
        sycl::half * Qh = (sycl::half *) extra.Q_buffer_ptr;
        if (!Qh) {
            Qh_pool.emplace(ctx.pool(), (size_t) D * nq * H * mb);
            Qh = Qh_pool->get();
        }
        fattn_xmx_q_to_f16((const char *) Q->data, Qh, D, nq, H, mb, Q->nb[1], Q->nb[2], Q->nb[3], stream);
        p.Q = Qh;
    }

    std::optional<ggml_sycl_pool_alloc<int>> kv_end_pool;
    if (mask && nq * nkv * (D + DV) >= XMX_KV_END_MIN) {
        kv_end_pool.emplace(ctx.pool(), (size_t) (nq + TM - 1) / TM);
        fattn_xmx_kv_end(p.mask, kv_end_pool->get(), p.nq, p.nkv, TM, p.mask_s1, stream);
        p.kv_end = kv_end_pool->get();
    }

    if (mla) {
        q_fused ? fattn_xmx_mla_launch<XMX_MLA_DK, XMX_MLA_DV, 32, true>(stream, p)
                : fattn_xmx_mla_launch<XMX_MLA_DK, XMX_MLA_DV, 32, false>(stream, p);
    } else if (D == 64) {
        q_fused ? fattn_xmx_launch<64, 64, 16, 32, true, true>(stream, p)
                : fattn_xmx_launch<64, 64, 16, 32, true, false>(stream, p);
    } else if (D == 128) {
        q_fused ? fattn_xmx_launch<128, 128, 8, 64, true, true>(stream, p)
                : fattn_xmx_launch<128, 128, 8, 64, true, false>(stream, p);
    } else {
        fattn_xmx_launch<256, 256, 8, 32, false, false>(stream, p);
    }
#endif
}
