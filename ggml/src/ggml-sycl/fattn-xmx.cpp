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
// Split-KV (head_dim 256 only): KV length below which a row block is never split (each split
// then gets at least half of it), and the most work-groups one row block is split across.
static constexpr int XMX_SPLIT_MIN_KV = 2048;
static constexpr int XMX_SPLIT_MAX    = 4;
// head_dim 256: heads of one KV head per work-group (each with 8 sub-groups of 8 rows), so that a
// work-group fills an Xe core and its sub-groups share K/V tiles and mask rows through L1.
static constexpr int XMX_D256_HPW     = 4;

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
    float *            part;
    sycl::float2 *     part_ml;
    int                nsplit, rb0, nrb;
    const int *        kv_zero;
};

template <typename T>
static auto fattn_xmx_global(const T * p) {
    return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(const_cast<T *>(p));
}

// Row block n of a packed launch as (batch, head, query block). Each run of four row blocks takes
// four heads of one KV head at the same query block where the GQA ratio allows, and the heads left
// over in each KV group at adjacent query blocks (27B, 6 heads per KV head: heads 0-3 at one
// block, then heads 4-5 at two).
static inline void fattn_xmx_unit(int n, int nqb, int H, int R, int & b, int & h, int & qb) {
    b = n / (nqb * H);
    n %= nqb * H;
    const int kvh  = n / (nqb * R);
    const int full = R / 4 * 4 * nqb;
    n %= nqb * R;
    if (n < full) {
        h  = kvh * R + n / (4 * nqb) * 4 + n % 4;
        qb = n / 4 % nqb;
    } else {
        n -= full;
        h  = kvh * R + R / 4 * 4 + n % (R % 4);
        qb = n / (R % 4);
    }
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
//
// With NSG = HPW * 8 a work-group runs HPW row blocks of 64 rows, picked by fattn_xmx_unit() to
// share one KV head. At head_dim 256 that is 32 sub-groups, a whole Xe core, so the K/V tiles and
// mask rows those row blocks have in common are shared through L1 whatever the dispatcher does.
// Head_dim 256 also skips the mask in full blocks inside the all-zero prefix kv_zero gives.
//
// SPLIT runs row blocks rb0 .. rb0 + nrb - 1 with the KV range of each shared out over nsplit
// work-groups. Each writes its unnormalized output and per-row max and sum to part / part_ml, and
// fattn_xmx_combine() merges them into dst.
template <int DK, int DV, int TM, int BC, bool QREG, bool QF32, bool SPLIT = false, int NSG = XMX_NSG>
struct fattn_xmx_kernel {
    static constexpr int DKT = DK / XMX_TK;
    static constexpr int DVT = DV / XMX_TN;
    static constexpr int CT  = BC / XMX_TN;
    static constexpr int PR  = BC / NSG;
    static constexpr int LB  = 2;
    static constexpr int HPW = NSG / XMX_NSG;

    using a_t = jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, TM, XMX_TK, jm::layout::row_major>;
    using c_t = jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, TM, XMX_TN>;
    using k_t = jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, XMX_TK, XMX_TN, jm::layout::col_major>;
    using v_t = jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, XMX_TK, XMX_TN, jm::layout::row_major>;

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
        const int gu  = it.get_group(0);
        const int g   = (SPLIT ? p.rb0 / HPW + gu % (p.nrb / HPW) : gu) * HPW + sgid / XMX_NSG;
        int       qb  = g % nqb;
        int       h   = (g / nqb) % p.H;
        int       b   = g / (nqb * p.H);
        if constexpr (HPW > 1) {
            fattn_xmx_unit(g, nqb, p.H, p.H / p.Hkv, b, h, qb);
        }
        const int q0  = (qb * XMX_NSG + sgid % XMX_NSG) * TM;
        if (q0 >= p.nq) {
            return;
        }
        const int hk     = h / (p.H / p.Hkv);
        const int kv_len  = p.kv_end ? p.kv_end[q0 / TM] : p.nkv;
        const int kv_zero = DK == 256 && p.kv_zero ? p.kv_zero[q0 / TM] : 0;
        int       kv_lo   = 0;
        int       kv_end  = kv_len;
        int       sp      = 0;
        if constexpr (SPLIT) {
            const int chunk = ((kv_len + p.nsplit - 1) / p.nsplit + BC - 1) / BC * BC;
            sp              = gu / (p.nrb / HPW);
            kv_lo           = sp * chunk;
            kv_end          = sycl::min(kv_len, kv_lo + chunk);
        }

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

        for (int kb = kv_lo; kb < kv_end; kb += BC) {
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
            if constexpr (QREG) {
#pragma unroll
                for (int c = 0; c < CT; ++c) {
                    jm::joint_matrix_fill(sg, s[c], 0.0f);
#pragma unroll
                    for (int t = 0; t < DKT; ++t) {
                        k_t kt;
                        jmi::joint_matrix_load_checked(sg, kt, fattn_xmx_global(Kh), p.k_s1, p.nkv, DK, kb + c * XMX_TN, t * XMX_TK);
                        jm::joint_matrix_mad(sg, s[c], qa[t], kt, s[c]);
                    }
                }
            } else {
#pragma unroll
                for (int c = 0; c < CT; ++c) {
                    jm::joint_matrix_fill(sg, s[c], 0.0f);
                }
#pragma unroll
                for (int t = 0; t < DKT; t += LB) {
                    a_t qt[LB];
                    k_t kt[LB][CT];
#pragma unroll
                    for (int j = 0; j < LB; ++j) {
                        load_q(sg, qt[j], Qh, Qfh, q0, t + j);
#pragma unroll
                        for (int c = 0; c < CT; ++c) {
                            jmi::joint_matrix_load_checked(sg, kt[j][c], fattn_xmx_global(Kh), p.k_s1, p.nkv, DK, kb + c * XMX_TN,
                                                           (t + j) * XMX_TK);
                        }
                    }
#pragma unroll
                    for (int j = 0; j < LB; ++j) {
#pragma unroll
                        for (int c = 0; c < CT; ++c) {
                            jm::joint_matrix_mad(sg, s[c], qt[j], kt[j][c], s[c]);
                        }
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
            if (TM == 8 && full_block && (!p.mask || (DK == 256 && kb + BC <= kv_zero))) {
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

            if constexpr (QREG) {
#pragma unroll
                for (int t = 0; t < DVT; ++t) {
                    int i = 0;
                    jm::joint_matrix_apply(sg, o[t], [&](float & x) {
                        x *= alpha[i];
                        ++i;
                    });
#pragma unroll
                    for (int c = 0; c < CT; ++c) {
                        v_t vb;
                        jmi::joint_matrix_load_checked(sg, vb, fattn_xmx_global(Vh), p.v_s1, p.nkv, DV, kb + c * XMX_TK, t * XMX_TN);
                        jm::joint_matrix_mad(sg, o[t], pa[c], vb, o[t]);
                    }
                }
            } else {
#pragma unroll
                for (int t = 0; t < DVT; t += LB) {
                    v_t vb[LB][CT];
#pragma unroll
                    for (int j = 0; j < LB; ++j) {
#pragma unroll
                        for (int c = 0; c < CT; ++c) {
                            jmi::joint_matrix_load_checked(sg, vb[j][c], fattn_xmx_global(Vh), p.v_s1, p.nkv, DV, kb + c * XMX_TK,
                                                           (t + j) * XMX_TN);
                        }
                    }
#pragma unroll
                    for (int j = 0; j < LB; ++j) {
                        int i = 0;
                        jm::joint_matrix_apply(sg, o[t + j], [&](float & x) {
                            x *= alpha[i];
                            ++i;
                        });
#pragma unroll
                        for (int c = 0; c < CT; ++c) {
                            jm::joint_matrix_mad(sg, o[t + j], pa[c], vb[j][c], o[t + j]);
                        }
                    }
                }
            }
        }

        if constexpr (SPLIT) {
            float * ph = p.part + ((size_t) (sp * p.mb + b) * p.nq * p.H + h) * DV;
#pragma unroll
            for (int t = 0; t < DVT; ++t) {
                jmi::joint_matrix_store_checked(sg, o[t], fattn_xmx_global(ph), (size_t) p.H * DV, jm::layout::row_major,
                                                p.nq, DV, q0, t * XMX_TN);
            }
            sycl::float2 * mlh = p.part_ml + ((size_t) (sp * p.mb + b) * p.H + h) * p.nq + q0;
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const float ls = reduce_over_group(sg, l[i], plus<float>());
                if (lane == 0 && q0 + i < p.nq) {
                    mlh[i] = sycl::float2(m[i], ls);
                }
            }
            return;
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

template <int DK, int DV, int TM, int BC, bool QREG, bool QF32, bool SPLIT = false, int NSG = XMX_NSG>
static void fattn_xmx_launch(dpct::queue_ptr stream, const fattn_xmx_params & p, size_t groups = 0) {
    constexpr int HPW = NSG / XMX_NSG;
    const int     nqb = (p.nq + TM * XMX_NSG - 1) / (TM * XMX_NSG);
    if (groups == 0) {
        groups = SPLIT ? (size_t) p.nrb / HPW * p.nsplit : (size_t) nqb * p.H * p.mb / HPW;
    }
    stream->parallel_for(sycl::nd_range<1>(groups * NSG * XMX_SG, NSG * XMX_SG),
                         fattn_xmx_kernel<DK, DV, TM, BC, QREG, QF32, SPLIT, NSG>{ p });
}

// Merges the nsplit partial outputs of a split-KV launch over row blocks rb0 .. rb0 + nrb - 1 into
// dst, weighting each by its row sum and the exp2 distance of its row max from the largest.
static void fattn_xmx_combine(const fattn_xmx_params & p, int64_t DV, bool packed, dpct::queue_ptr stream) {
    const float *        part   = p.part;
    const sycl::float2 * ml     = p.part_ml;
    float *              dst    = p.dst;
    const int            nsplit = p.nsplit;
    const int64_t        nq     = p.nq;
    const int64_t        H      = p.H;
    const int64_t        mb     = p.mb;
    const int64_t        rb0    = p.rb0;
    const int64_t        rows   = 8 * XMX_NSG;
    const int64_t        nqb    = (nq + rows - 1) / rows;
    const int64_t        ss     = mb * nq * H * DV;
    const int            R      = p.H / p.Hkv;
    stream->parallel_for(sycl::range<1>((size_t) p.nrb * rows * DV / 4), [=](sycl::id<1> id) {
        const int64_t d  = (int64_t) id[0] * 4 % DV;
        const int64_t r  = (int64_t) id[0] * 4 / DV;
        const int     n  = (int) (rb0 + r / rows);
        int           qb = n % nqb;
        int           h  = n / nqb % H;
        int           b  = n / (nqb * H);
        if (packed) {
            fattn_xmx_unit(n, (int) nqb, (int) H, R, b, h, qb);
        }
        const int64_t q = (int64_t) qb * rows + r % rows;
        if (q >= nq) {
            return;
        }
        const int64_t e  = ((b * nq + q) * H + h) * DV + d;
        float         mx = -INFINITY;
        for (int s = 0; s < nsplit; ++s) {
            mx = sycl::fmax(mx, ml[((s * mb + b) * H + h) * nq + q].x());
        }
        sycl::float4 acc(0.0f);
        float        ls = 0.0f;
        for (int s = 0; s < nsplit; ++s) {
            const sycl::float2 v = ml[((s * mb + b) * H + h) * nq + q];
            const float        w = v.x() == -INFINITY ? 0.0f : sycl::exp2(v.x() - mx);
            ls += w * v.y();
            acc += w * *(const sycl::float4 *) (part + s * ss + e);
        }
        *(sycl::float4 *) (dst + e) = ls > 0.0f ? acc / ls : sycl::float4(0.0f);
    });
}

// Work-groups left over past the last full wave would run in a half-empty wave as long as a full
// one (27B prefill: 24 heads, nq 512, 48 packed work-groups against 32 Xe cores). The full waves
// run as one launch and the leftover work-groups as a second one, with each KV range split over as
// many work-groups as fill a single wave. Work-groups that start together read K/V in step, which
// L3 rewards, so splitting every row block over more waves was measured slower.
struct fattn_xmx_plan {
    int64_t full;
    int64_t tail;
    int     nsplit;
};

static fattn_xmx_plan fattn_xmx_plan_split(int device, int nsg, int64_t groups, int64_t nkv) {
    const int64_t resident = (int64_t) ggml_sycl_info().devices[device].nsm * 16 * 4 / nsg;
    const int64_t full     = groups / resident * resident;
    const int64_t tail     = groups - full;
    int           nsplit   = tail > 0 ? (int) std::min<int64_t>(XMX_SPLIT_MAX, resident / tail) : 1;
    while (nsplit > 1 && nkv / nsplit < XMX_SPLIT_MIN_KV / 2) {
        --nsplit;
    }
    if (nkv < XMX_SPLIT_MIN_KV || nsplit < 2) {
        return { groups, 0, 1 };
    }
    return { full, tail, nsplit };
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

// fattn_xmx_q_to_f16() four elements at a time, for Q with 16-byte aligned rows. Near DRAM bandwidth:
// 38 us instead of 89 us for 27B prefill (24 heads, nq 512, head_dim 256).
static void fattn_xmx_q_to_f16_v4(const char * src, sycl::half * dst, int64_t D, int64_t nq, int64_t H, int64_t mb,
                                  size_t nb1, size_t nb2, size_t nb3, dpct::queue_ptr stream) {
    const int64_t n4 = D / 4 * nq * H * mb;
    stream->parallel_for(sycl::range<1>(n4), [=](sycl::id<1> id) {
        int64_t       i = id[0];
        const int64_t d = i % (D / 4) * 4;
        i /= D / 4;
        const int64_t q = i % nq;
        i /= nq;
        const int64_t h = i % H;
        const int64_t b = i / H;
        const sycl::float4 v = *(const sycl::float4 *) (src + d * sizeof(float) + q * nb1 + h * nb2 + b * nb3);
        *(sycl::vec<sycl::half, 4> *) (dst + id[0] * 4) = v.convert<sycl::half>();
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

// For each block of `rows` query rows: the KV bound fattn_xmx_kv_end() gives, and the length of the
// key prefix whose mask is exactly 0 in every row, which the head_dim 256 kernel runs without
// reading the mask. One work-group per block reads its mask rows once.
static void fattn_xmx_kv_bounds(const sycl::half * mask, int * end, int * zero, int nq, int nkv, int rows,
                                int64_t mask_s1, dpct::queue_ptr stream) {
    constexpr int WG   = 1024;
    const int     nblk = (nq + rows - 1) / rows;
    const int     nv   = (uintptr_t) mask % 16 == 0 && mask_s1 % 8 == 0 ? nkv / 8 : 0;
    stream->parallel_for(sycl::nd_range<1>((size_t) nblk * WG, WG), [=](sycl::nd_item<1> it) {
        const int r0    = it.get_group(0) * rows;
        const int nr    = sycl::min(rows, nq - r0);
        const int li    = it.get_local_linear_id();
        int       last  = 0;
        int       first = nkv;
        for (int k8 = li; k8 < nv; k8 += WG) {
            for (int r = 0; r < nr; ++r) {
                const sycl::uint4 v = *(const sycl::uint4 *) (mask + (int64_t) (r0 + r) * mask_s1 + k8 * 8);
                for (int e = 0; e < 8; ++e) {
                    const uint32_t x = (v[e / 2] >> (e % 2 * 16)) & 0xFFFF;
                    if (x != 0xFC00) {
                        last = sycl::max(last, k8 * 8 + e + 1);
                    }
                    if ((x & 0x7FFF) != 0) {
                        first = sycl::min(first, k8 * 8 + e);
                    }
                }
            }
        }
        for (int k = nv * 8 + li; k < nkv; k += WG) {
            for (int r = 0; r < nr; ++r) {
                const uint16_t x = sycl::bit_cast<uint16_t>(mask[(int64_t) (r0 + r) * mask_s1 + k]);
                if (x != 0xFC00) {
                    last = sycl::max(last, k + 1);
                }
                if ((x & 0x7FFF) != 0) {
                    first = sycl::min(first, k);
                }
            }
        }
        last  = sycl::reduce_over_group(it.get_group(), last, sycl::maximum<int>());
        first = sycl::reduce_over_group(it.get_group(), first, sycl::minimum<int>());
        if (li == 0) {
            end[it.get_group(0)]  = sycl::min(nkv, (last + 63) / 64 * 64);
            zero[it.get_group(0)] = first;
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
        if (D == 256 && (uintptr_t) Q->data % 16 == 0 && Q->nb[1] % 16 == 0 && Q->nb[2] % 16 == 0 && Q->nb[3] % 16 == 0) {
            fattn_xmx_q_to_f16_v4((const char *) Q->data, Qh, D, nq, H, mb, Q->nb[1], Q->nb[2], Q->nb[3], stream);
        } else {
            fattn_xmx_q_to_f16((const char *) Q->data, Qh, D, nq, H, mb, Q->nb[1], Q->nb[2], Q->nb[3], stream);
        }
        p.Q = Qh;
    }

    std::optional<ggml_sycl_pool_alloc<int>> kv_end_pool;
    std::optional<ggml_sycl_pool_alloc<int>> kv_zero_pool;
    if (mask && D == 256) {
        kv_end_pool.emplace(ctx.pool(), (size_t) (nq + TM - 1) / TM);
        kv_zero_pool.emplace(ctx.pool(), (size_t) (nq + TM - 1) / TM);
        fattn_xmx_kv_bounds(p.mask, kv_end_pool->get(), kv_zero_pool->get(), p.nq, p.nkv, TM, p.mask_s1, stream);
        p.kv_end  = kv_end_pool->get();
        p.kv_zero = kv_zero_pool->get();
    } else if (mask && nq * nkv * (D + DV) >= XMX_KV_END_MIN) {
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
        const int64_t nqb    = (nq + 8 * XMX_NSG - 1) / (8 * XMX_NSG);
        const bool    packed = (H / p.Hkv) * nqb % XMX_D256_HPW == 0;
        const int     nsg    = packed ? XMX_NSG * XMX_D256_HPW : XMX_NSG;
        const int64_t hpw    = packed ? XMX_D256_HPW : 1;
        const fattn_xmx_plan plan = fattn_xmx_plan_split(ctx.device, nsg, nqb * H * mb / hpw, nkv);
        if (plan.full > 0) {
            packed ? fattn_xmx_launch<256, 256, 8, 32, false, false, false, XMX_NSG * XMX_D256_HPW>(stream, p, plan.full)
                   : fattn_xmx_launch<256, 256, 8, 32, false, false>(stream, p, plan.full);
        }
        if (plan.tail > 0) {
            ggml_sycl_pool_alloc<float>        part(ctx.pool(), (size_t) plan.nsplit * mb * nq * H * DV);
            ggml_sycl_pool_alloc<sycl::float2> part_ml(ctx.pool(), (size_t) plan.nsplit * mb * H * nq);
            p.part    = part.get();
            p.part_ml = part_ml.get();
            p.nsplit  = plan.nsplit;
            p.rb0     = (int) (plan.full * hpw);
            p.nrb     = (int) (plan.tail * hpw);
            packed ? fattn_xmx_launch<256, 256, 8, 32, false, false, true, XMX_NSG * XMX_D256_HPW>(stream, p)
                   : fattn_xmx_launch<256, 256, 8, 32, false, false, true>(stream, p);
            fattn_xmx_combine(p, DV, packed, stream);
        }
    }
#endif
}
