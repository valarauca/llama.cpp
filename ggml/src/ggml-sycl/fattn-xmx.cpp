#include "fattn-xmx.hpp"
#include "fattn.hpp"
#include "fattn-tile.hpp"

#include <cmath>
#include <optional>

#ifdef GGML_SYCL_FA_XMX

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/experimental/prefetch.hpp>

namespace jm     = sycl::ext::oneapi::experimental::matrix;
namespace jmi    = sycl::ext::intel::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;

// DPAS tile on SIMD16 XMX: an 8x16 f32 accumulator from an 8x16 f16 A and a 16x16 f16 B.
static constexpr int XMX_SG  = 16;
static constexpr int XMX_TM  = 8;
static constexpr int XMX_TN  = 16;
static constexpr int XMX_TK  = 16;
static constexpr int XMX_NSG = 8;
// 2D block prefetch tiles are at most 64 bytes wide.
static constexpr int XMX_PC  = 32;

struct fattn_xmx_params {
    const sycl::half * Q;
    const sycl::half * K;
    const sycl::half * V;
    const sycl::half * mask;
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

// One sub-group owns XMX_TM query rows of one head and streams the KV sequence in blocks of BC
// keys with an online softmax. Per-row state is indexed by accumulator element, which relies on
// element i of an 8x16 accumulator (and of the 8x16 A operand) holding row i, column lane;
// fattn_xmx_layout_ok() checks that on the device before the kernel is first used.
// While a block is processed, each sub-group prefetches its slice of the next K/V block into L1.
// Staging K/V through SLM was measured slower on Xe2: the 2D block loads already share the
// tiles across sub-groups through L1, and SLM adds a copy and two barriers per block.
template <int D, int BC>
struct fattn_xmx_kernel {
    static constexpr int DT = D / XMX_TK;
    static constexpr int CT = BC / XMX_TN;
    static constexpr int PR = BC / XMX_NSG;

    fattn_xmx_params p;

    auto get(syclex::properties_tag) const {
        return syclex::properties{ sycl::ext::intel::experimental::grf_size<256>, syclex::sub_group_size<XMX_SG> };
    }

    void operator()(sycl::nd_item<1> it) const {
        using namespace sycl;
        auto      sg   = it.get_sub_group();
        const int sgid = sg.get_group_linear_id();
        const int lane = sg.get_local_linear_id();

        const int nqb = (p.nq + XMX_TM * XMX_NSG - 1) / (XMX_TM * XMX_NSG);
        const int g   = it.get_group(0);
        const int qb  = g % nqb;
        const int h   = (g / nqb) % p.H;
        const int b   = g / (nqb * p.H);
        const int q0  = qb * XMX_TM * XMX_NSG + sgid * XMX_TM;
        if (q0 >= p.nq) {
            return;
        }
        const int hk = h / (p.H / p.Hkv);

        const half * Qh = p.Q + ((size_t) b * p.H + h) * p.nq * D;
        const half * Kh = p.K + b * p.k_s3 + hk * p.k_s2;
        const half * Vh = p.V + b * p.v_s3 + hk * p.v_s2;

        const float log2e   = 1.4426950408889634f;
        const float s_scale = p.scale * log2e;

        jm::joint_matrix<sub_group, half, jm::use::a, XMX_TM, XMX_TK, jm::layout::row_major> qa[DT];
#pragma unroll
        for (int t = 0; t < DT; ++t) {
            jmi::joint_matrix_load_checked(sg, qa[t], fattn_xmx_global(Qh), D, p.nq, D, q0, t * XMX_TK);
        }

        jm::joint_matrix<sub_group, float, jm::use::accumulator, XMX_TM, XMX_TN> o[DT];
#pragma unroll
        for (int t = 0; t < DT; ++t) {
            jm::joint_matrix_fill(sg, o[t], 0.0f);
        }

        float m[XMX_TM];
        float l[XMX_TM];
#pragma unroll
        for (int i = 0; i < XMX_TM; ++i) {
            m[i] = -INFINITY;
            l[i] = 0.0f;
        }

        for (int kb = 0; kb < p.nkv; kb += BC) {
            if (kb + 2 * BC <= p.nkv) {
                const int64_t r = kb + BC + sgid * PR;
#pragma unroll
                for (int cc = 0; cc < D; cc += XMX_PC) {
                    jm::joint_matrix_prefetch<PR, XMX_PC>(sg, const_cast<half *>(Kh) + r * p.k_s1 + cc, p.k_s1,
                                                          jm::layout::row_major, syclex::properties{ syclex::prefetch_hint_L1 });
                    jm::joint_matrix_prefetch<PR, XMX_PC>(sg, const_cast<half *>(Vh) + r * p.v_s1 + cc, p.v_s1,
                                                          jm::layout::row_major, syclex::properties{ syclex::prefetch_hint_L1 });
                }
            }
            jm::joint_matrix<sub_group, float, jm::use::accumulator, XMX_TM, XMX_TN> s[CT];
#pragma unroll
            for (int c = 0; c < CT; ++c) {
                jm::joint_matrix_fill(sg, s[c], 0.0f);
#pragma unroll
                for (int t = 0; t < DT; ++t) {
                    jm::joint_matrix<sub_group, half, jm::use::b, XMX_TK, XMX_TN, jm::layout::col_major> kt;
                    jmi::joint_matrix_load_checked(sg, kt, fattn_xmx_global(Kh), p.k_s1, p.nkv, D, kb + c * XMX_TN, t * XMX_TK);
                    jm::joint_matrix_mad(sg, s[c], qa[t], kt, s[c]);
                }
            }

            float bmax[XMX_TM];
#pragma unroll
            for (int i = 0; i < XMX_TM; ++i) {
                bmax[i] = -INFINITY;
            }
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

            float alpha[XMX_TM];
            float mref[XMX_TM];
#pragma unroll
            for (int i = 0; i < XMX_TM; ++i) {
                const float mn = sycl::fmax(m[i], reduce_over_group(sg, bmax[i], maximum<float>()));
                mref[i]        = mn == -INFINITY ? 0.0f : mn;
                alpha[i]       = sycl::exp2(m[i] - mref[i]);
                m[i]           = mn;
                l[i] *= alpha[i];
            }

            jm::joint_matrix<sub_group, half, jm::use::a, XMX_TM, XMX_TK, jm::layout::row_major> pa[CT];
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
            for (int t = 0; t < DT; ++t) {
                int i = 0;
                jm::joint_matrix_apply(sg, o[t], [&](float & x) {
                    x *= alpha[i];
                    ++i;
                });
#pragma unroll
                for (int c = 0; c < CT; ++c) {
                    jm::joint_matrix<sub_group, half, jm::use::b, XMX_TK, XMX_TN, jm::layout::row_major> vb;
                    jmi::joint_matrix_load_checked(sg, vb, fattn_xmx_global(Vh), p.v_s1, p.nkv, D, kb + c * XMX_TK, t * XMX_TN);
                    jm::joint_matrix_mad(sg, o[t], pa[c], vb, o[t]);
                }
            }
        }

        float inv[XMX_TM];
#pragma unroll
        for (int i = 0; i < XMX_TM; ++i) {
            const float ls = reduce_over_group(sg, l[i], plus<float>());
            inv[i]         = ls > 0.0f ? 1.0f / ls : 0.0f;
        }

        float * dh = p.dst + ((size_t) b * p.nq * p.H + h) * D;
#pragma unroll
        for (int t = 0; t < DT; ++t) {
            int i = 0;
            jm::joint_matrix_apply(sg, o[t], [&](float & x) {
                x *= inv[i];
                ++i;
            });
            jmi::joint_matrix_store_checked(sg, o[t], fattn_xmx_global(dh), (size_t) p.H * D, jm::layout::row_major, p.nq, D,
                                            q0, t * XMX_TN);
        }
    }
};

template <int D, int BC>
static void fattn_xmx_launch(dpct::queue_ptr stream, const fattn_xmx_params & p) {
    const int    nqb    = (p.nq + XMX_TM * XMX_NSG - 1) / (XMX_TM * XMX_NSG);
    const size_t groups = (size_t) nqb * p.H * p.mb;
    stream->parallel_for(sycl::nd_range<1>(groups * XMX_NSG * XMX_SG, XMX_NSG * XMX_SG), fattn_xmx_kernel<D, BC>{ p });
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

// Checks that element i of the 8x16 accumulator and A operand is (row i, column lane). It shares
// the large-GRF property with fattn_xmx_kernel so both land in the same device image, which keeps
// joint_matrix code out of the image every other SYCL kernel is JIT-compiled from.
struct fattn_xmx_layout_probe {
    int * bad;

    auto get(syclex::properties_tag) const {
        return syclex::properties{ sycl::ext::intel::experimental::grf_size<256>, syclex::sub_group_size<XMX_SG> };
    }

    void operator()(sycl::nd_item<1> it) const {
        auto         sg   = it.get_sub_group();
        const size_t lane = sg.get_local_linear_id();
        jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, XMX_TM, XMX_TN>                            c;
        jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, XMX_TM, XMX_TK, jm::layout::row_major> a;
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
};

// The kernel indexes per-row softmax state by accumulator element, so verify the element layout
// once per device before first use.
static bool fattn_xmx_layout_ok(int device, dpct::queue_ptr stream) {
    static int8_t checked[GGML_SYCL_MAX_DEVICES] = {};
    if (checked[device] != 0) {
        return checked[device] > 0;
    }

    sycl::queue q(stream->get_context(), stream->get_device());
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
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || sinks) {
        return false;
    }
    const int64_t D = K->ne[0];
    if ((D != 64 && D != 128) || Q->ne[0] != D || V->ne[0] != D) {
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
        if (t->nb[0] != sizeof(sycl::half) || (uintptr_t) t->data % 64 != 0 || t->nb[1] % 16 != 0 ||
            t->nb[2] % 64 != 0 || t->nb[3] % 64 != 0) {
            return false;
        }
    }
    return Q->nb[0] == sizeof(float) && ggml_is_contiguous(dst);
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

    const int64_t D  = K->ne[0];
    const int64_t nq = Q->ne[1];
    const int64_t H  = Q->ne[2];
    const int64_t mb = Q->ne[3];

    const ggml_sycl_fattn_extra extra = ggml_sycl_fattn_get_extra(dst);
    std::optional<ggml_sycl_pool_alloc<sycl::half>> Qf_pool;
    sycl::half * Qf = (sycl::half *) extra.Q_buffer_ptr;
    if (!Qf) {
        Qf_pool.emplace(ctx.pool(), (size_t) D * nq * H * mb);
        Qf = Qf_pool->get();
    }
    fattn_xmx_q_to_f16((const char *) Q->data, Qf, D, nq, H, mb, Q->nb[1], Q->nb[2], Q->nb[3], stream);

    fattn_xmx_params p{};
    p.Q       = Qf;
    p.K       = (const sycl::half *) K->data;
    p.V       = (const sycl::half *) V->data;
    p.mask    = mask ? (const sycl::half *) mask->data : nullptr;
    p.dst     = (float *) dst->data;
    p.k_s1    = K->nb[1] / sizeof(sycl::half);
    p.k_s2    = K->nb[2] / sizeof(sycl::half);
    p.k_s3    = K->ne[3] == 1 ? 0 : K->nb[3] / sizeof(sycl::half);
    p.v_s1    = V->nb[1] / sizeof(sycl::half);
    p.v_s2    = V->nb[2] / sizeof(sycl::half);
    p.v_s3    = V->ne[3] == 1 ? 0 : V->nb[3] / sizeof(sycl::half);
    p.mask_s1 = mask ? mask->nb[1] / sizeof(sycl::half) : 0;
    p.nq      = (int) nq;
    p.nkv     = (int) K->ne[1];
    p.H       = (int) H;
    p.Hkv     = (int) K->ne[2];
    p.mb      = (int) mb;
    memcpy(&p.scale, (const float *) dst->op_params + 0, sizeof(float));

    if (D == 64) {
        fattn_xmx_launch<64, 128>(stream, p);
    } else {
        fattn_xmx_launch<128, 64>(stream, p);
    }
#endif
}
