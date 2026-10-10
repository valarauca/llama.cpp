// A/B harness for the SYCL flash-attention kernels.
//
// Runs GGML_OP_FLASH_ATTN_EXT on identical inputs through the CPU backend (the reference) and
// through SYCL0 once per kernel, forcing each with GGML_SYCL_FA_KERNEL. Reports the kernel that
// actually ran, the error against the CPU result, and the time per call.
//
// usage: test-fattn-ab [shape-name-filter]

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

struct fa_shape {
    const char * name;
    int64_t      D;
    int64_t      nq;
    int64_t      nkv;
    int64_t      H;
    int64_t      Hkv;
    bool         causal;
    bool         token_major;
};

struct fa_inputs {
    std::vector<float>       q;
    std::vector<ggml_fp16_t> k;
    std::vector<ggml_fp16_t> v;
    std::vector<ggml_fp16_t> mask;
};

struct fa_graph {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_cgraph *         gf  = nullptr;
    ggml_tensor *         q   = nullptr;
    ggml_tensor *         k   = nullptr;
    ggml_tensor *         v   = nullptr;
    ggml_tensor *         m   = nullptr;
    ggml_tensor *         out = nullptr;

    ~fa_graph() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
};

static std::string g_last_kernel;

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static void capture_dispatch_log(ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    const char * tag = strstr(text, "[FA-DISP]");
    if (tag) {
        char name[16] = {};
        if (sscanf(tag, "[FA-DISP] #%*d %15s", name) == 1) {
            g_last_kernel = name;
        }
        return;
    }
    if (level >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

static fa_inputs make_inputs(const fa_shape & s) {
    std::mt19937                    rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    fa_inputs                       in;
    in.q.resize(s.D * s.nq * s.H);
    in.k.resize(s.D * s.nkv * s.Hkv);
    in.v.resize(s.D * s.nkv * s.Hkv);
    for (auto & x : in.q) {
        x = nd(rng);
    }
    for (auto & x : in.k) {
        x = ggml_fp32_to_fp16(nd(rng));
    }
    for (auto & x : in.v) {
        x = ggml_fp32_to_fp16(nd(rng));
    }
    if (s.causal) {
        in.mask.resize(s.nkv * s.nq);
        for (int64_t i = 0; i < s.nq; ++i) {
            for (int64_t j = 0; j < s.nkv; ++j) {
                in.mask[i * s.nkv + j] = ggml_fp32_to_fp16(j <= i + (s.nkv - s.nq) ? 0.0f : -INFINITY);
            }
        }
    }
    return in;
}

static void build(fa_graph & g, const fa_shape & s, const fa_inputs & in, ggml_backend_t backend) {
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    g.ctx               = ggml_init(ip);

    ggml_tensor * q;
    ggml_tensor * k;
    ggml_tensor * v;
    if (s.token_major) {
        g.q = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, s.D, s.H, s.nq, 1);
        g.k = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, s.D, s.Hkv, s.nkv, 1);
        g.v = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, s.D, s.Hkv, s.nkv, 1);
        q   = ggml_permute(g.ctx, g.q, 0, 2, 1, 3);
        k   = ggml_permute(g.ctx, g.k, 0, 2, 1, 3);
        v   = ggml_permute(g.ctx, g.v, 0, 2, 1, 3);
    } else {
        g.q = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, s.D, s.nq, s.H, 1);
        g.k = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, s.D, s.nkv, s.Hkv, 1);
        g.v = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, s.D, s.nkv, s.Hkv, 1);
        q   = g.q;
        k   = g.k;
        v   = g.v;
    }
    if (s.causal) {
        g.m = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, s.nkv, s.nq, 1, 1);
    }

    g.out = ggml_flash_attn_ext(g.ctx, q, k, v, g.m, 1.0f / std::sqrt((float) s.D), 0.0f, 0.0f);
    ggml_prec_set_acc(g.out, GGML_PREC_F32);
    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, g.out);

    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, backend);
    ggml_backend_tensor_set(g.q, in.q.data(), 0, ggml_nbytes(g.q));
    ggml_backend_tensor_set(g.k, in.k.data(), 0, ggml_nbytes(g.k));
    ggml_backend_tensor_set(g.v, in.v.data(), 0, ggml_nbytes(g.v));
    if (g.m) {
        ggml_backend_tensor_set(g.m, in.mask.data(), 0, ggml_nbytes(g.m));
    }
}

static std::vector<float> read_out(const fa_graph & g) {
    std::vector<float> out(ggml_nelements(g.out));
    ggml_backend_tensor_get(g.out, out.data(), 0, ggml_nbytes(g.out));
    return out;
}

static void compare(const std::vector<float> & a, const std::vector<float> & ref, double & nmse, double & maxabs) {
    double err = 0.0, nrm = 0.0;
    maxabs     = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double e = (double) a[i] - (double) ref[i];
        err += e * e;
        nrm += (double) ref[i] * ref[i];
        maxabs = std::isnan(e) ? INFINITY : std::max(maxabs, std::fabs(e));
    }
    nmse = nrm > 0.0 ? err / nrm : err;
}

int main(int argc, char ** argv) {
    const char * filter = argc > 1 ? argv[1] : nullptr;

    set_env("GGML_SYCL_MKL_FA_DEBUG", "1");
    ggml_log_set(capture_dispatch_log, nullptr);

    ggml_backend_dev_t sycl_dev = ggml_backend_dev_by_name("SYCL0");
    if (!sycl_dev) {
        fprintf(stderr, "no SYCL0 device\n");
        return 1;
    }
    ggml_backend_t sycl = ggml_backend_dev_init(sycl_dev, nullptr);
    ggml_backend_t cpu  = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, (int) std::max(1u, std::thread::hardware_concurrency()));

    const fa_shape shapes[] = {
        { "dit-4096x4096-h24",        128, 4096, 4096, 24, 24, false, true  },
        { "dit-4096x4121-h24",        128, 4096, 4121, 24, 24, false, true  },
        { "llm-gqa-causal-512x1024",  128,  512, 1024, 32,  8, true,  false },
        { "llm-gqa-causal-77x333",    128,   77,  333, 32,  8, true,  false },
        { "d64-causal-1000x1000-h16",  64, 1000, 1000, 16, 16, true,  false },
        { "d64-unmasked-1024x1024",    64, 1024, 1024, 16, 16, false, true  },
    };
    const char * kernels[] = { "tile", "onednn", "xmx" };
    const double nmse_max  = 5e-4;
    const int    iters     = 5;

    int failures = 0;
    printf("%-26s %-7s %-7s %11s %11s %10s %8s\n", "shape", "forced", "ran", "nmse", "maxabs", "ms/call", "TFLOPS");
    for (const fa_shape & s : shapes) {
        if (filter && !strstr(s.name, filter)) {
            continue;
        }
        const fa_inputs in = make_inputs(s);

        fa_graph ref_g;
        build(ref_g, s, in, cpu);
        ggml_backend_graph_compute(cpu, ref_g.gf);
        const std::vector<float> ref = read_out(ref_g);

        const double flops = 4.0 * (double) s.H * s.nq * s.nkv * s.D;
        for (const char * kernel : kernels) {
            set_env("GGML_SYCL_FA_KERNEL", kernel);
            fa_graph g;
            build(g, s, in, sycl);

            g_last_kernel.clear();
            ggml_backend_graph_compute(sycl, g.gf);
            ggml_backend_synchronize(sycl);
            const std::string ran = g_last_kernel;

            const auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; ++i) {
                ggml_backend_graph_compute(sycl, g.gf);
            }
            ggml_backend_synchronize(sycl);
            const auto   t1 = std::chrono::high_resolution_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

            double nmse, maxabs;
            compare(read_out(g), ref, nmse, maxabs);
            const bool bad = !(nmse <= nmse_max);
            failures += bad;
            printf("%-26s %-7s %-7s %11.3e %11.3e %10.3f %8.1f%s\n", s.name, kernel, ran.c_str(), nmse, maxabs, ms,
                   flops / ms / 1e9, bad ? "  FAIL" : "");
        }
    }
    set_env("GGML_SYCL_FA_KERNEL", "");

    ggml_backend_free(sycl);
    ggml_backend_free(cpu);
    printf("%s\n", failures ? "FAIL" : "OK");
    return failures ? 1 : 0;
}
