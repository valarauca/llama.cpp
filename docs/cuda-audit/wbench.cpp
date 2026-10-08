// DRAM-bound mat-vec bandwidth: nw distinct [K x M] weights (> L2 in total), cycled through `cycles` times per graph,
// so every MUL_MAT reads its weight from DRAM. nw=1 keeps the weight L2-resident for comparison.
// usage: wbench <type> <n[,n...]> [nw=16] [reps=40]   prints us per MUL_MAT and weight GB/s
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static const int64_t K = 14336;
static int64_t M = 4096; // rows, override with WB_M

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: wbench <type> <n[,n...]> [nw=16] [reps=40]\n");
        return 1;
    }
    ggml_backend_load_all();
    if (getenv("WB_M")) {
        M = atoll(getenv("WB_M"));
    }
    ggml_type type = GGML_TYPE_COUNT;
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const char * nm = ggml_type_name((ggml_type) t);
        if (nm && !strcmp(nm, argv[1])) {
            type = (ggml_type) t;
        }
    }
    if (type == GGML_TYPE_COUNT) {
        fprintf(stderr, "unknown type\n");
        return 1;
    }
    std::vector<int64_t> ns;
    for (char * tok = strtok(argv[2], ","); tok; tok = strtok(nullptr, ",")) {
        ns.push_back(atoll(tok));
    }
    const int     nw     = argc > 3 ? atoi(argv[3]) : 16;
    const int     reps   = argc > 4 ? atoi(argv[4]) : 40;
    const int     cycles = nw == 1 ? 64 : 4;

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        fprintf(stderr, "no gpu\n");
        return 1;
    }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);

    // one quantized weight, copied into every slot
    ggml_quantize_init(type);
    std::mt19937 gen(42);
    std::normal_distribution<float> nd(0.0f, 0.02f);
    std::vector<float> wf(M * K);
    for (auto & v : wf) {
        v = nd(gen);
    }
    std::vector<float> ones(K, 1.0f);
    std::vector<uint8_t> wq(ggml_row_size(type, K) * M);
    ggml_quantize_chunk(type, wf.data(), wq.data(), 0, M, K, ggml_quantize_requires_imatrix(type) ? ones.data() : nullptr);
    const int n_nodes = nw * cycles;
    ggml_init_params wp = { ggml_tensor_overhead() * (nw + 1), nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    std::vector<ggml_tensor *> w(nw);
    for (auto & t : w) {
        t = ggml_new_tensor_2d(wctx, type, K, M);
    }
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, be);
    if (!wbuf) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }
    for (auto & t : w) {
        ggml_backend_tensor_set(t, wq.data(), 0, wq.size());
    }

    for (int64_t n : ns) {
        std::vector<float> x(K * n);
        std::uniform_real_distribution<float> ux(-1.0f, 1.0f);
        for (auto & v : x) {
            v = ux(gen);
        }
        ggml_init_params ip = { ggml_tensor_overhead() * (n_nodes + 8) + ggml_graph_overhead_custom(n_nodes + 8, false), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, n);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, n_nodes + 8, false);
        for (int i = 0; i < n_nodes; i++) {
            ggml_tensor * y = ggml_mul_mat(ctx, w[i % nw], b);
            ggml_set_output(y);
            ggml_build_forward_expand(gf, y);
        }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        ggml_backend_tensor_set(b, x.data(), 0, x.size() * sizeof(float));

        for (int i = 0; i < 5; i++) {
            ggml_backend_graph_compute(be, gf);
        }
        ggml_backend_synchronize(be);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; i++) {
            ggml_backend_graph_compute(be, gf);
        }
        ggml_backend_synchronize(be);
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / ((double) reps * n_nodes);

        const double bytes = (double) ggml_nbytes(w[0]);
        printf("%-7s n=%-3ld nw=%-2d bpw=%.3f  %8.2f us/op  %7.1f GB/s (weight bytes %.1f MB, total %.0f MB)\n", ggml_type_name(type), (long) n, nw,
               8.0 * bytes / (M * K), us, bytes / us * 1e-3, bytes / 1e6, bytes * nw / 1e6);
        fflush(stdout);

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    ggml_backend_free(be);
    return 0;
}
