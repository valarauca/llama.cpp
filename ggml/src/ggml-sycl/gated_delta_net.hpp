#pragma once

#include <sycl/sycl.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel).
// When state_idx is set, the input state of sequence s is read from state_src + state_idx[s] * state_row_stride
// (the cache rows the skipped get_rows would have gathered) instead of src[5].
struct ggml_sycl_gated_delta_net_fused_cache {
    float *         data             = nullptr; // rollback slot 0
    int64_t         slot_stride      = 0;       // between rollback slots (0 when K==1)
    const float *   state_src        = nullptr;
    const int32_t * state_idx        = nullptr;
    int64_t         state_row_stride = 0;
};

void ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_sycl_try_gdn_cache_fusion)
void ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache);

// Fuses the gated-delta-net gate projections of a decode step (at most 8 tokens) into one kernel when nodes
// node_idx.. hold MUL_MAT(alpha), RESHAPE, ADD(dt bias), SOFTPLUS, MUL(a), RESHAPE, MUL_MAT(beta), RESHAPE,
// SIGMOID with F32 weights. Returns the number of nodes to skip after node_idx, or 0 if the pattern does not apply.
int ggml_sycl_try_gdn_gate_proj_fusion(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int node_idx);
