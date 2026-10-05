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

#ifndef GGML_SYCL_MOE_CACHE_HPP
#define GGML_SYCL_MOE_CACHE_HPP

#include "common.hpp"

// Pinned host buffer type for routed expert tensors, named SYCL<n>_MOE and selected with
// -ot "\.ffn_(up|down|gate|gate_up)_(ch|)exps=SYCL0_MOE". The weights are copied out of the model file into
// SYCL-owned host memory, and MUL_MAT_ID pages the experts it needs into per-layer device slot pools of
// GGML_SYCL_MOE_CACHE_SLOTS experts each (default 32).
ggml_backend_buffer_type_t ggml_backend_sycl_moe_buffer_type(int device);

// True if the buffer was allocated from a SYCL MoE cache buffer type.
bool ggml_backend_buffer_is_sycl_moe(ggml_backend_buffer_t buffer);

// Lays out each cached expert tensor in device order (the MUL_MAT_ID reorder), called once per member before
// any of its experts are copied to the device.
typedef void (*ggml_sycl_moe_cache_reorder_t)(ggml_backend_sycl_context * ctx, const ggml_tensor * tensor);

// For a MUL_MAT_ID whose src0 lives in a MoE cache buffer: reads ids on the host, copies the experts the layer
// is missing into its slot pools and fills src0_view and ids_view (expert ids remapped to slots) so the regular
// MUL_MAT_ID paths can run on device memory.
void ggml_sycl_moe_cache_prepare(ggml_backend_sycl_context & ctx, const ggml_tensor * dst,
                                 ggml_sycl_moe_cache_reorder_t reorder, ggml_tensor & src0_view,
                                 ggml_tensor & ids_view);

#endif  // GGML_SYCL_MOE_CACHE_HPP
