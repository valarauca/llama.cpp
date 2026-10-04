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

#ifndef GGML_SYCL_MOE_GROUP_HPP
#define GGML_SYCL_MOE_GROUP_HPP

#include "common.hpp"

// True if ggml_sycl_moe_grouped can run src0 (expert weights) of this type on the device: a build with the
// grouped kernel, a device with 16-wide XMX, and a type with a grouped dequant. Q4_0 additionally needs the
// per-expert reorder layout, which the caller installs.
bool ggml_sycl_moe_grouped_supported(int device, const ggml_tensor * src0, const ggml_tensor * src1);

// MUL_MAT_ID for a batch of tokens without reading ids on the host: counts and sorts the routed rows per expert
// on the device, gathers them as fp16 and runs one XMX grouped GEMM over all experts, which writes dst directly.
// Returns false (having done nothing) if the shapes are outside what the kernel handles.
bool ggml_sycl_moe_grouped(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                           const ggml_tensor * ids, ggml_tensor * dst);

#endif  // GGML_SYCL_MOE_GROUP_HPP
