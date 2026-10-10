#ifndef GGML_SYCL_FATTN_XMX_HPP
#define GGML_SYCL_FATTN_XMX_HPP

#include "common.hpp"

// Static check for the joint_matrix (XMX/DPAS) flash-attention kernel: SIMD16 DPAS device,
// f32 Q, f16 K/V with head_dim 64 or 128, optional f16 mask, no sinks/ALiBi/softcap.
bool ggml_sycl_flash_attn_ext_xmx_supported(const ggml_tensor * dst);

// Run flash attention through the joint_matrix kernel. Falls back to the TILE kernel if the
// device's joint_matrix element layout does not match the one the kernel was written against.
void ggml_sycl_flash_attn_ext_xmx(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_FATTN_XMX_HPP
