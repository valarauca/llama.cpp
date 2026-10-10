#ifndef GGML_SYCL_FATTN_XMX_HPP
#define GGML_SYCL_FATTN_XMX_HPP

#include "common.hpp"

// Static check for the joint_matrix (XMX/DPAS) flash-attention kernel: SIMD16 DPAS device,
// f32 Q, f16/q8_0/q4_0 K/V (quantized K/V are converted to f16 first) with head_dim 64, 128, 256
// or MLA 576/512, optional f16 mask, no sinks/ALiBi/softcap.
bool ggml_sycl_flash_attn_ext_xmx_supported(const ggml_tensor * dst);

// True when the kernel stages a dense f16 copy of Q (head_dim 256, or Q strides the fused f32
// load cannot take), so the caller reserves scratch for it.
bool ggml_sycl_flash_attn_ext_xmx_needs_q_f16(const ggml_tensor * dst);

// Run flash attention through the joint_matrix kernel. Falls back to the TILE kernel if the
// device's joint_matrix element layout does not match the one the kernel was written against.
void ggml_sycl_flash_attn_ext_xmx(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_FATTN_XMX_HPP
