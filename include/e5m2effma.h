// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. E5M2EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
/* E5M2EffMA: FP8 (E5M2 and E4M3) GEMM / GEMV / cast / Adam for NVIDIA GPUs without native FP8
 * (sm_50 ... sm_87).
 *
 * GEMM semantics follow cuBLASLt / Transformer Engine (nvte_cublas_gemm_v2), column-major:
 *
 *     D = alpha * (sa * sb) * op(A) * op(B) + bias + beta * C          D, C: m x n
 *
 *   op(A) is m x k: transa = E5MA_OP_T -> A stored k x m (lda >= k, K contiguous)
 *                   transa = E5MA_OP_N -> A stored m x k (lda >= m)
 *   op(B) is k x n: transb = E5MA_OP_N -> B stored k x n (ldb >= k, K contiguous)
 *                   transb = E5MA_OP_T -> B stored n x k (ldb >= n)
 *   A, B: E5M2, E4M3, BF16 or FP32 in any combination -- e.g. the backward GEMMs of TE's HYBRID
 *   format (E4M3 weights / activations x E5M2 gradients), or FP8 weights x BF16 activations.
 *   sa, sb = scale_inv of FP8 operands (1 if NULL or not FP8): one factor per tensor, or with
 *   scaling = E5MA_SCALING_OUTER one per row of op(A) (m floats) / per column of op(B) (n floats),
 *   as the row-wise scaling of cuBLASLt / torch._scaled_mm.  bias has length m.
 *   "TN" (transa = T, transb = N) is the FP8 layout of cuBLASLt / TE: both operands K-contiguous.
 *
 * Accumulation (TE `accumulate`): C may alias D, and C == NULL with beta != 0 means C = D, so
 * beta = 1 adds the product into D in place -- e.g. FP32 main_grad += dY^T X.  C may have any
 * dtype (FP8 C is dequantized with C->scale_inv).
 *
 * Arithmetic: products of finite FP8 values (E5M2 / E4M3, any pairing) are exact in FP32, FP8 x BF16
 * products whenever they are normal FP32 numbers (FP32 x FP32 products round as in SGEMM);
 * accumulation is one FP32 FFMA per MAC in a fixed order (ascending k for the tiled kernel;
 * k = 256c + 8l + j per lane then an xor-butterfly for the GEMV kernel).  Epilogue, in FP32:
 *     v = acc * s  (s = (alpha * sa[i]) * sb[j]);  v = v + bias[i];  v = beta * C + v (one FMA);
 *     amax = max(amax, |v|);  D = convert(v)   (BF16 / FP16 / FP32: RNE; FP8: RNE satfinite of
 *                                               v * D->scale, NaN -> 0x7F)
 *
 * Dispatch: n <= 8, transa = T, transb = N and an FP8 A run the bandwidth-bound GEMV kernel;
 * everything else runs the tiled GEMM kernel.  Kernel configurations are chosen per compute
 * capability (tuned with `make tune`, defaults otherwise).
 *
 * Requirements (cuBLASLt-like FP8 constraints): A and B 16-byte aligned; k, lda, ldb multiples of 16.
 * m, n, ldc, ldd are unrestricted (ldc, ldd >= m).  A and B must hold finite values: no E5M2 Inf / NaN
 * codes and no E4M3 NaN code (not checked; they would be read as finite numbers).  Casts, C and bias
 * read Inf / NaN exactly.
 */
#ifndef E5M2EFFMA_H
#define E5M2EFFMA_H

#define E5M2EFFMA_VERSION "0.1.0"

#include <stddef.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { E5MA_E4M3 = 0, E5MA_BF16 = 1, E5MA_FP16 = 2, E5MA_FP32 = 3, E5MA_E5M2 = 4 } e5ma_dtype;
typedef enum { E5MA_OP_N = 0, E5MA_OP_T = 1 } e5ma_op;
typedef enum { E5MA_OK = 0, E5MA_INVALID_VALUE = 1, E5MA_NOT_SUPPORTED = 2, E5MA_CUDA_ERROR = 3 } e5ma_status;
/* How scale_inv of an FP8 GEMM operand is applied: one factor for the whole tensor, or one per
 * outer index (row of op(A), column of op(B)); never along K. */
typedef enum { E5MA_SCALING_TENSOR = 0, E5MA_SCALING_OUTER = 1 } e5ma_scaling;

typedef struct {
  void *data;              /* device pointer */
  e5ma_dtype dtype;
  const float *scale_inv;  /* FP8 input:  dequantization factor (device, 1 float), NULL = 1 */
  const float *scale;      /* FP8 output: quantization factor   (device, 1 float), NULL = 1 */
  float *amax;             /* optional output statistic: max |v| before quantization (device,
                              1 float, max-reduced into the existing value); NULL = off */
  e5ma_scaling scaling;    /* FP8 GEMM operands: E5MA_SCALING_OUTER makes scale_inv a vector
                              (m floats for A, n for B); everything else: E5MA_SCALING_TENSOR (0) */
} e5ma_tensor;

/* D = alpha * (sa * sb) * op(A) * op(B) + bias + beta * C; see the header comment. */
e5ma_status e5ma_gemm(e5ma_op transa, e5ma_op transb, int m, int n, int k, float alpha,
                      const e5ma_tensor *A, int lda, const e5ma_tensor *B, int ldb, float beta,
                      const e5ma_tensor *C, int ldc, const e5ma_tensor *D, int ldd,
                      const e5ma_tensor *bias, cudaStream_t stream);

/* Element-wise conversion of count values: v = src * src.scale_inv (FP8 source; Inf / NaN codes
 * decode to Inf / NaN); dst = convert(v) (FP8 destination: RNE satfinite of v * dst.scale, so an
 * overflow saturates -- watch dst.amax, which gets max |v| and becomes Inf / NaN with the input). */
e5ma_status e5ma_cast(const e5ma_tensor *src, const e5ma_tensor *dst, size_t count, cudaStream_t stream);

/* Adam / AdamW step over count elements, in place (PyTorch / TE fused-Adam semantics).
 *   param       FP32 or BF16 (updated); grad FP32, BF16 or FP16 (e.g. the BF16 / FP32 output of
 *               e5ma_gemm); exp_avg, exp_avg_sq FP32 or BF16 (updated)
 *   param_fp8   optional: E4M3 or E5M2 copy of the updated FP32 value for the next FP8 pass,
 *               quantized with param_fp8->scale; param_fp8->amax gets max |param|
 * Per element, in FP32 (bit-reproducible; host emulation in tests/test_adamw.cu):
 *   g = grad * grad_scale;  if (!adamw) g = weight_decay * p + g (one FMA)
 *   m = beta1 * m + (1 - beta1) * g;       v = beta2 * v + (1 - beta2) * (g * g)   (one FMA each)
 *   u = (m * rbc1) / (sqrt(v * rbc2) + eps)  with rbc = 1 / (1 - beta^step) (1 if no bias corr.)
 *   if (adamw) p = p - (lr * weight_decay) * p (one FMA);   p = p - lr * u (one FMA)
 * The scalar factors (1 - beta, rbc, lr * weight_decay) are formed on the host in double and
 * rounded to FP32 once. */
typedef struct {
  float lr, beta1, beta2, eps, weight_decay;
  int step;             /* 1-based step number */
  int adamw;            /* 1: decoupled weight decay (AdamW); 0: L2 term added to the gradient */
  int bias_correction;  /* 1: Adam bias correction */
  float grad_scale;     /* gradient multiplier, e.g. 1 / loss_scale (0 is treated as 1) */
} e5ma_adam_config;

e5ma_status e5ma_adam(const e5ma_adam_config *cfg, size_t count, const e5ma_tensor *param,
                      const e5ma_tensor *grad, const e5ma_tensor *exp_avg,
                      const e5ma_tensor *exp_avg_sq, const e5ma_tensor *param_fp8,
                      cudaStream_t stream);

const char *e5ma_status_string(e5ma_status status);

/* Description of the most recent failure of an e5ma_* call on the calling thread (which argument
 * was wrong, or the CUDA error); "" after a successful call.  Valid until the next e5ma_* call. */
const char *e5ma_last_error(void);

#ifdef __cplusplus
}
#endif
#endif /* E5M2EFFMA_H */
