# C API

Everything is declared in [`include/e5m2effma.h`](../include/e5m2effma.h), which also documents the
exact order of operations. The semantics follow cuBLASLt / Transformer Engine (`nvte_cublas_gemm_v2`):
column-major matrices, per-tensor or row-wise (outer) scales.

```c
typedef struct {
  void *data;              /* device pointer */
  e5ma_dtype dtype;        /* E5MA_E5M2, E5MA_E4M3, E5MA_BF16, E5MA_FP16, E5MA_FP32 */
  const float *scale_inv;  /* FP8 input:  dequantization factor (device, 1 float), NULL = 1 */
  const float *scale;      /* FP8 output: quantization factor   (device, 1 float), NULL = 1 */
  float *amax;             /* optional: max |value| before quantization, max-reduced into it */
  e5ma_scaling scaling;    /* E5MA_SCALING_TENSOR (0, default) or E5MA_SCALING_OUTER */
} e5ma_tensor;
```

With `scaling = E5MA_SCALING_OUTER` an FP8 GEMM operand's `scale_inv` is a vector along its outer
dimension: `m` floats for A (one per row of D), `n` floats for B (one per column of D); the factor of
`D[i, j]` is then `(α·sa[i])·sb[j]`, applied once to the FP32 sum (scales along K cannot be). C, D, the
bias and `e5ma_cast` / `e5ma_adam` operands take per-tensor scales only. Initializers that leave the
field out get per-tensor scaling.

| Function | Description |
|---|---|
| `e5ma_gemm` | `D = α·(sa·sb)·op(A)·op(B) + bias + β·C` (`sa`, `sb` per tensor or per row / column), column-major. A, B ∈ {E5M2, E4M3, BF16, FP32} in any combination and any N/T layout; D ∈ {E5M2, E4M3, BF16, FP16, FP32}. C may have any dtype, may alias D, and `C = NULL` with β ≠ 0 accumulates into D (TE `accumulate`). For `n ≤ 8`, TN layout and an FP8 A the bandwidth-bound GEMV kernel runs. |
| `e5ma_cast` | Element-wise conversion between E5M2, E4M3, BF16, FP16 and FP32 with `scale_inv` / `scale` / `amax`. FP8 sources decode exactly, Inf / NaN included; FP32 → FP8 is round-to-nearest-even with saturation (`cvt.rn.satfinite.e5m2x2` / `.e4m3x2` semantics), so an overflow shows in `amax`, not in the data. |
| `e5ma_adam` | In-place Adam / AdamW step. Params FP32 or BF16, grads FP32 / BF16 / FP16, moments FP32 or BF16, optional E5M2 or E4M3 copy of the updated weights with `scale` and `amax`. |
| `e5ma_status_string` | Name of a status code. |
| `e5ma_last_error` | Description of the last failure on the calling thread: the invalid argument, the CUDA error, or a device the build does not cover (sm_89+ with native FP8, or an architecture missing from `ARCHS`). |

**Constraints of `e5ma_gemm`** (cuBLASLt-like): A and B 16-byte aligned; `k`, `lda` and `ldb`
multiples of 16; `m`, `n`, `ldc`, `ldd` arbitrary (`ldc, ldd ≥ m`); A and B finite (no E5M2 Inf / NaN
codes, no E4M3 NaN code: not checked).
"TN" (`transa = T`, `transb = N`) is the FP8 layout of cuBLASLt / TE: both operands K-contiguous.

## Errors

Every function returns an `e5ma_status` (`E5MA_OK`, `E5MA_INVALID_VALUE`, `E5MA_NOT_SUPPORTED`,
`E5MA_CUDA_ERROR`) and validates its arguments before launching anything. On failure
`e5ma_last_error()` says what was wrong:

```
e5ma_gemm: k = 24 is not a multiple of 16
e5ma_cast: invalid dtype (src 3, dst 7)
device 0 is sm_61, but this build of the library has no code for it: rebuild with ARCHS including 61
```

## Example

```c
#include "e5m2effma.h"

// Linear layer backward in TE's HYBRID format: W [out, in] and X [in, batch] are E4M3 (forward
// operands), the incoming gradient dY [out, batch] is E5M2.  Column-major throughout.
e5ma_tensor W  = {d_w, E5MA_E4M3, d_w_scale_inv, NULL, NULL};    // data, dtype, scale_inv, scale, amax
e5ma_tensor X  = {d_x, E5MA_E4M3, d_x_scale_inv, NULL, NULL};
e5ma_tensor dY = {d_dy, E5MA_E5M2, d_dy_scale_inv, NULL, NULL};

// dgrad: dX [in, batch] = Wᵀ·dY in BF16 (W stored [out, in] row-major = in x out column-major)
e5ma_tensor dX = {d_dx, E5MA_BF16, NULL, NULL, NULL};
e5ma_status st = e5ma_gemm(E5MA_OP_N, E5MA_OP_N, in, batch, out, 1.0f, &W, in, &dY, out,
                           0.0f, NULL, 0, &dX, in, /*bias=*/NULL, stream);
if (st != E5MA_OK) fprintf(stderr, "%s: %s\n", e5ma_status_string(st), e5ma_last_error());

// wgrad accumulated into an FP32 main_grad (TE "accumulate"): G [out, in] += dY·Xᵀ
e5ma_tensor G = {d_main_grad, E5MA_FP32, NULL, NULL, NULL};
e5ma_gemm(E5MA_OP_N, E5MA_OP_T, out, in, batch, 1.0f, &dY, out, &X, in,
          1.0f, /*C = D*/ NULL, 0, &G, out, NULL, stream);

// AdamW step on BF16 weights + FP32 moments, refreshing the E4M3 copy used by the next forward pass
e5ma_tensor P = {d_param, E5MA_BF16, NULL, NULL, NULL};
e5ma_tensor M = {d_exp_avg, E5MA_FP32, NULL, NULL, NULL}, V = {d_exp_avg_sq, E5MA_FP32, NULL, NULL, NULL};
e5ma_adam_config cfg = {.lr = 1e-3f, .beta1 = 0.9f, .beta2 = 0.999f, .eps = 1e-8f,
                         .weight_decay = 0.01f, .step = t, .adamw = 1, .bias_correction = 1, .grad_scale = 1.0f};
e5ma_tensor Wq = {d_w, E5MA_E4M3, NULL, d_w_scale, d_w_amax};  // E4M3 copy: quantization scale + amax
e5ma_adam(&cfg, numel, &P, &G, &M, &V, &Wq, stream);
```

The library is a plain C ABI, so it can also be called from any language with a C FFI.
