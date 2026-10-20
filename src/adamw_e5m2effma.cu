// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. E5M2EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// E5M2EffMA Adam / AdamW (e5ma_adam): fused in-place step over FP32 / BF16 state with an optional
// E4M3 / E5M2 copy of the updated weights.  Operation order: include/e5m2effma.h.
#include <math.h>
#include "e5m2effma.cuh"

namespace e5ma {
namespace kern {

struct AdamArgs {
  float lr, beta1, beta2, omb1, omb2, rbc1, rbc2, eps, lr_wd, wd, gscale;
  int adamw;
};

// One element, in the documented order (include/e5m2effma.h); p, m, v updated in place.
__device__ __forceinline__ void adam_elem(const AdamArgs &a, float g, float &p, float &m, float &v) {
  g = __fmul_rn(g, a.gscale);
  if (!a.adamw) g = __fmaf_rn(a.wd, p, g);
  m = __fmaf_rn(a.beta1, m, __fmul_rn(a.omb1, g));
  v = __fmaf_rn(a.beta2, v, __fmul_rn(a.omb2, __fmul_rn(g, g)));
  const float u = __fdiv_rn(__fmul_rn(m, a.rbc1), __fadd_rn(__fsqrt_rn(__fmul_rn(v, a.rbc2)), a.eps));
  if (a.adamw) p = __fmaf_rn(-a.lr_wd, p, p);
  p = __fmaf_rn(-a.lr, u, p);
}

// 8 consecutive elements of an FP32 / BF16 / FP16 array at element i (16-byte aligned): 16-byte accesses
template <int T> __device__ __forceinline__ void ld8(const void *base, size_t i, float x[8]) {
  if (T == E5MA_FP32) {
    const float4 a = ((const float4 *)base)[i / 4], b = ((const float4 *)base)[i / 4 + 1];
    x[0] = a.x; x[1] = a.y; x[2] = a.z; x[3] = a.w; x[4] = b.x; x[5] = b.y; x[6] = b.z; x[7] = b.w;
  } else {
    const uint4 w = ((const uint4 *)base)[i / 8];
    const uint32_t q[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      if (T == E5MA_BF16) { x[2 * k] = __uint_as_float(q[k] << 16); x[2 * k + 1] = __uint_as_float(q[k] & 0xFFFF0000u); }
      else { x[2 * k] = __half2float(__ushort_as_half((unsigned short)q[k])); x[2 * k + 1] = __half2float(__ushort_as_half((unsigned short)(q[k] >> 16))); }
    }
  }
}
template <int T> __device__ __forceinline__ void st8(void *base, size_t i, const float x[8]) {
  if (T == E5MA_FP32) {
    ((float4 *)base)[i / 4] = make_float4(x[0], x[1], x[2], x[3]);
    ((float4 *)base)[i / 4 + 1] = make_float4(x[4], x[5], x[6], x[7]);
  } else {  // BF16 (round to nearest even, as store_any)
    uint32_t q[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) q[k] = __byte_perm(rne_hi(__float_as_uint(x[2 * k])), rne_hi(__float_as_uint(x[2 * k + 1])), 0x7632);
    ((uint4 *)base)[i / 8] = make_uint4(q[0], q[1], q[2], q[3]);
  }
}

// PT: param dtype (FP32 / BF16), GT: grad dtype (FP32 / BF16 / FP16), MT: moment dtype (FP32 / BF16).
// QT: dtype of the FP8 copy (E4M3 or E5M2; written only when q_ is set).
// VEC: every pointer is 16-byte aligned (8-byte for the FP8 copy), so each thread handles chunks of 8
// elements with 16-byte loads and stores; the last count % 8 elements, and unaligned tensors, go
// element by element.  Both paths compute each element with adam_elem: identical results.
__device__ __forceinline__ uint32_t to_fp8(int qt, float v) { return qt == E5MA_E5M2 ? f32_to_e5m2(v) : f32_to_e4m3(v); }

template <int PT, int GT, int MT, bool VEC>
__global__ void adam_kernel(AdamArgs a, size_t count, void *__restrict__ p_, const void *__restrict__ g_,
                            void *__restrict__ m_, void *__restrict__ v_, uint8_t *__restrict__ q_,
                            int qt, const float *qscale_p, float *amax) {
  const float qs = scale_or_one(qscale_p);
  float mx = 0.f;
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x, stride = (size_t)gridDim.x * blockDim.x;
  const size_t vec_end = count / 8 * 8 * VEC;  // elements done 8 at a time
  if (VEC) {
    for (size_t i = tid * 8; i < count / 8 * 8; i += stride * 8) {
      float g[8], p[8], m[8], v[8];
      ld8<GT>(g_, i, g); ld8<PT>(p_, i, p); ld8<MT>(m_, i, m); ld8<MT>(v_, i, v);
#pragma unroll
      for (int k = 0; k < 8; ++k) { adam_elem(a, g[k], p[k], m[k], v[k]); mx = fmaxf(mx, fabsf(p[k])); }
      st8<PT>(p_, i, p); st8<MT>(m_, i, m); st8<MT>(v_, i, v);
      if (q_) {
        uint32_t w[2];
#pragma unroll
        for (int h = 0; h < 2; ++h)
          w[h] = to_fp8(qt, __fmul_rn(p[4 * h], qs)) | to_fp8(qt, __fmul_rn(p[4 * h + 1], qs)) << 8 |
                 to_fp8(qt, __fmul_rn(p[4 * h + 2], qs)) << 16 | to_fp8(qt, __fmul_rn(p[4 * h + 3], qs)) << 24;
        *(uint2 *)(q_ + i) = make_uint2(w[0], w[1]);
      }
    }
  }
  for (size_t i = vec_end + tid; i < count; i += stride) {
    float p = load_any(PT, p_, i, 1.f), m = load_any(MT, m_, i, 1.f), v = load_any(MT, v_, i, 1.f);
    adam_elem(a, load_any(GT, g_, i, 1.f), p, m, v);
    store_any(PT, p_, i, p, 1.f);
    store_any(MT, m_, i, m, 1.f);
    store_any(MT, v_, i, v, 1.f);
    if (q_) q_[i] = (uint8_t)to_fp8(qt, __fmul_rn(p, qs));
    mx = fmaxf(mx, fabsf(p));
  }
  if (amax) amax_commit(amax, mx);
}

template <int PT, int GT, int MT>
static void adam_v(const AdamArgs &a, size_t count, void *p, const void *g, void *m, void *v, uint8_t *q,
                   int qt, const float *qs, float *amax, cudaStream_t s) {
  const auto a16 = [](const void *x) { return ((uintptr_t)x & 15u) == 0; };
  const bool vec = a16(p) && a16(g) && a16(m) && a16(v) && ((uintptr_t)q & 7u) == 0;
  const size_t work = vec ? (count + 7) / 8 : count;  // threads needed
  const int grid = (int)((work + 255) / 256 < 8192 ? (work + 255) / 256 : 8192);
  if (vec) adam_kernel<PT, GT, MT, true><<<grid, 256, 0, s>>>(a, count, p, g, m, v, q, qt, qs, amax);
  else adam_kernel<PT, GT, MT, false><<<grid, 256, 0, s>>>(a, count, p, g, m, v, q, qt, qs, amax);
}
template <int PT, int GT>
static void adam_m(int mt, const AdamArgs &a, size_t count, void *p, const void *g, void *m, void *v,
                   uint8_t *q, int qt, const float *qs, float *amax, int, cudaStream_t s) {
  if (mt == E5MA_BF16) adam_v<PT, GT, E5MA_BF16>(a, count, p, g, m, v, q, qt, qs, amax, s);
  else adam_v<PT, GT, E5MA_FP32>(a, count, p, g, m, v, q, qt, qs, amax, s);
}
template <int PT>
static void adam_g(int gt, int mt, const AdamArgs &a, size_t count, void *p, const void *g, void *m, void *v,
                   uint8_t *q, int qt, const float *qs, float *amax, int grid, cudaStream_t s) {
  if (gt == E5MA_BF16) adam_m<PT, E5MA_BF16>(mt, a, count, p, g, m, v, q, qt, qs, amax, grid, s);
  else if (gt == E5MA_FP16) adam_m<PT, E5MA_FP16>(mt, a, count, p, g, m, v, q, qt, qs, amax, grid, s);
  else adam_m<PT, E5MA_FP32>(mt, a, count, p, g, m, v, q, qt, qs, amax, grid, s);
}

}  // namespace kern
}  // namespace e5ma

extern "C" e5ma_status e5ma_adam(const e5ma_adam_config *cfg, size_t count, const e5ma_tensor *param,
                                 const e5ma_tensor *grad, const e5ma_tensor *exp_avg,
                                 const e5ma_tensor *exp_avg_sq, const e5ma_tensor *param_fp8,
                                 cudaStream_t stream) {
  using e5ma::fail;
  e5ma::clear_error();
  if (!cfg || !param || !grad || !exp_avg || !exp_avg_sq || !param->data || !grad->data || !exp_avg->data ||
      !exp_avg_sq->data)
    return fail(E5MA_INVALID_VALUE, "e5ma_adam: cfg, param, grad, exp_avg, exp_avg_sq or their data is NULL");
  if (cfg->step < 1) return fail(E5MA_INVALID_VALUE, "e5ma_adam: step = %d, must be >= 1", cfg->step);
  const auto state = [](e5ma_dtype t) { return t == E5MA_FP32 || t == E5MA_BF16; };
  if (!state(param->dtype)) return fail(E5MA_NOT_SUPPORTED, "e5ma_adam: param must be FP32 or BF16 (dtype %d)", (int)param->dtype);
  if (!state(exp_avg->dtype) || exp_avg_sq->dtype != exp_avg->dtype)
    return fail(E5MA_NOT_SUPPORTED, "e5ma_adam: exp_avg and exp_avg_sq must both be FP32 or both BF16 (dtypes %d, %d)",
                (int)exp_avg->dtype, (int)exp_avg_sq->dtype);
  if (grad->dtype < E5MA_BF16 || grad->dtype > E5MA_FP32)
    return fail(E5MA_NOT_SUPPORTED, "e5ma_adam: grad must be FP32, BF16 or FP16 (dtype %d)", (int)grad->dtype);
  if (param_fp8 && param_fp8->data && !e5ma::is_fp8(param_fp8->dtype))
    return fail(E5MA_INVALID_VALUE, "e5ma_adam: param_fp8 must have dtype E5MA_E4M3 or E5MA_E5M2 (dtype %d)", (int)param_fp8->dtype);
  if (param_fp8 && param_fp8->data && param_fp8->scaling != E5MA_SCALING_TENSOR)
    return fail(E5MA_INVALID_VALUE, "e5ma_adam: param_fp8 takes per-tensor scaling only");
  if (count == 0) return E5MA_OK;
  if (e5ma_status st = e5ma::check_device()) return st;
  using namespace e5ma::kern;
  AdamArgs a;
  a.lr = cfg->lr; a.beta1 = cfg->beta1; a.beta2 = cfg->beta2; a.eps = cfg->eps; a.wd = cfg->weight_decay;
  a.omb1 = (float)(1.0 - (double)cfg->beta1);
  a.omb2 = (float)(1.0 - (double)cfg->beta2);
  a.rbc1 = cfg->bias_correction ? (float)(1.0 / (1.0 - pow((double)cfg->beta1, cfg->step))) : 1.f;
  a.rbc2 = cfg->bias_correction ? (float)(1.0 / (1.0 - pow((double)cfg->beta2, cfg->step))) : 1.f;
  a.lr_wd = (float)((double)cfg->lr * cfg->weight_decay);
  a.gscale = cfg->grad_scale == 0.f ? 1.f : cfg->grad_scale;
  a.adamw = cfg->adamw;
  uint8_t *q = param_fp8 && param_fp8->data ? (uint8_t *)param_fp8->data : nullptr;
  const int qt = q ? param_fp8->dtype : E5MA_E4M3;
  const float *qs = q ? param_fp8->scale : nullptr;
  float *amax = q ? param_fp8->amax : nullptr;
  const int grid = (int)((count + 255) / 256 < 8192 ? (count + 255) / 256 : 8192);
  if (param->dtype == E5MA_BF16)
    adam_g<E5MA_BF16>(grad->dtype, exp_avg->dtype, a, count, param->data, grad->data, exp_avg->data, exp_avg_sq->data, q, qt, qs, amax, grid, stream);
  else
    adam_g<E5MA_FP32>(grad->dtype, exp_avg->dtype, a, count, param->data, grad->data, exp_avg->data, exp_avg_sq->data, q, qt, qs, amax, grid, stream);
  return e5ma::cuda_status(cudaGetLastError(), "e5ma_adam: kernel launch");
}
