// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. E5M2EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// E5M2EffMA core (device side): E5M2 / E4M3 decode / encode, BF16 rounding, typed element access, the
// GEMM / GEMV epilogue and the per-architecture kernel tables.  Shared by gemm_*, gemv_*, adamw_*.
//
//   arithmetic : one FP32 FFMA per MAC; FP8 x FP8 products (E5M2 or E4M3) are exact in FP32
//   rounding   : accumulate in FP32 over the whole reduction, round once when storing.  Per-MAC BF16
//                rounding drifts (5-136 ulp mean, stagnation on same-sign data) and rounding every N
//                MACs only adds error; one rounding at the end matched the correctly rounded exact
//                sum in 99.9-100 % of outputs (max 0.5 ulp) in the distributions studied (E4M3 data;
//                the argument does not depend on the input format).
//   contract   : no E5M2 Inf / NaN codes (0x7C..0x7F / 0xFC..0xFF) and no E4M3 NaN code (0x7F / 0xFF) in
//                GEMM operands -- the GEMM decode maps them to finite values.  Casts, C and bias decode
//                them exactly (Inf / NaN stay Inf / NaN).
#pragma once
#include <stdint.h>
#include <string.h>
#include <cuda_fp16.h>
#include "../include/e5m2effma.h"

// Build policy: only for targets without native FP8 (sm_89+ has it: use cuBLASLt / TE there).
#if defined(__CUDA_ARCH__)
#if __CUDA_ARCH__ < 500 || __CUDA_ARCH__ >= 890
#error "E5M2EffMA targets sm_50..sm_87 (no native FP8); exclude it from sm_89+ builds."
#endif
#endif

namespace e5ma {

// ---- arithmetic primitives -------------------------------------------------------------------
// explicit .rn without .ftz: denormals must be honoured
__device__ __forceinline__ float fma_rn(float a, float b, float c) {
  float d;
  asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}
__device__ __forceinline__ float mul_rn(float a, float b) {
  float d;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}

// Exact FP32 values of the four E4M3 bytes of w (byte k -> vk), as decoded for GEMM operands.
// raw(x) = sign | (x & 0x7F) << 20 is the FP32 encoding of value(x) * 2^-120 for every finite
// E4M3 x (E4M3 subnormals land exactly on FP32 subnormals); one FMUL by 2^120 restores value(x).
__device__ __forceinline__ void decode_e4m3x4(uint32_t w, float &v0, float &v1, float &v2, float &v3) {
  const float K = 0x1p120f;
  v0 = mul_rn(__uint_as_float(((w << 24) & 0x80000000u) | ((w << 20) & 0x07F00000u)), K);
  v1 = mul_rn(__uint_as_float(((w << 16) & 0x80000000u) | ((w << 12) & 0x07F00000u)), K);
  v2 = mul_rn(__uint_as_float(((w << 8) & 0x80000000u) | ((w << 4) & 0x07F00000u)), K);
  v3 = mul_rn(__uint_as_float((w & 0x80000000u) | ((w >> 4) & 0x07F00000u)), K);
}
// One E4M3 byte for casts, C and bias: the NaN code decodes to NaN.
__device__ __forceinline__ float e4m3_value(uint32_t b) {
  float v0, v1, v2, v3;
  decode_e4m3x4(b & 0xFFu, v0, v1, v2, v3);
  return (b & 0x7Fu) == 0x7Fu ? __int_as_float(0x7FFFFFFF) : v0;
}
// The same for E5M2 (bias 15, 2 mantissa bits): raw(x) = sign | (x & 0x7F) << 21 encodes
// value(x) * 2^-112 for every finite E5M2 x (subnormals included); one FMUL by 2^112 restores it.
// Inf / NaN codes (exponent 31) come out as finite values (contract: not in GEMM operands).
__device__ __forceinline__ void decode_e5m2x4(uint32_t w, float &v0, float &v1, float &v2, float &v3) {
  const float K = 0x1p112f;
  v0 = mul_rn(__uint_as_float(((w << 24) & 0x80000000u) | ((w << 21) & 0x0FE00000u)), K);
  v1 = mul_rn(__uint_as_float(((w << 16) & 0x80000000u) | ((w << 13) & 0x0FE00000u)), K);
  v2 = mul_rn(__uint_as_float(((w << 8) & 0x80000000u) | ((w << 5) & 0x0FE00000u)), K);
  v3 = mul_rn(__uint_as_float((w & 0x80000000u) | ((w >> 3) & 0x0FE00000u)), K);
}
// E5M2 is the upper byte of an FP16: exact, Inf / NaN included (casts, C, bias)
__device__ __forceinline__ float e5m2_value(uint32_t b) {
  return __half2float(__ushort_as_half((unsigned short)(b << 8)));
}
// 4 FP8 bytes of type T (E5MA_E5M2 or E5MA_E4M3) -> 4 exact FP32 values
template <int T>
__device__ __forceinline__ void decode_fp8x4(uint32_t w, float &v0, float &v1, float &v2, float &v3) {
  if (T == E5MA_E5M2) decode_e5m2x4(w, v0, v1, v2, v3);
  else decode_e4m3x4(w, v0, v1, v2, v3);
}

// FP32 bits -> BF16 bits in the high half, round-to-nearest-even; NaN -> canonical 0x7FC0.
// u + 0x7FFF + lsb(bit 16), with the LSB moved into the carry flag (t + 0xFFFF0000 carries iff
// t == 0x10000): LOP + IADD.CC + IADD.X, all full rate.
__device__ __forceinline__ uint32_t rne_hi(uint32_t u) {
  u = (u & 0x7FFFFFFFu) > 0x7F800000u ? 0x7FC00000u : u;  // NaN would otherwise wrap/round to Inf
  uint32_t t = u & 0x10000u, r;
  asm("add.cc.u32 %1, %1, 0xFFFF0000;\n\taddc.u32 %0, %2, 0x7FFF;" : "=r"(r), "+r"(t) : "r"(u));
  return r;
}

// FP32 -> E4M3 code, round-to-nearest-even, saturating to +-448 (cvt.rn.satfinite.e4m3 semantics);
// every NaN -> canonical 0x7F (GPU arithmetic does not keep NaN signs, so neither does this).
__host__ __device__ __forceinline__ uint32_t f32_to_e4m3(float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  const uint32_t s = (u >> 24) & 0x80u, a = u & 0x7FFFFFFFu;
  if (a > 0x7F800000u) return 0x7Fu;                  // NaN
  if (a >= 0x43E00000u) return s | 0x7Eu;             // |x| >= 448 (incl. Inf): saturate
  if (a < 0x3C800000u) {                              // |x| < 2^-6: subnormal grid 2^-9
    float m; memcpy(&m, &a, 4);
    m = m * 512.0f + 8388608.0f;                      // RNE to an integer in [0, 8] (exact scaling)
    uint32_t q; memcpy(&q, &m, 4);
    return s | (q - 0x4B000000u);                     // 8 -> 0x08 = smallest normal
  }
  uint32_t r = a + 0x7FFFFu + ((a >> 20) & 1u);       // RNE to 3 mantissa bits
  return s | (((r >> 23) - 120u) << 3) | ((r >> 20) & 7u);
}

// FP32 -> E5M2 code, round-to-nearest-even, saturating to +-57344 (cvt.rn.satfinite.e5m2x2 semantics:
// Inf -> +-57344); every NaN -> canonical 0x7F.
__host__ __device__ __forceinline__ uint32_t f32_to_e5m2(float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  const uint32_t s = (u >> 24) & 0x80u, a = u & 0x7FFFFFFFu;
  if (a > 0x7F800000u) return 0x7Fu;                  // NaN
  if (a >= 0x47600000u) return s | 0x7Bu;             // |x| >= 57344 (incl. Inf): saturate
  if (a < 0x38800000u) {                              // |x| < 2^-14: subnormal grid 2^-16
    float m; memcpy(&m, &a, 4);
    m = m * 65536.0f + 8388608.0f;                    // RNE to an integer in [0, 4] (exact scaling)
    uint32_t q; memcpy(&q, &m, 4);
    return s | (q - 0x4B000000u);                     // 4 -> 0x04 = smallest normal
  }
  uint32_t r = a + 0xFFFFFu + ((a >> 21) & 1u);       // RNE to 2 mantissa bits
  return s | (((r >> 23) - 112u) << 2) | ((r >> 21) & 3u);
}

__host__ __device__ constexpr bool is_fp8(int t) { return t == E5MA_E5M2 || t == E5MA_E4M3; }

namespace kern {

// ---- typed element access --------------------------------------------------------------------
template <int T> struct Elem { static constexpr int bytes = is_fp8(T) ? 1 : T == E5MA_FP32 ? 4 : 2; };

__device__ __forceinline__ float scale_or_one(const float *p) { return p ? *p : 1.0f; }

// value i of a typed array (FP8 values are multiplied by sinv; other types are converted exactly)
__device__ __forceinline__ float load_any(int t, const void *p, size_t i, float sinv) {
  switch (t) {
    case E5MA_E5M2: return __fmul_rn(e5m2_value(((const uint8_t *)p)[i]), sinv);
    case E5MA_E4M3: return __fmul_rn(e4m3_value(((const uint8_t *)p)[i]), sinv);
    case E5MA_BF16: return __uint_as_float((uint32_t)((const uint16_t *)p)[i] << 16);
    case E5MA_FP16: return __half2float(((const __half *)p)[i]);
    default: return ((const float *)p)[i];
  }
}
__device__ __forceinline__ void store_any(int t, void *p, size_t i, float v, float qscale) {
  switch (t) {
    case E5MA_E5M2: ((uint8_t *)p)[i] = (uint8_t)f32_to_e5m2(__fmul_rn(v, qscale)); break;
    case E5MA_E4M3: ((uint8_t *)p)[i] = (uint8_t)f32_to_e4m3(__fmul_rn(v, qscale)); break;
    case E5MA_BF16: ((uint16_t *)p)[i] = (uint16_t)(rne_hi(__float_as_uint(v)) >> 16); break;
    case E5MA_FP16: ((__half *)p)[i] = __float2half_rn(v); break;
    default: ((float *)p)[i] = v;
  }
}

// 4 consecutive GEMM operand elements of type T (E5M2, E4M3, BF16 or FP32; address aligned to
// 4 * bytes) -> 4 exact FP32 values
template <int T>
__device__ __forceinline__ void load4(const uint8_t *p, float v[4]) {
  if (is_fp8(T)) {
    decode_fp8x4<T>(*(const uint32_t *)p, v[0], v[1], v[2], v[3]);
  } else if (T == E5MA_BF16) {
    const uint2 w = *(const uint2 *)p;
    v[0] = __uint_as_float(w.x << 16); v[1] = __uint_as_float(w.x & 0xFFFF0000u);
    v[2] = __uint_as_float(w.y << 16); v[3] = __uint_as_float(w.y & 0xFFFF0000u);
  } else {
    const float4 a = *(const float4 *)p;
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
  }
}

// 8 consecutive GEMM / GEMV operand elements of type T (E5M2, E4M3, BF16 or FP32; address aligned
// to 8 * bytes) -> 8 exact FP32 values
template <int T>
__device__ __forceinline__ void load8(const uint8_t *p, float v[8]) {
  if (is_fp8(T)) {
    const uint2 w = *(const uint2 *)p;
    decode_fp8x4<T>(w.x, v[0], v[1], v[2], v[3]);
    decode_fp8x4<T>(w.y, v[4], v[5], v[6], v[7]);
  } else if (T == E5MA_BF16) {
    const uint4 w = *(const uint4 *)p;
    const uint32_t q[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) { v[2 * i] = __uint_as_float(q[i] << 16); v[2 * i + 1] = __uint_as_float(q[i] & 0xFFFF0000u); }
  } else {
    const float4 a = *(const float4 *)p, b = *(const float4 *)(p + 16);
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
  }
}

// Warp-reduce the amax statistic into *amax (non-negative floats order like ints).
__device__ __forceinline__ void amax_commit(float *amax, float m) {
#pragma unroll
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, o));
  if ((threadIdx.x & 31) == 0 && m > 0.f) atomicMax((int *)amax, __float_as_int(m));
}

// ---- GEMM / GEMV epilogue --------------------------------------------------------------------
struct Epilogue {
  const float *sa, *sb;  // device scale_inv of FP8 A / B (NULL = 1)
  bool sa_vec, sb_vec;   // sa[i] per row / sb[j] per column instead of one factor
  float alpha, beta;
  const void *c; int ldc; int c_dtype; const float *c_sinv;
  void *d; int ldd; int d_dtype; const float *d_scale; float *amax;
  const void *bias; int bias_dtype;
};

struct EpiState { float s, qscale, csinv, amax; };

__device__ __forceinline__ EpiState epi_begin(const Epilogue &e) {
  EpiState st;
  st.s = e.sa_vec || e.sb_vec ? 0.f : __fmul_rn(__fmul_rn(e.alpha, scale_or_one(e.sa)), scale_or_one(e.sb));
  st.qscale = scale_or_one(e.d_scale);
  st.csinv = scale_or_one(e.c_sinv);
  st.amax = 0.f;
  return st;
}

// output scale of element (i, j): (alpha * sa[i]) * sb[j]; precomputed once when both are per tensor
__device__ __forceinline__ float epi_scale(const Epilogue &e, float s, int i, int j) {
  if (!e.sa_vec && !e.sb_vec) return s;
  return __fmul_rn(__fmul_rn(e.alpha, e.sa_vec ? e.sa[i] : scale_or_one(e.sa)), e.sb_vec ? e.sb[j] : scale_or_one(e.sb));
}

// D[i + j*ldd] = convert(acc * s + bias[i] + beta * C[i + j*ldc]), returns |value| for amax.  C may
// alias D: each element is read and then written by the same thread.
__device__ __forceinline__ float epi_apply(const Epilogue &e, float s, float qscale, float csinv, int i, int j,
                                           float acc) {
  float v = __fmul_rn(acc, epi_scale(e, s, i, j));
  if (e.bias) v = __fadd_rn(v, load_any(e.bias_dtype, e.bias, i, 1.f));
  if (e.beta != 0.f) v = __fmaf_rn(e.beta, load_any(e.c_dtype, e.c, (size_t)j * e.ldc + i, csinv), v);
  store_any(e.d_dtype, e.d, (size_t)j * e.ldd + i, v, qscale);
  return fabsf(v);
}
// One output through the epilogue, max-reducing |value| into the amax statistic (GEMM: one call site in
// a loop over a staged column; GEMV: one or two call sites).
__device__ __forceinline__ void epi_store(const Epilogue &e, EpiState &st, int i, int j, float acc) {
  st.amax = fmaxf(st.amax, epi_apply(e, st.s, st.qscale, st.csinv, i, j, acc));
}
__device__ __forceinline__ void epi_end(const Epilogue &e, EpiState &st) {
  if (e.amax) amax_commit(e.amax, st.amax);
}

// ---- per-architecture kernel tables ----------------------------------------------------------
// GEMM / GEMV operand dtypes E5M2, E4M3, BF16, FP32 -> 0, 1, 2, 3
constexpr int kOperandTypes = 4;
constexpr int operand_index(int dtype) {
  return dtype == E5MA_E5M2 ? 0 : dtype == E5MA_E4M3 ? 1 : dtype == E5MA_BF16 ? 2 : 3;
}

struct ArchTable {
  int cc;                                // compute capability * 10 + minor (61 = sm_61); 0 = default
  int gemm_large, gemm_small;            // GEMM candidate ids (small: fewer than 2 x SMs large tiles)
  unsigned char gemv[kOperandTypes][8];  // GEMV candidate id per x dtype and n = 1..8 (both FP8 A types)
};

}  // namespace kern
}  // namespace e5ma

#include "e5m2effma_tuned.h"  // generated by `make tune`
#ifndef E5MA_TUNED_TABLES
#define E5MA_TUNED_TABLES
#endif
#ifndef E5MA_GEMM_USED
#define E5MA_GEMM_USED(X)
#endif
#ifndef E5MA_GEMV_USED
#define E5MA_GEMV_USED(X)
#endif

namespace e5ma {
namespace kern {

// Defaults (GPUs without a tuned entry): GEMM 128x128 for large problems, 64x64 for small ones; GEMV
// 4 rows per warp.
constexpr ArchTable kDefaultTable = {0, 0, 1, {{0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}}};
constexpr ArchTable kTables[] = {kDefaultTable E5MA_TUNED_TABLES};
constexpr int kNumTables = sizeof(kTables) / sizeof(kTables[0]);

constexpr int table_index(int cc) {
  int r = 0;
  for (int i = 1; i < kNumTables; ++i) if (kTables[i].cc == cc) r = i;
  return r;
}
// Configurations a device of compute capability cc can be dispatched to: those of its table plus the
// defaults (a GPU running binary-compatible SASS of another minor version uses the defaults).
constexpr bool gemm_selected(int cc, int id) {
  return id == kDefaultTable.gemm_large || id == kDefaultTable.gemm_small ||
         id == kTables[table_index(cc)].gemm_large || id == kTables[table_index(cc)].gemm_small;
}
constexpr bool gemv_selected(int cc, int id, int xt, int nb) {
  return id == kDefaultTable.gemv[operand_index(xt)][nb - 1] ||
         id == kTables[table_index(cc)].gemv[operand_index(xt)][nb - 1];
}

// compute capability of the device pass (0 in the host pass): kernel bodies of configurations not
// selected for it are compiled as trapping stubs
#if defined(__CUDA_ARCH__)
#define E5MA_ARCH_CC (__CUDA_ARCH__ / 10)
#else
#define E5MA_ARCH_CC 0
#endif

// Host: the current device's SM count and table (src/e5m2effma.cu).  supported = false when the device
// has native FP8 (sm_89+), predates sm_50, or the library carries no SASS that runs on it.
struct DeviceInfo { int cc, sms; bool supported; const ArchTable *table; };
const DeviceInfo &device_info();

}  // namespace kern

// Host: error reporting of the C API.  Every entry point starts with clear_error(); failures record a
// message for e5ma_last_error() (thread-local) and return their status.
void clear_error();
e5ma_status fail(e5ma_status s, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
e5ma_status cuda_status(cudaError_t e, const char *what);  // E5MA_OK, or E5MA_CUDA_ERROR with message
e5ma_status check_device();  // E5MA_OK, or E5MA_NOT_SUPPORTED / E5MA_CUDA_ERROR with message

namespace kern {

}  // namespace kern
}  // namespace e5ma

#define E5MA_CAT6_(a, b, c, d, e, f) a##b##c##d##e##f
#define E5MA_CAT6(a, b, c, d, e, f) E5MA_CAT6_(a, b, c, d, e, f)
#define E5MA_CAT8_(a, b, c, d, e, f, g, h) a##b##c##d##e##f##g##h
#define E5MA_CAT8(a, b, c, d, e, f, g, h) E5MA_CAT8_(a, b, c, d, e, f, g, h)
