// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. E5M2EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// E5M2EffMA core, exhaustively over 2^32 inputs each (src/e5m2effma.cuh), and e5ma_cast:
//  1. mac  : the kernels' MAC path -- decode FP8 A, B (E5M2 x E5M2, E5M2 x E4M3, E4M3 x E4M3); one
//            FFMA onto the BF16 accumulator; RNE to BF16 -- == RN_BF16(ACC + exact(A*B)) for every
//            (A, B, ACC) = 2^32 cases per pairing.  Reference: tests/e5m2effma_ref.h (exact integer
//            model).  Domain: finite FP8, no BF16 Inf/NaN accumulator (out-of-domain inputs are
//            mapped into the domain, so coverage stays complete).
//  2. rne  : FP32 -> BF16 rounding on every FP32 bit pattern vs an independent remainder-comparison
//            RNE (NaN -> 0x7FC0).
//  3. e5m2, e4m3 : FP32 -> FP8 encoding on every FP32 bit pattern vs an independent search for the
//            nearest finite value (midpoint table, ties to the even code, saturation to +-57344 /
//            +-448, every NaN -> 0x7F).
//  4. e5ma_cast: 1M random FP32 bit patterns (incl. NaN / Inf) -> E5M2 / E4M3 / BF16 / FP16, and every
//            FP8 code -> FP32 (Inf / NaN codes decode to Inf / NaN).
// Build with -DE5MA_NEGATIVE_CONTROLS to check that deliberately broken variants are rejected.
#include "test_util.h"
#include "e5m2effma_ref.h"

struct Err { unsigned long long count; uint32_t x[16], y[16], got[16], exp[16]; };

__device__ __forceinline__ void record(Err *err, uint32_t x, uint32_t y, uint32_t got, uint32_t exp) {
  unsigned long long e = atomicAdd(&err->count, 1ull);
  if (e < 16) { err->x[e] = x; err->y[e] = y; err->got[e] = got; err->exp[e] = exp; }
}

// ---- implementations under test -------------------------------------------------------------
// the GEMM decode (decode_fp8x4, as the kernels' tile loaders), one FFMA
template <int FA, int FB> __device__ __forceinline__ float mac_f32(uint32_t a, uint32_t b, uint32_t acc) {
  float va, vb, x;
  e5ma::decode_fp8x4<FA>(a, va, x, x, x);
  e5ma::decode_fp8x4<FB>(b, vb, x, x, x);
  return e5ma::fma_rn(va, vb, __uint_as_float(acc << 16));
}
template <int FA, int FB> __device__ __forceinline__ uint32_t mac_ok(uint32_t a, uint32_t b, uint32_t acc) {
  return e5ma::rne_hi(__float_as_uint(mac_f32<FA, FB>(a, b, acc))) >> 16;
}
__device__ __forceinline__ uint32_t rne_ok(uint32_t u) { return e5ma::rne_hi(u) >> 16; }
#ifdef E5MA_NEGATIVE_CONTROLS
__device__ __forceinline__ uint32_t mac_trunc(uint32_t a, uint32_t b, uint32_t acc) {  // truncation, no RNE
  return __float_as_uint(mac_f32<E5MA_E5M2, E5MA_E5M2>(a, b, acc)) >> 16;
}
__device__ __forceinline__ uint32_t mac_ftz(uint32_t a, uint32_t b, uint32_t acc) {  // flushes subnormal results
  const uint32_t r = mac_ok<E5MA_E5M2, E5MA_E5M2>(a, b, acc);
  return (r & 0x7F80u) == 0 && (r & 0x7Fu) ? r & 0x8000u : r;
}
__device__ __forceinline__ uint32_t rne_nonan(uint32_t u) { return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16; }
__device__ __forceinline__ uint32_t e4m3_ties_away(uint32_t u) {  // round half away instead of even
  const float x = __uint_as_float(u), a = fabsf(x);
  if (a != a || a >= 448.f || a < 0x1p-6f) return e5ma::f32_to_e4m3(x);
  const uint32_t b = __float_as_uint(a), r = b + 0x80000u;
  return ((u >> 24) & 0x80u) | (((r >> 23) - 120u) << 3) | ((r >> 20) & 7u);
}
__device__ __forceinline__ uint32_t e5m2_ties_away(uint32_t u) {
  const float x = __uint_as_float(u), a = fabsf(x);
  if (a != a || a >= 57344.f || a < 0x1p-14f) return e5ma::f32_to_e5m2(x);
  const uint32_t b = __float_as_uint(a), r = b + 0x100000u;
  return ((u >> 24) & 0x80u) | (((r >> 23) - 112u) << 2) | ((r >> 21) & 3u);
}
#endif

// ---- kernels ---------------------------------------------------------------------------------
// out-of-domain FP8 codes -> a finite neighbour: E4M3 NaN -> 0x7E, E5M2 Inf / NaN -> exponent 30
__device__ __forceinline__ uint32_t finite(uint32_t b, int t) {
  if (t == E5MA_E4M3) return (b & 0x7F) == 0x7F ? b ^ 1 : b;
  return (b & 0x7C) == 0x7C ? b ^ 4 : b;
}
template <int FA, int FB, uint32_t (*OP)(uint32_t, uint32_t, uint32_t)>
__global__ void k_mac(uint64_t base, uint64_t n, Err *err) {
  uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  const int ra = FA == E5MA_E5M2 ? e5ma_ref::E5M2 : e5ma_ref::E4M3, rb = FB == E5MA_E5M2 ? e5ma_ref::E5M2 : e5ma_ref::E4M3;
  for (uint64_t k = base + blockIdx.x * blockDim.x + threadIdx.x; k < base + n; k += stride) {
    const uint32_t i = (uint32_t)k;
    uint32_t a = finite(i & 0xFF, FA), b = finite((i >> 8) & 0xFF, FB), c = i >> 16;
    if ((c & 0x7F80) == 0x7F80) c = (c & 0x8000) | 0x7F7F;    // BF16 Inf / NaN -> max finite
    const uint32_t got = OP(a, b, c), exp = e5ma_ref::lane(a, b, c, ra, rb);
    if (got != exp) record(err, i, 0, got, exp);
  }
}

template <uint32_t (*OP)(uint32_t)>
__global__ void k_rne(uint64_t base, uint64_t n, Err *err) {
  uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  for (uint64_t k = base + blockIdx.x * blockDim.x + threadIdx.x; k < base + n; k += stride) {
    uint32_t u = (uint32_t)k, exp;
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) exp = 0x7FC0;  // NaN
    else {
      uint32_t hi = u >> 16, lo = u & 0xFFFF;
      exp = (lo > 0x8000 || (lo == 0x8000 && (hi & 1))) ? hi + 1 : hi;
    }
    uint32_t got = OP(u);
    if (got != exp) record(err, u, 0, got, exp);
  }
}

// Independent FP32 -> FP8 reference: binary search over the midpoints between consecutive
// non-negative finite FP8 values (midpoints have <= 5 significant bits: exact in FP32).
// E4M3: codes 0..126 (max 0x7E = 448); E5M2: codes 0..123 (max 0x7B = 57344).
__constant__ float c_mid4[126], c_mid5[123];  // c_mid[c] = (value(c) + value(c+1)) / 2
template <int F> __device__ __forceinline__ uint32_t fp8_enc_ref(float x) {
  const uint32_t u = __float_as_uint(x), s = (u >> 24) & 0x80u;
  const int top = F == E5MA_E5M2 ? 123 : 126;  // largest finite code
  const float *mids = F == E5MA_E5M2 ? c_mid5 : c_mid4;
  const float a = fabsf(x);
  if (a != a) return 0x7Fu;
  if (a >= (F == E5MA_E5M2 ? 57344.f : 448.f)) return s | (uint32_t)top;
  int lo = 0, hi = top;  // answer c in [0, top]: first c with a < mid[c] (or == and c even)
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (a < mids[mid] || (a == mids[mid] && (mid & 1) == 0)) hi = mid; else lo = mid + 1;
  }
  return s | (uint32_t)lo;
}
__device__ __forceinline__ uint32_t e4m3_ok(uint32_t u) { return e5ma::f32_to_e4m3(__uint_as_float(u)); }
__device__ __forceinline__ uint32_t e5m2_ok(uint32_t u) { return e5ma::f32_to_e5m2(__uint_as_float(u)); }

template <int F, uint32_t (*OP)(uint32_t)>
__global__ void k_enc(uint64_t base, uint64_t n, Err *err) {
  uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  for (uint64_t k = base + blockIdx.x * blockDim.x + threadIdx.x; k < base + n; k += stride) {
    const uint32_t u = (uint32_t)k, got = OP(u), exp = fp8_enc_ref<F>(__uint_as_float(u));
    if (got != exp) record(err, u, 0, got, exp);
  }
}

template <class LAUNCH>
static int run(const char *name, LAUNCH launch, uint64_t total) {
  Err *d; cudaMalloc(&d, sizeof(Err)); cudaMemset(d, 0, sizeof(Err));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  const uint64_t chunk = 1ull << 28;
  for (uint64_t b = 0; b < total; b += chunk) launch(b, total - b < chunk ? total - b : chunk, d);
  cudaEventRecord(e1);
  cudaError_t ce = cudaDeviceSynchronize();
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  Err h; cudaMemcpy(&h, d, sizeof(Err), cudaMemcpyDeviceToHost); cudaFree(d);
  if (ce) { printf("%-12s CUDA error %s\n", name, cudaGetErrorString(ce)); return 1; }
  printf("%-12s %s  cases=%llu  mismatches=%llu  (%.1f s)\n", name, h.count ? "FAIL" : "PASS",
         (unsigned long long)total, h.count, ms / 1e3);
  for (unsigned e = 0; e < h.count && e < 8; ++e)
    printf("   in=%08x %08x got=%08x exp=%08x\n", h.x[e], h.y[e], h.got[e], h.exp[e]);
  return h.count != 0;
}

#ifndef E5MA_NEGATIVE_CONTROLS
// ---- e5ma_cast ------------------------------------------------------------------------------
static int test_cast() {
  int fails = 0;
  const size_t N = 1 << 20;
  std::vector<float> hx(N);
  for (size_t i = 0; i < N; ++i) { uint32_t u = rnd(); memcpy(&hx[i], &u, 4); }
  for (int i = 0; i < 256; ++i) { hx[i] = e4((uint8_t)i); hx[256 + i] = e5((uint8_t)i); }
  float *dx = dev(hx), *dsc = devf(0.5f);
  for (e5ma_dtype t : {E5MA_E5M2, E5MA_E4M3, E5MA_BF16, E5MA_FP16}) {
    std::vector<uint8_t> out(N * esize(t)), exp(N * esize(t));
    uint8_t *dy; cudaMalloc(&dy, out.size());
    e5ma_tensor s = {dx, E5MA_FP32, nullptr, nullptr, nullptr}, d = {dy, t, nullptr, dsc, nullptr};
    e5ma_status st = e5ma_cast(&s, &d, N, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(out.data(), dy, out.size(), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < N; ++i) hstore(t, exp.data(), i, hx[i], 0.5f);
    size_t bad = 0; for (size_t i = 0; i < out.size(); ++i) bad += out[i] != exp[i];
    printf("cast FP32 -> %s (1M random bit patterns incl. NaN/Inf): %s\n", DN[t], (st == E5MA_OK && !bad) ? "ok" : "FAIL");
    fails += st != E5MA_OK || bad;
    cudaFree(dy);
  }
  // every FP8 code -> FP32 with scale_inv 0.5 (bit compare: NaN codes must give NaN)
  for (e5ma_dtype t : {E5MA_E5M2, E5MA_E4M3}) {
    std::vector<uint8_t> codes(256); for (int i = 0; i < 256; ++i) codes[i] = (uint8_t)i;
    uint8_t *dc = dev(codes); float *dy; cudaMalloc(&dy, 256 * 4);
    e5ma_tensor s = {dc, t, dsc, nullptr, nullptr}, d = {dy, E5MA_FP32, nullptr, nullptr, nullptr};
    e5ma_status st = e5ma_cast(&s, &d, 256, 0);
    float got[256]; cudaMemcpy(got, dy, sizeof got, cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int i = 0; i < 256; ++i) {
      const float e = hload(t, codes.data(), i, 0.5f);
      bad += !(got[i] == e || (got[i] != got[i] && e != e));
    }
    printf("cast %s -> FP32 (all 256 codes incl. Inf / NaN): %s\n", DN[t], (st == E5MA_OK && !bad) ? "ok" : "FAIL");
    fails += st != E5MA_OK || bad;
    cudaFree(dc); cudaFree(dy);
  }
  cudaFree(dx); cudaFree(dsc);
  return fails;
}
#endif

#define MAC(FA, FB, OP) [](uint64_t b, uint64_t n, Err *d) { k_mac<FA, FB, OP><<<1024, 256>>>(b, n, d); }
#define RNE(OP) [](uint64_t b, uint64_t n, Err *d) { k_rne<OP><<<1024, 256>>>(b, n, d); }
#define ENC(F, OP) [](uint64_t b, uint64_t n, Err *d) { k_enc<F, OP><<<1024, 256>>>(b, n, d); }
#define E5 E5MA_E5M2
#define E4 E5MA_E4M3

int main(int argc, char **argv) {
  {  // midpoint tables for the FP8 encoder references
    float m4[126], m5[123];
    for (int c = 0; c < 126; ++c) m4[c] = 0.5f * (e4((uint8_t)c) + e4((uint8_t)(c + 1)));
    for (int c = 0; c < 123; ++c) m5[c] = 0.5f * (e5((uint8_t)c) + e5((uint8_t)(c + 1)));
    cudaMemcpyToSymbol(c_mid4, m4, sizeof m4);
    cudaMemcpyToSymbol(c_mid5, m5, sizeof m5);
  }
  uint64_t total = (argc > 1 && !strcmp(argv[1], "--quick")) ? 1ull << 26 : 1ull << 32;
#ifndef E5MA_NEGATIVE_CONTROLS
  int fails = 0;
  fails += run("mac e5m2", MAC(E5, E5, (mac_ok<E5, E5>)), total);
  fails += run("mac e5 x e4", MAC(E5, E4, (mac_ok<E5, E4>)), total);
  fails += run("mac e4m3", MAC(E4, E4, (mac_ok<E4, E4>)), total);
  fails += run("rne", RNE(rne_ok), total);
  fails += run("e5m2", ENC(E5, e5m2_ok), total);
  fails += run("e4m3", ENC(E4, e4m3_ok), total);
  fails += test_cast();
  printf(fails ? "\nFAILED\n" : "\nALL PASS\n");
  return fails != 0;
#else
  int caught = 0;
  caught += run("neg_trunc", MAC(E5, E5, mac_trunc), total);
  caught += run("neg_ftz", MAC(E5, E5, mac_ftz), total);
  caught += run("neg_rne_nan", RNE(rne_nonan), total);
  caught += run("neg_e5m2_away", ENC(E5, e5m2_ties_away), total);
  caught += run("neg_e4m3_away", ENC(E4, e4m3_ties_away), total);
  printf(caught == 5 ? "\nall 5 broken variants rejected\n" : "\nNEGATIVE CONTROL NOT CAUGHT\n");
  return caught != 5;
#endif
}
