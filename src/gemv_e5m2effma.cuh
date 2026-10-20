// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. E5M2EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// E5M2EffMA GEMV (n <= 8, TN, FP8 A): bandwidth-bound kernel template, its configuration candidates
// and the entry points of the per-instance objects (gemv_e5m2effma_inst.cu).  Dispatch:
// gemv_e5m2effma.cu; the public entry is e5ma_gemm (gemm_e5m2effma.cu), which routes here.
#pragma once
#include "e5m2effma.cuh"

namespace e5ma {
namespace kern {

// ---------------------------------------------------------------------------------------------
// GEMV / small batch (TN, A E5M2 or E4M3 k x m, x k x NB, NB <= 8): bandwidth-bound, every byte of A
// read once.  A warp owns R rows; per pass a lane reads 8 consecutive bytes of each row and runs
// 8 x NB FFMAs against 8 x NB x values converted once per pass and reused for all R rows.
// Per output, lane l accumulates k = 256c + 8l + j in (c, j) order; lanes are then combined with
// a butterfly (xor 16, 8, 4, 2, 1) -- tests/test_gemv.cu emulates exactly this order.
// ---------------------------------------------------------------------------------------------
// MAXNB: largest n the configuration is a candidate for (beyond it R x NB accumulators and x values
// exceed the register budget on every target -- all have 64K registers per SM).
template <int R_, int MAXNB_>
struct GemvCfg {
  static constexpr int R = R_, WARPS = 8, MAXNB = MAXNB_;
  template <int NB, int XT> struct On { static constexpr bool value = true; };
};

template <class C, int NB, int AT, int XT>
__global__ void __launch_bounds__(C::WARPS * 32)
gemv_kernel(const uint8_t *__restrict__ A, int lda, const uint8_t *__restrict__ X, int ldx,
            int m, int k, Epilogue e) {
  if (!C::template On<NB, XT>::value) { __trap(); return; }
  constexpr int XB = Elem<XT>::bytes;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int row0 = (blockIdx.x * C::WARPS + warp) * C::R;
  float acc[C::R][NB];
#pragma unroll
  for (int r = 0; r < C::R; ++r)
#pragma unroll
    for (int b = 0; b < NB; ++b) acc[r][b] = 0.f;

  for (int kb = lane * 8; kb < k; kb += 256) {
    float xv[NB][8];
#pragma unroll
    for (int b = 0; b < NB; ++b) load8<XT>(X + ((size_t)b * ldx + kb) * XB, xv[b]);
#pragma unroll
    for (int r = 0; r < C::R; ++r) {
      if (row0 + r >= m) break;
      float a[8];
      load8<AT>(A + (size_t)(row0 + r) * lda + kb, a);
#pragma unroll
      for (int b = 0; b < NB; ++b)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[r][b] = fma_rn(a[j], xv[b][j], acc[r][b]);
    }
  }

  // After the butterfly every lane holds every sum; output o = r * NB + b is stored by lane o % 32,
  // so the stores run in parallel and the epilogue is inlined once (twice for R * NB > 32).
  constexpr int NO = C::R * NB, PER = (NO + 31) / 32;
  float mine[PER];
#pragma unroll
  for (int t = 0; t < PER; ++t) mine[t] = 0.f;
#pragma unroll
  for (int r = 0; r < C::R; ++r)
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      float v = acc[r][b];
#pragma unroll
      for (int o = 16; o; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
      if (((r * NB + b) & 31) == lane) mine[(r * NB + b) >> 5] = v;
    }
  EpiState st = epi_begin(e);
#pragma unroll
  for (int t = 0; t < PER; ++t) {
    const int o = t * 32 + lane, r = o / NB;
    if (o < NO && row0 + r < m) epi_store(e, st, row0 + r, o - r * NB, mine[t]);
  }
  epi_end(e, st);
}

template <class C, int NB, int AT, int XT>
cudaError_t gemv_launch(const uint8_t *A, int lda, const uint8_t *X, int ldx, int m, int k,
                        const Epilogue &e, cudaStream_t s) {
  const int rows = C::WARPS * C::R;
  gemv_kernel<C, NB, AT, XT><<<(m + rows - 1) / rows, C::WARPS * 32, 0, s>>>(A, lda, X, ldx, m, k, e);
  return cudaGetLastError();
}


// Candidates: id, rows per warp, largest n.  Not a candidate: x staged once per block in shared
// memory (lost at every n on sm_61; sm_70+ caches global loads in L1 anyway).
#define E5MA_GEMV_CANDIDATES(X) \
  X(0, 4, 8)                     \
  X(1, 8, 2)                     \
  X(2, 2, 8)
#define E5MA_DEF_GEMV(id, r, mx) using GemvCfg##id = GemvCfg<r, mx>;
E5MA_GEMV_CANDIDATES(E5MA_DEF_GEMV)
#undef E5MA_DEF_GEMV

// GEMV of FP8 A (dtype at) by n <= 8 columns of x (dtype xt) with the configuration the device's table
// selects.
cudaError_t launch_gemv(int at, int xt, const uint8_t *A, int lda, const uint8_t *X, int ldx, int m, int n, int k,
                        const Epilogue &e, cudaStream_t s);

}  // namespace kern
}  // namespace e5ma

// configurations the library builds: the default (0) and those the tuned tables use
#define E5MA_GEMV_LIB(X) X(0) E5MA_GEMV_USED(X)

// entry point of one kernel: config id x A dtype x x dtype x n
#define E5MA_GEMV_ENTRY(id, a, t, nb) E5MA_CAT8(gemv_entry_, id, _, a, _, t, _, nb)
#define E5MA_GEMV_PROTO(id, a, t, nb)                                                             \
  cudaError_t E5MA_GEMV_ENTRY(id, a, t, nb)(const uint8_t *A, int lda, const uint8_t *X, int ldx, int m, \
                                          int k, const e5ma::kern::Epilogue &e, cudaStream_t s)
