// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. E5M2EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// E5M2EffMA GEMM: tiled kernel template, its configuration candidates and the entry points of the
// per-instance objects (gemm_e5m2effma_inst.cu).  Host API and dispatch: gemm_e5m2effma.cu.
#pragma once
#include "e5m2effma.cuh"

namespace e5ma {
namespace kern {

// ---------------------------------------------------------------------------------------------
// GEMM: BM x BN block tile, BK k-slice, TM x TN FP32 register tile per thread (groups of 4 rows /
// columns spread over the block so that LDS.128 fragment reads are conflict-free).  Each operand
// element is converted to FP32 once per block into a single-buffered shared tile; the inner loop
// is LDS.128 + FFMA only, one FFMA per MAC, k ascending per output.
// ---------------------------------------------------------------------------------------------
// On<..>::value: whether the kernel body is compiled for the architecture of the current device
// pass.  The library (gemm_e5m2effma_inst.cu) compiles a configuration only where a table selects it
// and leaves a trapping stub elsewhere; the tuner keeps everything.
template <int BM_, int BN_, int BK_, int TM_, int TN_>
struct GemmCfg {
  static constexpr int BM = BM_, BN = BN_, BK = BK_, TM = TM_, TN = TN_;
  static constexpr int TX = BM / TM, TY = BN / TN, NT = TX * TY, PAD = 4;  // lanes run along m
  // resident blocks the register budget targets: 512 threads (<= 128 registers), e.g. 2 x 256 or 4 x 128;
  // 64-thread blocks (small problems, few blocks anyway) get 6 x 64, as 8 x 64 would make them spill
  static constexpr int MINB = NT >= 512 ? 1 : NT >= 128 ? 512 / NT : 6;
  template <int TA, int TB> struct On { static constexpr bool value = true; };
};

// R x BK tile of an operand of element type T -> FP32 shared s[BK][R + PAD].  The chunk mapping keeps
// the shared-memory stores free of bank conflicts:
//   KC = true : element (r, kk) at X[kk + r*ld] (K contiguous).  A chunk is 8 consecutive k of one row
//               and the BK / 8 chunks of a row go to adjacent lanes, so a warp reads whole row
//               segments.  Its transposing scalar stores would put the lanes of one row into one
//               bank; instead the column of k-group kk / 8 is XOR-ed with swz(kk) < 32, which only
//               moves it inside its aligned 32-column block: the 32 lanes hit 32 distinct banks, and
//               fragment reads apply the same XOR (a compile-time constant per kk).
//   KC = false: element (r, kk) at X[r + kk*ld] (M/N contiguous).  A chunk is 4 consecutive rows of
//               one k; the lanes of a warp take consecutive chunks: one STS.128 per chunk, 16-byte
//               stride, and coalesced global loads.
// Rows / columns past the end of the matrix read row 0 instead: they only feed outputs that the
// epilogue does not store (K has no tail), so neither a branch nor zeroing is needed.
template <class C, int R, int T, bool KC>
struct TileLoader {
  // elements per chunk: 4 consecutive rows (M/N contiguous) or up to 8 consecutive k (K contiguous)
  static constexpr int EB = Elem<T>::bytes, E = !KC ? 4 : (R * C::BK / C::NT < 8 ? R * C::BK / C::NT : 8);
  static constexpr int CH = R * C::BK / (E * C::NT), LPR = C::BK / E;  // chunks per thread, lanes per row
  static_assert(CH >= 1 && (R * C::BK) % (E * C::NT) == 0, "tile does not split into chunks");
  static_assert(!KC || (E == 4 || E == 8) && R % 32 == 0 && 32 % LPR == 0, "unsupported KC tile");
  // column XOR of row kk of the shared tile (0 for M/N-contiguous operands): the LPR lanes of a tile
  // row land in distinct 32 / LPR-column slices of its aligned 32-column block
  static __host__ __device__ constexpr int swz(int kk) { return KC ? (kk / E) * (32 / LPR) : 0; }
  const uint8_t *p[CH];
  int dst[CH];
  size_t step;

  __device__ void init(const uint8_t *X, int ld, int r0, int rmax) {
#pragma unroll
    for (int j = 0; j < CH; ++j) {
      const int w = threadIdx.x + C::NT * j;
      if (KC) {
        const int rl = w / LPR, kc = (w % LPR) * E;
        p[j] = X + ((size_t)(r0 + rl < rmax ? r0 + rl : 0) * ld + kc) * EB;
        dst[j] = kc * (R + C::PAD) + (rl ^ swz(kc));
      } else {
        const int kl = w / (R / 4), rc = (w % (R / 4)) * 4;
        p[j] = X + ((size_t)kl * ld + (r0 + rc < rmax ? r0 + rc : 0)) * EB;
        dst[j] = kl * (R + C::PAD) + rc;
      }
    }
    step = KC ? (size_t)C::BK * EB : (size_t)C::BK * ld * EB;
  }
  // Global data stays raw until the store: a chunk is RW 32-bit words (2 for 8 FP8 values), so the
  // next tile can be in flight in a few registers while the current one is computed.
  static constexpr int RW = E * EB / 4;
  uint32_t raw[CH][RW];

  __device__ __forceinline__ void fetch() {
#pragma unroll
    for (int j = 0; j < CH; ++j) {
      if (RW == 1) raw[j][0] = *(const uint32_t *)p[j];
      else if (RW == 2) { const uint2 w = *(const uint2 *)p[j]; raw[j][0] = w.x; raw[j][1] = w.y; }
      else {
#pragma unroll
        for (int q = 0; q < RW; q += 4) {
          const uint4 w = *(const uint4 *)(p[j] + 4 * q);
          raw[j][q] = w.x; raw[j][q + 1] = w.y; raw[j][q + 2] = w.z; raw[j][q + 3] = w.w;
        }
      }
    }
  }
  // next k-tile; on the last one the pointers stay, so the final (unused) fetch re-reads valid memory
  __device__ __forceinline__ void advance(bool more) {
#pragma unroll
    for (int j = 0; j < CH; ++j) p[j] += more ? step : 0;
  }
  __device__ __forceinline__ void store(float *s) const {
#pragma unroll
    for (int j = 0; j < CH; ++j) {
      float v[E];
#pragma unroll
      for (int q = 0; q < RW; ++q) {
        if (is_fp8(T)) decode_fp8x4<T>(raw[j][q], v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]);
        else if (T == E5MA_BF16) { v[2 * q] = __uint_as_float(raw[j][q] << 16); v[2 * q + 1] = __uint_as_float(raw[j][q] & 0xFFFF0000u); }
        else v[q] = __uint_as_float(raw[j][q]);
      }
      if (KC) {
#pragma unroll
        for (int q = 0; q < E; ++q) s[dst[j] + q * (R + C::PAD)] = v[q];
      } else {
        *reinterpret_cast<float4 *>(&s[dst[j]]) = make_float4(v[0], v[1], v[2], v[3]);
      }
    }
  }
};

template <class C, int TA, int TB, bool A_KC, bool B_KC>
__global__ void __launch_bounds__(C::NT, C::MINB)
gemm_kernel(const uint8_t *__restrict__ A, int lda, const uint8_t *__restrict__ B, int ldb,
            int m, int n, int k, Epilogue e) {
  if (!C::template On<TA, TB>::value) { __trap(); return; }
  __shared__ __align__(16) float sA[C::BK * (C::BM + C::PAD)];
  __shared__ __align__(16) float sB[C::BK * (C::BN + C::PAD)];
  const int tx = threadIdx.x % C::TX, ty = threadIdx.x / C::TX;
  const int m0 = blockIdx.y * C::BM, n0 = blockIdx.x * C::BN;
  using LA = TileLoader<C, C::BM, TA, A_KC>;
  using LB = TileLoader<C, C::BN, TB, B_KC>;
  static_assert(C::TX * 4 >= 32 && C::TY * 4 >= 32, "fragment XOR must stay inside a 32-column block");
  LA la; la.init(A, lda, m0, m);
  LB lb; lb.init(B, ldb, n0, n);

  float acc[C::TM][C::TN];
#pragma unroll
  for (int i = 0; i < C::TM; ++i)
#pragma unroll
    for (int j = 0; j < C::TN; ++j) acc[i][j] = 0.f;

  // The next tile's global loads are issued before the current tile is computed and consumed only
  // after it: their latency hides behind the block's own FFMAs, not only behind other blocks.
  const int kts = k / C::BK;
  la.fetch(); lb.fetch();
  for (int kt = 0; kt < kts; ++kt) {
    if (kt) __syncthreads();  // every warp finished reading the previous tile
    la.store(sA); lb.store(sB);
    __syncthreads();           // tile visible to every warp
    la.advance(kt + 1 < kts); lb.advance(kt + 1 < kts);
    la.fetch(); lb.fetch();
#pragma unroll
    for (int kk = 0; kk < C::BK; ++kk) {
      float a[C::TM], b[C::TN];
#pragma unroll
      for (int g = 0; g < C::TM / 4; ++g)
        *reinterpret_cast<float4 *>(&a[4 * g]) =
            *reinterpret_cast<const float4 *>(&sA[kk * (C::BM + C::PAD) + g * C::TX * 4 + ((tx * 4) ^ LA::swz(kk))]);
#pragma unroll
      for (int g = 0; g < C::TN / 4; ++g)
        *reinterpret_cast<float4 *>(&b[4 * g]) =
            *reinterpret_cast<const float4 *>(&sB[kk * (C::BN + C::PAD) + g * C::TY * 4 + ((ty * 4) ^ LB::swz(kk))]);
      // FFMA order: pairs of rows, zigzag over j (consecutive FFMAs share a or b); this halves the
      // register-bank conflicts ptxas leaves compared with row-by-row order
#pragma unroll
      for (int p = 0; p < C::TM / 2; ++p) {
        const int i0 = 2 * p, i1 = i0 + 1;
#pragma unroll
        for (int jj = 0; jj < C::TN; ++jj) {
          const int j = (p & 1) ? C::TN - 1 - jj : jj;
          if (jj & 1) { acc[i1][j] = fma_rn(a[i1], b[j], acc[i1][j]); acc[i0][j] = fma_rn(a[i0], b[j], acc[i0][j]); }
          else { acc[i0][j] = fma_rn(a[i0], b[j], acc[i0][j]); acc[i1][j] = fma_rn(a[i1], b[j], acc[i1][j]); }
        }
      }
    }
  }

  EpiState st = epi_begin(e);
  // Common case -- BF16 out, no bias, no C, D 8-byte aligned -- the 4 consecutive rows a thread owns
  // in each column go out as one 8-byte store.
  const bool packed = e.d_dtype == E5MA_BF16 && !e.bias && e.beta == 0.f &&
                      (((uintptr_t)e.d | (uintptr_t)e.ldd * 2) & 7) == 0;
  if (packed) {
#pragma unroll
    for (int g = 0; g < C::TM / 4; ++g) {
      const int gi = m0 + g * C::TX * 4 + tx * 4;
#pragma unroll
      for (int j = 0; j < C::TN; ++j) {
        const int gj = n0 + (j / 4) * C::TY * 4 + ty * 4 + (j & 3);
        if (gj >= n) continue;
        float v[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          v[i] = __fmul_rn(acc[4 * g + i][j], epi_scale(e, st.s, gi + i, gj));
          if (gi + i < m) st.amax = fmaxf(st.amax, fabsf(v[i]));  // rows past m are not outputs
        }
        uint16_t *d = (uint16_t *)e.d + (size_t)gj * e.ldd + gi;
        if (gi + 3 < m) {
          uint2 w;
          w.x = __byte_perm(rne_hi(__float_as_uint(v[0])), rne_hi(__float_as_uint(v[1])), 0x7632);
          w.y = __byte_perm(rne_hi(__float_as_uint(v[2])), rne_hi(__float_as_uint(v[3])), 0x7632);
          *reinterpret_cast<uint2 *>(d) = w;
        } else {
#pragma unroll
          for (int i = 0; i < 4; ++i)
            if (gi + i < m) d[i] = (uint16_t)(rne_hi(__float_as_uint(v[i])) >> 16);
        }
      }
    }
  } else {
    // Everything else (any output dtype, bias, C, FP8 output) goes column by column through shared
    // memory: pass j stages column j of every thread's tile -- BM rows x TY columns -- in sA, then
    // each thread stores TM elements, consecutive threads on consecutive rows.  One inlined epilogue
    // in a loop instead of a call per output: no call frame, no stack, and the stores coalesce along m.
    static_assert(C::TY <= C::BK && C::BM % 4 == 0, "a staged column must fit in sA");
#pragma unroll 1
    for (int j = 0; j < C::TN; ++j) {  // run-time j: one copy of the epilogue
      __syncthreads();  // sA free: the k loop, or the previous pass, is done with it
#pragma unroll
      for (int jj = 0; jj < C::TN; ++jj)  // column j of acc: one branch per column, acc stays in registers
        if (jj == j) {
#pragma unroll
          for (int g = 0; g < C::TM / 4; ++g)
            *reinterpret_cast<float4 *>(&sA[ty * C::BM + g * C::TX * 4 + tx * 4]) =
                make_float4(acc[4 * g][jj], acc[4 * g + 1][jj], acc[4 * g + 2][jj], acc[4 * g + 3][jj]);
        }
      __syncthreads();
#pragma unroll 1
      for (int q = 0; q < C::TM; ++q) {
        const int el = q * C::NT + threadIdx.x, r = el % C::BM, c = el / C::BM;
        const int gi = m0 + r, gj = n0 + (j / 4) * C::TY * 4 + c * 4 + (j & 3);
        if (gi < m && gj < n) epi_store(e, st, gi, gj, sA[c * C::BM + r]);
      }
    }
  }
  epi_end(e, st);
}

template <class C, int TA, int TB, bool A_KC, bool B_KC>
cudaError_t gemm_launch(const uint8_t *A, int lda, const uint8_t *B, int ldb, int m, int n, int k,
                        const Epilogue &e, cudaStream_t s) {
  dim3 grid((n + C::BN - 1) / C::BN, (m + C::BM - 1) / C::BM);
  gemm_kernel<C, TA, TB, A_KC, B_KC><<<grid, C::NT, 0, s>>>(A, lda, B, ldb, m, n, k, e);
  return cudaGetLastError();
}

// Candidates: id, BM, BN, BK, TM, TN.  All share the 8 x 8 thread tile (same FFMA : LDS ratio); the
// block tile only trades parallelism on small grids, which depends on the SM count.  Not a candidate:
// BK = 32 (fewer barriers, but there are 1024 FFMAs per thread between barriers already; twice the
// shared memory, one block per SM where an SM has 64 KB).  Ids are what src/e5m2effma_tuned.h stores.
#define E5MA_GEMM_CANDIDATES(X) \
  X(0, 128, 128, 16, 8, 8)       \
  X(1, 64, 64, 16, 8, 8)         \
  X(2, 128, 64, 16, 8, 8)        \
  X(3, 64, 128, 16, 8, 8)
#define E5MA_DEF_GEMM(id, bm, bn, bk, tm, tn) using GemmCfg##id = GemmCfg<bm, bn, bk, tm, tn>;
E5MA_GEMM_CANDIDATES(E5MA_DEF_GEMM)
#undef E5MA_DEF_GEMM

}  // namespace kern
}  // namespace e5ma

// configurations the library builds: the defaults (0, 1) and those the tuned tables use
#define E5MA_GEMM_LIB(X) X(0) X(1) E5MA_GEMM_USED(X)

// entry point of one instance object: config id x A dtype x B dtype, all 4 layouts
#define E5MA_GEMM_ENTRY(id, ta, tb) E5MA_CAT6(gemm_entry_, id, _, ta, _, tb)
#define E5MA_GEMM_PROTO(id, ta, tb)                                                               \
  cudaError_t E5MA_GEMM_ENTRY(id, ta, tb)(bool akc, bool bkc, const uint8_t *A, int lda, const uint8_t *B, \
                                          int ldb, int m, int n, int k, const e5ma::kern::Epilogue &e,     \
                                          cudaStream_t s)
