// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. E5M2EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// e5ma_adam bandwidth: bytes moved per step (param + grad + 2 moments read and written, plus the
// optional E5M2 copy) over time, for FP32 and BF16 state.  Median time per row after an idle pause
// (see bench_util.h).
#include "../include/e5m2effma.h"
#include "bench_util.h"

int main() {
  const size_t N = 64u << 20;
  cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
  int memory_clock_khz = 0;  // cudaDeviceProp::memoryClockRate is gone in CUDA 13
  cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate, 0);
  const double peak_bw = 2.0 * memory_clock_khz * 1e3 * (prop.memoryBusWidth / 8) / 1e9;
  Bench bench;
  void *p, *g, *m, *v, *q;
  cudaMalloc(&p, N * 4); cudaMalloc(&g, N * 4); cudaMalloc(&m, N * 4); cudaMalloc(&v, N * 4); cudaMalloc(&q, N);
  cudaMemset(p, 0, N * 4); cudaMemset(g, 0, N * 4); cudaMemset(m, 0, N * 4); cudaMemset(v, 0, N * 4);
  e5ma_adam_config cfg = {1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, 10, 1, 1, 1.f};
  printf("e5ma_adam (AdamW), %zu M parameters; DRAM peak %.0f GB/s\n", N >> 20, peak_bw);
  for (e5ma_dtype st : {E5MA_FP32, E5MA_BF16})
    for (int fp8 = 0; fp8 < 2; ++fp8) {
      e5ma_tensor P = {p, st, nullptr, nullptr, nullptr}, G = {g, st, nullptr, nullptr, nullptr};
      e5ma_tensor M = {m, st, nullptr, nullptr, nullptr}, V = {v, st, nullptr, nullptr, nullptr};
      e5ma_tensor Q = {q, E5MA_E5M2, nullptr, nullptr, nullptr};
      const BenchRow x = bench.run([&] { e5ma_adam(&cfg, N, &P, &G, &M, &V, fp8 ? &Q : nullptr, 0); }, 50);
      const double es = st == E5MA_FP32 ? 4 : 2, bytes = N * (es * 7 + (fp8 ? 1 : 0));  // p,m,v r+w; g r
      printf("  state %s%-13s %7.3f ms  %6.1f GB/s  %3.0f%% of DRAM peak  %5.0f MHz\n", st == E5MA_FP32 ? "FP32" : "BF16",
             fp8 ? " + E5M2 copy:" : ":", x.ms, bytes / x.ms / 1e6, 100 * bytes / x.ms / 1e6 / peak_bw, x.mhz);
      fflush(stdout);
    }
  return 0;
}
