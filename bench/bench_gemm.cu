// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. E5M2EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// e5ma_gemm throughput (E5M2 x E5M2 -> BF16, E5M2 x E4M3, mixed operands, FP32 accumulation) vs
// cuBLAS SGEMM of the same shape (FP32 in/out).  Median time per row; % of FP32 peak = SMs x 128 FFMA x the SM clock
// measured during that very row (see bench_util.h).
#include <vector>
#include <cublas_v2.h>
#include "../include/e5m2effma.h"
#include "bench_util.h"

int main(int argc, char **argv) {
  const int S = argc > 1 ? atoi(argv[1]) : 4096, M = S, N = S, K = S;
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  printf("%s, %d SMs; M=N=K=%d; median of %d runs per row after %d s idle; %% of FP32 peak at the row's SM clock\n\n",
         p.name, p.multiProcessorCount, S, Bench::env("E5MA_BENCH_REPS", 20), Bench::env("E5MA_BENCH_COOL", 2));
  printf("%-50s %9s %13s %9s %12s\n", "", "time", "", "SM clock", "of FP32 peak");
  Bench bench;

  std::vector<unsigned char> h((size_t)M * K);
  unsigned r = 1;
  for (auto &x : h) { do { r ^= r << 13; r ^= r >> 17; r ^= r << 5; x = r & 0xFF; } while ((x & 0x7C) == 0x7C); }  // finite E5M2 and E4M3
  void *dA, *dB, *dD; cudaMalloc(&dA, (size_t)M * K); cudaMalloc(&dB, (size_t)K * N); cudaMalloc(&dD, (size_t)M * N * 2);
  cudaMemcpy(dA, h.data(), h.size(), cudaMemcpyHostToDevice); cudaMemcpy(dB, h.data(), h.size(), cudaMemcpyHostToDevice);
  e5ma_tensor A = {dA, E5MA_E5M2, nullptr, nullptr, nullptr}, B = {dB, E5MA_E5M2, nullptr, nullptr, nullptr};
  e5ma_tensor D = {dD, E5MA_BF16, nullptr, nullptr, nullptr};
  const double macs = (double)M * N * K;
  auto row = [&](const char *name, BenchRow x) {
    const double gmacs = macs / x.ms / 1e6, peak = p.multiProcessorCount * 128 * x.mhz / 1e3;
    if (x.mhz > 0) printf("%-50s %6.3f ms %7.1f GMAC/s %5.0f MHz %10.1f%%\n", name, x.ms, gmacs, x.mhz, 100 * gmacs / peak);
    else printf("%-50s %6.3f ms %7.1f GMAC/s %9s %11s\n", name, x.ms, gmacs, "n/a", "n/a");
    fflush(stdout);
  };
  const e5ma_op ops[][2] = {{E5MA_OP_T, E5MA_OP_N}, {E5MA_OP_N, E5MA_OP_N}, {E5MA_OP_N, E5MA_OP_T}, {E5MA_OP_T, E5MA_OP_T}};
  for (auto &o : ops) {
    const int lda = o[0] == E5MA_OP_T ? K : M, ldb = o[1] == E5MA_OP_N ? K : N;
    e5ma_status st = E5MA_OK;
    const BenchRow x = bench.run([&] { st = e5ma_gemm(o[0], o[1], M, N, K, 1.f, &A, lda, &B, ldb, 0.f, nullptr, 0, &D, M, nullptr, 0); }, 20);
    char name[64]; snprintf(name, 64, "e5ma_gemm %c%c E5M2 -> BF16%s", o[0] == E5MA_OP_T ? 'T' : 'N', o[1] == E5MA_OP_T ? 'T' : 'N',
                            st == E5MA_OK ? "" : " (FAILED)");
    row(name, x);
  }
  {  // TE HYBRID backward pairing: E5M2 gradients x E4M3 activations / weights
    e5ma_tensor B4 = {dB, E5MA_E4M3, nullptr, nullptr, nullptr};
    row("e5ma_gemm TN E5M2 x E4M3 -> BF16",
        bench.run([&] { e5ma_gemm(E5MA_OP_T, E5MA_OP_N, M, N, K, 1.f, &A, K, &B4, K, 0.f, nullptr, 0, &D, M, nullptr, 0); }, 20));
  }
  {  // mixed operands (FP8 x BF16 / FP32) and TE-style FP32 accumulation
    void *dX; cudaMalloc(&dX, (size_t)K * N * 4); cudaMemset(dX, 0, (size_t)K * N * 4);
    void *dF; cudaMalloc(&dF, (size_t)M * N * 4); cudaMemset(dF, 0, (size_t)M * N * 4);
    e5ma_tensor F = {dF, E5MA_FP32, nullptr, nullptr, nullptr};
    for (e5ma_dtype bt : {E5MA_BF16, E5MA_FP32}) {
      e5ma_tensor X = {dX, bt, nullptr, nullptr, nullptr};
      char name[64]; snprintf(name, 64, "e5ma_gemm TN E5M2 x %s -> BF16", bt == E5MA_BF16 ? "BF16" : "FP32");
      row(name, bench.run([&] { e5ma_gemm(E5MA_OP_T, E5MA_OP_N, M, N, K, 1.f, &A, K, &X, K, 0.f, nullptr, 0, &D, M, nullptr, 0); }, 20));
    }
    e5ma_tensor X = {dX, E5MA_BF16, nullptr, nullptr, nullptr};
    row("e5ma_gemm TN E5M2 x BF16, FP32 D += (accumulate)",
        bench.run([&] { e5ma_gemm(E5MA_OP_T, E5MA_OP_N, M, N, K, 1.f, &A, K, &X, K, 1.f, nullptr, 0, &F, M, nullptr, 0); }, 20));
    cudaFree(dX); cudaFree(dF);
  }
  float *fA, *fB, *fC; cudaMalloc(&fA, (size_t)M * K * 4); cudaMalloc(&fB, (size_t)K * N * 4); cudaMalloc(&fC, (size_t)M * N * 4);
  cudaMemset(fA, 0, (size_t)M * K * 4); cudaMemset(fB, 0, (size_t)K * N * 4);
  cublasHandle_t hb; cublasCreate(&hb);
  const float one = 1, zero = 0;
  row("cuBLAS SGEMM NN FP32 -> FP32", bench.run([&] { cublasSgemm(hb, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, fA, M, fB, K, &zero, fC, M); }, 20));
  row("cuBLAS SGEMM TN FP32 -> FP32", bench.run([&] { cublasSgemm(hb, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, fA, K, fB, K, &zero, fC, M); }, 20));
  return 0;
}
