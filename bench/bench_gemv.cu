// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. E5M2EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// e5ma_gemm on the GEMV path (TN, n <= 8): E5M2 weights read once, so the figure of merit is the
// effective bandwidth over A.  Compared with the DRAM peak, a device-to-device copy (practical
// bandwidth) and cuBLAS SGEMV / SGEMM on FP32 weights of the same shape (4x the bytes).
// Median time per row after an idle pause (see bench_util.h).
#include <vector>
#include <cublas_v2.h>
#include "../include/e5m2effma.h"
#include "bench_util.h"

int main(int argc, char **argv) {
  const int M = argc > 1 ? atoi(argv[1]) : 8192, K = argc > 2 ? atoi(argv[2]) : 8192;
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  int memory_clock_khz = 0;  // cudaDeviceProp::memoryClockRate is gone in CUDA 13
  cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate, 0);
  const double peak_bw = 2.0 * memory_clock_khz * 1e3 * (p.memoryBusWidth / 8) / 1e9;
  Bench bench;
  void *dA, *dX, *dY; cudaMalloc(&dA, (size_t)M * K); cudaMalloc(&dX, (size_t)K * 8 * 4); cudaMalloc(&dY, (size_t)M * 8 * 4);
  std::vector<unsigned char> h((size_t)M * K);
  unsigned r = 7;
  for (auto &x : h) { do { r ^= r << 13; r ^= r >> 17; r ^= r << 5; x = r & 0xFF; } while ((x & 0x7C) == 0x7C); }  // finite E5M2
  cudaMemcpy(dA, h.data(), h.size(), cudaMemcpyHostToDevice);
  cudaMemset(dX, 0, (size_t)K * 8 * 4);
  void *cp; cudaMalloc(&cp, (size_t)M * K);
  const BenchRow copy = bench.run([&] { cudaMemcpyAsync(cp, dA, (size_t)M * K, cudaMemcpyDeviceToDevice); }, 200);
  const double copy_bw = 2.0 * M * K / copy.ms / 1e6;
  printf("%s: M=%d K=%d, A = %.0f MB E5M2; DRAM peak %.0f GB/s (%d-bit bus), D2D copy %.0f GB/s (read+write)\n\n",
         p.name, M, K, (double)M * K / 1e6, peak_bw, p.memoryBusWidth, copy_bw);
  printf("%-36s %9s %12s %8s %8s %9s\n", "", "time", "A bytes/s", "of DRAM", "of copy", "SM clock");
  auto row = [&](const char *name, BenchRow x, double bytes, const char *note) {
    const double gbs = bytes / x.ms / 1e6;
    printf("%-36s %6.3f ms %7.1f GB/s %7.0f%% %7.0f%% %5.0f MHz%s\n", name, x.ms, gbs, 100 * gbs / peak_bw, 100 * gbs / copy_bw,
           x.mhz, note);
    fflush(stdout);
  };
  e5ma_tensor A = {dA, E5MA_E5M2, nullptr, nullptr, nullptr}, Y = {dY, E5MA_BF16, nullptr, nullptr, nullptr};
  const e5ma_dtype xts[] = {E5MA_E5M2, E5MA_BF16, E5MA_FP32};
  const char *xn[] = {"E4M3", "BF16", "FP16", "FP32", "E5M2"};  // by e5ma_dtype
  for (e5ma_dtype xt : xts)
    for (int n : {1, 2, 4, 8}) {
      e5ma_tensor X = {dX, xt, nullptr, nullptr, nullptr};
      e5ma_status st = E5MA_OK;
      const BenchRow x = bench.run([&] { st = e5ma_gemm(E5MA_OP_T, E5MA_OP_N, M, n, K, 1.f, &A, K, &X, K, 0.f, nullptr, 0, &Y, M, nullptr, 0); }, 200);
      char name[64]; snprintf(name, 64, "e5ma gemv n=%d x=%s -> BF16%s", n, xn[xt], st == E5MA_OK ? "" : " FAILED");
      row(name, x, (double)M * K, "");
    }
  float *fA; cudaMalloc(&fA, (size_t)M * K * 4); cudaMemset(fA, 0, (size_t)M * K * 4);
  cublasHandle_t hb; cublasCreate(&hb);
  const float one = 1, zero = 0;
  row("cuBLAS SGEMV FP32 (same M, K)", bench.run([&] { cublasSgemv(hb, CUBLAS_OP_T, K, M, &one, fA, K, (float *)dX, 1, &zero, (float *)dY, 1); }, 200),
      4.0 * M * K, "   (FP32 weights: 4x bytes)");
  row("cuBLAS SGEMM n=8 FP32 (same M, K)",
      bench.run([&] { cublasSgemm(hb, CUBLAS_OP_T, CUBLAS_OP_N, M, 8, K, &one, fA, K, (float *)dX, K, &zero, (float *)dY, M); }, 200),
      4.0 * M * K, "   (FP32 weights: 4x bytes)");
  return 0;
}
