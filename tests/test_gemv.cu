// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. E5M2EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// e5ma_gemm on the GEMV path (n <= 8, TN, FP8 A), bit-exact against the host emulation of the
// kernel's lane order: A in {E5M2, E4M3} x x in {E5M2, E4M3, BF16, FP32} x n = 1..8, every output
// dtype, accumulate, bias, row-wise scales.
#include "test_util.h"

int main() {
  int fails = 0, idx = 0;
  const e5ma_dtype outs[] = {E5MA_BF16, E5MA_FP32, E5MA_FP16, E5MA_E5M2, E5MA_E4M3};
  const e5ma_dtype ops[] = {E5MA_E5M2, E5MA_E4M3, E5MA_BF16, E5MA_FP32};
  const int gs[][2] = {{1, 16}, {33, 272}, {300, 1024}, {1000, 4112}};
  for (e5ma_dtype at : {E5MA_E5M2, E5MA_E4M3})
    for (e5ma_dtype bt : ops)
      for (int n = 1; n <= 8; ++n) {
        const int *g = gs[(n + bt + at) % 4];
        Case c{E5MA_OP_T, E5MA_OP_N, g[0], n, g[1], at, bt, outs[idx % 5], n == 3 ? -1 : (int)outs[idx % 5],
               idx % 3 == 0 ? -1 : (int)E5MA_BF16, idx % 2 ? 0.75f : 1.f, n == 3 ? 1.f : (idx % 4 == 1 ? -1.5f : 0.f)};
        fails += run_case(c);
        if (n == 1 || n == 4 || n == 8) { c.vec = 1 + (n & 2) + (bt == E5MA_E5M2 || bt == E5MA_E4M3 ? 2 : 0); fails += run_case(c); }
        ++idx;
      }
  printf(fails ? "\n%d FAILED\n" : "\nALL PASS\n", fails);
  return fails != 0;
}
