// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. E5M2EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// e5ma_gemm, tiled kernel, bit-exact against the host emulation: all 4 layouts x A, B in
// {E5M2, E4M3, BF16, FP32}, m/n tails, ld padding, every output dtype, C of another dtype, TE-style
// accumulate (C = NULL, D += ...), alpha / beta / bias / scale / amax, larger problems including the
// E5M2 x E4M3 pairings of TE's HYBRID backward, and row-wise scales.
#include "test_util.h"

int main() {
  int fails = 0, idx = 0;
  const e5ma_dtype outs[] = {E5MA_BF16, E5MA_FP32, E5MA_FP16, E5MA_E5M2, E5MA_E4M3};
  const e5ma_dtype ops[] = {E5MA_E5M2, E5MA_E4M3, E5MA_BF16, E5MA_FP32};
  const int sizes[][3] = {{1, 9, 16}, {37, 45, 48}, {200, 300, 144}, {257, 129, 256}};
  // tiled GEMM: layouts x operand types x tails; outputs, C dtype, accumulate, bias rotate
  for (e5ma_op ta : {E5MA_OP_N, E5MA_OP_T})
    for (e5ma_op tb : {E5MA_OP_N, E5MA_OP_T})
      for (e5ma_dtype at : ops)
        for (e5ma_dtype bt : ops) {
          const int *sz = sizes[idx % 4];
          Case c{ta, tb, sz[0], sz[1] < 9 ? 9 : sz[1], sz[2], at, bt, outs[idx % 5],
                 idx % 5 == 1 ? -1 : (idx % 5 == 2 ? (int)outs[(idx + 1) % 5] : (int)outs[idx % 5]),
                 idx % 3 == 0 ? -1 : (idx % 3 == 1 ? (int)E5MA_BF16 : (int)E5MA_FP32),
                 idx % 2 ? 1.25f : 1.f, idx % 5 == 1 ? 1.f : (idx % 4 == 3 ? 0.5f : 0.f)};
          fails += run_case(c); ++idx;
        }
  // larger problems (large-tile configuration)
  fails += run_case(Case{E5MA_OP_T, E5MA_OP_N, 1100, 1300, 512, E5MA_E5M2, E5MA_E5M2, E5MA_BF16, -1, E5MA_FP32, 1.f, 1.f});
  fails += run_case(Case{E5MA_OP_N, E5MA_OP_N, 1029, 1500, 256, E5MA_E5M2, E5MA_BF16, E5MA_FP32, -1, -1, 0.5f, 1.f});
  // TE HYBRID backward: dgrad = dY (E5M2) x W (E4M3), wgrad += dY^T (E5M2) x X (E4M3) into FP32
  fails += run_case(Case{E5MA_OP_N, E5MA_OP_N, 768, 640, 512, E5MA_E5M2, E5MA_E4M3, E5MA_BF16, -1, -1, 1.f, 0.f});
  fails += run_case(Case{E5MA_OP_T, E5MA_OP_N, 512, 640, 768, E5MA_E4M3, E5MA_E5M2, E5MA_FP32, -1, -1, 1.f, 1.f});
  // outer (row-wise) scaling: per row of op(A), per column of op(B), both; packed BF16 and generic stores
  int v = 1;
  for (e5ma_op ta : {E5MA_OP_N, E5MA_OP_T})
    for (e5ma_op tb : {E5MA_OP_N, E5MA_OP_T})
      for (e5ma_dtype dt : {E5MA_BF16, E5MA_FP32}) {
        Case c{ta, tb, 257, 129, 256, v == 2 ? E5MA_E4M3 : E5MA_E5M2, v == 1 ? E5MA_E4M3 : E5MA_E5M2, dt, dt == E5MA_BF16 ? -1 : (int)E5MA_BF16,
               dt == E5MA_FP32 ? (int)E5MA_FP32 : -1, 1.25f, dt == E5MA_BF16 ? 0.f : 0.5f};
        c.vec = v; v = v % 3 + 1;
        fails += run_case(c);
      }
  {
    Case c{E5MA_OP_T, E5MA_OP_N, 1100, 1300, 512, E5MA_E5M2, E5MA_BF16, E5MA_BF16, -1, -1, 1.f, 0.f};
    c.vec = 3;  // E5M2 weights per output row x BF16 activations (B's flag is ignored for BF16)
    fails += run_case(c);
  }
  printf(fails ? "\n%d FAILED\n" : "\nALL PASS\n", fails);
  return fails != 0;
}
