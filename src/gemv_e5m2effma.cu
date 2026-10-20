// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. E5M2EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// E5M2EffMA GEMV host side: the kernel for the A dtype, x dtype and n, in the configuration the
// device's table selects per x dtype and n.
#include "gemv_e5m2effma.cuh"

namespace e5ma {
namespace kern {

// dtype codes in operand_index order: A E5M2 (4) / E4M3 (0); x E5M2 (4), E4M3 (0), BF16 (1), FP32 (3)
#define E5MA_DECL_GEMV_X(id, a, t)                                                                 \
  E5MA_GEMV_PROTO(id, a, t, 1); E5MA_GEMV_PROTO(id, a, t, 2); E5MA_GEMV_PROTO(id, a, t, 3); E5MA_GEMV_PROTO(id, a, t, 4); \
  E5MA_GEMV_PROTO(id, a, t, 5); E5MA_GEMV_PROTO(id, a, t, 6); E5MA_GEMV_PROTO(id, a, t, 7); E5MA_GEMV_PROTO(id, a, t, 8);
#define E5MA_DECL_GEMV_A(id, a) E5MA_DECL_GEMV_X(id, a, 4) E5MA_DECL_GEMV_X(id, a, 0) E5MA_DECL_GEMV_X(id, a, 1) E5MA_DECL_GEMV_X(id, a, 3)
#define E5MA_DECL_GEMV(id) E5MA_DECL_GEMV_A(id, 4) E5MA_DECL_GEMV_A(id, 0)
E5MA_GEMV_LIB(E5MA_DECL_GEMV)
#undef E5MA_DECL_GEMV
#undef E5MA_DECL_GEMV_A
#undef E5MA_DECL_GEMV_X

cudaError_t launch_gemv(int at, int xt, const uint8_t *A, int lda, const uint8_t *X, int ldx, int m, int n, int k,
                        const Epilogue &e, cudaStream_t s) {
  const int id = device_info().table->gemv[operand_index(xt)][n - 1];
  using Fn = cudaError_t (*)(const uint8_t *, int, const uint8_t *, int, int, int, const Epilogue &, cudaStream_t);
#define E5MA_ROW(cid, a, t)                                                                    \
  {E5MA_GEMV_ENTRY(cid, a, t, 1), E5MA_GEMV_ENTRY(cid, a, t, 2), E5MA_GEMV_ENTRY(cid, a, t, 3), E5MA_GEMV_ENTRY(cid, a, t, 4), \
   E5MA_GEMV_ENTRY(cid, a, t, 5), E5MA_GEMV_ENTRY(cid, a, t, 6), E5MA_GEMV_ENTRY(cid, a, t, 7), E5MA_GEMV_ENTRY(cid, a, t, 8)}
#define E5MA_ROWS(cid, a) {E5MA_ROW(cid, a, 4), E5MA_ROW(cid, a, 0), E5MA_ROW(cid, a, 1), E5MA_ROW(cid, a, 3)}
#define E5MA_CASE(cid)                                                                        \
  if (id == cid) {                                                                            \
    static const Fn f[2][kOperandTypes][8] = {E5MA_ROWS(cid, 4), E5MA_ROWS(cid, 0)};          \
    return f[at == E5MA_E5M2 ? 0 : 1][operand_index(xt)][n - 1](A, lda, X, ldx, m, k, e, s);  \
  }
  E5MA_GEMV_LIB(E5MA_CASE)
#undef E5MA_CASE
#undef E5MA_ROWS
#undef E5MA_ROW
  return cudaErrorInvalidConfiguration;
}

}  // namespace kern
}  // namespace e5ma
