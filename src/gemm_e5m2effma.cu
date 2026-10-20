// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. E5M2EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// E5M2EffMA GEMM host side: e5ma_gemm (validation, epilogue setup, routing of n <= 8 TN problems with
// an FP8 A to the GEMV kernel) and the choice of the tiled-kernel configuration.
#include "gemm_e5m2effma.cuh"
#include "gemv_e5m2effma.cuh"

namespace e5ma {
namespace kern {

// operand dtype codes in operand_index order: E5M2 (4), E4M3 (0), BF16 (1), FP32 (3)
#define E5MA_DECL_GEMM_T(id, ta) E5MA_GEMM_PROTO(id, ta, 4); E5MA_GEMM_PROTO(id, ta, 0); E5MA_GEMM_PROTO(id, ta, 1); E5MA_GEMM_PROTO(id, ta, 3);
#define E5MA_DECL_GEMM(id) E5MA_DECL_GEMM_T(id, 4) E5MA_DECL_GEMM_T(id, 0) E5MA_DECL_GEMM_T(id, 1) E5MA_DECL_GEMM_T(id, 3)
E5MA_GEMM_LIB(E5MA_DECL_GEMM)
#undef E5MA_DECL_GEMM
#undef E5MA_DECL_GEMM_T

static cudaError_t launch_gemm(int ta, int tb, bool akc, bool bkc, const uint8_t *A, int lda, const uint8_t *B,
                               int ldb, int m, int n, int k, const Epilogue &e, cudaStream_t s) {
  const DeviceInfo &d = device_info();
  int id = d.table->gemm_large, bm = 128, bn = 128;
#define E5MA_TILE(cid, cbm, cbn, cbk, ctm, ctn) if (id == cid) { bm = cbm; bn = cbn; }
  E5MA_GEMM_CANDIDATES(E5MA_TILE)
#undef E5MA_TILE
  if ((long long)((m + bm - 1) / bm) * ((n + bn - 1) / bn) < 2LL * d.sms) id = d.table->gemm_small;
  using Fn = cudaError_t (*)(bool, bool, const uint8_t *, int, const uint8_t *, int, int, int, int, const Epilogue &,
                             cudaStream_t);
  const int ia = operand_index(ta), ib = operand_index(tb);
#define E5MA_ROW(cid, t) {E5MA_GEMM_ENTRY(cid, t, 4), E5MA_GEMM_ENTRY(cid, t, 0), E5MA_GEMM_ENTRY(cid, t, 1), E5MA_GEMM_ENTRY(cid, t, 3)}
#define E5MA_CASE(cid)                                                                        \
  if (id == cid) {                                                                             \
    static const Fn f[kOperandTypes][kOperandTypes] = {E5MA_ROW(cid, 4), E5MA_ROW(cid, 0), E5MA_ROW(cid, 1), E5MA_ROW(cid, 3)}; \
    return f[ia][ib](akc, bkc, A, lda, B, ldb, m, n, k, e, s);                                 \
  }
  E5MA_GEMM_LIB(E5MA_CASE)
#undef E5MA_CASE
#undef E5MA_ROW
  return cudaErrorInvalidConfiguration;
}

}  // namespace kern
}  // namespace e5ma

extern "C" e5ma_status e5ma_gemm(e5ma_op transa, e5ma_op transb, int m, int n, int k, float alpha,
                                 const e5ma_tensor *A, int lda, const e5ma_tensor *B, int ldb, float beta,
                                 const e5ma_tensor *C, int ldc, const e5ma_tensor *D, int ldd,
                                 const e5ma_tensor *bias, cudaStream_t stream) {
  using namespace e5ma::kern;
  using e5ma::fail;
  e5ma::clear_error();
  const auto valid = [](const e5ma_tensor *t) { return t->dtype >= E5MA_E4M3 && t->dtype <= E5MA_E5M2; };
  const auto operand = [](e5ma_dtype t) { return e5ma::is_fp8(t) || t == E5MA_BF16 || t == E5MA_FP32; };
  const auto aligned16 = [](const void *p) { return ((uintptr_t)p & 15u) == 0; };
  const auto op_ok = [](e5ma_op o) { return o == E5MA_OP_N || o == E5MA_OP_T; };
  if (!A || !B || !D || !A->data || !B->data || !D->data)
    return fail(E5MA_INVALID_VALUE, "e5ma_gemm: A, B, D or their data is NULL");
  if (!op_ok(transa) || !op_ok(transb)) return fail(E5MA_INVALID_VALUE, "e5ma_gemm: transa / transb must be E5MA_OP_N or E5MA_OP_T");
  if (m < 0 || n < 0 || k < 0) return fail(E5MA_INVALID_VALUE, "e5ma_gemm: negative size (m=%d n=%d k=%d)", m, n, k);
  if (!valid(D)) return fail(E5MA_INVALID_VALUE, "e5ma_gemm: invalid D dtype %d", (int)D->dtype);
  if (!operand(A->dtype) || !operand(B->dtype))
    return fail(E5MA_NOT_SUPPORTED, "e5ma_gemm: A and B must be E5M2, E4M3, BF16 or FP32 (got dtypes %d, %d)", (int)A->dtype,
                (int)B->dtype);
  if (m == 0 || n == 0) return E5MA_OK;
  if (k % 16) return fail(E5MA_INVALID_VALUE, "e5ma_gemm: k = %d is not a multiple of 16", k);
  if (lda % 16 || ldb % 16) return fail(E5MA_INVALID_VALUE, "e5ma_gemm: lda = %d / ldb = %d must be multiples of 16", lda, ldb);
  if (!aligned16(A->data) || !aligned16(B->data)) return fail(E5MA_INVALID_VALUE, "e5ma_gemm: A and B must be 16-byte aligned");
  const bool a_kc = transa == E5MA_OP_T, b_kc = transb == E5MA_OP_N;
  if (lda < (a_kc ? k : m)) return fail(E5MA_INVALID_VALUE, "e5ma_gemm: lda = %d < %d", lda, a_kc ? k : m);
  if (ldb < (b_kc ? k : n)) return fail(E5MA_INVALID_VALUE, "e5ma_gemm: ldb = %d < %d", ldb, b_kc ? k : n);
  if (ldd < m) return fail(E5MA_INVALID_VALUE, "e5ma_gemm: ldd = %d < m = %d", ldd, m);
  if (bias && bias->data && !valid(bias)) return fail(E5MA_INVALID_VALUE, "e5ma_gemm: invalid bias dtype %d", (int)bias->dtype);
  if (beta != 0.f && C && C->data && !valid(C)) return fail(E5MA_INVALID_VALUE, "e5ma_gemm: invalid C dtype %d", (int)C->dtype);
  if (beta != 0.f && C && C->data && ldc < m) return fail(E5MA_INVALID_VALUE, "e5ma_gemm: ldc = %d < m = %d", ldc, m);
  const auto scaling_ok = [](const e5ma_tensor *t) { return t->scaling == E5MA_SCALING_TENSOR || t->scaling == E5MA_SCALING_OUTER; };
  if (!scaling_ok(A) || !scaling_ok(B)) return fail(E5MA_INVALID_VALUE, "e5ma_gemm: invalid scaling of A or B");
  for (const e5ma_tensor *t : {A, B})
    if (t->scaling == E5MA_SCALING_OUTER && (!e5ma::is_fp8(t->dtype) || !t->scale_inv))
      return fail(E5MA_INVALID_VALUE, "e5ma_gemm: E5MA_SCALING_OUTER needs an FP8 operand with a scale_inv vector");
  for (const e5ma_tensor *t : {C, D, bias})
    if (t && t->scaling != E5MA_SCALING_TENSOR)
      return fail(E5MA_INVALID_VALUE, "e5ma_gemm: C, D and bias take per-tensor scaling only");
  if (e5ma_status st = e5ma::check_device()) return st;
  const bool c_is_d = beta != 0.f && (!C || !C->data);  // TE "accumulate": D += ...

  Epilogue e;
  e.sa = e5ma::is_fp8(A->dtype) ? A->scale_inv : nullptr;
  e.sb = e5ma::is_fp8(B->dtype) ? B->scale_inv : nullptr;
  e.sa_vec = A->scaling == E5MA_SCALING_OUTER;
  e.sb_vec = B->scaling == E5MA_SCALING_OUTER;
  e.alpha = alpha; e.beta = beta;
  const e5ma_tensor *Cx = c_is_d ? D : C;
  e.c = beta != 0.f ? Cx->data : nullptr; e.ldc = beta != 0.f ? (c_is_d ? ldd : ldc) : 0;
  e.c_dtype = beta != 0.f ? Cx->dtype : E5MA_FP32;
  e.c_sinv = beta != 0.f && e5ma::is_fp8(Cx->dtype) ? Cx->scale_inv : nullptr;
  e.d = D->data; e.ldd = ldd; e.d_dtype = D->dtype; e.d_scale = e5ma::is_fp8(D->dtype) ? D->scale : nullptr;
  e.amax = D->amax;
  e.bias = bias && bias->data ? bias->data : nullptr; e.bias_dtype = bias ? bias->dtype : E5MA_FP32;

  const uint8_t *a = (const uint8_t *)A->data, *b = (const uint8_t *)B->data;
  const cudaError_t err = n <= 8 && a_kc && b_kc && e5ma::is_fp8(A->dtype)
                              ? launch_gemv(A->dtype, B->dtype, a, lda, b, ldb, m, n, k, e, stream)
                              : launch_gemm(A->dtype, B->dtype, a_kc, b_kc, a, lda, b, ldb, m, n, k, e, stream);
  return e5ma::cuda_status(err, "e5ma_gemm: kernel launch");
}
