// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. E5M2EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// One GEMM instance object: -DE5MA_ID=<config> -DE5MA_TA=<A dtype> -DE5MA_TB=<B dtype>, all 4
// layouts.  Small objects keep the build parallel; the kernel bodies are compiled only for the
// architectures whose table (or the defaults) select the configuration, trapping stubs elsewhere.
#include "gemm_e5m2effma.cuh"

namespace e5ma {
namespace kern {

#define E5MA_CFG_(id) GemmCfg##id
#define E5MA_CFG(id) E5MA_CFG_(id)
// (a template on the id, not a plain struct: kernel names must differ between instance objects)
template <int ID> struct LibGemm : E5MA_CFG(E5MA_ID) {
  template <int TA, int TB> struct On { static constexpr bool value = gemm_selected(E5MA_ARCH_CC, ID); };
};
using C = LibGemm<E5MA_ID>;

E5MA_GEMM_PROTO(E5MA_ID, E5MA_TA, E5MA_TB) {
  if (akc) return bkc ? gemm_launch<C, E5MA_TA, E5MA_TB, true, true>(A, lda, B, ldb, m, n, k, e, s)
                      : gemm_launch<C, E5MA_TA, E5MA_TB, true, false>(A, lda, B, ldb, m, n, k, e, s);
  return bkc ? gemm_launch<C, E5MA_TA, E5MA_TB, false, true>(A, lda, B, ldb, m, n, k, e, s)
             : gemm_launch<C, E5MA_TA, E5MA_TB, false, false>(A, lda, B, ldb, m, n, k, e, s);
}

}  // namespace kern
}  // namespace e5ma
