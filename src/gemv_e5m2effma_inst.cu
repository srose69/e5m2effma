// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. E5M2EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// One GEMV instance object: -DE5MA_ID=<config> -DE5MA_TA=<FP8 A dtype> -DE5MA_TX=<x dtype>, n = 1..8.
// The kernel bodies are compiled only for the architectures whose table (or the defaults) select the
// configuration for that x dtype and n, trapping stubs elsewhere.
#include "gemv_e5m2effma.cuh"

namespace e5ma {
namespace kern {

#define E5MA_CFG_(id) GemvCfg##id
#define E5MA_CFG(id) E5MA_CFG_(id)
// (a template on the id, not a plain struct: kernel names must differ between instance objects)
template <int ID> struct LibGemv : E5MA_CFG(E5MA_ID) {
  template <int NB, int XT> struct On { static constexpr bool value = gemv_selected(E5MA_ARCH_CC, ID, XT, NB); };
};
using C = LibGemv<E5MA_ID>;

#define E5MA_GEMV_DEF(nb) \
  E5MA_GEMV_PROTO(E5MA_ID, E5MA_TA, E5MA_TX, nb) { return gemv_launch<C, nb, E5MA_TA, E5MA_TX>(A, lda, X, ldx, m, k, e, s); }
E5MA_GEMV_DEF(1) E5MA_GEMV_DEF(2) E5MA_GEMV_DEF(3) E5MA_GEMV_DEF(4)
E5MA_GEMV_DEF(5) E5MA_GEMV_DEF(6) E5MA_GEMV_DEF(7) E5MA_GEMV_DEF(8)

}  // namespace kern
}  // namespace e5ma
