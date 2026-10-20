// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. E5M2EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// C entry of the exact reference for tests/check_ref.py.
#include "e5m2effma_ref.h"
extern "C" uint32_t ref_lane(uint32_t a, uint32_t b, uint32_t acc, int fa, int fb) { return e5ma_ref::lane(a, b, acc, fa, fb); }
