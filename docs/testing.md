# Testing

`make test` runs everything below on GPU `$(GPU)` (default 0); the exhaustive checks take a few
seconds on a GTX 1080.

| Test | What it checks |
|---|---|
| `tests/test_gemm.cu` | `e5ma_gemm` bit-exact against a host emulation of the documented order: 64 configurations (4 layouts × 16 A/B dtype pairs over E5M2 / E4M3 / BF16 / FP32, m/n tails, every output dtype, C of another dtype, `accumulate`, bias, scale, amax), 4 large problems (including the E5M2 × E4M3 pairings of TE's HYBRID backward) and 9 with row-wise scales. |
| `tests/test_gemv.cu` | The GEMV path, bit-exact against an emulation of its lane order: A in E5M2 / E4M3 × x in E5M2 / E4M3 / BF16 / FP32 × n = 1 … 8, plus row-wise scales. |
| `tests/test_adamw.cu` | `e5ma_adam` bit-exact: Adam and AdamW over 12 param / grad / moment dtype combinations with an E5M2 or E4M3 copy and amax, and two unaligned cases. |
| `tests/test_e5m2effma.cu` | Exhaustive over 2³² inputs each: one MAC (decode → FFMA → RNE) against an exact integer reference for E5M2 × E5M2, E5M2 × E4M3 and E4M3 × E4M3; FP32 → BF16 rounding and FP32 → E5M2 / E4M3 encoding on every FP32 bit pattern; `e5ma_cast` on 1 M random bit patterns and on every FP8 code (Inf / NaN included). |
| `test_e5m2effma -DE5MA_NEGATIVE_CONTROLS` | Five deliberately broken variants (truncation instead of RNE, flush-to-zero, RNE without NaN handling, E5M2 and E4M3 ties away from zero) must be rejected. |
| `tests/check_ref.py` | The integer reference model itself against Python `Fraction`, for all four FP8 pairings (2.35 M cases). |
| `binds/python/tests` | The PyTorch bindings (`pytest`): results against a float64 reference for every operand pairing, casts with Inf / NaN, `Float8Tensor`, `torch._scaled_mm` and its dispatch shared with other *EffMA packages, and rejection of bad input (dtype, device, shape, scales, overlapping output). |

The reference model (`tests/e5m2effma_ref.h`) holds the exact sum `ACC + A·B` as a 320-bit integer in
units of 2⁻¹³³ and rounds it with integer arithmetic only, so it shares no floating-point code with the
implementation it checks.
