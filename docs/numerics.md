# Numerics

## Guarantees

- **Products.** Every product of finite FP8 values is exact in FP32, for every pairing: E5M2 × E5M2
  (3 × 3 significant bits, magnitudes 2⁻³² … 3.3 · 10⁹), E5M2 × E4M3 and E4M3 × E4M3 all fit an FP32
  significand and lie inside its normal range. FP8 × BF16 and BF16 × BF16 products are exact whenever
  the mathematical product is a normal FP32 number: the significands fit, but BF16 has almost the full
  FP32 exponent range, so a large BF16 value times e.g. 57344 can overflow and a tiny one can
  underflow. FP32 × FP32 rounds as in SGEMM.
- **Accumulation.** One FP32 FFMA per MAC, `k` ascending per output. The GEMV kernel uses its own fixed
  order: lane `l` accumulates `k = 256c + 8l + j` in `(c, j)` order, then the 32 lanes are combined by an
  xor butterfly (16, 8, 4, 2, 1).
- **Epilogue.** In FP32, in this order, with a single rounding to the output type:
  `v = acc · s` (`s = (α · sa) · sb`, with `sa[i]` / `sb[j]` for row-wise scales); `v = v + bias[i]`; `v = β · C + v` (one FMA);
  `amax = max(amax, |v|)`; `D = convert(v)`. BF16 / FP16 / FP32 outputs round to nearest even; FP8
  outputs are `RNE_satfinite(v · scale)` (saturating to ±57344 for E5M2 and ±448 for E4M3, Inf
  included; NaN → `0x7F`).
- **Reproducibility.** Results are deterministic and bit-reproducible for a fixed kernel path (same
  configuration and architecture). Cross-architecture reproducibility has not been validated (execution
  is tested on sm_61 only), and future Tensor Core kernels may use a different reduction order.
- **Contract.** FP8 GEMM operands must be finite: no E5M2 Inf / NaN codes (exponent 31: `0x7C` … `0x7F`,
  `0xFC` … `0xFF`) and no E4M3 NaN code (`0x7F` / `0xFF`). The GEMM decode maps them to finite values
  and does not detect them (that would cost a pass over the data or a test per element).
- **Casts, C and bias** decode every FP8 code exactly, Inf and NaN included, and `amax` becomes Inf /
  NaN when such a value passes through. That is the overflow signal of loss scaling: an E5M2
  gradient that overflowed shows up in the `amax` of the cast that produced it or consumed it, before
  any GEMM sees it.

## Accuracy

On the distributions studied — N(0, 1) → E4M3, uniformly random finite E4M3 codes, and all-positive
data, K from 256 to 16384, 4096 outputs per point (measured with E4M3 during the development of the
kernels; the argument does not depend on the input format) — FP32 accumulation followed by a single BF16
rounding matched the correctly rounded exact sum in **99.9–100 %** of outputs, with a maximum error of
0.5 ulp. This is a measurement, not a guarantee for arbitrary inputs and K: heavy cancellation (an
exact sum much smaller than the sum of magnitudes) can cost more.

## Rounding policy: once, at the end

The project started as an attempt to keep the accumulator itself in BF16 (two BF16 lanes packed in one
32-bit register). Three policies were implemented and compared against the exact sum:

| Rounding to BF16 | Wrong vs. correctly rounded sum | Mean error | Drift |
|---|---:|---:|---|
| after every MAC | 79–100 % | 5–136 ulp | up to −136 ulp on all-positive data (stagnation) |
| every N MACs (N = 2 … 1024) | decreases with N | decreases with N | still −5.8 ulp at K = 16384, N = 128 |
| **once, when the result is stored** | **0–0.12 %** | **0.25 ulp (max 0.5)** | none |

**Why not after every MAC.** BF16 has an 8-bit significand. Once the running sum is large, every
product smaller than half an ulp of the accumulator is lost, and the rest is rounded with an error of up
to half an ulp per step:

- *Same-sign data stagnates*: with all-positive inputs the per-MAC BF16 sum ends up systematically too
  small — −32 ulp at K = 1024, −95 ulp at K = 4096, −136 ulp at K = 16384, wrong in 100 % of outputs.
- *Zero-mean data random-walks*: with N(0, 1) → E4M3 inputs the error grows like √K — mean 10 ulp at
  K = 256 and 72 ulp at K = 16384; 90–99 % of outputs differ from the correctly rounded sum.
- It is also about **4.5× slower**: the rounding is extra ALU work on every MAC.

**Why not every N MACs.** Each intermediate rounding adds an independent error of up to half an ulp.
On zero-mean data these errors are unbiased (the central limit theorem averages them), but their spread
still grows like √(K/N): at K = 4096, N = 128 the mean error is 4.6 ulp versus 0.25 ulp for a single
rounding. On same-sign data the bias does not average out at all. No N gave a speed or accuracy
advantage, so inside a kernel N = K. A period only makes sense as a **storage boundary** — split-K,
streaming over K, accumulation across launches: store BF16 there and resume exactly (BF16 → FP32 is a
shift), with chunks as long as the structure allows; or accumulate across launches in an FP32 `D`
(TE-style `accumulate`, e.g. FP32 `main_grad`) and round only when the value is consumed.

**Why once is enough.** A chain of FFMAs over exact products has a relative error of roughly
√K · 2⁻²⁴ for well-conditioned sums, far below half a BF16 ulp (2⁻⁹) for any practical K.

**Why the FFMA chain is a correct building block.** For a single step, rounding to FP32 and then to
BF16 equals rounding the exact value straight to BF16 (FP32 has more than 2 · 8 + 2 significand bits,
so the double rounding is harmless). `tests/test_e5m2effma.cu` checks the kernels' decode → FFMA → RNE
path for all 2³² (A, B, ACC) combinations of one MAC against an exact integer reference, for
E5M2 × E5M2, E5M2 × E4M3 and E4M3 × E4M3.

## Why BF16 as the output and storage format

- Same 8-bit exponent as FP32: no overflow, no scaling factor and no `amax` bookkeeping, unlike FP16 or
  FP8 outputs.
- FP32 → BF16 is rounding of the upper half; the kernels do round-to-nearest-even in three full-rate
  instructions by moving the tie bit into the carry flag (`LOP` + `IADD.CC` + `IADD.X`) instead of the
  half-rate `BFE` + `IADD3`; NaN maps to the canonical `0x7FC0`.
- BF16 → FP32 is exact (a shift or a mask), so a stored partial result resumes without loss.
- Half the bytes of FP32, and it is the activation / gradient format of TE-style training, so GEMM
  outputs feed the optimizer and the next layer directly.
- **It is the Tensor Core format of newer GPUs.** From sm_80 on, BF16 is a native `mma.sync` input type
  with FP32 accumulation. Both FP8 formats convert to BF16 exactly (3 or 4 significant bits, exponents
  well inside BF16's) and BF16 activations go in as they are, while an FP16 Tensor Core path would have
  to convert BF16 activations to FP16 with range loss — even though E5M2 itself is a truncated FP16 and
  converts to FP16 with a single byte permute. Keeping BF16 as the interchange format means a future Tensor Core backend changes
  only the inner MAC loop; operands, outputs and the optimizer stay as they are.
