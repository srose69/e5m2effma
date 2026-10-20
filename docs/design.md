# Design notes

Everything here was implemented and measured on a GTX 1080 (GP104) during development. What survived
is "exact FP8 decode, one FP32 FFMA per MAC, BF16 at the boundary". The rounding side of the story is
in [numerics.md](numerics.md).

## One FP32 FFMA per MAC

- A product of two FP8 values has at most 8 significant bits (6 for E5M2 × E5M2) and an exponent
  range inside FP32's normal range (2⁻³² … 2³² for E5M2 × E5M2), so the multiply part of an FFMA is
  **exact**. The only rounding is the add, i.e. the rounding of the accumulator.
- FP8 decodes into FP32 with integer bit placement and one FMUL. E5M2 is the upper byte of an FP16,
  and `sign | (x & 0x7F) << 21` is the FP32 encoding of value(x) · 2⁻¹¹², subnormals included —
  the same 13-bit shift that converts FP16 to FP32 — so multiplying by 2¹¹² restores it exactly. For
  E4M3 it is `<< 20` and 2¹²⁰. The two decoders differ only in these constants: the GEMM kernels compile
  to the same instruction sequence for every FP8 pairing. The GEMM kernel decodes each element once per
  block into FP32 shared-memory tiles, not once per MAC; the inner loop is `LDS.128` + `FFMA` only.
- Neither the FP16 hardware conversion (`F2F`, a quarter of the FFMA rate on sm_61 and slower on sm_7x)
  nor packed FP16 arithmetic (1/64 – 1/128 rate on GP104) is cheaper than this decode, and removing the
  FMUL breaks exactness: an integer exponent offset mishandles subnormals and zero, and folding 2¹¹² into
  the epilogue pushes small accumulators into FP32 subnormals.
- Consequence for performance: on Pascal the GEMM can at best match cuBLAS SGEMM in MAC rate, while
  reading 4× fewer bytes ([performance.md](performance.md)). Bandwidth-bound kernels (GEMV, the
  optimizer) gain the full 4×.

## Packed half-registers: not on Pascal, maybe on newer GPUs

Packing two values per 32-bit register only pays off if one instruction does the work of two. On GP104
it does not:

- **No separate integer pipe.** FFMA, IADD, LOP3, XMAD and IDP4A share one issue slot per scheduler per
  clock; shifts, PRMT, BFE and BFI cost about two slots each when issue is saturated. "Shifts are free"
  holds only when the ALU is idle, which it is not in a GEMM.
- **Packed BF16 adder.** It still needs, per lane, exponent comparison, an alignment shift,
  normalization and rounding. All six variants built (XMAD- or FMUL-based carrier × normalization by
  FFMA magic or by `FLO`) were 3–4× slower than one FFMA per MAC with the same results. One useful trick
  from that work: with each lane in its own 32-bit word and 14 guard bits, alignment needs one `shr` and
  no sticky bit.
- **"Two FP8 products per FP32 multiply"** (both mantissas in one FMUL with 2⁹ spacing) works only for
  the mantissas; each lane still needs its own exponent, split and add — at least ~3 ALU slots per MAC.
- **Dual issue.** What issues alongside FFMA (LD/ST, SFU, PRMT) cannot perform a MAC; splitting lanes
  into "multiply" and "correction" warps diverges the warp (40 vs 122 useful lane-ops per clock per SM).
- **DP4A** runs at full rate, but exact FP8 in int8 needs several digit planes: 2.25–4 issue slots per
  MAC versus 1 for FFMA. **HFMA2** runs at 1/128 of the FFMA rate on GP104.

So on Pascal one exact MAC costs at least one FP32 issue slot, and FFMA already does multiply, align,
add, normalize and round in that slot. Packing survives where it is free: four FP8 values per 32-bit
load, and two BF16 results per 32-bit store.

**Newer GPUs change the cost model, and the idea may come back there.** Volta, Turing and Ampere have a
separate INT32 datapath that issues alongside FP32, so the integer side of a packed adder (alignment,
lane split, correction) no longer has to steal FFMA slots; packed HFMA2 runs at twice the FFMA rate on
sm_60/62 and sm_70+, and sm_80+ has native BF16x2 FMA (`fma.rn.bf16x2`) next to BF16 Tensor Cores.
Two-lanes-per-register MACs could therefore beat one FFMA per MAC on those GPUs. This is untested — the
author has no such hardware yet — and the numerical argument still holds: whatever the packing, the
accumulator should be rounded to BF16 only at a storage boundary.

## Kernel configurations

Kernels are templates over a few parameters; the candidates are those that can win on some target:

- **GEMM**: 128 × 128, 64 × 64, 128 × 64 and 64 × 128 blocks, all with an 8 × 8 thread tile and a
  16-deep k-slice. They share the FFMA : LDS ratio and only trade parallelism on small grids, which
  depends on the SM count. A 32-deep k-slice is not a candidate: it halves the barriers, but there are
  already 1024 FFMAs per thread between barriers, and it doubles the shared memory (one block per SM
  where an SM has 64 KB).
- **GEMV**: 2, 4 or 8 rows per warp (8 only for n ≤ 2: beyond that the accumulators exceed the register
  budget on every target, all of which have 64 K registers per SM). Staging x in shared memory is not a
  candidate: it lost at every n on sm_61, and sm_70+ caches global loads in L1 anyway.
- **Tile loading.** Operands are decoded once per block into FP32 shared tiles `[k][row]`. Lanes run
  along m, so the column-major output is written contiguously. M/N-contiguous operands are loaded in
  chunks of 4 rows (one `STS.128` per chunk, 16-byte lane stride: no bank conflicts, coalesced global
  reads). K-contiguous operands (the TN layout of `linear`) are loaded as 8 consecutive k of one row
  and transposed with scalar stores; the k-half `kk ≥ 8` of each row is XOR-swizzled by 16 columns
  inside its aligned 32-column block, so the two lanes of a row never share a bank, without padding
  that would break the 16-byte alignment of the `LDS.128` fragment reads. Rows past the end of the
  matrix read row 0: they only feed outputs that are never stored, so the loader has no branches.
- **Epilogue.** For the common BF16 output without bias or C, each thread writes its 4 consecutive rows
  of a column as one 8-byte store. Every other combination (FP32 / FP16 / FP8 output, bias, C,
  accumulation) is staged through shared memory one column of the thread tile at a time: the block
  writes BM × TY accumulators into the free A tile, then every thread runs the generic epilogue on 8
  of them, consecutive threads on consecutive rows. The epilogue is inlined once, in a loop over the
  column passes: inlining it into each of the 64 outputs per thread made the kernels ~20× larger,
  and calling it out of line put a call frame on the stack. The stores now coalesce along m, which
  made TE-style FP32 accumulation 6–7 % faster than per-output calls. The GEMV kernel hands each
  output to a different lane after the butterfly, so stores run in parallel and its epilogue is
  inlined once.

- **Raw FP8 prefetch.** The next tile's global loads are issued right after the current tile is in
  shared memory and are decoded only after it has been computed. FP8 data stays raw meanwhile, so this
  costs 4 registers per thread (8 bytes of A and 8 of B), and the global latency hides behind the block's
  own FFMAs instead of only behind other blocks (+2–4 %).
- **Occupancy.** 256 threads × ~120 registers = 2 blocks (16 warps) per SM, the same as cuBLAS SGEMM on
  this GPU. The 64 × 64 configuration (64 threads, for grids too small to fill the GPU with large tiles)
  targets 6 blocks per SM: at 8 its doubled loader state spilled. A third block needs ≤ 80 registers per
  thread, but an 8 × 8 thread tile alone holds 64 accumulators: under that cap (or 96 registers with
  128-thread blocks, 20 warps) ptxas spills and leaves 30–45 % of the FFMAs bank-conflicting, and the
  kernel runs at 75 % of cuBLAS. An 8 × 4 thread tile (128 × 64 block) does reach 3 blocks without
  spills, but doubles decode, LDS and address work per FFMA and runs at 80 %. Two blocks of 8 × 8 tiles
  stay the best configuration on Pascal.

These changes took the library GEMM from 82–85 % to 94–100 % of cuBLAS SGEMM on the GTX 1080
(interleaved A/B runs in one process): coalesced output and the packed BF16 store ≈ 9 %, the
branch-free loader ≈ 5 %, conflict-free shared stores ≈ 2–3 %.

How configurations are chosen per architecture: [building.md](building.md#tuning).
