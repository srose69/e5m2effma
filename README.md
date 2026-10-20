<div align="center">

# E5M2EffMA

**FP8 (E5M2 and E4M3) GEMM, GEMV, casts and fused AdamW for NVIDIA GPUs without native FP8**

![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-76B900?logo=nvidia)
![Targets](https://img.shields.io/badge/targets-sm__50%E2%80%93sm__87-blue)
![Tested](https://img.shields.io/badge/tested%20on-GTX%201080%20%2B%20RTX%203090%20(community)-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch)
![API](https://img.shields.io/badge/API-C%20%2F%20cuBLASLt--style-lightgrey)
![License](https://img.shields.io/badge/license-AGPL--3.0%20%2B%20linking%20exception-blue)

</div>

NVIDIA's native FP8 compute stack (cuBLASLt FP8, Transformer Engine, `torch._scaled_mm`) starts at
Ada / Hopper-class GPUs (sm_89+). **E5M2EffMA** brings the same data formats and the same GEMM
semantics to the GPUs most people still have — Maxwell, Pascal, Volta, Turing, Ampere and Jetson Orin
(sm_50 … sm_87). It is built around **E5M2**, the FP8 format with the range of FP16 that TE uses for
gradients, and takes E4M3 as well, in any pairing: the backward GEMMs of TE's HYBRID recipe (E5M2
gradients × E4M3 weights / activations) run as they are. Products are exact, accumulation is FP32,
and outputs come out in BF16 / FP16 / FP32 / E5M2 / E4M3.

For E4M3 alone, see the sibling library [E4M3EffMA](https://github.com/srose69/e4m3effma); the two
share their kernels and can be used together (`torch._scaled_mm` dispatches to whichever covers a call).

> [!IMPORTANT]
> **Tested and tuned by the author on a GTX 1080 only.** The author has no GPU newer than Pascal;
> everything else is up to the community, or to time. So far an RTX 3090 (sm_86, CUDA 13.3) has
> passed the test suite of the sibling E4M3EffMA, which shares these kernels, in a community run
> (untuned). The library compiles for every
> architecture from sm_50 to sm_87, but it runs the same FP32-FFMA kernels everywhere: it does **not**
> use Tensor Cores, `cp.async`, `ldmatrix`, or any other Volta/Turing/Ampere-specific feature, although
> it could and should on those GPUs. Contributions are very welcome — and so is hardware, which is
> honestly the better way to get those kernels written :)  See [Contributing](#contributing).

## Performance at a glance

GTX 1080. The baseline is cuBLAS on **FP32 data** of the same shape (SGEMM / SGEMV) — the fastest
way to run the same model on this GPU without FP8 — and PyTorch's fused AdamW for the optimizer. Median times; the FP32 peak of each GEMM row is
taken at the SM clock measured while that row ran:

| Workload | E5M2EffMA | Utilization | Baseline: cuBLAS FP32 | Speed vs baseline |
|---|---:|---|---:|---:|
| GEMM 4096³, E5M2 × E5M2 → BF16 | 19.9 ms | **77 %** of FP32 peak | SGEMM 19.2 ms (76 % of FP32 peak) | 0.96× |
| GEMM 4096³, E5M2 × E4M3 → BF16 (HYBRID backward) | 20.0 ms | 76 % of FP32 peak | SGEMM 19.2 ms (76 % of FP32 peak) | 0.96× |
| GEMM 4096³, E5M2 weights × BF16 activations | 19.1 ms | **77 %** of FP32 peak | SGEMM 19.2 ms (76 % of FP32 peak) | **1.00×** |
| GEMV 8192 × 8192, batch 1 | 0.237 ms | 284 GB/s, **89 %** of DRAM peak | SGEMV 1.042 ms | **4.4×** |
| GEMV 8192 × 8192, batch 8 (BF16 x) | 0.403 ms | 166 GB/s, 52 % of DRAM peak | SGEMM 1.538 ms | **3.8×** |
| AdamW, 64 M parameters, FP32 params + state | 7.43 ms | 253 GB/s, **79 %** of DRAM peak | `torch.optim.AdamW(fused=True)` 9.25 ms | **1.24×** |
| AdamW, 64 M parameters, BF16 params + state | 3.35 ms | 281 GB/s, **88 %** of DRAM peak | `torch.optim.AdamW(fused=True)` 9.35 ms | **2.79×** |

FP32 peak: 20 SMs × 128 FFMA × SM clock (1751–1832 MHz in these rows); DRAM peak: 320 GB/s. Per clock the
FP8 GEMM matches cuBLAS SGEMM; with both operands in FP8 the kernel draws more power (up to 214 W) and
runs at a lower boost clock (1751 vs 1832 MHz), with one operand in BF16 it does not. On Pascal an
exact FP8 MAC costs one FP32 FFMA, so compute-bound GEMM tops out at SGEMM speed while reading 4× fewer
bytes; bandwidth-bound kernels (decoding, optimizer) get the full benefit of the smaller weights.
CUTLASS has not been measured.

Results by GPU: [benches/table.md](benches/table.md) (full GTX 1080 run: [benches/1080.md](benches/1080.md);
RTX 3090, community, with E4M3EffMA: [benches/rtx-3090.md](benches/rtx-3090.md));
methodology and how to add your GPU: [docs/performance.md](docs/performance.md).

## Features

- **Exact products, FP32 accumulation.** Every product of finite FP8 values (E5M2 or E4M3, any
  pairing) is exact in FP32; one FFMA per MAC in a fixed order, one rounding at the end —
  deterministic and bit-reproducible for a fixed kernel path. Why not BF16 accumulation:
  [docs/numerics.md](docs/numerics.md).
- **cuBLASLt / Transformer Engine semantics.** `D = α·op(A)·op(B) + bias + β·C`, per-tensor or
  row-wise (per output row / column) `scale_inv` for FP8 inputs, `scale` + `amax` for FP8 outputs,
  TE-style accumulation into an FP32 `main_grad`.
- **Mixed operands.** A and B can each be E5M2, E4M3, BF16 or FP32 — E5M2 gradients × E4M3 weights,
  FP8 weights × BF16 or FP32 activations.
- **Overflow-aware casts.** E5M2 Inf / NaN codes decode to Inf / NaN, FP32 → FP8 saturates as
  `cvt.rn.satfinite`, and `amax` turns Inf / NaN when the input had them — the signal loss scaling
  needs.
- **GEMV path for decoding**, picked automatically for up to 8 columns, reading each weight byte once.
- **Fused Adam / AdamW** on FP32 or BF16 state, with an optional E5M2 or E4M3 copy of the updated
  weights.
- **PyTorch bindings**: functional API, a `torch.optim` AdamW, `torch.e5m2effma`,
  `torch._scaled_mm` working on pre-Ada GPUs, and `Float8Tensor`, an FP8 tensor that carries its scale,
  so `@`, `F.linear` and `nn.Linear` run FP8 matmuls unchanged.
- **Strict input checking** with readable errors in C (`e5ma_last_error`) and Python (`TypeError` /
  `ValueError` / `NotImplementedError`).
- **Per-architecture tuning** (`make tune`) and a parallel build for all 12 architectures.
- **Heavily tested**: bit-exact against host emulation, exhaustive 2³² checks of the core arithmetic
  for every FP8 pairing and both encoders.

## Supported GPUs

| Architecture | Compute capability | Status |
|---|---|---|
| Pascal (GTX 10xx, Tesla P4/P40/P100) | sm_60, sm_61, sm_62 | ✅ tested and tuned by the author (GTX 1080, sm_61) |
| Maxwell | sm_50, sm_52, sm_53 | ⚙️ compiles, untested |
| Volta / Turing | sm_70, sm_72, sm_75 | ⚙️ compiles, untested, no Tensor Core path yet |
| Ampere / Orin | sm_80, sm_86, sm_87 | ✅ E4M3EffMA's tests pass on RTX 3090 (sm_86, community); untuned, no Tensor Core / `cp.async` path yet |
| Ada, Hopper, Blackwell | sm_89+ | ❌ excluded on purpose — native FP8, use cuBLASLt / TE / `torch._scaled_mm` |

## Installation

**Python (PyTorch ≥ 2.1, CUDA toolkit of the same major version as PyTorch's):**

```sh
git clone https://github.com/srose69/e5m2effma && cd e5m2effma
pip install ./binds/python --no-build-isolation          # builds for the GPUs visible right now
TORCH_CUDA_ARCH_LIST="6.1;7.5;8.6" pip install ./binds/python --no-build-isolation   # or chosen ones
```

**C / C++ (CUDA 12.0+, GNU make):**

```sh
make lib                                   # out/libe5m2effma.a for all of sm_50 … sm_87
make install PREFIX=$HOME/.local           # header + library + pkg-config (ARCHS="61 75" to narrow)
cc app.c $(pkg-config --cflags --libs e5m2effma)
```

Build options, the parallel build and tuning: [docs/building.md](docs/building.md).

## Usage

**PyTorch**

```python
import torch, e5m2effma as fx

x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
w = (torch.randn(4096, 4096, device="cuda") * 16).to(torch.float8_e4m3fn)   # E4M3 weight
w_scale_inv = torch.tensor([1 / 16], device="cuda")
dy = torch.randn(8, 4096, device="cuda").to(torch.e5m2effma)                 # = torch.float8_e5m2

y = fx.linear(x, w, scale_w=w_scale_inv)                     # BF16 [8, 4096], GEMV kernel
dx = fx.matmul(dy, w, scale_b=w_scale_inv)                   # E5M2 x E4M3: the HYBRID dgrad
dx = torch._scaled_mm(dy, w.t().contiguous().t(),            # PyTorch's own FP8 matmul, now on sm_61
                      torch.tensor(1.0, device="cuda"), w_scale_inv[0], out_dtype=torch.bfloat16)

mlp = torch.nn.Sequential(torch.nn.Linear(4096, 4096), torch.nn.GELU()).cuda().bfloat16()
fx.quantize_linear_(mlp)                                      # weights -> fx.Float8Tensor (E4M3, per row)
y = mlp(x)                                                    # FP8 x BF16 matmuls on E5M2EffMA

model = torch.nn.Linear(4096, 4096, device="cuda")
opt = fx.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, state_dtype=torch.bfloat16)
```

More (`matmul`, `cast`, `adam_step`, `Float8Tensor`, FP8 outputs with `amax`, error behaviour):
[binds/python/README.md](binds/python/README.md).

**C**

```c
#include "e5m2effma.h"

// dX[in, batch] (BF16) = Wᵀ (E4M3, stored [out, in]) · dY[out, batch] (E5M2), column-major as in cuBLASLt
e5ma_tensor W  = {d_w, E5MA_E4M3, d_w_scale_inv, NULL, NULL};    // data, dtype, scale_inv, scale, amax
e5ma_tensor dY = {d_dy, E5MA_E5M2, d_dy_scale_inv, NULL, NULL};
e5ma_tensor dX = {d_dx, E5MA_BF16, NULL, NULL, NULL};
e5ma_status st = e5ma_gemm(E5MA_OP_N, E5MA_OP_N, in, batch, out, 1.0f, &W, in, &dY, out,
                           0.0f, NULL, 0, &dX, in, /*bias=*/NULL, stream);
if (st != E5MA_OK) fprintf(stderr, "%s\n", e5ma_last_error());   // e.g. "k = 24 is not a multiple of 16"
```

Full API, constraints and more examples: [docs/c-api.md](docs/c-api.md).

**Together with E4M3EffMA**

E5M2EffMA handles E4M3 as well, so it is enough on its own. If a project already uses
[E4M3EffMA](https://github.com/srose69/e4m3effma) — the lighter E4M3-only build — for its forward pass,
the two work side by side: they have separate names everywhere (C prefixes `effma_` / `e5ma_`,
Python packages `e4m3effma` / `e5m2effma`), link into one binary and import into one process, and
`torch._scaled_mm` routes each call to the package that covers it:

```python
import torch
import e4m3effma as e4, e5m2effma as e5      # import order does not matter

w  = (torch.randn(4096, 4096, device="cuda") * 16).to(torch.e4m3effma)   # E4M3 weight [out, in]
sw = torch.tensor([1 / 16], device="cuda")                               # its scale_inv
x  = torch.randn(32, 4096, device="cuda").to(torch.e4m3effma)            # E4M3 activations
dy = torch.randn(32, 4096, device="cuda").to(torch.e5m2effma)            # E5M2 gradient of y

y  = e4.linear(x, w, scale_w=sw)                        # forward: E4M3 x E4M3        (E4M3EffMA)
dx = e5.matmul(dy, w, scale_b=sw)                       # dgrad:   E5M2 x E4M3        (E5M2EffMA)
dw = e5.matmul(dy.t(), x, out_dtype=torch.float32)      # wgrad:   E5M2 x E4M3 -> FP32

one = torch.tensor(1.0, device="cuda")                  # torch._scaled_mm: one override for both;
y  = torch._scaled_mm(x, w.t(), one, sw[0], out_dtype=torch.bfloat16)       # E4M3 x E4M3 -> e4m3effma
dx = torch._scaled_mm(dy, w.t().contiguous().t(), one, sw[0],               # E5M2 x E4M3 -> e5m2effma
                      out_dtype=torch.bfloat16)
```

```c
#include "e4m3effma.h"   /* effma_*, EFFMA_* */
#include "e5m2effma.h"   /* e5ma_*,  E5MA_*  */

// forward on E4M3EffMA: Y [out, batch] = W (E4M3, [out, in]) · X (E4M3, [in, batch])
effma_tensor W = {d_w, EFFMA_E4M3, d_w_sinv, NULL, NULL}, X = {d_x, EFFMA_E4M3, d_x_sinv, NULL, NULL};
effma_tensor Y = {d_y, EFFMA_BF16, NULL, NULL, NULL};
effma_gemm(EFFMA_OP_T, EFFMA_OP_N, out, batch, in, 1.0f, &W, in, &X, in, 0.0f, NULL, 0, &Y, out, NULL, stream);

// dgrad on E5M2EffMA: dX [in, batch] = Wᵀ (same E4M3 buffer) · dY (E5M2, [out, batch])
e5ma_tensor W5 = {d_w, E5MA_E4M3, d_w_sinv, NULL, NULL}, dY = {d_dy, E5MA_E5M2, d_dy_sinv, NULL, NULL};
e5ma_tensor dX = {d_dx, E5MA_BF16, NULL, NULL, NULL};
e5ma_gemm(E5MA_OP_N, E5MA_OP_N, in, batch, out, 1.0f, &W5, in, &dY, out, 0.0f, NULL, 0, &dX, in, NULL, stream);
```

```sh
cc app.c $(pkg-config --cflags --libs e4m3effma e5m2effma)    # both static libraries in one binary
```

## Documentation

| | |
|---|---|
| [docs/c-api.md](docs/c-api.md) | C API, constraints, error handling, examples |
| [binds/python/README.md](binds/python/README.md) | Python / PyTorch bindings, `torch._scaled_mm` integration, `Float8Tensor` |
| [docs/building.md](docs/building.md) | build options, parallel build, install, tuning, project layout |
| [benches/table.md](benches/table.md) | benchmark results by GPU (one file per GPU in `benches/`) |
| [docs/performance.md](docs/performance.md) | how the benchmarks are run and measured, how to add a GPU, GTX 1080 analysis |
| [docs/numerics.md](docs/numerics.md) | exactness, accuracy, why FP32 accumulation and one rounding at the end, why BF16 |
| [docs/design.md](docs/design.md) | why one FFMA per MAC, packed half-registers, kernel configurations |
| [docs/testing.md](docs/testing.md) | what the tests check |
| [NOTICE](NOTICE), [LICENSE](LICENSE), [LICENSE-EXCEPTION](LICENSE-EXCEPTION) | license: AGPL v3 + linking and attribution exception |

## Roadmap

Things that would make the library much faster on newer GPUs, none of which can be tested here:

- [ ] **Tensor Core path for sm_70+.** E5M2 is the upper byte of an FP16 — one byte permute gives two
      exact FP16 values — and E4M3 converts exactly to FP16 as well, so `mma.sync` with FP32
      accumulation preserves exact FP8 products while moving the MAC work onto Tensor Cores, with
      potentially much higher throughput than scalar FFMA.
- [ ] **`cp.async` + multi-stage pipelines for sm_80+**, and `ldmatrix` for operand fetch.
- [ ] **Packed two-lane MACs on sm_70+** (separate INT32 pipe, fast HFMA2 / BF16x2) — not worth it on
      Pascal, possibly worth it there ([docs/design.md](docs/design.md)).
- [ ] **Per-architecture tuning entries** for Maxwell, Volta, Turing and Ampere GPUs.

## Contributing

Issues and pull requests are welcome, especially from anyone who can run the code on something newer
than a GTX 1080:

1. Run `make test` (and `pytest binds/python/tests`) on your GPU and report the result with the GPU model.
2. Run `make tune` and open a pull request with your entry in `src/e5m2effma_tuned.h`.
3. Run `bench/run_bench.sh` and open a pull request with the `benches/<gpu>.md` file and the
   `benches/table.md` row it writes (see [docs/performance.md](docs/performance.md)).
4. Tensor Core / `cp.async` kernels for sm_70+ / sm_80+ are the most valuable contribution. They must
   pass `make test` bit-exactly, or document why a different accumulation order is needed.

If you would rather donate or lend a newer GPU (Turing, Ampere or Jetson Orin), please open an issue —
that is the fastest way for those kernels to get written and tested.

Contributions are accepted under the project's license (below).

## License

E5M2EffMA is licensed under the [GNU AGPL v3](LICENSE) with the
[Linking and Attribution Exception](LICENSE-EXCEPTION); details and the exact attribution text are in
[NOTICE](NOTICE).

- **Use it in anything.** Programs and services that use E5M2EffMA — link it, call it, load its Python
  bindings — can be under any license, including proprietary ones.
- **Changes to E5M2EffMA itself stay open**: modified versions remain under the AGPL with the exception,
  with their source available, and must be marked as modified.
- **Credit is required**: every work that uses E5M2EffMA shows the attribution from NOTICE (for example
  in its third-party notices, credits or README).
- **A request, not a condition**: if you train or fine-tune a model with E5M2EffMA, please mention it in
  the model card or paper. [CITATION.cff](CITATION.cff) has the citation.
