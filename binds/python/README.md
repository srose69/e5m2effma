# e5m2effma (PyTorch bindings)

FP8 (E5M2 and E4M3) matmul / linear, casts and a fused AdamW for NVIDIA GPUs **without** native FP8
(sm_50 … sm_87: Maxwell, Pascal, Volta, Turing, Ampere, Orin), on ordinary `torch` tensors.
FP8 tensors use `torch.float8_e5m2` and `torch.float8_e4m3fn`, in any pairing (e.g. E5M2 gradients ×
E4M3 weights, the backward of Transformer Engine's HYBRID format).

## Install

Requires PyTorch ≥ 2.1 with CUDA, a CUDA toolkit (`nvcc`) of the same major version, and GNU make.

```sh
git clone https://github.com/srose69/e5m2effma && cd e5m2effma
pip install ./binds/python --no-build-isolation
```

The library is compiled for the GPUs visible at build time (a few seconds). To build for other GPUs,
for example a wheel for several machines:

```sh
TORCH_CUDA_ARCH_LIST="6.1;7.5;8.6" pip install ./binds/python --no-build-isolation
E5M2EFFMA_ARCHS="50 52 53 60 61 62 70 72 75 80 86 87" pip wheel ./binds/python --no-build-isolation
```

Other build variables: `CUDA_HOME`, `E5M2EFFMA_CCBIN` (host compiler for nvcc, e.g. `g++-12` with
CUDA 12.0–12.3), `MAX_JOBS`.

## Use

```python
import torch, e5m2effma as fx

w = (torch.randn(4096, 4096, device="cuda") * 16).to(torch.float8_e4m3fn)   # E4M3 weight
w_scale_inv = torch.tensor([1 / 16], device="cuda")                          # dequantization factor
x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)

y = fx.linear(x, w, scale_w=w_scale_inv)                  # [8, 4096] bfloat16; <= 8 rows: GEMV kernel
y = fx.matmul(x, w.t(), scale_b=w_scale_inv, out_dtype=torch.float32)
y = fx.linear(x, w, scale_w=torch.rand(4096, device="cuda"))  # one factor per output feature

grad = torch.randn(8, 4096, device="cuda")                # e.g. dL/dy of this layer
amax = torch.zeros(1, device="cuda")                      # becomes Inf / NaN if the gradient overflowed
g8 = fx.cast(grad, torch.e5m2effma, scale=torch.tensor([4.0], device="cuda"), amax=amax)  # E5M2 gradient
dx = fx.matmul(g8, w, scale_a=torch.tensor([0.25], device="cuda"), scale_b=w_scale_inv)  # E5M2 x E4M3

model = torch.nn.Linear(1024, 1024).cuda()
opt = fx.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, state_dtype=torch.bfloat16)
```

| function | |
|---|---|
| `matmul(a, b, *, out_dtype, bias, out, alpha, beta, scale_a, scale_b, scale_out, amax_out)` | `alpha·(sa·sb)·(a @ b) + bias + beta·out`; a, b ∈ {E5M2, E4M3, BF16, FP32}; out ∈ {BF16 (default), FP16, FP32, E5M2, E4M3}; `beta ≠ 0` accumulates into `out` |
| `linear(x, weight, bias=None, *, out_dtype, scale_x, scale_w, scale_out, amax_out)` | `x @ weight.t() + bias`, `x` of shape `[..., K]` |
| `Float8Tensor.from_float(x, *, fp8=float8_e5m2, per_row=False, dtype=None, margin=1.0)` | FP8 tensor that carries its scale ([below](#float8tensor)) |
| `quantize_linear_(module, *, fp8=float8_e4m3fn, per_row=True, dtype=None)` | turns the weights of every `nn.Linear` in `module` into `Float8Tensor`s |
| `cast(x, dtype, *, scale_inv, scale, amax)` | E5M2 / E4M3 / BF16 / FP16 / FP32 conversion, RNE, FP8 saturating; FP8 Inf / NaN codes decode exactly |
| `adam_step(param, grad, exp_avg, exp_avg_sq, *, lr, betas, eps, weight_decay, step, adamw, ..., param_fp8)` | one fused step, optional E5M2 / E4M3 copy of the weights |
| `AdamW(params, lr, betas, eps, weight_decay, *, adamw=True, state_dtype=None)` | `torch.optim` optimizer on `adam_step` |

Scales are float32 CUDA tensors: one element (per tensor), or row-wise along the outer dimension of
an FP8 operand — `scale_a` with M elements (one per row of `a`), `scale_b` / `scale_w` with N (one per
output column / feature); the product is then `(alpha·sa[i])·sb[j]·(a @ b)[i, j]`. Scales along K
cannot be applied after an FP32-accumulated product and are rejected. `scale_x` of `linear` is per
tensor or one per row of `x`.

`matmul` / `linear` need K to be a multiple of 16 (pad with zeros otherwise). The kernels read
K-contiguous operands (as in `linear`) directly; other layouts, strided views and unaligned slices are
copied once into that layout. Wrong dtypes, devices, shapes, scales or an output that overlaps an
input raise `TypeError` / `ValueError` before anything runs; GPUs with native FP8 (sm_89+) raise
`NotImplementedError`; library or CUDA failures raise `RuntimeError` with the library's description.
FP8 matmul operands must be finite — no E5M2 Inf / NaN codes, no E4M3 NaN code (not checked: that
would cost a pass over the data); `cast` reads them exactly, and its `amax` reports them.

## PyTorch integration

`import e5m2effma` also

* adds **`torch.e5m2effma`**, an alias of `torch.float8_e5m2`. It is the same dtype on purpose: the
  bit encoding is identical, and PyTorch has no way to register a new dtype from outside its C++
  core, so the tensors stay interoperable with everything that understands `float8_e5m2`;
* makes **`torch._scaled_mm`** (PyTorch's FP8 matmul, natively sm_89+ only) run on this library when
  every visible GPU is sm_50 … sm_87, so existing FP8 code works unchanged on those GPUs:

```python
import torch, e5m2effma

a = torch.randn(64, 128, device="cuda").to(torch.e5m2effma)              # E5M2
b = torch.randn(32, 128, device="cuda").to(torch.float8_e4m3fn).t()      # E4M3
y = torch._scaled_mm(a, b, torch.tensor(0.5, device="cuda"), torch.tensor(2.0, device="cuda"),
                     out_dtype=torch.bfloat16)                     # on a GTX 1080
```

E5M2 and E4M3 operands in any pairing (including E5M2 × E5M2, which PyTorch's own kernel rejects);
per-tensor scales or row-wise ones (`scale_a` of shape `[M, 1]`, `scale_b` of shape `[1, N]`); other
scalings (block-wise, along K) and non-FP8 operands raise `NotImplementedError`. `use_fast_accum` is
ignored because accumulation is always full FP32. With a GPU that has native FP8 in the machine,
PyTorch's own kernel is left in place; `E5M2EFFMA_NO_TORCH_OPS=1` disables the override altogether,
and `e5m2effma.torch_ops_enabled()` tells whether it is active. Plain `a @ b` and element-wise
arithmetic on raw FP8 tensors are not implemented by PyTorch on any GPU; use `e5m2effma.matmul` /
`linear` / `cast`, or `Float8Tensor`.

The override is shared with the sibling package [`e4m3effma`](https://github.com/srose69/e4m3effma):
whichever is imported first registers one `aten::_scaled_mm` kernel, and each package adds a handler
to it; a call goes to the first handler that covers it. Both packages can be imported in any order.

## Float8Tensor

`e5m2effma.Float8Tensor` is an FP8 tensor that carries its own scale, like `Float8Tensor` in
Transformer Engine: a `__torch_dispatch__` wrapper subclass that holds `float8_e5m2` or
`float8_e4m3fn` data and its
dequantization factor `scale_inv` (one per tensor, or one per row or column of a 2-D tensor), and
presents itself as a tensor of a high-precision dtype (BF16 by default) — so ordinary PyTorch code
runs FP8 matmuls without changes:

```python
import torch, torch.nn.functional as F, e5m2effma as fx

w8 = fx.Float8Tensor.from_float(w_bf16, fp8=torch.float8_e4m3fn, per_row=True)  # 448 / amax per row
y = F.linear(x, w8, bias)                               # = fx.linear(x, w8._data, scale_w=w8._scale_inv, bias=bias)
y = x @ w8.t()                                          # same kernel; .t() is a view, the scales follow it

model = fx.quantize_linear_(model)                      # every nn.Linear weight -> Float8Tensor (E4M3)
```

* `mm` / `addmm` — what `@`, `torch.matmul`, `F.linear` and `nn.Linear` reduce to — run on this
  library whenever the scales are outer (per tensor, per row of the left operand, per column of the
  right one) and K is a multiple of 16. Both operands may be `Float8Tensor`s, or one of them a BF16 /
  FP32 tensor. The result is a plain tensor of the promoted nominal dtype.
* `t` / `transpose`, `detach`, `clone` keep the FP8 data (transposes are views).
* `.float()`, `.to(dtype)` and `dequantize()` dequantize; every other operation (and a matmul with
  scales along K) dequantizes its FP8 arguments and runs in the nominal dtype.
* Inference only: no autograd. `quantize_linear_` keeps layers whose `in_features` is not a multiple
  of 16 in high precision.

## Test

```sh
pip install pytest && pytest binds/python/tests
```

## License

GNU AGPL v3 with the Linking and Attribution Exception (see the repository's `LICENSE`,
`LICENSE-EXCEPTION` and `NOTICE`): programs using this package may be under any license, but must
show the attribution given in `NOTICE`; changes to the package itself stay under the AGPL.
