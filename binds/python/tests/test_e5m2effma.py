# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. E5M2EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""Tests of the PyTorch bindings: results against a float64 reference, and rejection of bad input.

Bit-exactness of the kernels themselves is covered by the C tests (make test); here the bound is the
output rounding plus the FP32 accumulation error.
"""
import pytest
import torch

import e5m2effma as fx

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
DEV = "cuda"
F8 = torch.float8_e5m2
F4 = torch.float8_e4m3fn
FP8_MAX = {F8: 57344.0, F4: 448.0}


def rand(shape, dtype, scale=1.0, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(shape, generator=g) * scale
    return x.to(DEV).to(dtype)


def check_close(y, ref, bound_scale, k, out_dtype):
    """|y - ref| <= half an ulp of the output type + FP32 accumulation error (k * 2^-23 * sum |a b|)."""
    eps = {torch.bfloat16: 2.0 ** -8, torch.float16: 2.0 ** -11, torch.float32: 2.0 ** -24, F8: 2.0 ** -3, F4: 2.0 ** -4}[out_dtype]
    err = (y.double() - ref).abs()
    bound = eps * ref.abs() + k * 2.0 ** -23 * bound_scale + 1e-30
    assert bool((err <= bound * 1.0001).all()), f"max excess {float((err - bound).max())}"


def ref_mm(a, b, sa=1.0, sb=1.0, bias=None):
    r = (a.double() * sa) @ (b.double() * sb)
    mag = (a.double().abs() * abs(sa)) @ (b.double().abs() * abs(sb))
    if bias is not None:
        r = r + bias.double()
    return r, mag


# ---- results ------------------------------------------------------------------------------------
@pytest.mark.parametrize("ta", [F8, F4, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tb", [F8, F4, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("b_layout", ["row", "col"])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_matmul(ta, tb, b_layout, out_dtype):
    M, K, N = 70, 96, 50
    a = rand((M, K), ta, seed=1)
    b = rand((K, N), tb, seed=2) if b_layout == "row" else rand((N, K), tb, seed=2).t()
    y = fx.matmul(a, b, out_dtype=out_dtype)
    assert y.shape == (M, N) and y.dtype == out_dtype
    ref, mag = ref_mm(a, b)
    check_close(y, ref, mag, K, out_dtype)


@pytest.mark.parametrize("rows", [1, 3, 8, 9, 64])
@pytest.mark.parametrize("tx", [F8, F4, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tw", [F8, F4])
def test_linear_fp8_weight(rows, tx, tw):  # rows <= 8: GEMV kernel
    N, K = 300, 512
    w = rand((N, K), tw, seed=3)
    x = rand((rows, K), tx, seed=4)
    bias = rand((N,), torch.bfloat16, seed=5)
    sw = torch.tensor([0.25], device=DEV)
    y = fx.linear(x, w, bias, scale_w=sw)
    ref, mag = ref_mm(x, w.t(), sb=0.25, bias=bias)
    check_close(y, ref, mag, K, torch.bfloat16)


def test_linear_batched_shape():
    x = rand((2, 3, 64), torch.bfloat16)
    w = rand((40, 64), F8, seed=1)
    assert fx.linear(x, w).shape == (2, 3, 40)


def test_accumulate_fp32_out():
    M, K, N = 33, 64, 47
    a, b = rand((M, K), torch.bfloat16, seed=1), rand((K, N), torch.bfloat16, seed=2)
    out = rand((M, N), torch.float32, seed=3)
    start = out.clone()
    fx.matmul(a, b, out=out, beta=1.0)
    ref, mag = ref_mm(a, b)
    check_close(out, ref + start.double(), mag + start.double().abs(), K, torch.float32)


@pytest.mark.parametrize("out", [F8, F4])
def test_fp8_output_scale_amax(out):
    M, K, N = 16, 64, 32
    a, b = rand((M, K), F8, seed=1), rand((K, N), F4, seed=2)
    scale, amax = torch.tensor([4.0], device=DEV), torch.zeros(1, device=DEV)
    y = fx.matmul(a, b, out_dtype=out, scale_out=scale, amax_out=amax)
    ref, _ = ref_mm(a, b)
    assert y.dtype == out
    assert float(amax) == pytest.approx(float(ref.abs().max()), rel=2 ** -20)
    top = FP8_MAX[out]
    expect = (ref * 4.0).clamp(-top, top).float().to(out)  # torch's own RNE conversion, in range
    assert torch.equal(y.view(torch.uint8), expect.view(torch.uint8))


@pytest.mark.parametrize("dtype", [F8, F4, torch.bfloat16, torch.float16])
def test_cast_matches_torch(dtype):
    x = rand((1000, 37), torch.float32, scale=50.0).clamp(-448, 448)
    y = fx.cast(x, dtype)
    assert torch.equal(y.view(torch.uint8), x.to(dtype).view(torch.uint8))


@pytest.mark.parametrize("dtype", [F8, F4])
def test_cast_scale_roundtrip(dtype):
    x = rand((4096,), torch.float32)
    amax, scale = torch.zeros(1, device=DEV), torch.tensor([64.0], device=DEV)
    q = fx.cast(x, dtype, scale=scale, amax=amax)
    assert float(amax) == float(x.abs().max())
    back = fx.cast(q, torch.float32, scale_inv=torch.tensor([1 / 64.0], device=DEV))
    assert torch.equal(back, (x * 64).to(dtype).float() / 64)


def test_cast_e5m2_inf_nan():
    codes = torch.arange(256, device=DEV, dtype=torch.uint8).view(F8)
    y = fx.cast(codes, torch.float32)
    assert torch.equal(y.isinf(), codes.float().isinf()) and torch.equal(y.isnan(), codes.float().isnan())
    x = torch.tensor([1.0, float("inf"), -1e6, 3.0], device=DEV)
    amax = torch.zeros(1, device=DEV)
    q = fx.cast(x, F8, amax=amax)  # saturating, as cvt.rn.satfinite; the overflow shows in amax
    assert q.view(torch.uint8).tolist() == [0x3C, 0x7B, 0xFB, 0x42] and float(amax) == float("inf")


@pytest.mark.parametrize("pdtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_adamw_against_torch(pdtype, state_dtype):
    torch.manual_seed(0)
    ref = torch.nn.Parameter(torch.randn(1000, device=DEV))
    mine = torch.nn.Parameter(ref.detach().clone().to(pdtype))
    opt_ref = torch.optim.AdamW([ref], lr=1e-2, weight_decay=0.1, foreach=False)
    opt = fx.AdamW([mine], lr=1e-2, weight_decay=0.1, state_dtype=state_dtype)
    for _ in range(5):
        g = torch.randn(1000, device=DEV)
        ref.grad, mine.grad = g.clone(), g.to(pdtype)
        opt_ref.step()
        opt.step()
    tol = 1e-6 if pdtype == torch.float32 and state_dtype == torch.float32 else 2e-2
    torch.testing.assert_close(mine.detach().float(), ref.detach(), rtol=tol, atol=tol)


@pytest.mark.parametrize("qt", [F8, F4])
def test_adam_step_fp8_copy(qt):
    p = rand((256,), torch.float32)
    g, m, v = rand((256,), torch.bfloat16, seed=1), torch.zeros(256, device=DEV), torch.zeros(256, device=DEV)
    q, amax = torch.empty(256, device=DEV, dtype=qt), torch.zeros(1, device=DEV)
    fx.adam_step(p, g, m, v, lr=1e-3, step=1, param_fp8=q, fp8_scale=torch.tensor([8.0], device=DEV), fp8_amax=amax)
    assert torch.equal(q.view(torch.uint8), (p * 8).to(qt).view(torch.uint8))
    assert float(amax) == float(p.abs().max())


@pytest.mark.parametrize("which", ["a", "b", "both"])
def test_matmul_rowwise_scales(which):
    M, K, N = 70, 96, 50
    a, b = rand((M, K), F8, seed=1), rand((N, K), F4, seed=2).t()  # E5M2 x E4M3
    sa = torch.rand(M, device=DEV) + 0.5 if which in ("a", "both") else torch.tensor([0.75], device=DEV)
    sb = torch.rand(N, device=DEV) + 0.5 if which in ("b", "both") else torch.tensor([1.5], device=DEV)
    for out_dtype in (torch.bfloat16, torch.float32):
        y = fx.matmul(a, b, scale_a=sa, scale_b=sb, out_dtype=out_dtype)
        sa2, sb2 = sa.double().reshape(-1, 1) if sa.numel() > 1 else sa.double(), sb.double().reshape(1, -1) if sb.numel() > 1 else sb.double()
        ref = (a.double() * sa2) @ (b.double() * sb2)
        mag = (a.double().abs() * sa2) @ (b.double().abs() * sb2)
        check_close(y, ref, mag, K, out_dtype)


@pytest.mark.parametrize("rows", [1, 8, 64])
def test_linear_per_channel_weight_scale(rows):  # rows <= 8: GEMV kernel
    N, K = 300, 512
    w, x = rand((N, K), F8, seed=3), rand((rows, K), torch.bfloat16, seed=4)
    sw = torch.rand(N, device=DEV) + 0.5
    y = fx.linear(x, w, scale_w=sw)
    ref = x.double() @ (w.double() * sw.double().reshape(-1, 1)).t()
    mag = x.double().abs() @ (w.double().abs() * sw.double().reshape(-1, 1)).t()
    check_close(y, ref, mag, K, torch.bfloat16)


def test_rejects_bad_rowwise_scale():
    a, b = rand((8, 16), F8), rand((16, 6), F8)
    with pytest.raises(ValueError, match="one element"):
        fx.matmul(a, b, scale_a=torch.ones(6, device=DEV))  # length of b's N, not a's M


# ---- Float8Tensor ------------------------------------------------------------------------------
@pytest.mark.parametrize("fp8", [F8, F4])
@pytest.mark.parametrize("per_row", [False, True])
@pytest.mark.parametrize("rows", [1, 8, 64])
def test_float8tensor_linear(fp8, per_row, rows):  # rows <= 8: GEMV kernel
    import torch.nn.functional as Fn
    w, x = rand((300, 512), torch.bfloat16, seed=3), rand((rows, 512), torch.bfloat16, seed=4)
    w8 = fx.Float8Tensor.from_float(w, fp8=fp8, per_row=per_row)
    assert isinstance(w8, fx.Float8Tensor) and w8.dtype == torch.bfloat16 and w8.shape == w.shape
    assert w8._data.dtype == fp8
    bias = rand((300,), torch.bfloat16, seed=5)
    y = Fn.linear(x, w8, bias)
    assert y.dtype == torch.bfloat16 and not isinstance(y, fx.Float8Tensor)
    s = w8._scale_inv if per_row else w8._scale_inv.reshape(())
    assert torch.equal(y, fx.linear(x, w8._data, scale_w=s, bias=bias))
    assert torch.equal(x @ w8.t() + 0, fx.linear(x, w8._data, scale_w=s))
    wd = w8.dequantize(torch.float64)
    check_close(y, x.double() @ wd.t() + bias.double(), x.double().abs() @ wd.abs().t(), 512, torch.bfloat16)


def test_float8tensor_both_operands_and_views():
    a8 = fx.Float8Tensor.from_float(rand((70, 96), torch.float32, seed=1), per_row=True, dtype=torch.float32)
    b8 = fx.Float8Tensor.from_float(rand((50, 96), torch.float32, seed=2), per_row=True, dtype=torch.float32)
    y = a8 @ b8.t()  # per-row scales of both operands are outer: one FP8 x FP8 GEMM
    assert y.dtype == torch.float32
    assert torch.equal(y, fx.matmul(a8._data, b8._data.t(), scale_a=a8._scale_inv, scale_b=b8._scale_inv, out_dtype=torch.float32))
    assert b8.t()._data.data_ptr() == b8._data.data_ptr() and b8.t()._scale_dim == 1
    ad, bd = a8.dequantize(torch.float64), b8.dequantize(torch.float64)
    check_close(y, ad @ bd.t(), ad.abs() @ bd.abs().t(), 96, torch.float32)
    # scales along K cannot be applied after the product: falls back to dequantization, same values
    z = b8 @ a8.t().t().t()
    assert torch.allclose(z, b8.float() @ a8.float().t(), rtol=1e-5, atol=1e-4)
    # other ops and conversions dequantize
    assert torch.equal(a8.float(), a8.dequantize(torch.float32)) and a8.to(torch.bfloat16).dtype == torch.bfloat16
    assert torch.equal(a8 * 2, a8.dequantize() * 2) and type(a8 * 2) is torch.Tensor
    assert isinstance(a8.detach(), fx.Float8Tensor) and isinstance(a8.clone(), fx.Float8Tensor)


def test_quantize_linear_module():
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(512, 256), torch.nn.GELU(), torch.nn.Linear(256, 64), torch.nn.Linear(64, 8, bias=False),
                            torch.nn.Linear(8, 8)).to(DEV, torch.bfloat16)  # the last one: K = 8, kept in BF16
    x = rand((4, 7, 512), torch.bfloat16, seed=6)
    with torch.no_grad():
        ref = m(x).float()
        fx.quantize_linear_(m)
        y = m(x).float()
    assert [isinstance(l.weight, fx.Float8Tensor) for l in m if isinstance(l, torch.nn.Linear)] == [True, True, True, False]
    assert float((y - ref).norm() / ref.norm()) < 0.1


def test_float8tensor_rejects_bad_input():
    d = torch.zeros(4, 16, device=DEV, dtype=F8)
    with pytest.raises(TypeError, match="float8_e5m2"):
        fx.Float8Tensor(d.float(), torch.ones(1, device=DEV))
    with pytest.raises(ValueError, match="scale_dim"):
        fx.Float8Tensor(d, torch.ones(16, device=DEV), 0)
    with pytest.raises(ValueError, match="per_row"):
        fx.Float8Tensor.from_float(torch.ones(2, 3, 16, device=DEV), per_row=True)


# ---- rejection of bad input -------------------------------------------------------------------
def test_rejects_cpu_tensor():
    with pytest.raises(ValueError, match="CUDA"):
        fx.matmul(torch.zeros(16, 16), torch.zeros(16, 16))


def test_rejects_bad_dtype():
    with pytest.raises(TypeError, match="dtype|must be"):
        fx.matmul(torch.zeros(16, 16, device=DEV, dtype=torch.float64), rand((16, 16), F8))
    with pytest.raises(TypeError):
        fx.matmul(rand((16, 16), torch.float16), rand((16, 16), F8))  # FP16 operands are not supported


def test_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shapes"):
        fx.matmul(rand((8, 32), F8), rand((16, 8), F8))
    with pytest.raises(ValueError, match="in_features|shape"):
        fx.linear(rand((4, 32), torch.bfloat16), rand((8, 48), F8))


def test_rejects_k_not_multiple_of_16():
    with pytest.raises(ValueError, match="multiple of 16"):
        fx.matmul(rand((8, 24), F8), rand((24, 8), F8))


@pytest.mark.parametrize("view", ["offset", "strided", "row_major_n_not_16"])
def test_awkward_layouts_are_repacked(view):
    base = rand((40, 130), F8, seed=7)
    a = {"offset": base[:, 1:129], "strided": base[:, ::2][:, :64], "row_major_n_not_16": base[:, :128]}[view]
    b = rand((a.shape[1], 50), torch.bfloat16, seed=8)  # row-major, ld = 50
    y = fx.matmul(a, b)
    assert torch.equal(y, fx.matmul(a.contiguous(), b.t().contiguous().t()))
    ref, mag = ref_mm(a, b)
    check_close(y, ref, mag, a.shape[1], torch.bfloat16)


def test_rejects_misused_scales():
    with pytest.raises(ValueError, match="scale_a"):
        fx.matmul(rand((8, 16), torch.bfloat16), rand((16, 8), F8), scale_a=torch.ones(1, device=DEV))
    with pytest.raises(TypeError, match="float32"):
        fx.matmul(rand((8, 16), F8), rand((16, 8), F8), scale_a=torch.ones(1, device=DEV, dtype=torch.float64))
    with pytest.raises(ValueError, match="one element"):
        fx.matmul(rand((8, 16), F8), rand((16, 8), F8), scale_a=torch.ones(2, device=DEV))


def test_rejects_out_problems():
    a, b = rand((8, 16), F8), rand((16, 8), F8)
    with pytest.raises(ValueError, match="out must be given"):
        fx.matmul(a, b, beta=1.0)
    with pytest.raises(ValueError, match="shape"):
        fx.matmul(a, b, out=torch.empty(8, 9, device=DEV, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="row-contiguous"):
        fx.matmul(a, b, out=torch.empty(8, 8, device=DEV, dtype=torch.bfloat16).t())
    x = rand((16, 16), torch.float32)
    with pytest.raises(ValueError, match="share memory"):
        fx.matmul(x, rand((16, 16), torch.float32, seed=1), out=x)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_rejects_mixed_devices():
    with pytest.raises(ValueError, match="cuda:1"):
        fx.matmul(rand((8, 16), F8), rand((16, 8), F8).to("cuda:1"))


def test_runs_on_second_gpu():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs")
    a, b = rand((8, 16), F8).to("cuda:1"), rand((16, 8), F8).to("cuda:1")
    y = fx.matmul(a, b)
    assert y.device == a.device
    ref, mag = ref_mm(a, b)
    check_close(y, ref, mag, 16, torch.bfloat16)


def test_rejects_adam_problems():
    p, g = rand((10,), torch.float32), rand((10,), torch.float32)
    m, v = torch.zeros(10, device=DEV), torch.zeros(10, device=DEV)
    with pytest.raises(ValueError, match="step"):
        fx.adam_step(p, g, m, v, lr=1e-3, step=0)
    with pytest.raises(ValueError, match="elements"):
        fx.adam_step(p, g[:5], m, v, lr=1e-3, step=1)
    with pytest.raises(TypeError, match="exp_avg_sq"):
        fx.adam_step(p, g, m, v.bfloat16(), lr=1e-3, step=1)
    with pytest.raises(ValueError, match="share memory"):
        fx.adam_step(p, g, m, m, lr=1e-3, step=1)
    with pytest.raises(ValueError, match="betas"):
        fx.adam_step(p, g, m, v, lr=1e-3, betas=(0.9, 1.0), step=1)
    with pytest.raises(ValueError):
        fx.AdamW([torch.nn.Parameter(p)], lr=-1.0)


# ---- torch integration ----------------------------------------------------------------------------
def test_dtype_alias():
    assert torch.e5m2effma is torch.float8_e5m2 and fx.e5m2 is torch.float8_e5m2 and fx.e4m3 is torch.float8_e4m3fn


needs_torch_ops = pytest.mark.skipif(not fx.torch_ops_enabled(), reason="a GPU with native FP8 is visible")


@needs_torch_ops
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_torch_scaled_mm(out_dtype):
    a = rand((64, 128), torch.e5m2effma, seed=1)
    b = rand((40, 128), torch.e5m2effma, seed=2).t()  # column-major mat2, as PyTorch requires
    sa, sb = torch.tensor(0.5, device=DEV), torch.tensor(2.0, device=DEV)
    bias = rand((40,), out_dtype, seed=3) if out_dtype != torch.float32 else None
    y = torch._scaled_mm(a, b, sa, sb, bias=bias, out_dtype=out_dtype)
    assert torch.equal(y, fx.matmul(a, b, bias=bias, out_dtype=out_dtype, scale_a=sa.reshape(1), scale_b=sb.reshape(1)))


@needs_torch_ops
def test_torch_scaled_mm_fp8_out():
    a, b = rand((32, 64), torch.e5m2effma, seed=1), rand((48, 64), torch.e5m2effma, seed=2).t()
    one, sr = torch.tensor(1.0, device=DEV), torch.tensor(0.25, device=DEV)
    y32 = torch._scaled_mm(a, b, one, one, out_dtype=torch.float32)
    y8 = torch._scaled_mm(a, b, one, one, scale_result=sr)
    assert y8.dtype == torch.e5m2effma
    assert torch.equal(y8.view(torch.uint8), (y32 * 0.25).clamp(-57344, 57344).to(torch.e5m2effma).view(torch.uint8))


@needs_torch_ops
def test_torch_scaled_mm_rowwise():
    a, b = rand((48, 64), F4, seed=1), rand((40, 64), F8, seed=2).t()  # E4M3 x E5M2
    sa = torch.rand(48, 1, device=DEV) + 0.5
    sb = torch.rand(1, 40, device=DEV) + 0.5
    y = torch._scaled_mm(a, b, sa, sb, out_dtype=torch.bfloat16)
    assert torch.equal(y, fx.matmul(a, b, scale_a=sa.reshape(-1), scale_b=sb.reshape(-1)))
    ref = (a.double() * sa.double()) @ (b.double() * sb.double())
    mag = (a.double().abs() * sa.double()) @ (b.double().abs() * sb.double())
    check_close(y, ref, mag, 64, torch.bfloat16)


@needs_torch_ops
def test_torch_scaled_mm_rejects_unsupported():
    a, b = rand((32, 64), F8, seed=1), rand((48, 64), F8, seed=2).t()
    with pytest.raises(NotImplementedError, match="row-wise"):  # scaling along K
        torch._scaled_mm(a, b, torch.ones(1, 64, device=DEV), torch.ones(64, 1, device=DEV), out_dtype=torch.bfloat16)
    one = torch.tensor(1.0, device=DEV)
    with pytest.raises(NotImplementedError, match="float8_e5m2"):
        torch._scaled_mm(a.to(torch.bfloat16), b, one, one, out_dtype=torch.bfloat16)


@needs_torch_ops
def test_torch_scaled_mm_shared_dispatch():
    """One aten override for every *EffMA package: a handler added by another package (here a stand-in
    that only takes float16 operands) is tried after this one, and this one stays in place."""
    shared = torch._effma_scaled_mm
    assert fx._scaled_mm in shared["handlers"]

    def other(self, mat2, *args, **kwargs):
        if self.dtype != torch.float16:
            raise NotImplementedError("other: float16 only")
        return torch.full((self.shape[0], mat2.shape[1]), 7.0, device=self.device)

    shared["handlers"].append(other)
    try:
        one = torch.tensor(1.0, device=DEV)
        h = torch.zeros(16, 16, device=DEV, dtype=torch.float16)
        assert float(torch._scaled_mm(h, h.t(), one, one)[0, 0]) == 7.0
        a, b = rand((16, 32), F8, seed=1), rand((16, 32), F4, seed=2).t()
        assert torch.equal(torch._scaled_mm(a, b, one, one, out_dtype=torch.float32),
                           fx.matmul(a, b, out_dtype=torch.float32))
        with pytest.raises(NotImplementedError, match="float16 only"):  # nobody takes it: the last error
            torch._scaled_mm(h.bfloat16(), h.bfloat16().t(), one, one)
    finally:
        shared["handlers"].remove(other)


def test_torch_ops_can_be_disabled():
    import subprocess, sys
    code = ("import torch, e5m2effma as fx; assert not fx.torch_ops_enabled(); assert torch.e5m2effma is torch.float8_e5m2")
    env = dict(__import__("os").environ, E5M2EFFMA_NO_TORCH_OPS="1")
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
