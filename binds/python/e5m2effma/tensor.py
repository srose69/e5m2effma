# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. E5M2EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""``Float8Tensor``: FP8 data that carries its own scale, so ordinary PyTorch code runs FP8 matmuls.

A wrapper tensor subclass in the spirit of Transformer Engine's ``Float8Tensor``: it reports a nominal
high-precision dtype (BF16 by default) and shape, holds ``torch.float8_e5m2`` or ``torch.float8_e4m3fn``
data plus its dequantization factor ``scale_inv`` (value = code * scale_inv), and intercepts operations:

* ``mm`` / ``addmm`` -- which ``@``, ``torch.matmul``, ``F.linear`` and ``nn.Linear`` reduce to -- run
  on E5M2EffMA, with per-tensor or per-row / per-column (outer) scales;
* ``t`` / ``transpose`` / ``detach`` / ``clone`` keep the FP8 data (no copy, no dequantization);
* ``.to(dtype)`` / ``.float()`` dequantize; every other operation dequantizes its FP8 arguments and
  runs in the nominal dtype.

Inference only: there is no autograd support (keep ``requires_grad=False``).
"""
from typing import Optional

import torch
from torch.utils._pytree import tree_map

FP8_MAX = {torch.float8_e5m2: 57344.0, torch.float8_e4m3fn: 448.0}  # largest finite value
aten = torch.ops.aten


class Float8Tensor(torch.Tensor):
    """FP8 data ``_data`` (E5M2 or E4M3) with dequantization factor ``_scale_inv``: one float32 for the
    tensor, or one per index along ``_scale_dim`` of a 2-D tensor (0: per row, 1: per column)."""

    @staticmethod
    def __new__(cls, data: torch.Tensor, scale_inv: torch.Tensor, scale_dim: Optional[int] = None,
                dtype: torch.dtype = torch.bfloat16):
        if data.dtype not in FP8_MAX:
            raise TypeError(f"Float8Tensor data must be torch.float8_e5m2 or torch.float8_e4m3fn, got {data.dtype}")
        if scale_inv.dtype != torch.float32 or scale_inv.device != data.device:
            raise TypeError("scale_inv must be a float32 tensor on the device of the data")
        if scale_dim is None:
            if scale_inv.numel() != 1:
                raise ValueError(f"a per-tensor scale_inv has one element, got shape {tuple(scale_inv.shape)}")
        elif data.dim() != 2 or scale_dim not in (0, 1) or scale_inv.numel() != data.shape[scale_dim]:
            raise ValueError(f"scale_dim={scale_dim} needs 2-D data and {data.shape[scale_dim] if data.dim() == 2 and scale_dim in (0, 1) else '?'} "
                             f"factors, got data {tuple(data.shape)} and scale_inv {tuple(scale_inv.shape)}")
        if dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError(f"nominal dtype must be bfloat16, float16 or float32, got {dtype}")
        self = torch.Tensor._make_wrapper_subclass(cls, data.shape, strides=data.stride(), storage_offset=0,
                                                   dtype=dtype, device=data.device, requires_grad=False)
        self._data, self._scale_inv, self._scale_dim = data, scale_inv.reshape(-1) if scale_dim is not None else scale_inv.reshape(1), scale_dim
        return self

    # ---- construction / conversion -------------------------------------------------------------
    @classmethod
    def from_float(cls, x: torch.Tensor, *, fp8: torch.dtype = torch.float8_e5m2, per_row: bool = False,
                   dtype: Optional[torch.dtype] = None, margin: float = 1.0) -> "Float8Tensor":
        """Quantize ``x`` to ``fp8`` (``float8_e5m2`` or ``float8_e4m3fn``) with scale = max / amax, max
        being 57344 (E5M2) or 448 (E4M3), per tensor or per row of a 2-D ``x``; ``margin`` < 1 leaves
        headroom.  The nominal dtype defaults to ``x``'s (or BF16 for other dtypes)."""
        if not x.is_cuda:
            raise ValueError("Float8Tensor.from_float needs a CUDA tensor")
        if fp8 not in FP8_MAX:
            raise TypeError(f"fp8 must be torch.float8_e5m2 or torch.float8_e4m3fn, got {fp8}")
        top = FP8_MAX[fp8]
        dtype = dtype or (x.dtype if x.dtype in (torch.bfloat16, torch.float16, torch.float32) else torch.bfloat16)
        xf = x.detach().float()
        if per_row:
            if x.dim() != 2:
                raise ValueError(f"per_row quantization needs a 2-D tensor, got shape {tuple(x.shape)}")
            amax = xf.abs().amax(dim=1).clamp_min(1e-12)
            scale = margin * top / amax
            data = (xf * scale[:, None]).clamp(-top, top).to(fp8)
            return cls(data, (1.0 / scale).float(), 0, dtype)
        from . import cast
        amax = xf.abs().amax().clamp_min(1e-12).reshape(1)
        scale = (margin * top / amax).float()
        return cls(cast(xf.contiguous(), fp8, scale=scale), 1.0 / scale, None, dtype)

    def dequantize(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        s = self._scale_inv
        if self._scale_dim == 0: s = s[:, None]
        elif self._scale_dim == 1: s = s[None, :]
        return (self._data.float() * s).to(dtype or self.dtype)

    def __repr__(self):
        how = "per tensor" if self._scale_dim is None else ("per row" if self._scale_dim == 0 else "per column")
        return (f"Float8Tensor(shape={tuple(self.shape)}, dtype={self.dtype}, {self._data.dtype}, scale {how}, "
                f"data={self._data})")

    __torch_function__ = torch._C._disabled_torch_function_impl

    # ---- operations ----------------------------------------------------------------------------
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        handler = _HANDLERS.get(func)
        if handler is not None:
            out = handler(*args, **kwargs)
            if out is not NotImplemented:
                return out
        unwrap = lambda t: t.dequantize() if isinstance(t, Float8Tensor) else t
        return func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))


def _wrap(data, scale_inv, scale_dim, dtype):
    return Float8Tensor(data, scale_inv, scale_dim, dtype)


def _transpose2d(x: Float8Tensor):
    if x.dim() != 2:
        return NotImplemented
    return _wrap(x._data.t(), x._scale_inv, None if x._scale_dim is None else 1 - x._scale_dim, x.dtype)


def _operand(x, outer_dim):
    """(data, scale or None, nominal dtype) of a matmul operand whose outer dimension is outer_dim, or
    None when its scale runs along K (only possible by dequantizing)."""
    if not isinstance(x, Float8Tensor):
        return x, None, x.dtype
    if x._scale_dim is not None and x._scale_dim != outer_dim:
        return None
    return x._data, x._scale_inv, x.dtype


def _mm(a, b, bias=None):
    from . import matmul
    pa, pb = _operand(a, 0), _operand(b, 1)
    if pa is None or pb is None or a.dim() != 2 or b.dim() != 2 or a.shape[1] % 16:
        return NotImplemented
    (da, sa, ta), (db, sb, tb) = pa, pb
    if not all(t.dtype in FP8_MAX or t.dtype in (torch.bfloat16, torch.float32) for t in (da, db)):
        return NotImplemented  # e.g. float16 / float64 plain operands: dequantize instead
    out_dtype = torch.promote_types(ta, tb)
    if bias is not None and (bias.dim() != 1 or bias.shape[0] != b.shape[1]):
        return NotImplemented  # broadcasts other than a per-column vector
    return matmul(da, db, out_dtype=out_dtype, bias=bias, scale_a=sa, scale_b=sb)


def _addmm(bias, a, b, *, beta=1, alpha=1):
    if beta != 1 or alpha != 1:
        return NotImplemented
    return _mm(a, b, bias)


_HANDLERS = {
    aten.mm.default: _mm,
    aten.addmm.default: _addmm,
    aten.t.default: _transpose2d,
    aten.transpose.int: lambda x, d0, d1: _transpose2d(x) if x.dim() == 2 and {d0 % 2, d1 % 2} == {0, 1} else NotImplemented,
    aten.detach.default: lambda x: _wrap(x._data, x._scale_inv, x._scale_dim, x.dtype),
    aten.alias.default: lambda x: _wrap(x._data, x._scale_inv, x._scale_dim, x.dtype),
    aten.clone.default: lambda x, memory_format=None: _wrap(x._data.clone(), x._scale_inv.clone(), x._scale_dim, x.dtype),
    aten._to_copy.default: lambda x, dtype=None, **kw: x.dequantize(dtype).to(**{k: v for k, v in kw.items() if k in ("device",) and v is not None}),
}


def quantize_linear_(module: torch.nn.Module, *, fp8: torch.dtype = torch.float8_e4m3fn, per_row: bool = True,
                     dtype: Optional[torch.dtype] = None) -> torch.nn.Module:
    """Replace the weight of every ``nn.Linear`` in ``module`` (in place) by a ``Float8Tensor`` (per output
    feature scales by default), so the module's matmuls run in FP8 on E5M2EffMA.  Inference only.

    Weights default to E4M3, whose extra mantissa bit suits them better (E5M2's range is for gradients);
    ``fp8=torch.float8_e5m2`` stores them in E5M2."""
    for m in module.modules():
        if isinstance(m, torch.nn.Linear) and not isinstance(m.weight, Float8Tensor):
            if m.in_features % 16:
                continue  # K must be a multiple of 16: keep this layer in high precision
            w = Float8Tensor.from_float(m.weight.data, fp8=fp8, per_row=per_row, dtype=dtype or m.weight.dtype)
            m.weight = torch.nn.Parameter(w, requires_grad=False)
    return module
