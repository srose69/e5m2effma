# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. E5M2EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""E5M2EffMA: FP8 (E5M2 and E4M3) GEMM / GEMV / cast / Adam for NVIDIA GPUs without native FP8
(sm_50..sm_87).

Tensors are ordinary CUDA tensors; FP8 tensors use ``torch.float8_e5m2`` or ``torch.float8_e4m3fn``,
freely mixed (e.g. E5M2 gradients x E4M3 weights, the backward of Transformer Engine's HYBRID format).
Scales follow Transformer
Engine: ``scale_a`` / ``scale_b`` / ``scale_inv`` dequantize an FP8 input (value = code * scale_inv),
``scale_out`` / ``scale`` quantize an FP8 output (code = RNE_satfinite(value * scale)), ``amax`` tensors
receive max |value| before quantization (max-reduced into their current content; Inf / NaN there
signal an overflow upstream, as for loss scaling).  All of them are one-element float32 CUDA tensors on
the device of the operation (``matmul`` / ``linear`` also take row-wise vectors).

Every call checks its arguments and raises ``TypeError`` (dtype), ``ValueError`` (device, shape, K not
a multiple of 16, overlapping output), ``NotImplementedError`` (GPU with native FP8, or older than
sm_50) or ``RuntimeError`` (library / CUDA failure) instead of running on bad input.  The one thing
that is not checked is the content of FP8 matmul operands: they must be finite (no E5M2 Inf / NaN
codes, no E4M3 NaN code); casts read Inf / NaN exactly.

On import the package also
  * adds ``torch.e5m2effma`` (and ``e5m2effma.e5m2``): an alias of ``torch.float8_e5m2``, whose bit
    encoding is exactly the E5M2 of this library (a separate torch dtype cannot be added from outside
    PyTorch, and is not needed: the tensors are the same);
  * provides ``torch._scaled_mm`` -- PyTorch's FP8 matmul, which needs sm_89+ natively -- on top of this
    library when every visible GPU lacks native FP8 (sm_50..sm_87), so FP8 code written for PyTorch
    runs unchanged on those GPUs.  Set E5M2EFFMA_NO_TORCH_OPS=1 to leave PyTorch untouched.
"""
import os
import warnings
from typing import Optional

import torch

from . import _C
from .optim import AdamW
from .tensor import Float8Tensor, quantize_linear_

__version__ = _C.__version__
e5m2 = torch.float8_e5m2
e4m3 = torch.float8_e4m3fn
if not hasattr(torch, "e5m2effma"):
    torch.e5m2effma = e5m2
__all__ = ["matmul", "linear", "cast", "adam_step", "AdamW", "Float8Tensor", "quantize_linear_", "e5m2", "e4m3",
           "torch_ops_enabled", "__version__"]
_FP8 = (torch.float8_e5m2, torch.float8_e4m3fn)


def matmul(a: torch.Tensor, b: torch.Tensor, *, out_dtype: Optional[torch.dtype] = None,
           bias: Optional[torch.Tensor] = None, out: Optional[torch.Tensor] = None, alpha: float = 1.0,
           beta: float = 0.0, scale_a: Optional[torch.Tensor] = None, scale_b: Optional[torch.Tensor] = None,
           scale_out: Optional[torch.Tensor] = None, amax_out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``out = alpha * (scale_a * scale_b) * (a @ b) + bias + beta * out`` for 2-D ``a`` [M, K], ``b`` [K, N].

    * ``a``, ``b``: ``float8_e5m2``, ``float8_e4m3fn``, ``bfloat16`` or ``float32`` in any combination,
      K a multiple of 16.  Products of FP8 values (any pairing) are exact in FP32, FP8 x BF16 ones too
      unless they underflow; accumulation is FP32 with a single rounding to the output type.  The kernels read K-contiguous operands directly (and row- or column-contiguous ones whose
      leading dimension is a multiple of 16, 16-byte aligned); any other operand is copied once into
      that layout first.
    * ``scale_a`` / ``scale_b`` (FP8 operands only): one float32, or one per row of ``a`` (``[M]`` /
      ``[M, 1]``) / per column of ``b`` (``[N]`` / ``[1, N]``) — row-wise scaling as ``torch._scaled_mm``.
    * ``out_dtype``: ``bfloat16`` (default), ``float16``, ``float32``, ``float8_e5m2`` or
      ``float8_e4m3fn`` (quantized with ``scale_out``; ``amax_out`` gets max |value|).
    * ``bias``: vector of length N.  ``beta != 0`` accumulates into ``out`` (e.g. an FP32 gradient
      buffer), which must then be given; it must be row-contiguous [M, N].
    * For M <= 8 with ``b`` a column-contiguous FP8 matrix (``W.t()``) a bandwidth-bound GEMV kernel runs.
    """
    return _C.mm(a, b, out_dtype, bias, out, alpha, beta, scale_a, scale_b, scale_out, amax_out)


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, *,
           out_dtype: Optional[torch.dtype] = None, scale_x: Optional[torch.Tensor] = None,
           scale_w: Optional[torch.Tensor] = None, scale_out: Optional[torch.Tensor] = None,
           amax_out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``x @ weight.t() + bias`` like ``torch.nn.functional.linear``: ``x`` [..., K], ``weight`` [N, K].

    The FP8 layout of cuBLASLt / TE (both operands K-contiguous); ``weight`` is typically FP8 with its
    dequantization factor in ``scale_w``: one float32 or one per output feature (``[N]``); ``scale_x``
    is one float32 or one per row of ``x`` flattened to 2-D.  Up to 8 rows of ``x`` (decoding) run the
    GEMV kernel, which reads every weight byte once.
    """
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
        raise TypeError("x and weight must be tensors")
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2-D [out_features, in_features], got shape {tuple(weight.shape)}")
    if x.dim() < 1 or x.shape[-1] != weight.shape[1]:
        raise ValueError(f"x must have shape [..., {weight.shape[1]}], got {tuple(x.shape)}")
    x2 = x.reshape(-1, x.shape[-1])
    if scale_x is not None and scale_x.numel() > 1:
        scale_x = scale_x.reshape(-1)
    y = _C.mm(x2, weight.t(), out_dtype, bias, None, 1.0, 0.0, scale_x, scale_w, scale_out, amax_out)
    return y.reshape(*x.shape[:-1], weight.shape[0])


def cast(x: torch.Tensor, dtype: torch.dtype, *, scale_inv: Optional[torch.Tensor] = None,
         scale: Optional[torch.Tensor] = None, amax: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise conversion between ``float8_e5m2``, ``float8_e4m3fn``, ``bfloat16``, ``float16`` and
    ``float32``.

    ``value = x * scale_inv`` (FP8 ``x``; Inf / NaN codes decode to Inf / NaN); the result is
    ``RNE_satfinite(value * scale)`` for an FP8 ``dtype`` (saturating to +-57344 for E5M2, +-448 for
    E4M3, Inf included; NaN -> 0x7F), round-to-nearest-even otherwise; ``amax`` gets max |value| (Inf /
    NaN when the input had them).  Returns a new contiguous tensor of the shape of ``x``.
    """
    return _C.cast(x, dtype, scale_inv, scale, amax)


def adam_step(param: torch.Tensor, grad: torch.Tensor, exp_avg: torch.Tensor, exp_avg_sq: torch.Tensor, *,
              lr: float, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0, step: int,
              adamw: bool = True, bias_correction: bool = True, grad_scale: float = 1.0,
              param_fp8: Optional[torch.Tensor] = None, fp8_scale: Optional[torch.Tensor] = None,
              fp8_amax: Optional[torch.Tensor] = None) -> None:
    """One fused in-place Adam / AdamW step (``step`` is 1-based).

    ``param`` float32 or bfloat16; ``grad`` float32, bfloat16 or float16; ``exp_avg`` / ``exp_avg_sq``
    float32 or bfloat16 (same dtype); all contiguous with the same number of elements.  ``param_fp8``
    (``float8_e5m2`` or ``float8_e4m3fn``) optionally receives an FP8 copy of the updated weights
    quantized with ``fp8_scale``, and ``fp8_amax`` their max |value|.  ``grad_scale`` multiplies the gradient (e.g. 1 / loss_scale).
    """
    beta1, beta2 = betas
    _C.adam_step(param, grad, exp_avg, exp_avg_sq, lr, beta1, beta2, eps, weight_decay, step, adamw,
                 bias_correction, grad_scale, param_fp8, fp8_scale, fp8_amax)


# ---- torch._scaled_mm on GPUs without native FP8 -------------------------------------------------
def _scaled_mm(self, mat2, scale_a, scale_b, bias=None, scale_result=None, out_dtype=None, use_fast_accum=False):
    """aten::_scaled_mm for sm_50..sm_87: out = (self * scale_a) @ (mat2 * scale_b) + bias, FP8 out
    quantized with scale_result.  Operands float8_e5m2 / float8_e4m3fn in any pairing; per-tensor
    scales, or row-wise ones (scale_a [M, 1], scale_b [1, N]); use_fast_accum is ignored (accumulation
    is always full FP32, one FFMA per MAC)."""
    M, N = self.shape[0], mat2.shape[1]
    tensorwise = scale_a.numel() == 1 and scale_b.numel() == 1
    rowwise = tuple(scale_a.shape) == (M, 1) and tuple(scale_b.shape) == (1, N)
    if not (tensorwise or rowwise):
        raise NotImplementedError("e5m2effma implements torch._scaled_mm with per-tensor or row-wise scales "
                                  f"(scale_a [{M}, 1], scale_b [1, {N}]); got scale_a {tuple(scale_a.shape)}, "
                                  f"scale_b {tuple(scale_b.shape)}")
    for name, t in (("self", self), ("mat2", mat2)):
        if t.dtype not in _FP8:
            raise NotImplementedError(f"e5m2effma implements torch._scaled_mm for float8_e5m2 / float8_e4m3fn "
                                      f"operands, {name} is {t.dtype}")
    out_dtype = out_dtype or self.dtype
    if scale_result is not None and out_dtype not in _FP8:
        scale_result = None  # only applied to FP8 outputs, as in PyTorch
    return _C.mm(self, mat2, out_dtype, bias, None, 1.0, 0.0, scale_a.reshape(-1).float(), scale_b.reshape(-1).float(),
                 None if scale_result is None else scale_result.reshape(1).float(), None)


# One aten::_scaled_mm override shared by the *EffMA packages (E4M3EffMA, E5M2EffMA): the first one
# imported registers a dispatcher, every package appends its handler.  A handler raises
# NotImplementedError for calls it does not cover; the next one is tried, and the last error is raised
# when none applies.  Keep this protocol identical in all packages.
_SHARED = "_effma_scaled_mm"


def _register_torch_ops():
    if os.environ.get("E5M2EFFMA_NO_TORCH_OPS") == "1" or not torch.cuda.is_available():
        return None
    caps = [10 * a + b for a, b in (torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count()))]
    if not caps or any(cc < 50 or cc >= 89 for cc in caps):
        return None  # a GPU with native FP8 (or an unsupported one) is present: keep PyTorch's own kernel
    shared = getattr(torch, _SHARED, None)
    if shared is None:
        handlers = []

        def dispatch(*args, **kwargs):
            err = None
            for h in handlers:
                try:
                    return h(*args, **kwargs)
                except NotImplementedError as e:
                    err = e
            raise err

        lib = torch.library.Library("aten", "IMPL")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # "overriding a previously registered kernel"
            lib.impl("_scaled_mm", dispatch, "CUDA")
        shared = {"lib": lib, "handlers": handlers}  # the registration lives as long as lib
        setattr(torch, _SHARED, shared)
    shared["handlers"].append(_scaled_mm)
    return shared


_torch_ops = _register_torch_ops()


def torch_ops_enabled() -> bool:
    """Whether ``torch._scaled_mm`` runs on this library in this process."""
    return _torch_ops is not None
