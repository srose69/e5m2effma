# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. E5M2EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""``torch.optim``-compatible AdamW on the fused E5M2EffMA kernel."""
from typing import Iterable, Optional, Tuple

import torch


class AdamW(torch.optim.Optimizer):
    """AdamW (or Adam with ``adamw=False``) with the fused E5M2EffMA kernel, one launch per parameter.

    Same hyper-parameters and semantics as ``torch.optim.AdamW``; parameters may be float32 or
    bfloat16, gradients float32, bfloat16 or float16.  ``state_dtype`` sets the dtype of the moments
    (float32 by default; bfloat16 halves the optimizer memory).  Parameters must be contiguous CUDA
    tensors on a supported GPU (sm_50..sm_87).
    """

    def __init__(self, params: Iterable, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 1e-2, *, adamw: bool = True,
                 state_dtype: Optional[torch.dtype] = None):
        if lr < 0.0 or eps < 0.0 or weight_decay < 0.0:
            raise ValueError(f"lr, eps and weight_decay must be non-negative (got {lr}, {eps}, {weight_decay})")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"betas must be in [0, 1), got {betas}")
        if state_dtype not in (None, torch.float32, torch.bfloat16):
            raise TypeError(f"state_dtype must be torch.float32 or torch.bfloat16, got {state_dtype}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, adamw=adamw,
                        state_dtype=state_dtype or torch.float32)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        from . import adam_step
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise ValueError("AdamW does not support sparse gradients")
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=group["state_dtype"], memory_format=torch.contiguous_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=group["state_dtype"], memory_format=torch.contiguous_format)
                state["step"] += 1
                adam_step(p, p.grad, state["exp_avg"], state["exp_avg_sq"], lr=group["lr"], betas=group["betas"],
                          eps=group["eps"], weight_decay=group["weight_decay"], step=state["step"],
                          adamw=group["adamw"])
        return loss
