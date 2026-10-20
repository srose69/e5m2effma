#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. E5M2EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""e5m2effma.AdamW vs torch.optim.AdamW (for-loop, foreach, fused) on 64 M parameters.

Two layouts: one 64 M tensor, and 200 tensors of 320 K (a model-like parameter list, where torch's
foreach / fused implementations batch many tensors per launch).  Parameters FP32 or BF16; torch keeps
the moments in the parameter dtype, e5m2effma is shown with FP32 and with BF16 moments.  Median step
time over E5MA_BENCH_REPS steps after E5MA_BENCH_COOL s idle, as the C benchmarks (bench_util.h).
Needs PyTorch and the e5m2effma package (binds/python).
"""
import os, statistics, time
import torch
import e5m2effma as fx

REPS = int(os.environ.get('E5MA_BENCH_REPS', '30'))
COOL = float(os.environ.get('E5MA_BENCH_COOL', '2'))
dev = 'cuda'

def params(layout, dtype):
    sizes = [64 << 20] if layout == '1 x 64M' else [327680] * 200
    ps = [torch.nn.Parameter(torch.randn(n, device=dev, dtype=dtype) * 0.1) for n in sizes]
    for p in ps: p.grad = torch.randn_like(p) * 0.01
    return ps

def step_ms(opt):
    opt.step(); torch.cuda.synchronize()          # creates the state; untimed
    time.sleep(COOL)
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t = []
    for _ in range(REPS):
        a.record(); opt.step(); b.record(); b.synchronize(); t.append(a.elapsed_time(b))
    return statistics.median(t)

def main():
    print(f'{torch.cuda.get_device_name()}: AdamW step on 64 M parameters, median of {REPS} steps after {COOL:g} s idle; '
          f'torch {torch.__version__}, e5m2effma {fx.__version__}\n')
    print(f"{'':58s} {'time':>9s} {'vs torch fused':>15s}")
    hp = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    for dtype, dn in ((torch.float32, 'FP32'), (torch.bfloat16, 'BF16')):
        for layout in ('1 x 64M', '200 x 320K'):
            res = {}
            for name, make in (('torch AdamW (for-loop)', lambda ps: torch.optim.AdamW(ps, foreach=False, **hp)),
                               ('torch AdamW (foreach)', lambda ps: torch.optim.AdamW(ps, foreach=True, **hp)),
                               ('torch AdamW (fused)', lambda ps: torch.optim.AdamW(ps, fused=True, **hp)),
                               ('e5m2effma AdamW, FP32 moments', lambda ps: fx.AdamW(ps, **hp)),
                               ('e5m2effma AdamW, BF16 moments', lambda ps: fx.AdamW(ps, state_dtype=torch.bfloat16, **hp))):
                ps = params(layout, dtype)
                try: res[name] = step_ms(make(ps))
                except (RuntimeError, ValueError) as e: res[name] = None; print(f'   ({name}: {str(e).splitlines()[0]})')
                del ps
                torch.cuda.empty_cache()
            base = res.get('torch AdamW (fused)')
            for name, ms in res.items():
                label = f'{name}, {dn} params, {layout}'
                if ms is None: print(f'{label:58s} {"n/a":>9s}')
                else: print(f'{label:58s} {ms:6.3f} ms {base / ms if base else 0:13.2f}×', flush=True)

if __name__ == '__main__':
    main()
