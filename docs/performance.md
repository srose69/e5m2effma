# Performance

Results per GPU are in [benches/table.md](../benches/table.md) (one row per GPU) and in
`benches/<gpu>.md` (full tables and the conditions of each run). This page describes how they are
measured, how to add a GPU, and what the GTX 1080 numbers say.

## Running the benchmarks

```sh
bench/run_bench.sh                                          # GPU 0
GPU=1 COOLING="open bench, 100 % fan" NOTES="..." bench/run_bench.sh
```

The script builds the benchmarks, sets the board up (below), runs `bench_gemm`, `bench_gemv` and
`bench_adamw`, and writes `benches/<gpu>.md` plus that GPU's row in `benches/table.md`
(`bench/report.py`). Raw output, including clock / temperature / power logs, stays in
`out/bench_<gpu>/`. It needs `sudo` for `nvidia-smi` and restores every setting on exit.
`make bench` runs the same benchmarks without touching the board settings. With PyTorch and the
bindings importable (`PYTHON=...`), the script also compares AdamW with `torch.optim.AdamW`
(`bench/bench_torch_adamw.py`).

**Adding a GPU:** run the script on it and open a pull request with `benches/<gpu>.md` and the
updated `benches/table.md`. Results from any pre-FP8 GPU are welcome (sm_50 … sm_87); please describe
the cooling in `COOLING`.

## Methodology

- **Median per row.** Every row (one kernel, one shape) gets one untimed call, then 2 s of idle GPU
  (`E5MA_BENCH_COOL`), then 50–2000 timed calls (`E5MA_BENCH_REPS`); the median is reported. The
  benchmarks are separated by 10 s of idle GPU and there is no warm-up, so every row starts from a
  similar temperature.
- **FP32 peak at the row's own clock.** % of FP32 peak = GMAC/s / (SMs × 128 × SM clock), one FFMA = one
  MAC, with the SM clock sampled through NVML every 5 ms while the row runs (`bench/bench_util.h`).
  GPU Boost moves the clock with temperature and power, and kernels draw different power, so one clock
  for the whole run would credit some rows with a clock they never had.
- **Board settings.** The power limit is raised to its maximum and the SM clock is allowed up to its
  top bin; the memory clock is locked at its maximum. Without a display, GeForce drivers run CUDA work
  in the P2 state with a reduced memory clock (4513 instead of 5005 MHz on a GTX 1080, about 10 % of
  bandwidth); the lock avoids that. The per-GPU file records what was actually set and the clock,
  temperature, power and throttling seen during each benchmark.
- **Bandwidth.** DRAM peak = memory clock × bus width; a device-to-device copy (read + write) is
  measured as the practical ceiling for streaming kernels.
- **Baselines.** cuBLAS on FP32 data of the same shape: SGEMM for GEMM, SGEMV and SGEMM n = 8 for the
  GEMV path. CUTLASS is not measured.

## What the GTX 1080 numbers say

Full tables: [benches/1080.md](../benches/1080.md).

- **GEMM matches cuBLAS SGEMM per clock** — 77.1 % of the FP32 peak for E5M2 × E5M2 (TN) vs 76.4 % for
  SGEMM — with operands 4× smaller and the output half the size. On Pascal an exact FP8 MAC costs one
  FFMA issue slot at best ([design.md](design.md)), so SGEMM speed is the ceiling. All four layouts
  run at 75–77 %; the E5M2 × E4M3 pairing of TE's HYBRID backward runs like E5M2 × E5M2 (the kernels
  are the same instructions with different decode constants).
- **Power, not code, sets the remaining gap.** With both operands in FP8 the kernel decodes both and
  draws up to ~214 W, hits the power limit and runs at 1751–1792 MHz instead of cuBLAS's 1832 MHz, so it
  takes 3–4 % longer (19.9 vs 19.2 ms). With one operand in BF16 or FP32 it runs near the cuBLAS clock
  and finishes as fast as SGEMM (19.1 ms).
- **GEMV is bandwidth-bound**: 0.237 ms for an 8192 × 8192 E5M2 matrix, 89 % of the DRAM peak (as fast
  as a device-to-device copy) and 4.4× cuBLAS SGEMV on FP32 weights. At n = 8 the kernel becomes
  compute- and register-bound: 3.8× faster than SGEMM on FP32 weights with BF16 x, 3.0× with FP8 x
  (whose decode then costs ALU time).
- **AdamW** works on 8 elements per thread with 16-byte loads and stores. With BF16 state it runs at
  281 GB/s, 88 % of the DRAM peak (as fast as a device-to-device copy): 3.3 ms for 64 M parameters, 2.2×
  faster than FP32 state (7.4 ms, 79 %). Writing the E5M2 weight copy adds 0.25 ms.
- **AdamW vs `torch.optim.AdamW`** (`bench/bench_torch_adamw.py`, same 64 M parameters, as one tensor or
  as 200 tensors of 320 K): against torch's fastest variant (`fused=True`) e5m2effma is 1.20–1.24×
  faster with FP32 parameters and state, and 2.65–2.79× faster with BF16 parameters and state — torch's
  fused kernel barely gains from BF16 (9.3–9.6 vs 9.0–9.3 ms), this one halves the time. With many
  tensors torch batches launches while e5m2effma launches one kernel per parameter, and it is still
  ahead.

## Other GPUs

- **RTX 3090** (sm_86, community run (measured with E4M3EffMA, which shares the kernels); [benches/rtx-3090.md](../benches/rtx-3090.md)): the
  bandwidth-bound kernels carry over — GEMV batch 1 at 799 GB/s, 85 % of the 936 GB/s DRAM peak and
  3.7× cuBLAS SGEMV; AdamW with BF16 state at 87 %. GEMM does not: 1.10× cuBLAS SGEMM in the TN layout
  but 0.89× its faster NN layout, at about 50 % of the FP32 peak. These are the Pascal-tuned FP32-FFMA
  kernels with no sm_86 tuning entry; Ampere's FP32 peak counts its second FP32 datapath, which the
  kernels' LDS-heavy inner loop probably does not keep fed (not profiled), and the Tensor Core path
  (roadmap) is where the remaining speed is.
