# Results by GPU

One row per GPU, produced by `bench/run_bench.sh` (see [docs/performance.md](../docs/performance.md)); the
linked file has the full tables and the conditions of the run. Percentages of the FP32 peak use the SM
clock measured during each row; the baseline is cuBLAS on FP32 data of the same shape.

| GPU | CC | GEMM TN E5M2: % of FP32 peak (vs cuBLAS SGEMM) | GEMM TN E5M2 × E5M2 vs SGEMM | GEMM TN E5M2 × BF16 vs SGEMM | GEMV n = 1 (% of DRAM peak) | GEMV n = 1 vs SGEMV | AdamW FP32 state (% of DRAM peak) | AdamW vs torch fused (FP32 / BF16 params and state) | Date |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| [NVIDIA GeForce GTX 1080](1080.md) | 6.1 | 77 % vs 75 % | 0.98× | 1.01× | 284 GB/s (89 %) | 4.4× | 253 GB/s (79 %) | 1.24× / 2.79× | 2026-10-20 |
| [NVIDIA GeForce RTX 3090](rtx-3090.md) (community, measured with E4M3EffMA 0.1.0: E4M3 operands) | 8.6 | 49 % vs 44 % | 1.10× | 1.11× | 799 GB/s (85 %) | 3.7× | 713 GB/s (76 %) | — / — | 2026-10-20 |
