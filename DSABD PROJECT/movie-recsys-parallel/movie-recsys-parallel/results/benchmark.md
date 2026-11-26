### Benchmark (668 users x 10,147 items, 98,542 ratings; rank=32, 10 ALS iters; median of 3)

Cores available to this run: **1**. Outputs identical across thread counts: **True**.

| Threads | Similarity (s) | ALS (s) | Total compute (s) | Speed-up | Efficiency |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.368 | 0.687 | 1.065 | 1.00x | 100% |
| 2 * | 0.350 | 0.694 | 1.044 | 1.02x | 51% |
| 4 * | 0.362 | 0.696 | 1.058 | 1.01x | 25% |
| 8 * | 0.364 | 0.716 | 1.080 | 0.99x | 12% |

\* more threads than available cores (1): speed-up cannot exceed ~1x here. Re-run on a multi-core machine for meaningful scaling numbers.
