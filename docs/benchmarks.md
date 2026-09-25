# Heliax benchmark notes

These are development baselines, not a promise of superiority over PyTorch. Run them on your own hardware before making performance claims.

Environment used for the initial baseline:

- Python 3.12
- NumPy 2.5.3
- CPU-only container
- 512 × 512 float32 matrices

Initial output from `examples/benchmark.py`:

| operation | approximate time |
| --- | ---: |
| matrix multiply forward | 1.4–5.5 ms |
| elementwise add | 0.8–0.9 ms |
| mean reduction | 0.2–0.3 ms |
| forward + backward autograd step | ~5.4 ms |

The range reflects container CPU scheduling and shared-runner variance. The benchmark is intentionally explicit about the workload instead of comparing unlike systems.

## Benchmark rules for future kernels

1. Use the same dtype, shape, batch size, and warm-up policy.
2. Report median and p95 latency, not only the best result.
3. Compare against the NumPy baseline and a relevant mature framework.
4. Include memory traffic and compilation/caching status.
5. Never enable a fused kernel by default unless it is faster on the supported platform.
