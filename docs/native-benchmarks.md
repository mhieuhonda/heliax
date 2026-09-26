# Native kernel benchmark notes

The optional C kernels are deliberately opt-in. The numbers below are an example from one CPU container, not a universal claim.

Current native kernels: add+ReLU, GELU, last-dimension softmax, fused last-axis cross entropy, LayerNorm, and AdamW. The fused cross-entropy kernel returns the normalized gradient buffer so the autograd backward avoids rebuilding softmax probabilities.

## Default portable build

```bash
python scripts/build_native.py
python examples/native_benchmark.py
python scripts/native_report.py
```

`native_report.py` emits a JSON median-timing comparison for every available kernel and never changes the active backend. A scalar C loop can be slower than NumPy/BLAS for small or memory-bound arrays. Heliax therefore keeps NumPy as the default and never silently switches kernels.

## Host-tuned OpenMP build

```bash
HELIAX_NATIVE_OPENMP=1 HELIAX_NATIVE_NATIVE_ARCH=1 python scripts/build_native.py
python examples/native_benchmark.py
```

On the development container this reduced the native gap for larger elementwise workloads and made the softmax path competitive. The result is hardware/compiler dependent.

## Benchmark protocol

1. Build with the same compiler and flags recorded.
2. Warm the kernel before timing.
3. Use float32 contiguous arrays of the documented shape.
4. Report median and p95, not one best sample.
5. Compare against the same operation through the portable NumPy path.
6. Enable native dispatch only with `HELIAX_NATIVE=1` after the comparison is favorable.
