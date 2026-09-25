# Heliax roadmap

## v0.1 — a trustworthy core

- Tensor broadcasting, views, reductions, and reverse-mode autograd
- Linear/Conv2d/LayerNorm/Embedding/attention and foundational activations
- AdamW, SGD, RMSProp, schedules, clipping
- Checkpoint/state-dict serialization
- Quantization helpers, NumPy backend, and benchmark harness

## v0.2 — less Python overhead

- Operation counters and memory/latency profiler
- Fused NumPy kernels for common forward/backward paths
- Better in-place-safe internal buffers
- Strided-layout benchmark suite

## v0.3 — native acceleration

- Optional C ABI extension boundary
- Fused linear + activation and normalized cross-entropy candidates
- Quantized int8 weights for CPU inference
- Pluggable accelerator capability detection

## v1.0 — ecosystem depth

- Convolution and attention building blocks
- Recurrent cells
- Distributed tensor/process primitives
- Stable API, migration guides, and performance guarantees
