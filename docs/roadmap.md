# Heliax roadmap

## v0.1 — a trustworthy core

- Tensor broadcasting, views, reductions, and reverse-mode autograd
- Linear/Conv2d/LayerNorm/Embedding/attention and foundational activations
- AdamW, SGD, RMSProp, schedules, clipping
- Checkpoint/state-dict serialization
- Quantization helpers, NumPy backend, and benchmark harness

## v0.2 — less Python overhead

- Fused linear+bias and linear+GELU autograd nodes
- Masked softmax and scaled dot-product attention helpers
- Contiguous/clone/in-place tensor utilities
- Optional TorchAccelerator and Helianthus namespace
- Pinned BSD PyTorch reference snapshot with provenance checks

## v0.3 — native acceleration

- Optional C ABI extension boundary
- Fused native linear + activation and normalized cross-entropy candidates
- Quantized int8 weights for CPU inference
- Pluggable accelerator capability detection
- Benchmark-gated native/Torch kernel selection

## v1.0 — ecosystem depth

- Convolution and attention building blocks
- Recurrent cells
- Distributed tensor/process primitives
- Stable API, migration guides, and performance guarantees
