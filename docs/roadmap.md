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

## v0.3 — native acceleration and depth

- Optional C/ctypes kernels for fused elementwise, normalization, softmax, and AdamW paths
- Pooling, BatchNorm/GroupNorm, transformer encoder, GRU/LSTM, and module containers
- Anomaly detection, graph release, memory/op diagnostics
- Gradient accumulation, schedulers, Adagrad, bounded fit loop, distributed sampler
- Fan-aware initialization, extra losses, and broader edge-case tests

## v0.3.7 — memory-aware training

- Activation checkpointing with backward recomputation
- QuantizedLinear inference and half-precision dtype preservation
- Causal masks, linalg helpers, ModelEMA, GradScaler, and CI benchmark gate
- Shape-operation autograd coverage

## v0.4 — memory and kernels

- Strided/contiguous layout policies and reusable workspace buffers
- More fused forward/backward kernels with benchmark gates
- Optional quantized inference modules
- Better mixed-precision scaffolding and dtype policies
- Native backend packaging and per-platform artifact matrix

## v1.0 — ecosystem depth

- Convolution and attention building blocks
- Recurrent cells
- Distributed tensor/process primitives
- Stable API, migration guides, and performance guarantees
