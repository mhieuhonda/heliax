# Changelog

## 0.2.0 — 2026-09-25

### Added

- Pinned BSD PyTorch reference snapshot with preserved LICENSE/NOTICE and provenance.
- Optional `TorchAccelerator` interop behind the `torch` extra.
- `helianthus` compatibility namespace.
- Fused linear+bias and linear+GELU autograd nodes.
- Numerically safe masked softmax and scaled dot-product attention.
- Contiguous/clone/in-place tensor utilities for lower-allocation loops.

## 0.1.0 — 2026-09-25

### Added

- NumPy-backed n-dimensional Tensor and reverse-mode autograd.
- Broadcasting, reductions, views, indexing, matmul, activations, softmax, and losses.
- Linear, Conv2d, LayerNorm, Embedding, MultiheadAttention, Dropout, and Sequential modules.
- SGD, Nesterov SGD, Adam, AdamW, RMSProp, schedules, and gradient clipping.
- NPZ state-dict/checkpoint serialization and small training utilities.
- 4-bit/8-bit symmetric and affine quantization helpers.
- Backend registry, operation profiler, benchmark example, tests, and Python 3.10–3.12 CI.
