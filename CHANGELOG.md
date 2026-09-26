# Changelog

## 0.3.7 — 2026-09-25

### Added

- Activation checkpointing for compute/memory tradeoffs.
- Additional gradient replay and shape-operation coverage.

## 0.3.6 — 2026-09-25

### Added

- Differentiable `expand`, `repeat`, `roll`, `squeeze`, and `unsqueeze` tensor operations.
- Additional shape-operation gradient regression coverage.

## 0.3.5 — 2026-09-25

### Added

- Causal mask helper, differentiable linalg helpers, and half-precision dtype preservation.
- CI-safe benchmark smoke gate.
- Additional regression coverage for precision, quantized inference, and convolution paths.

## 0.3.4 — 2026-09-25

### Added

- QuantizedLinear inference module with persistent integer weights and state-dict support.
- Additional linalg and quantization regression coverage.

## 0.3.3 — 2026-09-25

### Added

- Grouped and dilated Conv2d paths.
- QuantizedLinear inference module with persistent integer weights.
- ModelEMA, explicit precision policy, and dynamic gradient scaling.
- Fresh-optimizer checkpoint state restoration and expanded serialization tests.
- Additional elementwise, pooling, normalization, recurrent, and attention regression coverage.

## 0.3.2 — 2026-09-25

### Added

- GroupedConv2d built from composable Conv2d blocks.
- Dilated Conv2d support and composable grouped convolution.
- ModelEMA parameter averaging with state-dict support.
- Explicit autocast policy context and dynamic GradScaler scaffolding.
- Additional container, convolution, and regression coverage.

## 0.3.1 — 2026-09-25

### Added

- Fused cross-entropy forward/backward node and `CrossEntropyLoss` module.
- GroupedConv2d, Adagrad/ExponentialLR, Huber loss, and checkpoint/model diagnostics.
- PrefetchLoader, elementwise `where`, comparison Tensor operators, and scalar conveniences.
- Checkpoint format validation, fresh-optimizer state restoration, and model summary diagnostics.
- Additional layer, optimizer, serialization, and regression tests.

## 0.3.0 — 2026-09-25

### Added

- Opt-in C/ctypes native kernels for add+ReLU, GELU, softmax, LayerNorm, and AdamW.
- `MaxPool2d`, `AvgPool2d`, BatchNorm1d/2d, GroupNorm, GRU, LSTM, transformer encoder, and positional encoding.
- Graph/memory diagnostics, anomaly detection, graph release, and op histograms.
- Gradient accumulation, schedulers, Adagrad, ExponentialLR, and a bounded `fit` loop.
- DistributedSampler and in-process gradient reduction helpers.
- Fan-aware parameter initialization utilities.
- Huber loss, elementwise comparison/autograd helpers, and broader edge-case coverage.

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
