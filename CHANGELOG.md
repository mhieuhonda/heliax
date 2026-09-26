# Changelog

## 0.6.14 — 2026-09-25

### Added

- Module-level `MSELoss`, `L1Loss`, `HuberLoss`, `BCEWithLogitsLoss`, and `BCELoss` APIs.
- Regression coverage for common loss module usage.

## 0.6.13 — 2026-09-25

### Added

- Native SiLU activation kernel and backend dispatch.
- Native report now covers twelve kernels.

## 0.6.12 — 2026-09-25

### Fixed

- Native softmax dispatch now supports arbitrary axes through moveaxis.
- Added arbitrary-axis native softmax regression coverage.

## 0.6.11 — 2026-09-25

### Added

- Native fused MAE kernel with broadcast-aware targets and sign gradients.
- Native report now covers eleven kernels.

## 0.6.10 — 2026-09-25

### Fixed

- Native log-softmax dispatch now supports arbitrary class axes through moveaxis.
- Added arbitrary-axis native log-softmax regression coverage.

## 0.6.9 — 2026-09-25

### Added

- Native last-dimension log-softmax kernel with backend dispatch.
- Native report now covers ten kernels.

## 0.6.8 — 2026-09-25

### Added

- Native fused BCE-with-logits kernel with broadcast-aware targets and fused gradients.
- Native report now covers nine kernels.

## 0.6.7 — 2026-09-25

### Added

- Native fused Huber loss kernel with broadcast-aware targets and fused gradients.
- Native report now covers eight kernels including Huber and MSE.

## 0.6.6 — 2026-09-25

### Added

- Native fused MSE kernel with broadcast-aware target handling.
- MSE native report coverage and backward autograd path.

## 0.6.5 — 2026-09-25

### Added

- Refreshed README feature matrix for the 0.6 baseline.
- Consolidated documentation for einsum, workspaces, memory accounting, Trainer, and native reporting.

## 0.6.4 — 2026-09-25

### Added

- Three additional pinned PyTorch module references (convolution, normalization, activation).
- Expanded provenance documentation and verified 20-file BSD snapshot.

## 0.6.3 — 2026-09-25

### Added

- Workspace memory reporting in the profiler for pooled scratch buffers.
- `memory_report` now includes workspace bytes in total storage accounting.

## 0.6.2 — 2026-09-25

### Added

- Tensor layout diagnostics with strides, dtype, contiguity, and storage bytes.
- `Module.freeze()` / `Module.unfreeze()` controls for inference and fine-tuning workflows.

## 0.6.1 — 2026-09-25

### Added

- Bounded shape/dtype workspace buffer pool with hit/miss reporting.
- Conv2d im2col scratch buffers now reuse pooled workspace storage in forward and backward.

## 0.6.0 — 2026-09-25

### Added

- Rich JSON checkpoint metadata for nested hyperparameters, lists, and dictionaries.
- Atomic serialization and non-strict loading retained as the stable checkpoint baseline.

## 0.5.9 — 2026-09-25

### Added

- JSON native-vs-portable kernel timing report with CI coverage.
- Native benchmark documentation now includes a reproducible report command.

## 0.5.8 — 2026-09-25

### Added

- Non-strict `Module.load_state_dict` for checkpoint migration workflows.
- Roadmap refresh for the 0.5 baseline and v0.6 layout/tooling plans.

## 0.5.7 — 2026-09-25

### Added

- Explicit `torch:<device>` backend selection in the backend registry.
- Device-selection tests and Torch compatibility documentation.

## 0.5.6 — 2026-09-25

### Added

- Explicit CPU device validation for `tensor(..., device=...)`, `Tensor.to_device`, and `Module.to_device`.
- Clear guidance for accelerator devices through Torch conversion/backends.

## 0.5.5 — 2026-09-25

### Added

- Explicit `device` arguments for `to_torch`/`from_torch` conversions.
- Torch interop documentation for CPU storage and accelerator device copies.

## 0.5.4 — 2026-09-25

### Added

- Native fused cross-entropy dispatch for arbitrary class axes via moveaxis.
- Atomic checkpoint writes and gradient/training memory accounting retained as the 0.5 baseline.

## 0.5.3 — 2026-09-25

### Fixed

- State-dict and checkpoint writes are atomic and preserve non-`.npz` path suffixes.
- Added serialization regression coverage for crash-safe writes and path handling.

## 0.5.2 — 2026-09-25

### Added

- Gradient memory accounting with missing-gradient detection.
- `training_memory_report` combining retained graph, parameter, and gradient storage.
- Profiler regression coverage and documentation refresh.

## 0.5.1 — 2026-09-25

### Added

- Native C fused last-axis cross-entropy kernel with normalized gradient buffer.
- Native dispatch for float32 fused cross entropy and direct kernel regression coverage.
- Expanded einsum gradients and axis-generic classification coverage.

## 0.5.0 — 2026-09-25

### Added

- Repeated-label single-operand einsum contractions and gradients.
- Complete axis-generic cross entropy coverage.
- Trainer, autograd buffer reuse, native kernels, Torch backend, quantization, EMA, and precision tooling retained as the stable 0.5 baseline.

## 0.4.9 — 2026-09-25

### Fixed

- `cross_entropy` and `fused_cross_entropy` now support arbitrary class axes.
- Single-operand einsum supports common unique-label reductions in addition to trace/permutation.
- Added regression coverage for axis-generic classification and reductions.

## 0.4.8 — 2026-09-25

### Added

- High-level `Trainer` with accumulation, scheduling, evaluation, and checkpoint helpers.
- Training loop regression coverage.

## 0.4.7 — 2026-09-25

### Fixed

- Reuse gradient buffers in-place during autograd accumulation to reduce hot-path allocations.
- Preserve independent gradient storage for alias-safe numerical checks.

## 0.4.6 — 2026-09-25

### Added

- Differentiable `gather` and `scatter_add` tensor operations.
- Advanced indexing regression coverage.

## 0.4.5 — 2026-09-25

### Added

- In-place `requires_grad_` and `detach_` tensor controls.
- Additional autograd lifecycle regression coverage.

## 0.4.4 — 2026-09-25

### Fixed

- Tensor and fused-linear matmul now dispatch through the active numerical backend.
- Verified Torch backend and interop paths remain optional and explicit.

## 0.4.3 — 2026-09-25

### Added

- Differentiable one- and two-operand einsum with explicit output specifications.
- Additional einsum contraction, trace, and transpose regression coverage.

## 0.4.2 — 2026-09-25

### Fixed

- Checkpoint keyword-tensor detection and repeat axis=None gradient reduction.
- Updated CI badge and Torch backend documentation.

## 0.4.1 — 2026-09-25

### Added

- Explicit opt-in TorchBackend numerical dispatch with tests.
- Validated CPU PyTorch backend and interop paths.

## 0.4.0 — 2026-09-25

### Added

- Repeatable autotuning helpers for benchmark-gated backend selection.
- Native artifact workflow for Linux/macOS shared libraries.
- Scheduled/manual PyTorch compatibility workflow and validated interop notes.
- Attention weight introspection and composable ModuleList execution.

## 0.3.9 — 2026-09-25

### Added

- Dilation support through GroupedConv2d.
- Validated Torch compatibility notes and composable ModuleList forward execution.
- Additional convolution and observability regression coverage.

## 0.3.8 — 2026-09-25

### Added

- MultiheadAttention can return attention weights for interpretability.
- Scheduled/manual PyTorch compatibility CI and native artifact builds.
- Causal mask and memory-aware checkpointing documentation.

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
