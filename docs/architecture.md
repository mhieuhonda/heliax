# Heliax architecture

## Design goals

Heliax separates **tensor semantics**, **automatic differentiation**, **neural modules**, and **backend execution**. A native backend can replace NumPy without changing model code.

```text
User code
  ├── Tensor / Parameter
  │     └── graph nodes + reverse-mode traversal
  ├── functional math and losses
  ├── nn.Module tree
  ├── optim.Optimizer
  ├── quantization / inference utilities
  ├── torch_interop (optional accelerator facade)
  ├── native_ops (opt-in C/ctypes kernels)
  ├── distributed helpers / initialization
  └── serialization / profiler

External reference
  └── third_party/pytorch (pinned BSD files, not runtime-imported)

Backend boundary
  └── NumPyBackend now
      ├── future fused C++ extension
      ├── future Rust/C ABI kernel registry
      └── optional TorchAccelerator for compatibility/benchmarks
```

## Tensor model

`Tensor.data` is a NumPy array. `Tensor.grad` is another Tensor, preserving the same shape and dtype metadata. A graph node stores its input tensors and a backward closure. The traversal is topological and visits each node once, so shared subexpressions are not recomputed.

Broadcasting follows NumPy semantics. When a gradient is reduced to the shape of an input, Heliax sums the broadcast dimensions before adding it to the parameter/input gradient.

## Backend boundary

The backend provides array construction, mathematical kernels, matrix multiplication, and optimizer updates. The public API uses backend-neutral operations; it does not leak NumPy into model modules except at explicit interoperability boundaries.

## Safety and correctness

- Gradient tracking is disabled inside `no_grad()`.
- Parameters are ordinary Tensors with `requires_grad=True`.
- The graph is not mutated during a forward pass; gradients are accumulated separately.
- `gradcheck` compares analytical gradients against finite differences.
- Checkpoint loading validates the expected state-dict keys before applying values.

## Native extension path

`src/heliax/native_ops.c` currently provides opt-in C kernels for add+ReLU, GELU, last-dimension softmax, LayerNorm, and AdamW. `native_ops.py` loads the shared library through `ctypes`; the NumPy path remains the default and is selected unless `HELIAX_NATIVE=1` is set.

A future extension can add selected kernels behind the same backend interface. Each candidate must beat the NumPy baseline on a published benchmark before being enabled by default.

## Diagnostics

`profiler.py` exposes graph node/edge counts, retained storage, operation histograms, and parameter/buffer memory reports. `anomaly_detection()` checks gradient finiteness and `retain_graph=False` releases a traversed graph.
