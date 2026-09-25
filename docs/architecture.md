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
  └── serialization / profiler

Backend boundary
  └── NumPyBackend now
      ├── future fused C++ extension
      ├── future Rust/C ABI kernel registry
      └── future accelerator backend
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

## Native extension plan

A future extension can implement selected kernels behind the same backend interface. Candidate first targets are fused `linear + activation`, normalized cross-entropy, AdamW updates, and contiguous transpose/layout transforms. Each candidate must beat the NumPy baseline on a published benchmark before being enabled by default.
