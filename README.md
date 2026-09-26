# Heliax

[![Heliax CI](https://github.com/mhieuhonda/heliax/actions/workflows/ci.yml/badge.svg)](https://github.com/mhieuhonda/heliax/actions/workflows/ci.yml)

> **A performance-first tensor, autodiff, and neural network library for Python.**

Heliax is a new, original Python deep-learning library built around a simple promise: keep the numerical work close to NumPy/BLAS, make the graph explicit, and leave room for a native compiler later.

Heliax does **not** claim to beat PyTorch on every workload. PyTorch has years of ecosystem investment and a mature CUDA/distributed stack. Heliax is being built as a focused alternative for people who want a readable core, a small dependency surface, and a path toward fused/native kernels.

## What is implemented in 0.6.4

- N-dimensional `Tensor` with broadcasting, views, dtype conversion, comparisons, and low-allocation in-place operations.
- Reverse-mode automatic differentiation with graph traversal, gradient accumulation, `no_grad`, `gradcheck`, anomaly detection, and graph release.
- `Linear`, `Conv2d`, pooling, `LayerNorm`, `BatchNorm1d/2d`, `GroupNorm`, `Embedding`, `MultiheadAttention`, transformer encoder, `GRU`, `LSTM`, `FusedLinearGELU`, and foundational activations.
- `SGD`, `Nesterov SGD`, `Adagrad`, `Adam`, `AdamW`, `RMSProp`; gradient clipping; cosine/step/exponential schedules.
- MSE, Huber, cross-entropy, binary cross-entropy, softmax, masked softmax, scaled dot-product attention, and functional APIs.
- Complete one/two-operand einsum contractions, gathers, scatters, linalg, and causal masks.
- 4-bit/8-bit symmetric and affine weight quantization with compression reporting.
- Atomic NPZ checkpoints with rich JSON metadata and non-strict migration loading.
- High-level `Trainer` with accumulation, scheduling, evaluation, and checkpoint helpers.
- Bounded workspace buffers reused by Conv2d, with layout diagnostics and memory accounting.
- Optional `TorchAccelerator` interop behind the `torch` extra; PyTorch is never a required dependency.
- Opt-in C/ctypes native kernels for add+ReLU, GELU, softmax, fused cross entropy, LayerNorm, and AdamW.
- Graph/memory diagnostics, `DistributedSampler`, gradient reduction, ModelEMA, fan-aware initialization, and freeze controls.
- Activation checkpointing, explicit `autocast`/`GradScaler` policies, and QuantizedLinear inference.
- A `helianthus` compatibility namespace re-exports the same core.
- A pinned, license-preserving PyTorch reference snapshot under `third_party/pytorch/`.
- Backend registry and environment reporting. NumPy is the default; native dispatch requires an explicit opt-in and benchmark.
- Focused tests, gradient checks, benchmark scripts, packaging, and CI.

## Quick start

```python
import heliax as hx

x = hx.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
w = hx.Parameter(hx.randn(3, 2))
b = hx.Parameter(hx.zeros(3))

y = hx.nn.Sequential(
    hx.nn.Linear(2, 3),
    hx.nn.GELU(),
)(x)
loss = y.square().mean()
loss.backward()

print(loss.item())
print(x.grad.shape, w.grad.shape)
```

Train a tiny model:

```python
import heliax as hx

model = hx.nn.MLP(4, [32, 16], 2, activation=hx.nn.GELU)
optimizer = hx.optim.AdamW(model.parameters(), lr=1e-3)

for x, y in loader:
    optimizer.zero_grad()
    loss = hx.functional.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()
```

## Performance philosophy

1. **Vectorize first.** The default backend uses NumPy/BLAS for elementwise work and matrix multiplication.
2. **Avoid hidden global state.** Context managers explicitly control gradient tracking.
3. **Keep the graph inspectable.** Every operation can be tested, profiled, or replaced.
4. **Make fusion a future primitive, not a rewrite.** The backend boundary is designed for Triton, Rust, C++, or accelerator kernels later.
5. **Measure honestly.** `examples/benchmark.py` reports throughput and memory shape instead of promising magic.

## PyTorch and Helianthus

Heliax vendors a small, pinned set of BSD-3-Clause PyTorch reference files with full provenance in `third_party/pytorch/`; it does not copy the whole framework or hide its dependencies. The adapted paths are written in Heliax itself: fused linear+bias, fused linear+GELU, masked softmax, attention composition, and low-allocation tensor utilities.

The optional interop is explicit:

```python
import heliax as hx

if hx.torch_available():
    accelerator = hx.TorchAccelerator(device="cpu")
    print(accelerator.info())
```

Projects that want the Helianthus namespace can use:

```python
import helianthus as hx
```

## Optional native CPU kernels

Build the small dependency-free C library and opt in only after benchmarking it on your machine:

```bash
python scripts/build_native.py
HELIAX_NATIVE=1 python examples/native_benchmark.py
# optional host-tuned OpenMP build
HELIAX_NATIVE_OPENMP=1 HELIAX_NATIVE_NATIVE_ARCH=1 python scripts/build_native.py
```

Native dispatch is never automatic. The portable NumPy/BLAS path remains the default correctness reference; set `HELIAX_DISABLE_NATIVE=1` to force it.

See [`docs/pytorch-integration.md`](docs/pytorch-integration.md), [`docs/torch-compat.md`](docs/torch-compat.md), and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Run tests

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
pytest
```

Build a wheel:

```bash
pip install build
python -m build
```

## Project layout

```text
heliax/
├── src/heliax/
│   ├── tensor.py       # Tensor, graph, gradient traversal
│   ├── backend.py      # swappable numerical backend
│   ├── functional.py   # differentiable math and losses
│   ├── nn.py           # Module and neural building blocks
│   ├── optim.py        # optimizers and schedulers
│   ├── data.py         # batching, prefetch, distributed sampler
│   ├── distributed.py  # data-parallel helpers
│   ├── initialization.py # fan-aware initializers
│   ├── precision.py    # autocast policy and GradScaler
│   ├── profiler.py     # graph/memory/model diagnostics
│   ├── native_ops.py   # opt-in ctypes native kernels
│   ├── torch_interop.py # optional PyTorch accelerator facade
│   └── serialization.py
├── src/helianthus/     # compatibility namespace
├── third_party/pytorch # pinned BSD reference snapshot
├── tests/
├── examples/
├── docs/
└── pyproject.toml
```

## Roadmap

- Vectorized backward kernels with fewer temporary allocations.
- Optional native extension ABI and benchmark-driven fused ops.
- CPU inference quantization and mixed-precision scaffolding.
- Distributed tensor/process primitives.
- More layers: convolution, attention, normalization variants, and recurrent cells.
- Stable API and compatibility guarantees after the core settles.

## License

MIT — see [`LICENSE`](LICENSE).
