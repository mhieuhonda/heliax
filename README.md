# Heliax

> **A performance-first tensor, autodiff, and neural network library for Python.**

Heliax is a new, original Python deep-learning library built around a simple promise: keep the numerical work close to NumPy/BLAS, make the graph explicit, and leave room for a native compiler later.

Heliax does **not** claim to beat PyTorch on every workload. PyTorch has years of ecosystem investment and a mature CUDA/distributed stack. Heliax is being built as a focused alternative for people who want a readable core, a small dependency surface, and a path toward fused/native kernels.

## What is implemented in 0.1

- N-dimensional `Tensor` with broadcasting, views, dtype conversion, and in-place-free operations.
- Reverse-mode automatic differentiation with graph traversal, gradient accumulation, `no_grad`, and `gradcheck`.
- `Linear`, `Conv2d`, `LayerNorm`, `Embedding`, `MultiheadAttention`, `ReLU`, `GELU`, `SiLU`, `Sigmoid`, `Tanh`, `Dropout`, and `Sequential`.
- `SGD`, `Nesterov SGD`, `Adam`, `AdamW`, and `RMSProp`; gradient clipping; cosine/step schedules.
- MSE, cross-entropy, binary cross-entropy, softmax, log-softmax, and functional APIs.
- 4-bit/8-bit symmetric and affine weight quantization with compression reporting.
- NPZ state-dict/checkpoint serialization and a small training loop helper.
- Backend registry and environment reporting. NumPy is the default; a native backend can be added without changing the public Tensor API.
- Focused tests, gradient checks, a benchmark script, packaging, and CI.

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
│   ├── data.py         # batching and seed utilities
│   ├── profiler.py     # lightweight operation profiler
│   └── serialization.py
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
