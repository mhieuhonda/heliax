# PyTorch compatibility notes

Heliax keeps PyTorch optional. The `torch` extra enables explicit conversion, the `TorchAccelerator` facade, and an opt-in `torch` numerical backend; it never changes the default Heliax backend.

```python
import heliax as hx

hx.set_backend("torch")       # explicit, never automatic
print(hx.backend_info())
```

## Validated smoke path

In an isolated CPU environment, the following passed:

```text
python -m pytest -q tests/test_torch_backend.py tests/test_torch_interop.py
python scripts/compare_torch.py
```

The optional conversion path is tested in a scheduled/manual workflow:

- `.github/workflows/torch-compat.yml`

## Benchmark interpretation

The first comparison run reported the following indicative numbers on the same container:

| operation | Heliax | PyTorch |
| --- | ---: | ---: |
| 512×512 float32 matmul | ~5.2 ms | ~12.6 ms |
| 1024×16 softmax | ~0.7 ms | ~5.7 ms |

These are not a claim of universal superiority. The workloads are small, CPU-only, measured through different dispatch layers, and the values vary with BLAS threading and container scheduling. Use `scripts/compare_torch.py` on the target machine before drawing conclusions.

## Conversion rules

- `from_torch` copies any source device into Heliax CPU storage and detaches the PyTorch autograd graph by default.
- `to_torch` accepts an explicit `device` (CPU, CUDA, MPS, ...) and copies Heliax storage there.
- `TorchAccelerator` exposes explicit matmul, softmax, GELU, layer norm, and cross-entropy calls.
- A future gradient bridge must be opt-in and must not silently mix graphs.
