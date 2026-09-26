# PyTorch compatibility notes

Heliax keeps PyTorch optional. The `torch` extra enables explicit conversion and the `TorchAccelerator` facade; it never changes the default Heliax backend.

## Validated smoke path

In an isolated CPU environment, the following passed:

```text
python -m pytest -q tests/test_torch_interop.py
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

- `from_torch` detaches the PyTorch autograd graph by default.
- `to_torch` creates a CPU PyTorch tensor from Heliax storage.
- `TorchAccelerator` exposes explicit matmul, softmax, GELU, layer norm, and cross-entropy calls.
- A future gradient bridge must be opt-in and must not silently mix graphs.
