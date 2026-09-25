# PyTorch integration and vendored references

Heliax uses a deliberate hybrid strategy instead of copying the whole PyTorch project.

## What is vendored

A small pinned reference snapshot lives in `third_party/pytorch/` at upstream commit `8bcb4a343ff8e8e5e7c384614bf445164eada82b`:

- CPU softmax declarations and softmax operator structure.
- GELU and fused-Adam declarations.
- Quantized GELU reference.
- TorchScript/tensor-expr softmax and matmul references.
- LayerNorm, log-softmax, and embedding reference interfaces.
- AdamW reference and operator benchmark structure.
- PyTorch `LICENSE` and `NOTICE`.

The files are unmodified and covered by PyTorch's BSD-3-Clause license. See `THIRD_PARTY_NOTICES.md` and `third_party/pytorch/README.md`.

The snapshot is **not compiled** by Heliax. Copying the full ATen/Caffe2/CUDA tree would add a large dependency surface without making the Python library more correct or faster. Instead, Heliax implements the ideas in its own NumPy backend and tests them directly.

## What was adapted

Heliax now includes:

- `fused_linear_bias` — one autograd node for projection plus bias.
- `FusedLinearGELU` — fused linear + GELU module.
- `masked_softmax` — stable masked softmax that safely handles fully masked rows.
- `scaled_dot_product_attention` — differentiable attention composition.
- `contiguous`, `clone`, and in-place tensor utilities for lower-allocation loops.
- `TorchAccelerator` — an explicit optional facade for PyTorch's optimized kernels.

The Helianthus namespace (`import helianthus`) re-exports the same core for projects that want to grow the library under that name.

## Optional native CPU path

The repository includes a small dependency-free C kernel file and an explicit opt-in loader:

```bash
python scripts/build_native.py
HELIAX_NATIVE=1 python -c 'import heliax as hx; print(hx.native_info())'
```

Native dispatch is **not automatic**: it is enabled only with `HELIAX_NATIVE=1`, and the portable NumPy/BLAS path remains the correctness and default reference. Run `python examples/native_benchmark.py` to measure whether a native kernel is actually faster on the target machine. `HELIAX_DISABLE_NATIVE=1` forces the fallback even when a shared library exists.## Optional PyTorch path

```bash
pip install 'heliax[torch]'
```

```python
import heliax as hx

if hx.torch_available():
    accelerated = hx.TorchAccelerator(device="cpu")
    print(accelerated.info())
```

PyTorch is never imported by default and Heliax does not silently replace its Tensor implementation. Conversions are explicit:

```python
heli = hx.from_torch(torch_cpu_tensor)
torch_cpu_tensor = hx.to_torch(heli, requires_grad=True)
```

## Performance boundary

The NumPy path is the correctness reference. The optional Torch path can be benchmarked against it, but a future native/Triton/Rust backend must beat the reference on a published benchmark before becoming the default. This keeps the library fast without hiding a dependency or copying an entire framework blindly.
