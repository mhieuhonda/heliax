# PyTorch reference snapshot

This directory contains a small, pinned snapshot of reference files from the upstream PyTorch repository.

- Upstream: https://github.com/pytorch/pytorch
- Commit: `8bcb4a343ff8e8e5e7c384614bf445164eada82b`
- Retrieved: 2026-09-25
- License: BSD-3-Clause; see `LICENSE` and `NOTICE` in this directory.

These files are vendored for design comparison and future native-kernel work. They are **not compiled by Heliax** and are not a replacement for PyTorch's build. Heliax implements its own vectorized NumPy kernels and optional PyTorch interop without importing these files at runtime.

Selected references cover:

- CPU softmax dispatch and last-dimension kernel declarations.
- GELU approximate/erf kernel declarations.
- Fused Adam interfaces.
- Quantized GELU reference.
- TorchScript/tensor-expr softmax and matmul operator structure.
- AdamW reference implementation.
- Softmax and matmul benchmark structure.

The copy is kept byte-for-byte at the pinned commit so provenance can be checked. Any Heliax adaptation belongs in `src/heliax/`, not in this directory.
