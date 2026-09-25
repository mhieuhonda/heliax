"""Benchmark the optional native kernels against the portable NumPy path."""

from __future__ import annotations

from time import perf_counter

import numpy as np

import heliax as hx


def timed(function, repeats: int = 30) -> float:
    function()
    started = perf_counter()
    for _ in range(repeats):
        function()
    return (perf_counter() - started) * 1000.0 / repeats


def numpy_gelu(value: np.ndarray) -> np.ndarray:
    coefficient = np.sqrt(np.asarray(2.0 / np.pi, dtype=value.dtype))
    cubic = value * value * value
    return 0.5 * value * (1.0 + np.tanh(coefficient * (value + 0.044715 * cubic)))


def numpy_softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - np.max(value, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=-1, keepdims=True)


def main() -> None:
    print("native:", hx.native_info())
    if not hx.native_available():
        print("Build it first with: python scripts/build_native.py")
        return
    rng = np.random.default_rng(0)
    array = rng.normal(size=(256, 256)).astype(np.float32)
    print(f"add_relu native: {timed(lambda: hx.native_ops.add_relu(array, array)):.3f} ms")
    print(f"add_relu numpy:  {timed(lambda: np.maximum(array + array, 0)):.3f} ms")
    print(f"gelu native:     {timed(lambda: hx.native_ops.gelu(array)):.3f} ms")
    print(f"gelu numpy:      {timed(lambda: numpy_gelu(array)):.3f} ms")
    print(f"softmax native:  {timed(lambda: hx.native_ops.softmax_lastdim(array)):.3f} ms")
    print(f"softmax numpy:   {timed(lambda: numpy_softmax(array)):.3f} ms")


if __name__ == "__main__":
    main()
