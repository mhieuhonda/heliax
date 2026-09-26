"""Numerical backend boundary for Heliax.

The default backend is NumPy.  Keeping kernels behind this small interface makes
it possible to add native C++, Rust, Triton, or accelerator implementations
without changing the public Tensor/Module API.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import native_ops

DEFAULT_DTYPE = np.float32


def _normalize_dtype(dtype: Any | None) -> np.dtype:
    if dtype is None:
        return np.dtype(DEFAULT_DTYPE)
    return np.dtype(dtype)


def _as_array(value: Any, dtype: Any | None = None) -> np.ndarray:
    if isinstance(value, np.ndarray):
        array = value
    else:
        array = np.asarray(value)
    if dtype is None:
        dtype = array.dtype
    dtype = np.dtype(dtype)
    if array.dtype == dtype:
        return array
    return array.astype(dtype, copy=False)


def unbroadcast(gradient: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Reduce a NumPy-broadcast gradient back to ``shape``."""

    target = tuple(shape)
    if gradient.shape == target:
        return gradient
    if gradient.ndim == 0:
        return np.broadcast_to(gradient, target).copy()
    if not target:
        return np.asarray(gradient, dtype=DEFAULT_DTYPE).sum().reshape(())
    extra = len(gradient.shape) - len(target)
    if extra < 0:
        raise ValueError("gradient has fewer dimensions than its input")
    reduced = gradient
    for axis in range(extra):
        reduced = reduced.sum(axis=0)
    for axis, size in enumerate(target):
        if size == 1 and reduced.shape[axis] != 1:
            reduced = reduced.sum(axis=axis, keepdims=True)
    return np.asarray(reduced, dtype=gradient.dtype).reshape(target)


@dataclass
class NumpyBackend:
    """Vectorized reference implementation used by Heliax 0.1."""

    name: str = "numpy"
    supports_autograd: bool = True
    supports_fused_kernels: bool = False

    def array(self, value: Any, dtype: Any | None = None) -> np.ndarray:
        array = _as_array(value, dtype)
        if dtype is None and array.dtype == np.float64:
            return array.astype(DEFAULT_DTYPE, copy=False)
        return np.ascontiguousarray(array) if array.ndim > 0 else array

    def zeros(self, shape: tuple[int, ...] | int, dtype: Any = DEFAULT_DTYPE) -> np.ndarray:
        return np.zeros(shape, dtype=_normalize_dtype(dtype))

    def ones(self, shape: tuple[int, ...] | int, dtype: Any = DEFAULT_DTYPE) -> np.ndarray:
        return np.ones(shape, dtype=_normalize_dtype(dtype))

    def full(
        self, shape: tuple[int, ...] | int, fill_value: Any, dtype: Any = DEFAULT_DTYPE
    ) -> np.ndarray:
        return np.full(shape, fill_value, dtype=_normalize_dtype(dtype))

    def arange(
        self, start: int, stop: int | None = None, step: int = 1, dtype: Any = DEFAULT_DTYPE
    ) -> np.ndarray:
        return np.arange(start, stop, step, dtype=_normalize_dtype(dtype))

    def randn(
        self, *shape: int, rng: np.random.Generator | None = None, dtype: Any = DEFAULT_DTYPE
    ) -> np.ndarray:
        generator = rng or np.random.default_rng()
        return generator.standard_normal(shape).astype(_normalize_dtype(dtype), copy=False)

    def matmul(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.matmul(left, right)

    def exp(self, value: np.ndarray) -> np.ndarray:
        return np.exp(value, dtype=DEFAULT_DTYPE)

    def log(self, value: np.ndarray) -> np.ndarray:
        return np.log(value, dtype=DEFAULT_DTYPE)

    def sqrt(self, value: np.ndarray) -> np.ndarray:
        return np.sqrt(value, dtype=DEFAULT_DTYPE)

    def tanh(self, value: np.ndarray) -> np.ndarray:
        return np.tanh(value)

    def sigmoid(self, value: np.ndarray) -> np.ndarray:
        # Stable sigmoid for float32 inference and training.
        output = np.empty_like(value, dtype=DEFAULT_DTYPE)
        positive = value >= 0
        output[positive] = 1.0 / (1.0 + np.exp(-value[positive], dtype=DEFAULT_DTYPE))
        exp_value = np.exp(value[~positive], dtype=DEFAULT_DTYPE)
        output[~positive] = exp_value / (1.0 + exp_value)
        return output

    def relu(self, value: np.ndarray) -> np.ndarray:
        return np.maximum(value, 0, dtype=DEFAULT_DTYPE)

    def add_relu(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        if (
            left.dtype == np.float32
            and right.dtype == np.float32
            and native_ops.native_enabled()
            and native_ops.native_available()
        ):
            try:
                return native_ops.add_relu(left, right)
            except (RuntimeError, ValueError):
                pass
        return np.maximum(left + right, 0, dtype=DEFAULT_DTYPE)

    def gelu(self, value: np.ndarray) -> np.ndarray:
        if (
            value.dtype == np.float32
            and native_ops.native_enabled()
            and native_ops.native_available()
        ):
            try:
                return native_ops.gelu(value)
            except RuntimeError:
                pass
        # tanh approximation; deterministic and differentiable in the graph.
        coefficient = np.sqrt(np.asarray(2.0 / np.pi, dtype=value.dtype))
        cubic = value * value * value
        return 0.5 * value * (1.0 + np.tanh(coefficient * (value + 0.044715 * cubic)))

    def silu(self, value: np.ndarray) -> np.ndarray:
        return value * self.sigmoid(value)

    def softmax(self, value: np.ndarray, axis: int = -1) -> np.ndarray:
        if (
            value.ndim >= 1
            and axis in {-1, value.ndim - 1}
            and value.dtype == np.float32
            and native_ops.native_enabled()
            and native_ops.native_available()
        ):
            try:
                return native_ops.softmax_lastdim(value)
            except RuntimeError:
                pass
        shifted = value - np.max(value, axis=axis, keepdims=True)
        exponent = np.exp(shifted, dtype=DEFAULT_DTYPE)
        return exponent / np.sum(exponent, axis=axis, keepdims=True, dtype=DEFAULT_DTYPE)

    def log_softmax(self, value: np.ndarray, axis: int = -1) -> np.ndarray:
        if (
            value.ndim >= 1
            and value.dtype == np.float32
            and native_ops.native_enabled()
            and native_ops.native_available()
        ):
            try:
                moved = np.moveaxis(value, axis, -1)
                result = native_ops.log_softmax_lastdim(np.ascontiguousarray(moved))
                return np.moveaxis(result, -1, axis)
            except (RuntimeError, ValueError):
                pass
        shifted = value - np.max(value, axis=axis, keepdims=True)
        return shifted - np.log(
            np.sum(
                np.exp(shifted, dtype=DEFAULT_DTYPE), axis=axis, keepdims=True, dtype=DEFAULT_DTYPE
            )
        )

    def layernorm(
        self,
        value: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray,
        axes: tuple[int, ...],
        eps: float,
    ) -> np.ndarray:
        if (
            len(axes) == 1
            and axes[0] == value.ndim - 1
            and value.dtype == np.float32
            and native_ops.native_enabled()
            and native_ops.native_available()
        ):
            try:
                return native_ops.layernorm_lastdim(value, weight, bias, eps)
            except (RuntimeError, ValueError):
                pass
        mean = np.mean(value, axis=axes, keepdims=True, dtype=DEFAULT_DTYPE)
        variance = np.mean((value - mean) ** 2, axis=axes, keepdims=True, dtype=DEFAULT_DTYPE)
        normalized = (value - mean) / np.sqrt(variance + eps, dtype=DEFAULT_DTYPE)
        return normalized * weight + bias

    def embedding(self, indices: np.ndarray, weight: np.ndarray) -> np.ndarray:
        return weight[indices.astype(np.int64, copy=False)]

    def cross_entropy(self, logits: np.ndarray, targets: np.ndarray, axis: int = -1) -> np.ndarray:
        log_probs = self.log_softmax(logits, axis=axis)
        if targets.ndim == logits.ndim - 1:
            flat_targets = targets.reshape(-1).astype(np.int64, copy=False)
        else:
            flat_targets = targets.astype(np.int64, copy=False)
        return -np.take_along_axis(
            log_probs, flat_targets.reshape(logits.shape[:-1] + (1,)), axis=axis
        ).reshape(-1)

    def adamw_update(
        self,
        parameter: np.ndarray,
        gradient: np.ndarray,
        first_moment: np.ndarray,
        second_moment: np.ndarray,
        learning_rate: float,
        beta1: float,
        beta2: float,
        epsilon: float,
        weight_decay: float,
        bias_correction1: float,
        bias_correction2: float,
    ) -> None:
        if (
            parameter.dtype == np.float32
            and gradient.dtype == np.float32
            and native_ops.native_enabled()
            and native_ops.native_available()
        ):
            try:
                native_ops.adamw(
                    parameter,
                    gradient,
                    first_moment,
                    second_moment,
                    learning_rate=learning_rate,
                    beta1=beta1,
                    beta2=beta2,
                    epsilon=epsilon,
                    weight_decay=weight_decay,
                    bias_correction1=bias_correction1,
                    bias_correction2=bias_correction2,
                )
                return
            except (RuntimeError, ValueError):
                pass
        first_moment *= beta1
        first_moment += (1.0 - beta1) * gradient
        second_moment *= beta2
        second_moment += (1.0 - beta2) * (gradient * gradient)
        first_hat = first_moment / bias_correction1
        second_hat = second_moment / bias_correction2
        parameter -= learning_rate * (
            first_hat / (np.sqrt(second_hat) + epsilon) + weight_decay * parameter
        )

    def sgd_update(
        self,
        parameter: np.ndarray,
        gradient: np.ndarray,
        learning_rate: float,
        weight_decay: float,
        momentum_buffer: np.ndarray | None,
        momentum: float,
        nesterov: bool,
    ) -> np.ndarray | None:
        update = gradient + weight_decay * parameter
        if momentum_buffer is None:
            momentum_buffer = np.zeros_like(parameter, dtype=DEFAULT_DTYPE)
        if nesterov:
            momentum_buffer = momentum * momentum_buffer + update
            update = update + momentum * momentum_buffer
        else:
            momentum_buffer = momentum * momentum_buffer + update
            update = momentum_buffer
        parameter -= learning_rate * update
        return momentum_buffer

    def rmsprop_update(
        self,
        parameter: np.ndarray,
        gradient: np.ndarray,
        square_average: np.ndarray,
        learning_rate: float,
        decay: float,
        epsilon: float,
    ) -> None:
        square_average *= decay
        square_average += (1.0 - decay) * (gradient * gradient)
        parameter -= learning_rate * gradient / (np.sqrt(square_average) + epsilon)

    def info(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "numpy": np.__version__,
            "supports_fused_kernels": native_ops.native_enabled() and native_ops.native_available(),
            "native": native_ops.native_info(),
            "threading": "NumPy/BLAS delegated",
        }


_BACKENDS: dict[str, NumpyBackend] = {"numpy": NumpyBackend()}
_ACTIVE_BACKEND = "numpy"


def register_backend(name: str, backend: NumpyBackend) -> None:
    _BACKENDS[name] = backend


def set_backend(name: str) -> None:
    global _ACTIVE_BACKEND
    if name == "torch" and name not in _BACKENDS:
        from .torch_backend import get_torch_backend

        register_backend(name, get_torch_backend())
    elif name.startswith("torch:") and name not in _BACKENDS:
        from .torch_backend import get_torch_backend

        register_backend(name, get_torch_backend(name.split(":", 1)[1]))
    if name not in _BACKENDS:
        raise KeyError(f"Unknown Heliax backend: {name}")
    _ACTIVE_BACKEND = name


def get_backend(name: str | None = None) -> NumpyBackend:
    return _BACKENDS[name or _ACTIVE_BACKEND]


def available_backends() -> tuple[str, ...]:
    return tuple(sorted(_BACKENDS))


def backend_info() -> Mapping[str, Any]:
    return get_backend().info()
