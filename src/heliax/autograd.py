"""Finite-difference gradient checks for Heliax graphs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from .tensor import Tensor, no_grad


def gradcheck(
    function: Callable[..., Tensor | float],
    inputs: Tensor | list[Tensor] | tuple[Tensor, ...],
    *,
    eps: float = 1e-3,
    rtol: float = 2e-2,
    atol: float = 2e-2,
) -> Mapping[str, Any]:
    """Compare analytical gradients with a central finite difference."""

    tensors = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]
    for value in tensors:
        if not value.requires_grad:
            raise ValueError("gradcheck inputs must require gradients")
    for value in tensors:
        value.zero_grad()
    output = function(*tensors)
    if not isinstance(output, Tensor):
        output = Tensor(output)
    if output.size != 1:
        raise ValueError("gradcheck function must return a scalar Tensor")
    output.backward()
    analytical = [None if value.grad is None else value.grad.numpy().copy() for value in tensors]
    numerical: list[np.ndarray] = []
    with no_grad():
        for tensor in tensors:
            original = tensor.numpy().copy()
            perturbation = np.zeros_like(original, dtype=np.float64)
            for index in np.ndindex(*original.shape):
                plus = original.copy()
                minus = original.copy()
                plus[index] += eps
                minus[index] -= eps
                tensor._data = tensor._data.dtype.type(plus)
                plus_value = float(function(*tensors).item())
                tensor._data = tensor._data.dtype.type(minus)
                minus_value = float(function(*tensors).item())
                perturbation[index] = (plus_value - minus_value) / (2.0 * eps)
            tensor._data = tensor._data.dtype.type(original)
            numerical.append(perturbation)
    errors = []
    for expected, actual in zip(analytical, numerical):
        if expected is None:
            errors.append(np.inf)
        else:
            errors.append(float(np.max(np.abs(expected - actual))))
    max_error = max(errors) if errors else 0.0
    return {
        "max_error": max_error,
        "errors": tuple(errors),
        "passed": bool(
            max_error
            <= atol + rtol * max((float(np.max(np.abs(value))) for value in numerical), default=0.0)
        ),
    }


def checkpoint(function: Callable[..., Tensor], *args: Tensor, **kwargs: Any) -> Tensor:
    """Trade compute for memory by recomputing a function during backward."""

    requires_grad = any(
        isinstance(value, Tensor) and value.requires_grad for value in (*args, *kwargs.values())
    )
    with no_grad():
        forward = function(*args, **kwargs)
    if not isinstance(forward, Tensor):
        raise TypeError("checkpointed functions must return a Tensor")
    result = Tensor(forward.numpy().copy(), requires_grad=requires_grad)

    if requires_grad:

        def run_backward() -> None:
            from .tensor import enable_grad

            with enable_grad():
                recomputed = function(*args, **kwargs)
                recomputed.backward(result.grad.numpy())

        result._backward = run_backward
        result._op = "checkpoint"
    return result
