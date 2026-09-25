"""Differentiable functions and losses."""

from __future__ import annotations

from typing import Any

import numpy as np

from .backend import get_backend
from .tensor import Parameter, Tensor, _accumulate


def relu(value: Tensor) -> Tensor:
    return value.relu()


def gelu(value: Tensor) -> Tensor:
    return value.gelu()


def silu(value: Tensor) -> Tensor:
    return value.silu()


def sigmoid(value: Tensor) -> Tensor:
    return value.sigmoid()


def tanh(value: Tensor) -> Tensor:
    return value.tanh()


def softmax(value: Tensor, axis: int = -1) -> Tensor:
    backend = get_backend()
    data = backend.softmax(value.numpy(), axis=axis)
    output = value._make(data, (value,), lambda: None, "softmax")
    if output.requires_grad:

        def run_backward() -> None:
            grad = output.grad.numpy()
            weighted = np.sum(grad * data, axis=axis, keepdims=True, dtype=data.dtype)
            _accumulate(value, (data * (grad - weighted)).astype(value.dtype, copy=False))

        output._backward = run_backward
    return output


def log_softmax(value: Tensor, axis: int = -1) -> Tensor:
    backend = get_backend()
    data = backend.log_softmax(value.numpy(), axis=axis)
    output = value._make(data, (value,), lambda: None, "log_softmax")
    if output.requires_grad:

        def run_backward() -> None:
            grad = output.grad.numpy()
            weighted = np.sum(grad, axis=axis, keepdims=True, dtype=data.dtype)
            _accumulate(value, grad - np.exp(data) * weighted)

        output._backward = run_backward
    return output


def mean_squared_error(prediction: Tensor, target: Any) -> Tensor:
    target_data = (
        target.numpy() if isinstance(target, Tensor) else np.asarray(target, dtype=prediction.dtype)
    )
    return (prediction - target_data).square().mean()


def mean_absolute_error(prediction: Tensor, target: Any) -> Tensor:
    target_data = (
        target.numpy() if isinstance(target, Tensor) else np.asarray(target, dtype=prediction.dtype)
    )
    difference = prediction - target_data
    output = prediction._make(np.abs(difference.numpy()), (prediction,), lambda: None, "abs")
    if output.requires_grad:

        def run_backward() -> None:
            _accumulate(
                prediction, output.grad.numpy() * np.sign(difference.numpy()) / prediction.size
            )

        output._backward = run_backward
    return output.mean()


def binary_cross_entropy(logits: Tensor, target: Any, from_logits: bool = True) -> Tensor:
    target_data = (
        target.numpy() if isinstance(target, Tensor) else np.asarray(target, dtype=logits.dtype)
    )
    data = logits.numpy()
    if from_logits:
        loss_data = np.maximum(data, 0) - data * target_data + np.log1p(np.exp(-np.abs(data)))
        derivative = 1.0 / (1.0 + np.exp(-data)) - target_data
    else:
        probabilities = np.clip(data, 1e-7, 1.0 - 1e-7)
        loss_data = -(
            target_data * np.log(probabilities) + (1.0 - target_data) * np.log(1.0 - probabilities)
        )
        derivative = (probabilities - target_data) / (probabilities * (1.0 - probabilities))
    output = logits._make(
        np.asarray(loss_data, dtype=logits.dtype), (logits,), lambda: None, "binary_cross_entropy"
    )
    if output.requires_grad:

        def run_backward() -> None:
            _accumulate(logits, output.grad.numpy() * derivative / logits.size)

        output._backward = run_backward
    return output.mean()


def cross_entropy(logits: Tensor, target: Any, axis: int = -1) -> Tensor:
    target_data = target.numpy() if isinstance(target, Tensor) else np.asarray(target)
    if target_data.ndim == logits.ndim:
        if target_data.shape[-1] != logits.shape[-1] or target_data.shape[:-1] != logits.shape[:-1]:
            raise ValueError("one-hot cross_entropy targets must match logits shape")
        target_data = np.argmax(target_data, axis=axis)
    if logits.ndim == 0:
        raise ValueError("cross_entropy expects a logits tensor with a class dimension")
    if target_data.shape != logits.shape[:-1]:
        target_data = target_data.reshape(logits.shape[:-1])
    if not np.issubdtype(target_data.dtype, np.integer):
        rounded = np.rint(target_data)
        if not np.allclose(target_data, rounded):
            raise ValueError("class targets must be integer indices")
        target_data = rounded.astype(np.int64)
    log_probs = log_softmax(logits, axis=axis)
    selected = np.take_along_axis(
        log_probs.numpy(), target_data.astype(np.int64)[..., np.newaxis], axis=axis
    )
    output_data = -selected.mean()
    output = logits._make(
        np.asarray(output_data, dtype=logits.dtype), (logits,), lambda: None, "cross_entropy"
    )
    if output.requires_grad:

        def run_backward() -> None:
            probabilities = get_backend().softmax(logits.numpy(), axis=axis)
            probabilities /= max(int(target_data.size), 1)
            target_index = target_data.astype(np.int64)[..., np.newaxis]
            selected_probability = np.take_along_axis(probabilities, target_index, axis=axis)
            np.put_along_axis(
                probabilities,
                target_index,
                selected_probability - (1.0 / max(int(target_data.size), 1)),
                axis=axis,
            )
            _accumulate(logits, probabilities)

        output._backward = run_backward
    return output


def embedding(indices: Tensor | np.ndarray | list[int], weight: Parameter | Tensor) -> Tensor:
    index_data = (
        indices.numpy() if isinstance(indices, Tensor) else np.asarray(indices, dtype=np.int64)
    )
    data = get_backend().embedding(index_data, weight.numpy())
    output = weight._make(data, (weight,), lambda: None, "embedding")
    if output.requires_grad:

        def run_backward() -> None:
            gradient = np.zeros_like(weight.numpy(), dtype=np.float32)
            np.add.at(gradient, index_data.astype(np.int64), output.grad.numpy())
            _accumulate(weight, gradient)

        output._backward = run_backward
    return output


def dropout(
    value: Tensor,
    probability: float = 0.5,
    *,
    training: bool = True,
    rng: np.random.Generator | None = None,
) -> Tensor:
    if not training or probability <= 0.0:
        return value
    if probability >= 1.0:
        raise ValueError("dropout probability must be in [0, 1)")
    generator = rng or np.random.default_rng()
    mask = (generator.random(value.shape) >= probability).astype(value.dtype) / (1.0 - probability)
    return (
        value._make(value.numpy() * mask, (value,), lambda: None, "dropout")
        if not value.requires_grad
        else value * Tensor(mask, requires_grad=False)
    )


def clip(value: Tensor, minimum: Any, maximum: Any) -> Tensor:
    minimum_data = (
        minimum.numpy() if isinstance(minimum, Tensor) else np.asarray(minimum, dtype=value.dtype)
    )
    maximum_data = (
        maximum.numpy() if isinstance(maximum, Tensor) else np.asarray(maximum, dtype=value.dtype)
    )
    data = np.clip(value.numpy(), minimum_data, maximum_data)
    output = value._make(data, (value,), lambda: None, "clip")
    if output.requires_grad:
        mask = ((value.numpy() >= minimum_data) & (value.numpy() <= maximum_data)).astype(
            value.dtype
        )

        def run_backward() -> None:
            _accumulate(value, output.grad.numpy() * mask)

        output._backward = run_backward
    return output


def concatenate(values: list[Tensor], axis: int = 0) -> Tensor:
    if not values:
        raise ValueError("concatenate requires at least one tensor")
    data = np.concatenate([value.numpy() for value in values], axis=axis)
    output = values[0]._make(data, tuple(values), lambda: None, "concatenate")
    if output.requires_grad:

        def run_backward() -> None:
            offset = 0
            for value in values:
                size = value.shape[axis]
                _accumulate(
                    value, output.grad.numpy().take(range(offset, offset + size), axis=axis)
                )
                offset += size

        output._backward = run_backward
    return output


def stack(values: list[Tensor], axis: int = 0) -> Tensor:
    if not values:
        raise ValueError("stack requires at least one tensor")
    data = np.stack([value.numpy() for value in values], axis=axis)
    output = values[0]._make(data, tuple(values), lambda: None, "stack")
    if output.requires_grad:

        def run_backward() -> None:
            for index, value in enumerate(values):
                key = [slice(None)] * output.ndim
                key[axis] = index
                _accumulate(value, output.grad.numpy()[tuple(key)])

        output._backward = run_backward
    return output
