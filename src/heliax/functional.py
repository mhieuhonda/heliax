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


def masked_softmax(value: Tensor, mask: Tensor | np.ndarray, axis: int = -1) -> Tensor:
    """Softmax with ``True`` mask entries excluded from the distribution."""

    mask_data = mask.numpy() if isinstance(mask, Tensor) else np.asarray(mask, dtype=bool)
    try:
        broadcast_mask = np.broadcast_to(mask_data, value.shape)
    except ValueError as error:
        raise ValueError(
            f"mask shape {mask_data.shape} is not broadcastable to {value.shape}"
        ) from error
    data = np.where(broadcast_mask, -np.inf, value.numpy())
    maximum = np.max(data, axis=axis, keepdims=True)
    maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    exponent = np.where(broadcast_mask, 0.0, np.exp(data - maximum))
    denominator = np.sum(exponent, axis=axis, keepdims=True, dtype=value.dtype)
    probabilities = np.divide(
        exponent,
        denominator,
        out=np.zeros_like(exponent, dtype=value.dtype),
        where=denominator > 0,
    )
    output = value._make(probabilities, (value,), lambda: None, "masked_softmax")
    if output.requires_grad:

        def run_backward() -> None:
            _accumulate(value, np.where(broadcast_mask, 0.0, output.grad.numpy()))

        output._backward = run_backward
    return output


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | np.ndarray | None = None,
    *,
    scale: float | None = None,
) -> Tensor:
    """Differentiable scaled dot-product attention over trailing dimensions."""

    if query.ndim < 2 or key.ndim < 2 or value.ndim < 2:
        raise ValueError("attention inputs need at least two dimensions")
    if query.shape[-1] != key.shape[-1] or key.shape[-2] != value.shape[-2]:
        raise ValueError("query/key/value shapes are incompatible")
    factor = float(scale if scale is not None else query.shape[-1] ** -0.5)
    scores = (
        query @ key.transpose(tuple(range(key.ndim - 2)) + (key.ndim - 1, key.ndim - 2))
    ) * factor
    weights = masked_softmax(scores, mask) if mask is not None else softmax(scores, axis=-1)
    return weights @ value


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


def fused_linear_bias(
    value: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    """Compute ``value @ weight.T + bias`` as one autograd node.

    Fusing the matmul and bias addition reduces graph nodes and temporary
    arrays on a hot inference/training path while keeping the same semantics
    as separate Tensor operations.
    """

    input_data = value.numpy()
    weight_data = weight.numpy()
    output_data = np.matmul(input_data, np.swapaxes(weight_data, -1, -2))
    if bias is not None:
        output_data = output_data + bias.numpy()
    parents = (value, weight) + ((bias,) if bias is not None else ())
    output = value._make(output_data, parents, lambda: None, "fused_linear_bias")
    if output.requires_grad:

        def run_backward() -> None:
            grad = output.grad.numpy()
            _accumulate(value, np.matmul(grad, weight_data))
            flat_grad = grad.reshape(-1, grad.shape[-1])
            flat_input = input_data.reshape(-1, input_data.shape[-1])
            _accumulate(weight, np.matmul(np.swapaxes(flat_grad, 0, 1), flat_input))
            if bias is not None:
                _accumulate(bias, np.sum(grad, axis=tuple(range(grad.ndim - 1)), dtype=np.float32))

        output._backward = run_backward
    return output


def fused_linear_gelu(
    value: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    """Fused linear projection plus the tanh approximation of GELU."""

    backend = get_backend()
    linear = fused_linear_bias(value, weight, bias)
    data = backend.gelu(linear.numpy())
    output = linear._make(data, (linear,), lambda: None, "fused_linear_gelu")
    if output.requires_grad:
        coefficient = np.sqrt(np.asarray(2.0 / np.pi, dtype=linear.dtype))

        def run_backward() -> None:
            x = linear.numpy()
            cubic = x**3
            inner = coefficient * (x + 0.044715 * cubic)
            tanh_value = np.tanh(inner)
            derivative = 0.5 * (1.0 + tanh_value) + 0.5 * x * (
                1.0 - tanh_value**2
            ) * coefficient * (1.0 + 3 * 0.044715 * x**2)
            _accumulate(linear, output.grad.numpy() * derivative)

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
