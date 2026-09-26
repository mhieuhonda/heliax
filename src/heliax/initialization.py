"""Parameter initialization helpers with fan-aware scaling."""

from __future__ import annotations

import math

import numpy as np

from .tensor import Parameter, Tensor, no_grad


def _fan(value: Tensor, mode: str) -> int:
    shape = value.shape
    if len(shape) < 2:
        return int(shape[0]) if shape else 1
    receptive = int(np.prod(shape[1:], dtype=np.int64))
    return receptive if mode == "fan_in" else int(shape[0]) * receptive


def _gain(nonlinearity: str) -> float:
    return {
        "linear": 1.0,
        "relu": math.sqrt(2.0),
        "leaky_relu": math.sqrt(2.0 / (1 + 0.01**2)),
        "tanh": 5.0 / 3.0,
    }.get(nonlinearity, 1.0)


def kaiming_uniform_(
    value: Parameter | Tensor,
    mode: str = "fan_in",
    nonlinearity: str = "relu",
    *,
    rng: np.random.Generator | None = None,
) -> Parameter | Tensor:
    generator = rng or np.random.default_rng()
    bound = _gain(nonlinearity) * math.sqrt(3.0 / max(1, _fan(value, mode)))
    with no_grad():
        value.data = generator.uniform(-bound, bound, size=value.shape).astype(value.dtype)
    return value


def xavier_uniform_(
    value: Parameter | Tensor, gain: float = 1.0, *, rng: np.random.Generator | None = None
) -> Parameter | Tensor:
    generator = rng or np.random.default_rng()
    fan_in, fan_out = _fan(value, "fan_in"), _fan(value, "fan_out")
    bound = gain * math.sqrt(6.0 / max(1, fan_in + fan_out))
    with no_grad():
        value.data = generator.uniform(-bound, bound, size=value.shape).astype(value.dtype)
    return value


def uniform_(
    value: Parameter | Tensor, bound: float = 0.0, *, rng: np.random.Generator | None = None
) -> Parameter | Tensor:
    generator = rng or np.random.default_rng()
    with no_grad():
        value.data = generator.uniform(-bound, bound, size=value.shape).astype(value.dtype)
    return value


def normal_(
    value: Parameter | Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    *,
    rng: np.random.Generator | None = None,
) -> Parameter | Tensor:
    generator = rng or np.random.default_rng()
    with no_grad():
        value.data = generator.normal(mean, std, size=value.shape).astype(value.dtype)
    return value


def zeros_(value: Parameter | Tensor) -> Parameter | Tensor:
    with no_grad():
        value.data = np.zeros_like(value.numpy())
    return value


def ones_(value: Parameter | Tensor) -> Parameter | Tensor:
    with no_grad():
        value.data = np.ones_like(value.numpy())
    return value
