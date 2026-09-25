"""Weight quantization utilities for CPU inference experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .nn import Module
from .tensor import Tensor


@dataclass(frozen=True)
class QuantizedTensor:
    """Symmetric or affine integer tensor metadata plus its compact payload."""

    data: np.ndarray
    scale: float
    zero_point: int = 0
    bits: int = 8
    symmetric: bool = True

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes)

    def dequantize(self) -> np.ndarray:
        return (self.data.astype(np.float32) - self.zero_point) * self.scale


def quantize(
    value: Tensor | np.ndarray, *, bits: int = 8, symmetric: bool = True
) -> QuantizedTensor:
    if bits not in (4, 8):
        raise ValueError("Heliax quantization currently supports 4-bit and 8-bit payloads")
    data = value.numpy() if isinstance(value, Tensor) else np.asarray(value)
    if not np.issubdtype(data.dtype, np.floating):
        raise TypeError("quantization expects a floating-point tensor")
    if symmetric:
        levels = (1 << (bits - 1)) - 1
        scale = max(float(np.max(np.abs(data))) / levels, 1e-12)
        quantized = np.clip(np.rint(data / scale), -levels - 1, levels).astype(np.int8)
        return QuantizedTensor(quantized, scale, 0, bits, True)
    minimum = float(np.min(data))
    maximum = float(np.max(data))
    levels = (1 << bits) - 1
    scale = max((maximum - minimum) / levels, 1e-12)
    zero_point = round(-minimum / scale)
    quantized = np.clip(np.rint(data / scale) + zero_point, 0, levels).astype(np.uint8)
    return QuantizedTensor(quantized, scale, zero_point, bits, False)


def dequantize(value: QuantizedTensor) -> Tensor:
    return Tensor(value.dequantize(), requires_grad=False)


def quantize_state_dict(
    module: Module, *, bits: int = 8, symmetric: bool = True
) -> dict[str, QuantizedTensor]:
    return {
        name: quantize(value, bits=bits, symmetric=symmetric)
        for name, value in module.state_dict().items()
    }


def dequantize_state_dict(state: Mapping[str, QuantizedTensor]) -> dict[str, np.ndarray]:
    return {name: value.dequantize() for name, value in state.items()}


def compression_ratio(module: Module, *, bits: int = 8, symmetric: bool = True) -> float:
    original_bytes = sum(value.nbytes for value in module.state_dict().values())
    compressed_bytes = sum(
        value.nbytes + 32
        for value in quantize_state_dict(module, bits=bits, symmetric=symmetric).values()
    )
    return float(original_bytes / max(compressed_bytes, 1))
