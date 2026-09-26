"""Weight quantization utilities for CPU inference experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

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


class QuantizedEmbedding(Module):
    """An inference-only embedding table backed by integer weights."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bits: int = 8,
        *,
        rng: np.random.Generator | None = None,
        dtype: Any = np.float32,
    ) -> None:
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("QuantizedEmbedding dimensions must be positive")
        generator = rng or np.random.default_rng()
        weight = generator.normal(size=(num_embeddings, embedding_dim)).astype(dtype) * dtype(
            np.sqrt(2.0 / embedding_dim)
        )
        packed = quantize(weight, bits=bits, symmetric=True)
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.bits = int(bits)
        self.register_buffer("weight_int8", packed.data)
        self.register_buffer("scale", np.asarray(packed.scale, dtype=np.float32))
        self.register_buffer("zero_point", np.asarray(packed.zero_point, dtype=np.int32))

    def forward(self, indices: Tensor | np.ndarray | list[int]) -> Tensor:
        index_data = (
            indices.numpy() if isinstance(indices, Tensor) else np.asarray(indices, dtype=np.int64)
        )
        if np.any(index_data < 0) or np.any(index_data >= self.num_embeddings):
            raise IndexError("embedding index out of range")
        weights = (
            self._buffers["weight_int8"][index_data].astype(np.float32)
            - int(self._buffers["zero_point"])
        ) * float(self._buffers["scale"])
        return Tensor(weights, requires_grad=False)


class QuantizedLinear(Module):
    """An inference-only linear module backed by integer weights."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 8,
        bias: bool = True,
        *,
        rng: np.random.Generator | None = None,
        dtype: Any = np.float32,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("QuantizedLinear dimensions must be positive")
        generator = rng or np.random.default_rng()
        weight = generator.normal(size=(out_features, in_features)).astype(dtype) * np.sqrt(
            2.0 / in_features
        )
        packed = quantize(weight, bits=bits, symmetric=True)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.bits = int(bits)
        self.register_buffer("weight_int8", packed.data)
        self.register_buffer("scale", np.asarray(packed.scale, dtype=np.float32))
        self.register_buffer("zero_point", np.asarray(packed.zero_point, dtype=np.int32))
        self.register_buffer(
            "bias", np.zeros(out_features, dtype=dtype) if bias else np.zeros(0, dtype=dtype)
        )

    def dequantized_weight(self) -> np.ndarray:
        return (
            self._buffers["weight_int8"].astype(np.float32) - int(self._buffers["zero_point"])
        ) * float(self._buffers["scale"])

    def forward(self, value: Tensor) -> Tensor:
        weight = Tensor(self.dequantized_weight(), requires_grad=False)
        output = value @ weight.transpose((1, 0))
        if self._buffers["bias"].size:
            output = output + Tensor(self._buffers["bias"], requires_grad=False)
        return output


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
