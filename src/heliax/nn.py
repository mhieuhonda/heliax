"""Neural network modules built on Heliax tensors."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from . import functional as F
from .backend import DEFAULT_DTYPE, get_backend
from .tensor import Parameter, Tensor, _accumulate


class Module:
    """Minimal tree-aware module base with state-dict support."""

    training = True

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        return self.forward(*args, **kwargs)

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, Module]]:
        yield prefix, self
        for name, value in list(self.__dict__.items()):
            if isinstance(value, Module):
                child_prefix = f"{prefix}.{name}" if prefix else name
                yield from value.named_modules(child_prefix)
            elif isinstance(value, (list, tuple)):
                for index, child in enumerate(value):
                    if isinstance(child, Module):
                        child_prefix = f"{prefix}.{name}.{index}" if prefix else f"{name}.{index}"
                        yield from child.named_modules(child_prefix)
            elif isinstance(value, dict):
                for key, child in value.items():
                    if isinstance(child, Module):
                        child_prefix = f"{prefix}.{name}.{key}" if prefix else f"{name}.{key}"
                        yield from child.named_modules(child_prefix)

    def modules(self) -> Iterator[Module]:
        for _, module in self.named_modules():
            yield module

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        seen: set[int] = set()
        for module_name, module in self.named_modules(prefix):
            for name, value in list(module.__dict__.items()):
                if isinstance(value, Parameter) and id(value) not in seen:
                    seen.add(id(value))
                    full_name = f"{module_name}.{name}" if module_name else name
                    yield full_name, value

    def parameters(self) -> list[Parameter]:
        return [parameter for _, parameter in self.named_parameters()]

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def state_dict(self) -> dict[str, np.ndarray]:
        return {name: parameter.numpy().copy() for name, parameter in self.named_parameters()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = set(self.state_dict())
        received = set(state)
        missing = expected - received
        unexpected = received - expected
        if missing or unexpected:
            raise ValueError(
                f"state_dict mismatch; missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        for name, parameter in self.named_parameters():
            parameter.data = np.asarray(state[name])

    def train(self, mode: bool = True) -> Module:
        self.training = mode
        for module in self.modules():
            module.training = mode
        return self

    def eval(self) -> Module:
        return self.train(False)

    def dtype(self, dtype: Any) -> Module:
        for parameter in self.parameters():
            parameter.data = parameter.numpy().astype(dtype, copy=False)
        return self

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        rng: np.random.Generator | None = None,
        dtype: Any = DEFAULT_DTYPE,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Linear dimensions must be positive")
        generator = rng or np.random.default_rng()
        scale = np.sqrt(2.0 / in_features)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = Parameter(
            generator.standard_normal((out_features, in_features)).astype(dtype) * dtype(scale)
        )
        self.bias = Parameter(np.zeros(out_features, dtype=dtype)) if bias else None

    def forward(self, value: Tensor) -> Tensor:
        if value.shape[-1] != self.in_features:
            raise ValueError(
                f"Linear expected last dimension {self.in_features}, got {value.shape[-1] if value.shape else None}"
            )
        output = value @ self.weight.transpose((1, 0))
        if self.bias is not None:
            output = output + self.bias
        return output


class LayerNorm(Module):
    def __init__(
        self,
        normalized_shape: int | Sequence[int],
        eps: float = 1e-5,
        *,
        dtype: Any = DEFAULT_DTYPE,
    ) -> None:
        super().__init__()
        shape = (
            (int(normalized_shape),)
            if isinstance(normalized_shape, int)
            else tuple(normalized_shape)
        )
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("normalized_shape must contain positive dimensions")
        self.normalized_shape = shape
        self.eps = float(eps)
        self.weight = Parameter(np.ones(shape, dtype=dtype))
        self.bias = Parameter(np.zeros(shape, dtype=dtype))

    def forward(self, value: Tensor) -> Tensor:
        if tuple(value.shape[-len(self.normalized_shape) :]) != self.normalized_shape:
            raise ValueError(
                f"LayerNorm expected trailing shape {self.normalized_shape}, got {value.shape}"
            )
        axes = tuple(range(value.ndim - len(self.normalized_shape), value.ndim))
        data = value.numpy()
        mean = np.mean(data, axis=axes, keepdims=True, dtype=np.float32)
        variance = np.mean((data - mean) ** 2, axis=axes, keepdims=True, dtype=np.float32)
        normalized = (data - mean) / np.sqrt(variance + self.eps)
        output_data = get_backend().layernorm(
            data, self.weight.numpy(), self.bias.numpy(), axes, self.eps
        )
        output = value._make(
            output_data, (value, self.weight, self.bias), lambda: None, "layer_norm"
        )
        if output.requires_grad:

            def run_backward() -> None:
                grad = output.grad.numpy()
                gamma = self.weight.numpy()
                batch_axes = tuple(range(value.ndim - len(self.normalized_shape)))
                size = (
                    max(1, np.prod([value.shape[axis] for axis in batch_axes])) if batch_axes else 1
                )
                weighted = grad * gamma
                mean_weighted = (
                    np.mean(weighted, axis=batch_axes, keepdims=True, dtype=np.float32)
                    if batch_axes
                    else weighted
                )
                mean_weighted_normalized = (
                    np.mean(weighted * normalized, axis=batch_axes, keepdims=True, dtype=np.float32)
                    if batch_axes
                    else weighted * normalized
                )
                grad_value = (
                    weighted - mean_weighted - normalized * mean_weighted_normalized
                ) / size
                grad_weight = (
                    np.sum(grad, axis=batch_axes, dtype=np.float32) if batch_axes else grad
                )
                grad_bias = np.sum(grad, axis=batch_axes, dtype=np.float32) if batch_axes else grad
                _accumulate(value, grad_value)
                _accumulate(self.weight, grad_weight)
                _accumulate(self.bias, grad_bias)

            output._backward = run_backward
        return output


class Embedding(Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        rng: np.random.Generator | None = None,
        dtype: Any = DEFAULT_DTYPE,
    ) -> None:
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("Embedding dimensions must be positive")
        generator = rng or np.random.default_rng()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.weight = Parameter(
            generator.standard_normal((num_embeddings, embedding_dim)).astype(dtype) * dtype(0.05)
        )

    def forward(self, indices: Tensor | np.ndarray | list[int]) -> Tensor:
        return F.embedding(indices, self.weight)


class Conv2d(Module):
    """A small im2col convolution for NCHW tensors (groups=1)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = True,
        *,
        rng: np.random.Generator | None = None,
        dtype: Any = DEFAULT_DTYPE,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Conv2d channels must be positive")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        )
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
        if len(self.kernel_size) != 2 or len(self.stride) != 2 or len(self.padding) != 2:
            raise ValueError("Conv2d kernel, stride, and padding must have two dimensions")
        generator = rng or np.random.default_rng()
        fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.weight = Parameter(
            generator.standard_normal((out_channels, in_channels, *self.kernel_size)).astype(dtype)
            * dtype(np.sqrt(2.0 / fan_in))
        )
        self.bias = Parameter(np.zeros(out_channels, dtype=dtype)) if bias else None

    def _dimensions(self, value: Tensor) -> tuple[int, int, int, int]:
        if value.ndim != 4:
            raise ValueError(f"Conv2d expects NCHW input, got shape {value.shape}")
        batch, channels, height, width = value.shape
        if channels != self.in_channels:
            raise ValueError(f"Conv2d expected {self.in_channels} channels, got {channels}")
        padded_height = height + 2 * self.padding[0]
        padded_width = width + 2 * self.padding[1]
        output_height = (padded_height - self.kernel_size[0]) // self.stride[0] + 1
        output_width = (padded_width - self.kernel_size[1]) // self.stride[1] + 1
        if output_height <= 0 or output_width <= 0:
            raise ValueError("Conv2d kernel is larger than the padded input")
        return batch, output_height, output_width, channels

    def _columns(self, data: np.ndarray) -> np.ndarray:
        batch, _, height, width = data.shape
        output_height = (height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        output_width = (width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        padded = np.pad(
            data,
            (
                (0, 0),
                (0, 0),
                (self.padding[0], self.padding[0]),
                (self.padding[1], self.padding[1]),
            ),
        )
        columns = np.empty(
            (
                batch,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
                output_height,
                output_width,
            ),
            dtype=data.dtype,
        )
        for row in range(self.kernel_size[0]):
            for column in range(self.kernel_size[1]):
                patch = padded[
                    :,
                    :,
                    row : row + (output_height - 1) * self.stride[0] + 1 : self.stride[0],
                    column : column + (output_width - 1) * self.stride[1] + 1 : self.stride[1],
                ]
                columns[:, :, row, column] = patch
        return columns.reshape(
            batch,
            self.in_channels * self.kernel_size[0] * self.kernel_size[1],
            output_height * output_width,
        )

    def forward(self, value: Tensor) -> Tensor:
        batch, output_height, output_width, _ = self._dimensions(value)
        columns = self._columns(value.numpy())
        weight = self.weight.numpy().reshape(self.out_channels, -1)
        output = (
            np.matmul(weight, columns)
            .reshape(self.out_channels, batch, output_height, output_width)
            .transpose(1, 0, 2, 3)
        )
        if self.bias is not None:
            output = output + self.bias.reshape(1, -1, 1, 1)
        parents = (value, self.weight) + ((self.bias,) if self.bias is not None else ())
        result = value._make(output, parents, lambda: None, "conv2d")
        if result.requires_grad:

            def run_backward() -> None:
                gradient = result.grad.numpy()
                grad_output = gradient.transpose(1, 0, 2, 3).reshape(self.out_channels, batch, -1)
                current_columns = self._columns(value.numpy())
                weight_flat = self.weight.numpy().reshape(self.out_channels, -1)
                grad_weight = np.einsum("obl,bfl->of", grad_output, current_columns).reshape(
                    self.weight.shape
                )
                grad_columns = np.einsum("obl,of->bfl", grad_output, weight_flat).reshape(
                    batch,
                    self.in_channels,
                    self.kernel_size[0],
                    self.kernel_size[1],
                    output_height,
                    output_width,
                )
                _, _, input_height, input_width = value.shape
                grad_padded = np.zeros(
                    (
                        batch,
                        self.in_channels,
                        input_height + 2 * self.padding[0],
                        input_width + 2 * self.padding[1],
                    ),
                    dtype=value.dtype,
                )
                for channel in range(self.in_channels):
                    for row in range(self.kernel_size[0]):
                        for column in range(self.kernel_size[1]):
                            grad_padded[
                                :,
                                channel,
                                row : row + (output_height - 1) * self.stride[0] + 1 : self.stride[
                                    0
                                ],
                                column : column
                                + (output_width - 1) * self.stride[1]
                                + 1 : self.stride[1],
                            ] += grad_columns[:, channel, row, column]
                grad_value = grad_padded[
                    :,
                    :,
                    self.padding[0] : self.padding[0] + input_height,
                    self.padding[1] : self.padding[1] + input_width,
                ]
                _accumulate(value, grad_value)
                _accumulate(self.weight, grad_weight)
                if self.bias is not None:
                    _accumulate(self.bias, gradient.sum(axis=(0, 2, 3)))

            result._backward = run_backward
        return result


class MultiheadAttention(Module):
    """Scaled dot-product multi-head attention for N x L x D tensors."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        *,
        rng: np.random.Generator | None = None,
        dtype: Any = DEFAULT_DTYPE,
    ) -> None:
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads:
            raise ValueError("embed_dim must be positive and divisible by num_heads")
        generator = rng or np.random.default_rng()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.in_proj_weight = Parameter(
            generator.standard_normal((3 * self.embed_dim, self.embed_dim)).astype(dtype)
            * dtype(np.sqrt(1.0 / self.embed_dim))
        )
        self.in_proj_bias = Parameter(np.zeros(3 * self.embed_dim, dtype=dtype)) if bias else None
        self.out_proj_weight = Parameter(
            generator.standard_normal((self.embed_dim, self.embed_dim)).astype(dtype)
            * dtype(np.sqrt(1.0 / self.embed_dim))
        )
        self.out_proj_bias = Parameter(np.zeros(self.embed_dim, dtype=dtype)) if bias else None

    def forward(self, value: Tensor, mask: Tensor | None = None) -> Tensor:
        if value.ndim != 3 or value.shape[-1] != self.embed_dim:
            raise ValueError(
                f"MultiheadAttention expected N x L x {self.embed_dim}, got {value.shape}"
            )
        batch, length, _ = value.shape
        projected = value @ self.in_proj_weight.transpose((1, 0))
        if self.in_proj_bias is not None:
            projected = projected + self.in_proj_bias
        query, key, val = projected.split(self.embed_dim, axis=-1)
        query = query.reshape(batch, length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        key = key.reshape(batch, length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        val = val.reshape(batch, length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        scores = (query @ key.transpose((0, 1, 3, 2))) * self.head_dim**-0.5
        if mask is not None:
            scores = scores + mask
        weights = F.softmax(scores, axis=-1)
        context = weights @ val
        context = context.transpose(0, 2, 1, 3).reshape(batch, length, self.embed_dim)
        output = context @ self.out_proj_weight.transpose((1, 0))
        if self.out_proj_bias is not None:
            output = output + self.out_proj_bias
        return output


class ReLU(Module):
    def forward(self, value: Tensor) -> Tensor:
        return F.relu(value)


class GELU(Module):
    def forward(self, value: Tensor) -> Tensor:
        return F.gelu(value)


class SiLU(Module):
    def forward(self, value: Tensor) -> Tensor:
        return F.silu(value)


class Sigmoid(Module):
    def forward(self, value: Tensor) -> Tensor:
        return F.sigmoid(value)


class Tanh(Module):
    def forward(self, value: Tensor) -> Tensor:
        return F.tanh(value)


class Softmax(Module):
    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, value: Tensor) -> Tensor:
        return F.softmax(value, self.axis)


class LogSoftmax(Module):
    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, value: Tensor) -> Tensor:
        return F.log_softmax(value, self.axis)


class Dropout(Module):
    def __init__(self, probability: float = 0.5, *, rng: np.random.Generator | None = None) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("Dropout probability must be in [0, 1)")
        self.probability = float(probability)
        self.rng = rng

    def forward(self, value: Tensor) -> Tensor:
        return F.dropout(value, self.probability, training=self.training, rng=self.rng)


class Flatten(Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, value: Tensor) -> Tensor:
        if self.start_dim != 1:
            raise NotImplementedError("Heliax 0.1 Flatten currently supports start_dim=1")
        if self.end_dim not in (-1, value.ndim - 1):
            raise NotImplementedError("Heliax 0.1 Flatten currently supports the last dimension")
        return value.reshape((value.shape[0], -1))


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        super().__init__()
        self.layers = list(layers)

    def forward(self, value: Tensor) -> Tensor:
        for layer in self.layers:
            value = layer(value)
        return value

    def __repr__(self) -> str:
        body = ",\n  ".join(repr(layer) for layer in self.layers)
        return f"Sequential(\n  {body}\n)" if body else "Sequential()"


class MLP(Module):
    """A small fully connected network builder."""

    def __init__(
        self,
        input_features: int,
        hidden_features: Sequence[int],
        output_features: int,
        *,
        activation: type[Module] = GELU,
        dropout: float = 0.0,
        bias: bool = True,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        dimensions = [
            int(input_features),
            *[int(size) for size in hidden_features],
            int(output_features),
        ]
        layers: list[Module] = []
        for index in range(len(dimensions) - 1):
            layers.append(Linear(dimensions[index], dimensions[index + 1], bias=bias, rng=rng))
            if index < len(dimensions) - 2:
                layers.append(activation())
                if dropout:
                    layers.append(Dropout(dropout, rng=rng))
        self.network = Sequential(*layers)

    def forward(self, value: Tensor) -> Tensor:
        return self.network(value)
