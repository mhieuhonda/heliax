"""Neural network modules built on Heliax tensors."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from . import functional as F
from .backend import DEFAULT_DTYPE, get_backend
from .tensor import Parameter, Tensor, _accumulate, zeros


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

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters():
            if set_to_none:
                parameter.zero_grad()
            elif parameter.grad is not None:
                parameter.grad.zero_()

    def register_buffer(self, name: str, value: np.ndarray, *, persistent: bool = True) -> None:
        if not hasattr(self, "_buffers"):
            self._buffers: dict[str, np.ndarray] = {}
        self._buffers[name] = np.asarray(value)

    def named_buffers(self, prefix: str = "") -> Iterator[tuple[str, np.ndarray]]:
        for module_name, module in self.named_modules(prefix):
            for name, value in getattr(module, "_buffers", {}).items():
                if value is None or name.endswith("_persistent"):
                    continue
                full_name = f"{module_name}.{name}" if module_name else name
                yield full_name, value

    def state_dict(self) -> dict[str, np.ndarray]:
        state = {name: parameter.numpy().copy() for name, parameter in self.named_parameters()}
        state.update({name: value.copy() for name, value in self.named_buffers()})
        return state

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
        for name, value in self.named_buffers():
            target = dict(self.named_buffers())[name]
            target[...] = np.asarray(state[name])

    def train(self, mode: bool = True) -> Module:
        self.training = mode
        for module in self.modules():
            module.training = mode
        return self

    def eval(self) -> Module:
        return self.train(False)

    def apply(self, function: Any) -> Module:
        for module in self.modules():
            function(module)
        return self

    def to(self, dtype: Any) -> Module:
        return self.dtype(dtype)

    def cpu(self) -> Module:
        return self

    def half(self) -> Module:
        return self.dtype(np.float16)

    def float(self) -> Module:
        return self.dtype(np.float32)

    def dtype(self, dtype: Any) -> Module:
        for parameter in self.parameters():
            parameter.data = parameter.numpy().astype(dtype, copy=False)
        for _, buffer in self.named_buffers():
            buffer[...] = buffer.astype(dtype, copy=False)
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
        return F.fused_linear_bias(value, self.weight, self.bias)


class FusedLinearGELU(Linear):
    """Linear + GELU with one fused forward node."""

    def forward(self, value: Tensor) -> Tensor:
        if value.shape[-1] != self.in_features:
            raise ValueError(
                f"FusedLinearGELU expected last dimension {self.in_features}, got {value.shape[-1] if value.shape else None}"
            )
        return F.fused_linear_gelu(value, self.weight, self.bias)


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


class _BatchNorm(Module):
    def __init__(
        self,
        num_features: int,
        eps: float,
        momentum: float,
        affine: bool,
        track_running_stats: bool,
        dtype: Any,
    ) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("BatchNorm features must be positive")
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)
        self.track_running_stats = bool(track_running_stats)
        self.weight = Parameter(np.ones(num_features, dtype=dtype)) if affine else None
        self.bias = Parameter(np.zeros(num_features, dtype=dtype)) if affine else None
        if track_running_stats:
            self.register_buffer("running_mean", np.zeros(num_features, dtype=dtype))
            self.register_buffer("running_var", np.ones(num_features, dtype=dtype))

    def _forward(self, value: Tensor, required_ndim: int) -> Tensor:
        if value.ndim != required_ndim or value.shape[1] != self.num_features:
            raise ValueError(
                f"BatchNorm expected shape (N, {self.num_features}, ...) with {required_ndim} dimensions"
            )
        data = value.numpy()
        reduce_axes = tuple(axis for axis in range(value.ndim) if axis != 1)
        if self.training or not self.track_running_stats:
            mean = np.mean(data, axis=reduce_axes, keepdims=True, dtype=np.float32)
            variance = np.mean(
                (data - mean) ** 2, axis=reduce_axes, keepdims=True, dtype=np.float32
            )
            if self.training and self.track_running_stats:
                self._buffers["running_mean"][...] = self.momentum * self._buffers[
                    "running_mean"
                ] + (1.0 - self.momentum) * mean.reshape(self.num_features)
                self._buffers["running_var"][...] = self.momentum * self._buffers["running_var"] + (
                    1.0 - self.momentum
                ) * variance.reshape(self.num_features)
        else:
            mean = self._buffers["running_mean"].reshape(1, -1, *([1] * (value.ndim - 2)))
            variance = self._buffers["running_var"].reshape(1, -1, *([1] * (value.ndim - 2)))
        inverse_std = 1.0 / np.sqrt(variance + self.eps)
        normalized = (data - mean) * inverse_std
        output_data = normalized
        if self.affine:
            output_data = normalized * self.weight.numpy().reshape(
                1, -1, *([1] * (value.ndim - 2))
            ) + self.bias.numpy().reshape(1, -1, *([1] * (value.ndim - 2)))
        parents = (value,) + ((self.weight, self.bias) if self.affine else ())
        output = value._make(output_data, parents, lambda: None, "batch_norm")
        if output.requires_grad:

            def run_backward() -> None:
                grad = output.grad.numpy()
                gamma = (
                    self.weight.numpy().reshape(1, -1, *([1] * (value.ndim - 2)))
                    if self.affine
                    else 1.0
                )
                count = max(int(np.prod([data.shape[axis] for axis in reduce_axes])), 1)
                weighted = grad * gamma
                mean_weighted = np.mean(weighted, axis=reduce_axes, keepdims=True, dtype=np.float32)
                mean_weighted_normalized = np.mean(
                    weighted * normalized, axis=reduce_axes, keepdims=True, dtype=np.float32
                )
                grad_value = (
                    (weighted - mean_weighted - normalized * mean_weighted_normalized)
                    * inverse_std
                    / count
                )
                _accumulate(value, grad_value)
                if self.affine:
                    _accumulate(
                        self.weight, np.sum(grad * normalized, axis=reduce_axes, dtype=np.float32)
                    )
                    _accumulate(self.bias, np.sum(grad, axis=reduce_axes, dtype=np.float32))

            output._backward = run_backward
        return output


class BatchNorm1d(_BatchNorm):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        *,
        dtype: Any = DEFAULT_DTYPE,
    ) -> None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats, dtype)

    def forward(self, value: Tensor) -> Tensor:
        return self._forward(value, 2)


class BatchNorm2d(_BatchNorm):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        *,
        dtype: Any = DEFAULT_DTYPE,
    ) -> None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats, dtype)

    def forward(self, value: Tensor) -> Tensor:
        return self._forward(value, 4)


class GroupNorm(Module):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        *,
        dtype: Any = DEFAULT_DTYPE,
    ) -> None:
        super().__init__()
        if num_groups <= 0 or num_channels <= 0 or num_channels % num_groups:
            raise ValueError("GroupNorm requires channels divisible by groups")
        self.num_groups = int(num_groups)
        self.num_channels = int(num_channels)
        self.group_size = self.num_channels // self.num_groups
        self.eps = float(eps)
        self.affine = bool(affine)
        self.weight = Parameter(np.ones(num_channels, dtype=dtype)) if affine else None
        self.bias = Parameter(np.zeros(num_channels, dtype=dtype)) if affine else None

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim < 2 or value.shape[1] != self.num_channels:
            raise ValueError(f"GroupNorm expected channel dimension {self.num_channels}")
        data = value.numpy()
        grouped = data.reshape(value.shape[0], self.num_groups, -1)
        mean = np.mean(grouped, axis=2, keepdims=True, dtype=np.float32)
        variance = np.mean((grouped - mean) ** 2, axis=2, keepdims=True, dtype=np.float32)
        normalized_grouped = (grouped - mean) / np.sqrt(variance + self.eps, dtype=np.float32)
        normalized = normalized_grouped.reshape(data.shape)
        output_data = normalized
        if self.affine:
            shape = (1, -1, *([1] * (value.ndim - 2)))
            output_data = normalized * self.weight.numpy().reshape(
                shape
            ) + self.bias.numpy().reshape(shape)
        parents = (value,) + ((self.weight, self.bias) if self.affine else ())
        output = value._make(output_data, parents, lambda: None, "group_norm")
        if output.requires_grad:

            def run_backward() -> None:
                grad = output.grad.numpy()
                gamma = 1.0
                if self.affine:
                    spatial = max(int(np.prod(value.shape[2:])), 1)
                    gamma_channels = self.weight.numpy().reshape(
                        1, self.num_groups, self.group_size, *([1] * (value.ndim - 2))
                    )
                    gamma = np.broadcast_to(
                        gamma_channels, (1, self.num_groups, self.group_size, spatial)
                    ).reshape(1, self.num_groups, -1)
                grad_grouped = grad.reshape(value.shape[0], self.num_groups, -1)
                weighted = grad_grouped * gamma
                mean_weighted = np.mean(weighted, axis=2, keepdims=True, dtype=np.float32)
                mean_weighted_normalized = np.mean(
                    weighted * normalized_grouped, axis=2, keepdims=True, dtype=np.float32
                )
                grad_value = (
                    (weighted - mean_weighted - normalized_grouped * mean_weighted_normalized)
                    / np.sqrt(variance + self.eps, dtype=np.float32)
                    / self.group_size
                ).reshape(data.shape)
                _accumulate(value, grad_value)
                if self.affine:
                    _accumulate(
                        self.weight,
                        np.sum(
                            grad * normalized,
                            axis=tuple(axis for axis in range(value.ndim) if axis != 1),
                            dtype=np.float32,
                        ),
                    )
                    _accumulate(
                        self.bias,
                        np.sum(
                            grad,
                            axis=tuple(axis for axis in range(value.ndim) if axis != 1),
                            dtype=np.float32,
                        ),
                    )

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
        dilation: int | tuple[int, int] = 1,
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
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
        if (
            len(self.kernel_size) != 2
            or len(self.stride) != 2
            or len(self.padding) != 2
            or len(self.dilation) != 2
            or any(item < 1 for item in self.dilation)
        ):
            raise ValueError(
                "Conv2d kernel, stride, padding, and dilation must be valid two-dimensional values"
            )
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
        output_height = (
            padded_height - self.dilation[0] * (self.kernel_size[0] - 1) - 1
        ) // self.stride[0] + 1
        output_width = (
            padded_width - self.dilation[1] * (self.kernel_size[1] - 1) - 1
        ) // self.stride[1] + 1
        if output_height <= 0 or output_width <= 0:
            raise ValueError("Conv2d kernel is larger than the padded input")
        return batch, output_height, output_width, channels

    def _columns(self, data: np.ndarray) -> np.ndarray:
        batch, _, height, width = data.shape
        output_height = (
            height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1
        ) // self.stride[0] + 1
        output_width = (
            width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1
        ) // self.stride[1] + 1
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


class MaxPool2d(Module):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, value: Tensor) -> Tensor:
        return F.max_pool2d(value, self.kernel_size, self.stride, self.padding)


class AvgPool2d(Module):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, value: Tensor) -> Tensor:
        return F.avg_pool2d(value, self.kernel_size, self.stride, self.padding)


class GroupedConv2d(Module):
    """Grouped convolution composed from ordinary Conv2d blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = True,
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        if groups <= 0 or in_channels % groups or out_channels % groups:
            raise ValueError("GroupedConv2d channels must be divisible by groups")
        self.groups = int(groups)
        self.blocks = ModuleList(
            *(
                Conv2d(
                    in_channels // groups,
                    out_channels // groups,
                    kernel_size,
                    stride,
                    padding,
                    bias,
                    rng=rng,
                )
                for _ in range(groups)
            )
        )

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim != 4:
            raise ValueError("GroupedConv2d expects NCHW input")
        group_size = value.shape[1] // self.groups
        outputs = [
            block(value[:, index * group_size : (index + 1) * group_size])
            for index, block in enumerate(self.blocks.items)
        ]
        return F.concatenate(outputs, axis=1)


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


class TransformerEncoderLayer(Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: type[Module] | Module | None = None,
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        self.self_attention = MultiheadAttention(d_model, nhead, rng=rng)
        self.norm1 = LayerNorm(d_model)
        self.linear1 = Linear(d_model, dim_feedforward, rng=rng)
        self.linear2 = Linear(dim_feedforward, d_model, rng=rng)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout, rng=rng)
        self.dropout2 = Dropout(dropout, rng=rng)
        self.activation = (
            (activation or GELU)()
            if isinstance(activation or GELU, type)
            else (activation or GELU())
        )

    def forward(self, value: Tensor, mask: Tensor | None = None) -> Tensor:
        attention = self.norm1(value + self.dropout1(self.self_attention(value, mask)))
        feedforward = self.linear2(self.dropout1(self.activation(self.linear1(attention))))
        return self.norm2(attention + self.dropout2(feedforward))


class TransformerEncoder(Module):
    def __init__(self, layer: TransformerEncoderLayer, num_layers: int = 1) -> None:
        super().__init__()
        self.layers = [layer] + [
            TransformerEncoderLayer(
                layer.norm1.normalized_shape[0],
                layer.self_attention.num_heads,
                layer.linear1.out_features,
                layer.dropout1.probability,
                type(layer.activation),
                rng=np.random.default_rng(index + 1),
            )
            for index in range(1, num_layers)
        ]

    def forward(self, value: Tensor, mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            value = layer(value, mask)
        return value


class PositionalEncoding(Module):
    def __init__(self, d_model: int, max_length: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model <= 0 or max_length <= 0:
            raise ValueError("PositionalEncoding dimensions must be positive")
        positions = np.arange(max_length, dtype=np.float32)[:, None]
        frequencies = np.exp(
            np.arange(0, d_model, 2, dtype=np.float32) * (-np.log(10000.0) / d_model)
        )
        encoding = np.zeros((max_length, d_model), dtype=np.float32)
        encoding[:, 0::2] = np.sin(positions * frequencies)
        encoding[:, 1::2] = np.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding)
        self.dropout = Dropout(dropout) if dropout else None

    def forward(self, value: Tensor) -> Tensor:
        if (
            value.ndim != 3
            or value.shape[1] > self._buffers["encoding"].shape[0]
            or value.shape[2] != self._buffers["encoding"].shape[1]
        ):
            raise ValueError("PositionalEncoding expects N x L x D with a supported length")
        encoded = value + Tensor(
            self._buffers["encoding"][: value.shape[1]].reshape(1, value.shape[1], value.shape[2])
        )
        return self.dropout(encoded) if self.dropout is not None else encoded


class GRU(Module):
    """Single-layer GRU for sequence tensors shaped (time, batch, input)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        *,
        rng: np.random.Generator | None = None,
        dtype: Any = DEFAULT_DTYPE,
    ) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("GRU dimensions must be positive")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        generator = rng or np.random.default_rng()
        scale = dtype(np.sqrt(1.0 / hidden_size))
        self.weight_ih = Parameter(
            generator.standard_normal((3 * hidden_size, input_size)).astype(dtype) * scale
        )
        self.weight_hh = Parameter(
            generator.standard_normal((3 * hidden_size, hidden_size)).astype(dtype) * scale
        )
        self.bias_ih = Parameter(np.zeros(3 * hidden_size, dtype=dtype)) if bias else None
        self.bias_hh = Parameter(np.zeros(3 * hidden_size, dtype=dtype)) if bias else None

    def forward(self, value: Tensor, hidden: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if value.ndim != 3 or value.shape[2] != self.input_size:
            raise ValueError(f"GRU expected (time, batch, {self.input_size})")
        batch = value.shape[1]
        h = hidden if hidden is not None else zeros(batch, self.hidden_size)
        if h.shape != (batch, self.hidden_size):
            raise ValueError("GRU hidden state has the wrong shape")
        outputs = []
        for time in range(value.shape[0]):
            gi = value[time] @ self.weight_ih.transpose((1, 0))
            gh = h @ self.weight_hh.transpose((1, 0))
            if self.bias_ih is not None:
                gi = gi + self.bias_ih
                gh = gh + self.bias_hh
            gi_r, gi_z, gi_n = gi.split(self.hidden_size, axis=-1)
            gh_r, gh_z, gh_n = gh.split(self.hidden_size, axis=-1)
            reset = (gi_r + gh_r).sigmoid()
            update = (gi_z + gh_z).sigmoid()
            candidate = (gi_n + reset * gh_n).tanh()
            h = update * h + (1.0 - update) * candidate
            outputs.append(h)
        return F.stack(outputs, axis=0), h


class LSTM(Module):
    """Single-layer LSTM for sequence tensors shaped (time, batch, input)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        *,
        rng: np.random.Generator | None = None,
        dtype: Any = DEFAULT_DTYPE,
    ) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("LSTM dimensions must be positive")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        generator = rng or np.random.default_rng()
        scale = dtype(np.sqrt(1.0 / hidden_size))
        self.weight_ih = Parameter(
            generator.standard_normal((4 * hidden_size, input_size)).astype(dtype) * scale
        )
        self.weight_hh = Parameter(
            generator.standard_normal((4 * hidden_size, hidden_size)).astype(dtype) * scale
        )
        self.bias_ih = Parameter(np.zeros(4 * hidden_size, dtype=dtype)) if bias else None
        self.bias_hh = Parameter(np.zeros(4 * hidden_size, dtype=dtype)) if bias else None

    def forward(
        self, value: Tensor, hidden: Tensor | None = None, cell: Tensor | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        if value.ndim != 3 or value.shape[2] != self.input_size:
            raise ValueError(f"LSTM expected (time, batch, {self.input_size})")
        batch = value.shape[1]
        h = hidden if hidden is not None else zeros(batch, self.hidden_size)
        c = cell if cell is not None else zeros(batch, self.hidden_size)
        if h.shape != (batch, self.hidden_size) or c.shape != (batch, self.hidden_size):
            raise ValueError("LSTM state has the wrong shape")
        outputs = []
        for time in range(value.shape[0]):
            gi = value[time] @ self.weight_ih.transpose((1, 0))
            gh = h @ self.weight_hh.transpose((1, 0))
            if self.bias_ih is not None:
                gi = gi + self.bias_ih
                gh = gh + self.bias_hh
            gi_i, gi_f, gi_g, gi_o = gi.split(self.hidden_size, axis=-1)
            gh_i, gh_f, gh_g, gh_o = gh.split(self.hidden_size, axis=-1)
            input_gate = (gi_i + gh_i).sigmoid()
            forget_gate = (gi_f + gh_f).sigmoid()
            cell_candidate = (gi_g + gh_g).tanh()
            output_gate = (gi_o + gh_o).sigmoid()
            c = forget_gate * c + input_gate * cell_candidate
            h = output_gate * c.tanh()
            outputs.append(h)
        return F.stack(outputs, axis=0), (h, c)


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


class CrossEntropyLoss(Module):
    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, logits: Tensor, target: Any) -> Tensor:
        return F.fused_cross_entropy(logits, target, self.axis)


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
        start = self.start_dim if self.start_dim >= 0 else value.ndim + self.start_dim
        end = self.end_dim if self.end_dim >= 0 else value.ndim + self.end_dim
        if start < 0 or end < start or end >= value.ndim:
            raise ValueError("Flatten dimensions are out of range")
        shape = (*value.shape[:start], -1, *value.shape[end + 1 :])
        return value.reshape(shape)


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


class ModuleList(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.items = list(modules)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, index: int) -> Module:
        return self.items[index]

    def append(self, module: Module) -> Module:
        self.items.append(module)
        return self


class ModuleDict(Module):
    def __init__(self, modules: dict[str, Module] | None = None) -> None:
        super().__init__()
        self.items = dict(modules or {})

    def __getitem__(self, key: str) -> Module:
        return self.items[key]

    def __setitem__(self, key: str, module: Module) -> None:
        if not isinstance(module, Module):
            raise TypeError("ModuleDict values must be Module instances")
        self.items[key] = module

    def keys(self):
        return self.items.keys()

    def values(self):
        return self.items.values()


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
