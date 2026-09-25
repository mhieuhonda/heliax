"""N-dimensional tensors and reverse-mode automatic differentiation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np

from .backend import DEFAULT_DTYPE, get_backend, unbroadcast

_GRAD_ENABLED = True


@contextmanager
def no_grad() -> Iterator[None]:
    """Temporarily disable graph construction."""

    global _GRAD_ENABLED
    previous = _GRAD_ENABLED
    _GRAD_ENABLED = False
    try:
        yield
    finally:
        _GRAD_ENABLED = previous


@contextmanager
def enable_grad() -> Iterator[None]:
    """Temporarily enable graph construction."""

    global _GRAD_ENABLED
    previous = _GRAD_ENABLED
    _GRAD_ENABLED = True
    try:
        yield
    finally:
        _GRAD_ENABLED = previous


def is_grad_enabled() -> bool:
    return _GRAD_ENABLED


def _reduce_gradient(
    gradient: np.ndarray,
    shape: tuple[int, ...],
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
) -> np.ndarray:
    gradient = np.asarray(gradient)
    if axis is None:
        return np.broadcast_to(gradient, shape).copy()
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    normalized = tuple(sorted(a if a >= 0 else a + len(shape) for a in axes))
    if keepdims:
        return np.broadcast_to(gradient, shape).copy()
    expanded = gradient
    for a in normalized:
        expanded = np.expand_dims(expanded, axis=a)
    return np.broadcast_to(expanded, shape).copy()


def _accumulate(tensor: Tensor, gradient: np.ndarray) -> None:
    if not tensor.requires_grad:
        return
    reduced = unbroadcast(np.asarray(gradient, dtype=tensor._data.dtype), tensor.shape)
    if tensor.grad is None:
        tensor.grad = Tensor(reduced, requires_grad=False)
    else:
        tensor.grad = Tensor(tensor.grad._data + reduced, requires_grad=False)


class Tensor:
    """A NumPy-backed value with an optional reverse-mode gradient graph."""

    def __init__(self, data: Any, requires_grad: bool = False) -> None:
        if isinstance(data, Tensor):
            data = data._data
        self._data = get_backend().array(data)
        self.requires_grad = bool(requires_grad) and _GRAD_ENABLED
        self._prev: set[Tensor] = set()
        self._backward: Any = lambda: None
        self._op = ""
        self.grad: Tensor | None = None

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, value: Any) -> None:
        self._data = get_backend().array(value)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._data.shape)

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def size(self) -> int:
        return int(self._data.size)

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    @property
    def device(self) -> str:
        return "cpu"

    def numpy(self) -> np.ndarray:
        """Return a NumPy view for explicit interoperability."""

        return self._data

    def item(self) -> Any:
        return self._data.item()

    def __array__(self, dtype: Any | None = None) -> np.ndarray:
        return np.asarray(self._data, dtype=dtype)

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[Tensor]:
        for index in range(len(self)):
            yield self[index]

    def __repr__(self) -> str:
        preview = np.array2string(self._data, threshold=18, edgeitems=3)
        suffix = f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        return f"Tensor({preview}{suffix})"

    def _make(self, data: Any, inputs: Iterable[Tensor], backward: Any, op: str) -> Tensor:
        parents = tuple(inputs)
        requires_grad = _GRAD_ENABLED and any(parent.requires_grad for parent in parents)
        output = Tensor(data, requires_grad=requires_grad)
        if requires_grad:
            output._prev = set(parents)
            output._backward = backward
            output._op = op
        return output

    def detach(self) -> Tensor:
        return Tensor(self._data.copy(), requires_grad=False)

    def clone(self) -> Tensor:
        result = self.detach()
        if self.requires_grad and _GRAD_ENABLED:
            return self._make(
                self._data.copy(),
                (self,),
                lambda: _accumulate(
                    self, result.grad.numpy() if result.grad else np.zeros_like(self._data)
                ),
                "clone",
            )
        return result

    def zero_grad(self) -> None:
        self.grad = None

    def backward(self, gradient: Any | None = None) -> None:
        if not self.requires_grad:
            raise RuntimeError("backward() called on a tensor that does not require gradients")
        if gradient is None:
            if self.size != 1:
                raise RuntimeError("a gradient is required for a non-scalar output")
            seed = np.ones_like(self._data)
        else:
            seed = np.asarray(gradient, dtype=self._data.dtype)
            if seed.shape != self.shape:
                seed = np.broadcast_to(seed, self.shape).copy()
        self.zero_grad()
        self.grad = Tensor(seed.copy(), requires_grad=False)
        nodes: list[Tensor] = []
        visited: set[int] = set()

        def visit(node: Tensor) -> None:
            identity = id(node)
            if identity in visited:
                return
            visited.add(identity)
            for parent in node._prev:
                visit(parent)
            nodes.append(node)

        visit(self)
        for node in reversed(nodes):
            if node.grad is not None and node._op:
                node._backward()

    def _binary(
        self, other: Any, operation: Any, backward: Any, op: str, reverse: bool = False
    ) -> Tensor:
        other_tensor = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        if reverse:
            left, right = other_tensor, self
            data = operation(left.data, right.data)
        else:
            left, right = self, other_tensor
            data = operation(left.data, right.data)
        parents = tuple(parent for parent in (left, right) if parent.requires_grad)
        output = self._make(data, parents, lambda: None, op)
        if output.requires_grad:

            def run_backward() -> None:
                grad = output.grad.numpy()
                left_grad, right_grad = backward(grad, left.data, right.data)
                _accumulate(left, left_grad)
                if right.requires_grad:
                    _accumulate(right, right_grad)

            output._backward = run_backward
        return output

    def _unary(self, operation: Any, backward: Any, op: str) -> Tensor:
        output = self._make(operation(self._data), (self,), lambda: None, op)
        if output.requires_grad:

            def run_backward() -> None:
                _accumulate(self, backward(output.grad.numpy(), self._data))

            output._backward = run_backward
        return output

    def __add__(self, other: Any) -> Tensor:
        return self._binary(other, lambda a, b: a + b, lambda g, a, b: (g, g), "add")

    def __radd__(self, other: Any) -> Tensor:
        return self._binary(other, lambda a, b: a + b, lambda g, a, b: (g, g), "add", True)

    def __sub__(self, other: Any) -> Tensor:
        return self._binary(other, lambda a, b: a - b, lambda g, a, b: (g, -g), "sub")

    def __rsub__(self, other: Any) -> Tensor:
        return self._binary(other, lambda a, b: a - b, lambda g, a, b: (g, -g), "sub", True)

    def __mul__(self, other: Any) -> Tensor:
        return self._binary(other, lambda a, b: a * b, lambda g, a, b: (g * b, g * a), "mul")

    def __rmul__(self, other: Any) -> Tensor:
        return self._binary(other, lambda a, b: a * b, lambda g, a, b: (g * b, g * a), "mul", True)

    def __truediv__(self, other: Any) -> Tensor:
        return self._binary(
            other, lambda a, b: a / b, lambda g, a, b: (g / b, -g * a / (b * b)), "div"
        )

    def __rtruediv__(self, other: Any) -> Tensor:
        return self._binary(
            other, lambda a, b: a / b, lambda g, a, b: (-g * b / (a * a), g / a), "div", True
        )

    def __pow__(self, other: Any) -> Tensor:
        return self._binary(
            other,
            lambda a, b: a**b,
            lambda g, a, b: (g * b * a ** (b - 1), g * a**b * np.log(np.maximum(a, 1e-7))),
            "pow",
        )

    def __rpow__(self, other: Any) -> Tensor:
        return self._binary(
            other,
            lambda a, b: a**b,
            lambda g, a, b: (g * b * a ** (b - 1), g * a**b * np.log(np.maximum(a, 1e-7))),
            "pow",
            True,
        )

    def __neg__(self) -> Tensor:
        return self._unary(lambda x: -x, lambda g, x: -g, "neg")

    def __matmul__(self, other: Any) -> Tensor:
        other_value = other.data if isinstance(other, Tensor) else other
        output = self._make(
            np.matmul(self._data, other_value),
            (self, other) if isinstance(other, Tensor) else (self,),
            lambda: None,
            "matmul",
        )
        if output.requires_grad:

            def run_backward() -> None:
                grad = output.grad.numpy()
                _accumulate(self, np.matmul(grad, np.swapaxes(other_value, -1, -2)))
                if isinstance(other, Tensor):
                    _accumulate(other, np.matmul(np.swapaxes(self._data, -1, -2), grad))

            output._backward = run_backward
        return output

    def __rmatmul__(self, other: Any) -> Tensor:
        other_tensor = Tensor(other)
        return other_tensor.__matmul__(self)

    def square(self) -> Tensor:
        return self._unary(lambda x: x * x, lambda g, x: g * 2 * x, "square")

    def exp(self) -> Tensor:
        backend = get_backend()
        return self._unary(backend.exp, lambda g, x: g * backend.exp(x), "exp")

    def log(self) -> Tensor:
        backend = get_backend()
        return self._unary(backend.log, lambda g, x: g / np.maximum(x, 1e-12), "log")

    def sqrt(self) -> Tensor:
        backend = get_backend()
        return self._unary(
            backend.sqrt, lambda g, x: g * 0.5 / np.maximum(backend.sqrt(x), 1e-12), "sqrt"
        )

    def tanh(self) -> Tensor:
        backend = get_backend()
        return self._unary(backend.tanh, lambda g, x: g * (1.0 - backend.tanh(x) ** 2), "tanh")

    def sigmoid(self) -> Tensor:
        backend = get_backend()
        return self._unary(
            backend.sigmoid,
            lambda g, x: g * backend.sigmoid(x) * (1.0 - backend.sigmoid(x)),
            "sigmoid",
        )

    def relu(self) -> Tensor:
        return self._unary(get_backend().relu, lambda g, x: g * (x > 0), "relu")

    def gelu(self) -> Tensor:
        backend = get_backend()
        coefficient = np.sqrt(np.asarray(2.0 / np.pi, dtype=self.dtype))

        def derivative(value: np.ndarray) -> np.ndarray:
            cubic = value**3
            inner = coefficient * (value + 0.044715 * cubic)
            tanh_value = np.tanh(inner)
            return 0.5 * (1.0 + tanh_value) + 0.5 * value * (1.0 - tanh_value**2) * coefficient * (
                1.0 + 3 * 0.044715 * value**2
            )

        return self._unary(backend.gelu, lambda g, x: g * derivative(x), "gelu")

    def silu(self) -> Tensor:
        backend = get_backend()

        def derivative(value: np.ndarray) -> np.ndarray:
            sigmoid = backend.sigmoid(value)
            return sigmoid * (1.0 + value * (1.0 - sigmoid))

        return self._unary(backend.silu, lambda g, x: g * derivative(x), "silu")

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        output = self._make(
            np.sum(self._data, axis=axis, keepdims=keepdims, dtype=self._data.dtype),
            (self,),
            lambda: None,
            "sum",
        )
        if output.requires_grad:

            def run_backward() -> None:
                _accumulate(self, _reduce_gradient(output.grad.numpy(), self.shape, axis, keepdims))

            output._backward = run_backward
        return output

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        count = (
            self.size
            if axis is None
            else np.prod([self.shape[a] for a in ((axis,) if isinstance(axis, int) else axis)])
        )
        output = self._make(
            np.mean(self._data, axis=axis, keepdims=keepdims, dtype=self._data.dtype),
            (self,),
            lambda: None,
            "mean",
        )
        if output.requires_grad:

            def run_backward() -> None:
                _accumulate(
                    self,
                    _reduce_gradient(output.grad.numpy(), self.shape, axis, keepdims)
                    / max(int(count), 1),
                )

            output._backward = run_backward
        return output

    def max(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        data = np.max(self._data, axis=axis, keepdims=keepdims)
        output = self._make(data, (self,), lambda: None, "max")
        if output.requires_grad:

            def run_backward() -> None:
                expanded = np.expand_dims(self._data, axis=axis) if axis is not None else self._data
                mask = expanded == data if axis is not None else self._data == data
                gradient = np.asarray(output.grad.numpy(), dtype=self._data.dtype)
                if axis is not None and not keepdims:
                    gradient = np.expand_dims(gradient, axis=axis)
                gradient = np.broadcast_to(gradient, expanded.shape) * mask
                if axis is not None and not keepdims:
                    gradient = gradient.sum(axis=axis, keepdims=True)
                _accumulate(self, gradient)

            output._backward = run_backward
        return output

    def split(self, split_size_or_sections: int | Sequence[int], axis: int = 0) -> list[Tensor]:
        if not self.shape:
            raise ValueError("cannot split a scalar tensor")
        normalized_axis = axis if axis >= 0 else axis + self.ndim
        if normalized_axis < 0 or normalized_axis >= self.ndim:
            raise ValueError("split axis is out of range")
        size = self.shape[normalized_axis]
        if isinstance(split_size_or_sections, int):
            if split_size_or_sections <= 0 or size % split_size_or_sections:
                raise ValueError("split size must divide the selected dimension")
            sections = [split_size_or_sections] * (size // split_size_or_sections)
        else:
            sections = list(split_size_or_sections)
            if not sections or sum(sections) != size or any(section <= 0 for section in sections):
                raise ValueError("split sections must cover the selected dimension")
        outputs: list[Tensor] = []
        start = 0
        for section in sections:
            key = [slice(None)] * self.ndim
            key[normalized_axis] = slice(start, start + section)
            outputs.append(self[tuple(key)])
            start += section
        return outputs

    def reshape(self, *shape: int | tuple[int, ...]) -> Tensor:
        target = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        output = self._make(self._data.reshape(target), (self,), lambda: None, "reshape")
        if output.requires_grad:

            def run_backward() -> None:
                _accumulate(self, output.grad.numpy().reshape(self.shape))

            output._backward = run_backward
        return output

    def transpose(self, *axes: int | tuple[int, ...]) -> Tensor:
        if not axes:
            target = tuple(reversed(range(self.ndim)))
        elif len(axes) == 1 and isinstance(axes[0], tuple):
            target = tuple(axes[0])
        else:
            target = tuple(axes)
        output = self._make(np.transpose(self._data, target), (self,), lambda: None, "transpose")
        if output.requires_grad:

            def run_backward() -> None:
                inverse = np.argsort(target)
                _accumulate(self, output.grad.numpy().transpose(inverse))

            output._backward = run_backward
        return output

    def permute(self, *axes: int | tuple[int, ...]) -> Tensor:
        return self.transpose(*axes)

    def flatten(self) -> Tensor:
        return self.reshape((self.shape[0] if self.shape else 1, -1))

    def astype(self, dtype: Any) -> Tensor:
        return self._unary(
            lambda x: x.astype(dtype, copy=False),
            lambda g, x: g.astype(x.dtype, copy=False),
            "astype",
        )

    def to(self, dtype: Any) -> Tensor:
        return self.astype(dtype)

    def __getitem__(self, key: Any) -> Tensor:
        output = self._make(self._data[key], (self,), lambda: None, "getitem")
        if output.requires_grad:

            def run_backward() -> None:
                gradient = np.zeros_like(self._data)
                selected = output.grad.numpy()
                normalized_key = key if isinstance(key, tuple) else (key,)
                advanced = any(isinstance(item, (list, np.ndarray)) for item in normalized_key)
                if advanced:
                    add_key = tuple(
                        np.asarray(item) if isinstance(item, list) else item
                        for item in normalized_key
                    )
                    try:
                        np.add.at(gradient, add_key, selected)
                    except (ValueError, IndexError):
                        mask = np.zeros_like(self._data, dtype=bool)
                        mask[key] = True
                        np.add.at(gradient, np.nonzero(mask), selected.reshape(-1))
                else:
                    gradient[key] = selected
                _accumulate(self, gradient)

            output._backward = run_backward
        return output

    def __hash__(self) -> int:
        return id(self)


class Parameter(Tensor):
    """A trainable Tensor."""

    def __init__(self, data: Any) -> None:
        super().__init__(data, requires_grad=True)

    def __repr__(self) -> str:
        return f"Parameter(shape={self.shape}, dtype={self.dtype}, requires_grad=True)"


def tensor(
    data: Any, *, dtype: Any | None = None, requires_grad: bool = False, device: str = "cpu"
) -> Tensor:
    if device != "cpu":
        raise NotImplementedError("Heliax 0.1 currently exposes the CPU backend only")
    return Tensor(
        data if dtype is None else np.asarray(data, dtype=dtype), requires_grad=requires_grad
    )


def from_numpy(array: np.ndarray, *, requires_grad: bool = False) -> Tensor:
    return Tensor(array, requires_grad=requires_grad)


def zeros(*shape: int, dtype: Any = DEFAULT_DTYPE, requires_grad: bool = False) -> Tensor:
    return Tensor(np.zeros(shape, dtype=dtype), requires_grad=requires_grad)


def ones(*shape: int, dtype: Any = DEFAULT_DTYPE, requires_grad: bool = False) -> Tensor:
    return Tensor(np.ones(shape, dtype=dtype), requires_grad=requires_grad)


def full(
    shape: Sequence[int], value: Any, *, dtype: Any = DEFAULT_DTYPE, requires_grad: bool = False
) -> Tensor:
    return Tensor(np.full(tuple(shape), value, dtype=dtype), requires_grad=requires_grad)


def arange(*args: int, dtype: Any = DEFAULT_DTYPE, requires_grad: bool = False) -> Tensor:
    return Tensor(np.arange(*args, dtype=dtype), requires_grad=requires_grad)


def randn(
    *shape: int,
    rng: np.random.Generator | None = None,
    dtype: Any = DEFAULT_DTYPE,
    requires_grad: bool = False,
) -> Tensor:
    return Tensor(get_backend().randn(*shape, rng=rng, dtype=dtype), requires_grad=requires_grad)
