"""Optional PyTorch interoperability and accelerated reference kernels.

PyTorch is never a required Heliax dependency. When it is installed, this
module provides explicit conversions and a small accelerated-kernel facade.
The default Heliax Tensor remains NumPy-backed and does not silently switch
backends.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .backend import DEFAULT_DTYPE
from .tensor import Tensor


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise RuntimeError("PyTorch interop requires the optional 'torch' extra") from error
    return torch


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def torch_version() -> str | None:
    if not torch_available():
        return None
    return str(_torch().__version__)


def from_torch(
    value: Any, *, requires_grad: bool = False, detach: bool = True, device: str = "cpu"
) -> Tensor:
    """Convert a PyTorch tensor to a Heliax CPU Tensor."""

    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise TypeError("from_torch expects a torch.Tensor")
    data = value.detach() if detach else value
    if device != "cpu":
        raise ValueError("from_torch writes into Heliax CPU storage; use device='cpu'")
    array = data.to("cpu").numpy()
    if not array.flags.writeable:
        array = np.array(array, copy=True)
    return Tensor(array, requires_grad=requires_grad)


def to_torch(value: Tensor, *, requires_grad: bool = False, device: str = "cpu") -> Any:
    """Convert a Heliax Tensor to a PyTorch tensor on ``device``."""

    torch = _torch()
    array = np.array(value.numpy(), copy=True)
    result = torch.from_numpy(array).to(device)
    if requires_grad:
        result = result.detach().requires_grad_(True)
    return result


class TorchAccelerator:
    """Explicit opt-in facade for PyTorch's optimized kernels.

    It is useful for a compatibility layer or a native integration without
    making the rest of Heliax depend on PyTorch's dispatcher.
    """

    def __init__(self, device: str = "cpu", dtype: Any = None) -> None:
        self.device = device
        self.dtype = dtype or DEFAULT_DTYPE
        self._torch = _torch()

    def _check(self) -> None:
        if self.device == "cpu" and not hasattr(self._torch, "cpu"):  # pragma: no cover
            raise RuntimeError("invalid PyTorch build")

    def to(self, value: Any) -> Any:
        self._check()
        return value.to(device=self.device)

    def matmul(self, left: Any, right: Any) -> Any:
        self._check()
        return self._torch.matmul(self.to(left), self.to(right))

    def softmax(self, value: Any, axis: int = -1) -> Any:
        self._check()
        return self._torch.softmax(self.to(value), dim=axis)

    def gelu(self, value: Any) -> Any:
        self._check()
        return self._torch.nn.functional.gelu(self.to(value), approximate="tanh")

    def layer_norm(
        self,
        value: Any,
        normalized_shape: tuple[int, ...],
        weight: Any,
        bias: Any,
        eps: float = 1e-5,
    ) -> Any:
        self._check()
        return self._torch.nn.functional.layer_norm(
            self.to(value), normalized_shape, self.to(weight), self.to(bias), eps
        )

    def cross_entropy(self, logits: Any, target: Any) -> Any:
        self._check()
        return self._torch.nn.functional.cross_entropy(self.to(logits), self.to(target))

    def info(self) -> dict[str, Any]:
        return {
            "backend": "pytorch-interop",
            "version": torch_version(),
            "device": self.device,
            "dtype": str(self.dtype),
            "cuda_available": bool(self._torch.cuda.is_available()),
        }
