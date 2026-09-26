"""Optional PyTorch-backed numerical kernels for Heliax.

This backend is explicit: install the ``torch`` extra, then call
``hx.set_backend("torch")``. Heliax tensors still expose NumPy storage, so the
adapter is a compatibility/performance bridge rather than a silent device
abstraction.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .backend import NumpyBackend


class TorchBackend(NumpyBackend):
    """NumPy-shaped adapter that dispatches math through PyTorch."""

    def __init__(self, device: str = "cpu") -> None:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("TorchBackend requires the optional 'torch' extra") from error
        self._torch = torch
        self.device = device
        super().__init__(name="torch", supports_fused_kernels=True)

    def _to(self, value: Any):
        return self._torch.as_tensor(np.asarray(value), device=self.device)

    def _out(self, value: Any) -> np.ndarray:
        return value.detach().cpu().numpy()

    def matmul(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return self._out(self._torch.matmul(self._to(left), self._to(right)))

    def exp(self, value: np.ndarray) -> np.ndarray:
        return self._out(self._torch.exp(self._to(value)))

    def log(self, value: np.ndarray) -> np.ndarray:
        return self._out(self._torch.log(self._to(value)))

    def sqrt(self, value: np.ndarray) -> np.ndarray:
        return self._out(self._torch.sqrt(self._to(value)))

    def tanh(self, value: np.ndarray) -> np.ndarray:
        return self._out(self._torch.tanh(self._to(value)))

    def sigmoid(self, value: np.ndarray) -> np.ndarray:
        return self._out(self._torch.sigmoid(self._to(value)))

    def relu(self, value: np.ndarray) -> np.ndarray:
        return self._out(self._torch.relu(self._to(value)))

    def gelu(self, value: np.ndarray) -> np.ndarray:
        return self._out(self._torch.nn.functional.gelu(self._to(value), approximate="tanh"))

    def silu(self, value: np.ndarray) -> np.ndarray:
        return self._out(self._torch.nn.functional.silu(self._to(value)))

    def softmax(self, value: np.ndarray, axis: int = -1) -> np.ndarray:
        return self._out(self._torch.softmax(self._to(value), dim=axis))

    def log_softmax(self, value: np.ndarray, axis: int = -1) -> np.ndarray:
        return self._out(self._torch.log_softmax(self._to(value), dim=axis))

    def layernorm(
        self,
        value: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray,
        axes: tuple[int, ...],
        eps: float,
    ) -> np.ndarray:
        tensor = self._to(value)
        reduce_axes = tuple(axis for axis in range(tensor.ndim) if axis in axes)
        shape = [1] * tensor.ndim
        for axis in reduce_axes:
            shape[axis] = tensor.shape[axis]
        mean = tensor.mean(dim=reduce_axes, keepdim=True)
        variance = (tensor - mean).pow(2).mean(dim=reduce_axes, keepdim=True)
        gamma = self._to(weight).reshape(shape)
        beta = self._to(bias).reshape(shape)
        return self._out((tensor - mean) / self._torch.sqrt(variance + eps) * gamma + beta)

    def adamw_update(
        self,
        parameter: np.ndarray,
        gradient: np.ndarray,
        first_moment: np.ndarray,
        second_moment: np.ndarray,
        learning_rate: float,
        beta1: float,
        beta2: float,
        epsilon: float,
        weight_decay: float,
        bias_correction1: float,
        bias_correction2: float,
    ) -> None:
        with self._torch.no_grad():
            param = self._to(parameter)
            grad = self._to(gradient)
            first = self._to(first_moment)
            second = self._to(second_moment)
            first.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            second.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            first_hat = first / bias_correction1
            second_hat = second / bias_correction2
            param.addcdiv_(first_hat, second_hat.sqrt().add_(epsilon), value=-learning_rate)
            param.add_(param, alpha=-learning_rate * weight_decay)
            parameter[...] = self._out(param)
            first_moment[...] = self._out(first)
            second_moment[...] = self._out(second)


_BACKEND: TorchBackend | None = None


def get_torch_backend(device: str = "cpu") -> TorchBackend:
    global _BACKEND
    if _BACKEND is None or _BACKEND.device != device:
        _BACKEND = TorchBackend(device=device)
    return _BACKEND


def torch_backend_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True
