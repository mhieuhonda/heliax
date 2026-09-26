"""Precision policies and gradient scaling for mixed-precision experiments."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from .tensor import Parameter, Tensor, no_grad


@contextmanager
def autocast(dtype: Any = np.float16, enabled: bool = True) -> Iterator[Any]:
    """Record a precision policy without silently changing Tensor storage.

    Heliax 0.3 keeps the portable backend in float32 by default. This context
    is a deliberate hook for future accelerator backends and makes the policy
    explicit instead of hiding dtype changes.
    """

    previous = dtype if enabled else None
    yield previous


class GradScaler:
    """Dynamic loss scaling compatible with the Heliax autograd engine."""

    def __init__(
        self,
        initial_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ) -> None:
        if (
            initial_scale <= 0
            or growth_factor <= 1
            or not 0 < backoff_factor < 1
            or growth_interval <= 0
        ):
            raise ValueError("invalid GradScaler configuration")
        self.scale = float(initial_scale)
        self.growth_factor = float(growth_factor)
        self.backoff_factor = float(backoff_factor)
        self.growth_interval = int(growth_interval)
        self._growth_tracker = 0
        self._found_inf = False

    def scale_loss(self, loss: Tensor) -> Tensor:
        return loss * self.scale

    def unscale_gradients(self, parameters: list[Parameter]) -> None:
        self._found_inf = False
        with no_grad():
            for parameter in parameters:
                if parameter.grad is None:
                    continue
                if not np.all(np.isfinite(parameter.grad.numpy())):
                    self._found_inf = True
                parameter.grad._data /= self.scale

    def step(self, optimizer: Any, parameters: list[Parameter]) -> None:
        if self._found_inf:
            self.scale *= self.backoff_factor
            self._growth_tracker = 0
            optimizer.zero_grad()
            return
        optimizer.step()
        self._growth_tracker += 1
        if self._growth_tracker >= self.growth_interval:
            self.scale *= self.growth_factor
            self._growth_tracker = 0

    def state_dict(self) -> dict[str, float | int]:
        return {
            "scale": self.scale,
            "growth_factor": self.growth_factor,
            "backoff_factor": self.backoff_factor,
            "growth_interval": self.growth_interval,
            "growth_tracker": self._growth_tracker,
        }

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        self.scale = float(state["scale"])
        self.growth_factor = float(state["growth_factor"])
        self.backoff_factor = float(state["backoff_factor"])
        self.growth_interval = int(state["growth_interval"])
        self._growth_tracker = int(state["growth_tracker"])
