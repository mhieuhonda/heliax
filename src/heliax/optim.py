"""Optimizers, gradient clipping, and learning-rate schedules."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

from . import native_ops
from .backend import get_backend
from .tensor import Parameter, Tensor, no_grad


class Optimizer:
    def __init__(
        self, parameters: Iterable[Parameter], defaults: dict[str, Any] | None = None
    ) -> None:
        self.parameters = list(parameters)
        if not self.parameters:
            raise ValueError("optimizer requires at least one parameter")
        self.defaults = defaults or {}
        self.state: dict[int, dict[str, Any]] = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            if set_to_none:
                parameter.zero_grad()
            elif parameter.grad is not None:
                parameter.grad._data.fill(0)

    def step(self) -> None:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        return {
            "defaults": dict(self.defaults),
            "state": {
                str(index): {
                    key: (value.copy() if isinstance(value, np.ndarray) else value)
                    for key, value in state.items()
                }
                for index, state in self.state.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.defaults = dict(state.get("defaults", self.defaults))
        self.state = {int(index): dict(values) for index, values in state.get("state", {}).items()}


class SGD(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float = 1e-2,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        super().__init__(
            parameters,
            {
                "lr": float(lr),
                "momentum": float(momentum),
                "weight_decay": float(weight_decay),
                "nesterov": bool(nesterov),
            },
        )
        if momentum < 0 or weight_decay < 0:
            raise ValueError("momentum and weight_decay must be non-negative")
        if nesterov and momentum <= 0:
            raise ValueError("nesterov SGD requires momentum > 0")

    def step(self) -> None:
        options = self.defaults
        backend = get_backend()
        for index, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue
            gradient = parameter.grad.numpy()
            state = self.state.setdefault(index, {})
            momentum_buffer = state.get("momentum_buffer")
            with no_grad():
                state["momentum_buffer"] = backend.sgd_update(
                    parameter._data,
                    gradient,
                    float(options["lr"]),
                    float(options["weight_decay"]),
                    momentum_buffer,
                    float(options["momentum"]),
                    bool(options["nesterov"]),
                )


class Adagrad(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float = 1e-2,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        initial_accumulator: float = 0.0,
    ) -> None:
        super().__init__(
            parameters,
            {
                "lr": float(lr),
                "eps": float(eps),
                "weight_decay": float(weight_decay),
                "initial_accumulator": float(initial_accumulator),
            },
        )
        if lr < 0 or eps <= 0 or weight_decay < 0 or initial_accumulator < 0:
            raise ValueError("invalid Adagrad hyperparameters")

    def step(self) -> None:
        options = self.defaults
        for index, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue
            state = self.state.setdefault(index, {})
            if "sum" not in state:
                state["sum"] = np.full_like(
                    parameter._data, options["initial_accumulator"], dtype=np.float32
                )
            gradient = parameter.grad.numpy() + options["weight_decay"] * parameter.numpy()
            if (
                parameter._data.dtype == np.float32
                and native_ops.native_enabled()
                and native_ops.native_available()
            ):
                with no_grad():
                    native_ops.adagrad(
                        parameter._data,
                        gradient,
                        state["sum"],
                        learning_rate=options["lr"],
                        epsilon=options["eps"],
                    )
            else:
                state["sum"] += gradient * gradient
                with no_grad():
                    parameter._data -= (
                        options["lr"] * gradient / (np.sqrt(state["sum"]) + options["eps"])
                    )


class Adam(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(
            parameters,
            {
                "lr": float(lr),
                "beta1": float(betas[0]),
                "beta2": float(betas[1]),
                "eps": float(eps),
                "weight_decay": float(weight_decay),
            },
        )
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1 or eps <= 0 or weight_decay < 0:
            raise ValueError("invalid Adam hyperparameters")

    def step(self) -> None:
        options = self.defaults
        backend = get_backend()
        for index, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue
            state = self.state.setdefault(index, {})
            if "step" not in state:
                state["step"] = 0
                state["first_moment"] = np.zeros_like(parameter._data, dtype=np.float32)
                state["second_moment"] = np.zeros_like(parameter._data, dtype=np.float32)
            state["step"] += 1
            beta1, beta2 = options["beta1"], options["beta2"]
            with no_grad():
                backend.adam_update(
                    parameter._data,
                    parameter.grad.numpy(),
                    state["first_moment"],
                    state["second_moment"],
                    float(options["lr"]),
                    beta1,
                    beta2,
                    float(options["eps"]),
                    float(options["weight_decay"]),
                    1.0 - beta1 ** state["step"],
                    1.0 - beta2 ** state["step"],
                )


class AdamW(Optimizer):
    """AdamW with decoupled weight decay."""

    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        super().__init__(
            parameters,
            {
                "lr": float(lr),
                "beta1": float(betas[0]),
                "beta2": float(betas[1]),
                "eps": float(eps),
                "weight_decay": float(weight_decay),
            },
        )
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1 or eps <= 0 or weight_decay < 0:
            raise ValueError("invalid AdamW hyperparameters")

    def step(self) -> None:
        options = self.defaults
        backend = get_backend()
        for index, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue
            state = self.state.setdefault(index, {})
            if "step" not in state:
                state["step"] = 0
                state["first_moment"] = np.zeros_like(parameter._data, dtype=np.float32)
                state["second_moment"] = np.zeros_like(parameter._data, dtype=np.float32)
            state["step"] += 1
            beta1, beta2 = options["beta1"], options["beta2"]
            with no_grad():
                backend.adamw_update(
                    parameter._data,
                    parameter.grad.numpy(),
                    state["first_moment"],
                    state["second_moment"],
                    float(options["lr"]),
                    beta1,
                    beta2,
                    float(options["eps"]),
                    float(options["weight_decay"]),
                    1.0 - beta1 ** state["step"],
                    1.0 - beta2 ** state["step"],
                )


class RMSProp(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float = 1e-3,
        decay: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(parameters, {"lr": float(lr), "decay": float(decay), "eps": float(eps)})
        if decay < 0 or eps <= 0 or lr < 0:
            raise ValueError("invalid RMSProp hyperparameters")

    def step(self) -> None:
        options = self.defaults
        backend = get_backend()
        for index, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue
            state = self.state.setdefault(index, {})
            if "square_average" not in state:
                state["square_average"] = np.zeros_like(parameter._data, dtype=np.float32)
            with no_grad():
                backend.rmsprop_update(
                    parameter._data,
                    parameter.grad.numpy(),
                    state["square_average"],
                    float(options["lr"]),
                    float(options["decay"]),
                    float(options["eps"]),
                )


class ModelEMA:
    """Exponential moving average of model parameters."""

    def __init__(self, module: Any, decay: float = 0.999, warmup_steps: int = 0) -> None:
        if not 0.0 <= decay < 1.0 or warmup_steps < 0:
            raise ValueError("ModelEMA decay/warmup configuration is invalid")
        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)
        self.step_count = 0
        self.shadow = {
            name: parameter.numpy().copy() for name, parameter in module.named_parameters()
        }

    def update(self, module: Any) -> None:
        self.step_count += 1
        decay = (
            min(self.decay, (1.0 + self.step_count) / (10.0 + self.step_count))
            if self.warmup_steps
            else self.decay
        )
        for name, parameter in module.named_parameters():
            if name in self.shadow:
                self.shadow[name] = decay * self.shadow[name] + (1.0 - decay) * parameter.numpy()

    def copy_to(self, module: Any) -> None:
        with no_grad():
            for name, parameter in module.named_parameters():
                if name in self.shadow:
                    parameter.data = self.shadow[name].copy()

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "step_count": self.step_count,
            "shadow": {key: value.copy() for key, value in self.shadow.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.warmup_steps = int(state["warmup_steps"])
        self.step_count = int(state["step_count"])
        self.shadow = {key: np.asarray(value).copy() for key, value in state["shadow"].items()}


def clip_grad_norm_(
    parameters: Iterable[Parameter], max_norm: float, norm_type: float = 2.0
) -> Tensor:
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    parameters = list(parameters)
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(np.sum(np.abs(parameter.grad.numpy()) ** norm_type, dtype=np.float64))
    total = total ** (1.0 / norm_type)
    scale = min(1.0, max_norm / (total + 1e-12))
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad._data *= dtype_scale(parameter.grad.dtype, scale)
    return Tensor(np.asarray(total, dtype=np.float32), requires_grad=False)


def dtype_scale(dtype: np.dtype, scale: float) -> np.dtype:
    return np.asarray(scale, dtype=dtype)


class LRScheduler:
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None:
        self.optimizer = optimizer
        self.last_epoch = int(last_epoch)
        self.base_lrs = [float(optimizer.defaults["lr"])]

    def get_lr(self) -> float:
        return float(self.optimizer.defaults["lr"])

    def step(self) -> float:
        self.last_epoch += 1
        value = self._compute_lr()
        self.optimizer.defaults["lr"] = value
        return value

    def _compute_lr(self) -> float:
        return self.base_lrs[0]


class StepLR(LRScheduler):
    def __init__(
        self, optimizer: Optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1
    ) -> None:
        super().__init__(optimizer, last_epoch)
        self.step_size = int(step_size)
        self.gamma = float(gamma)

    def _compute_lr(self) -> float:
        return self.base_lrs[0] * (self.gamma ** (self.last_epoch // self.step_size))


class ExponentialLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, gamma: float = 0.9, last_epoch: int = -1) -> None:
        super().__init__(optimizer, last_epoch)
        if gamma <= 0:
            raise ValueError("ExponentialLR gamma must be positive")
        self.gamma = float(gamma)

    def _compute_lr(self) -> float:
        return self.base_lrs[0] * (self.gamma**self.last_epoch)


class ReduceLROnPlateau:
    def __init__(
        self,
        optimizer: Optimizer,
        *,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 5,
        min_lr: float = 0.0,
        threshold: float = 1e-4,
    ) -> None:
        if mode not in {"min", "max"} or not 0.0 < factor < 1.0 or patience < 0 or min_lr < 0:
            raise ValueError("invalid ReduceLROnPlateau hyperparameters")
        self.optimizer = optimizer
        self.mode = mode
        self.factor = float(factor)
        self.patience = int(patience)
        self.min_lr = float(min_lr)
        self.threshold = float(threshold)
        self.best = float("inf") if mode == "min" else float("-inf")
        self.num_bad_epochs = 0

    def step(self, metric: float) -> float:
        metric = float(metric)
        improved = (
            metric < self.best - self.threshold
            if self.mode == "min"
            else metric > self.best + self.threshold
        )
        if improved:
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        if self.num_bad_epochs > self.patience:
            current = max(self.optimizer.defaults["lr"] * self.factor, self.min_lr)
            self.optimizer.defaults["lr"] = current
            self.num_bad_epochs = 0
        return float(self.optimizer.defaults["lr"])


class CosineAnnealingLR(LRScheduler):
    def __init__(
        self, optimizer: Optimizer, t_max: int, eta_min: float = 0.0, last_epoch: int = -1
    ) -> None:
        super().__init__(optimizer, last_epoch)
        self.t_max = int(t_max)
        self.eta_min = float(eta_min)

    def _compute_lr(self) -> float:
        if self.t_max <= 0:
            return self.eta_min
        progress = min(1.0, max(0.0, self.last_epoch / self.t_max))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.eta_min + (self.base_lrs[0] - self.eta_min) * cosine


def step_scheduler(scheduler: Any, metric: float | None = None) -> float | None:
    """Advance fixed-step or metric-driven schedulers uniformly."""

    if hasattr(scheduler, "patience") and hasattr(scheduler, "mode"):
        if metric is None:
            raise ValueError("metric-driven schedulers require a metric value")
        return float(scheduler.step(float(metric)))
    return float(scheduler.step())


class CosineAnnealingWarmRestarts(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        first_cycle_steps: int,
        *,
        min_lr: float = 0.0,
        factor: float = 0.5,
        warmup_steps: int = 0,
        last_epoch: int = -1,
    ) -> None:
        super().__init__(optimizer, last_epoch)
        if first_cycle_steps <= 0 or min_lr < 0 or factor <= 0 or warmup_steps < 0:
            raise ValueError("invalid warm-restart scheduler hyperparameters")
        self.first_cycle_steps = int(first_cycle_steps)
        self.min_lr = float(min_lr)
        self.factor = float(factor)
        self.warmup_steps = int(warmup_steps)

    def _compute_lr(self) -> float:
        if self.warmup_steps and self.last_epoch < self.warmup_steps:
            progress = self.last_epoch / max(self.warmup_steps, 1)
            return self.min_lr + (self.base_lrs[0] - self.min_lr) * progress
        cycle = (self.last_epoch - self.warmup_steps) // self.first_cycle_steps
        cycle_max = self.base_lrs[0] * (self.factor**cycle)
        position = (self.last_epoch - self.warmup_steps) % self.first_cycle_steps
        progress = position / self.first_cycle_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (cycle_max - self.min_lr) * cosine
