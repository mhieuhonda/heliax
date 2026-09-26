"""High-level training loop with accumulation, scheduling, and evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .optim import clip_grad_norm_
from .tensor import Tensor, no_grad


class Trainer:
    """Small explicit trainer for Heliax models.

    It intentionally exposes the loop instead of hiding it behind a framework
    object, so custom losses, samplers, and backend selection remain easy to
    override.
    """

    def __init__(
        self,
        model: Any,
        optimizer: Any,
        loss_fn: Any,
        *,
        epochs: int = 1,
        gradient_accumulation_steps: int = 1,
        grad_clip: float | None = None,
        scheduler: Any = None,
        max_steps: int | None = None,
    ) -> None:
        if epochs <= 0 or gradient_accumulation_steps <= 0:
            raise ValueError("epochs and gradient_accumulation_steps must be positive")
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be positive when provided")
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = int(epochs)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.grad_clip = grad_clip
        self.scheduler = scheduler
        self.max_steps = max_steps
        self.epoch = 0
        self.global_step = 0
        self.history: list[dict[str, float]] = []

    def _apply_step(self, loss: Tensor) -> None:
        parameters = self.model.parameters()
        if self.grad_clip is not None:
            clip_grad_norm_(parameters, self.grad_clip)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.global_step += 1

    def train_epoch(self, loader: Iterable[tuple[Tensor, ...]]) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        total = 0.0
        examples = 0
        pending = 0
        for batch in loader:
            if not batch:
                continue
            inputs, targets = batch[0], batch[1]
            loss = self.loss_fn(self.model(inputs), targets)
            (loss / self.gradient_accumulation_steps).backward()
            total += float(loss.item()) * (inputs.shape[0] if inputs.shape else 1)
            examples += inputs.shape[0] if inputs.shape else 1
            pending += 1
            if pending == self.gradient_accumulation_steps:
                self._apply_step(loss)
                self.optimizer.zero_grad()
                pending = 0
                if self.max_steps is not None and self.global_step >= self.max_steps:
                    break
        if pending and (self.max_steps is None or self.global_step < self.max_steps):
            self._apply_step(loss)
        average = total / max(examples, 1)
        return {"loss": average, "steps": float(self.global_step), "examples": float(examples)}

    def fit(self, loader: Iterable[tuple[Tensor, ...]]) -> list[dict[str, float]]:
        results = []
        for _ in range(self.epochs):
            metrics = self.train_epoch(loader)
            self.epoch += 1
            metrics["epoch"] = float(self.epoch)
            results.append(metrics)
            self.history.append(metrics)
            if self.max_steps is not None and self.global_step >= self.max_steps:
                break
        return results

    def evaluate(self, loader: Iterable[tuple[Tensor, ...]]) -> dict[str, float]:
        self.model.eval()
        total = 0.0
        examples = 0
        with no_grad():
            for batch in loader:
                if not batch:
                    continue
                inputs, targets = batch[0], batch[1]
                loss = self.loss_fn(self.model(inputs), targets)
                total += float(loss.item()) * (inputs.shape[0] if inputs.shape else 1)
                examples += inputs.shape[0] if inputs.shape else 1
        return {"loss": total / max(examples, 1), "examples": float(examples)}

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "history": list(self.history),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.epoch = int(state.get("epoch", 0))
        self.global_step = int(state.get("global_step", 0))
        self.history = list(state.get("history", []))

    def save(self, path: str) -> str:
        from .serialization import save_checkpoint

        save_checkpoint(
            path,
            self.model,
            self.optimizer,
            metadata={"epoch": self.epoch, "global_step": self.global_step},
        )
        return path

    def load(self, path: str) -> dict[str, Any]:
        from .serialization import load_checkpoint

        return load_checkpoint(path, self.model, self.optimizer)
