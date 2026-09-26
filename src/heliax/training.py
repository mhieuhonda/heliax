"""High-level training loop with accumulation, scheduling, and evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .optim import clip_grad_norm_, step_scheduler
from .tensor import Tensor, no_grad


class CheckpointManager:
    """Keep recent and best atomic NPZ checkpoints for a training run."""

    def __init__(
        self,
        root: str | Any,
        model: Any,
        optimizer: Any,
        *,
        keep_last: int = 2,
        mode: str = "min",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        from pathlib import Path

        if keep_last <= 0 or mode not in {"min", "max"}:
            raise ValueError("CheckpointManager requires keep_last > 0 and mode min/max")
        self.root = Path(root)
        self.model = model
        self.optimizer = optimizer
        self.keep_last = int(keep_last)
        self.mode = mode
        self.metadata = dict(metadata or {})
        self.counter = 0
        self.best_metric: float | None = None
        self.best_path: Any | None = None
        self.paths: list[Any] = []

    def save(self, metric: float | None = None, metadata: dict[str, Any] | None = None) -> Any:
        from .serialization import save_checkpoint

        self.root.mkdir(parents=True, exist_ok=True)
        self.counter += 1
        path = self.root / f"checkpoint-{self.counter:06d}.npz"
        values = dict(self.metadata)
        values.update(metadata or {})
        if metric is not None:
            values["metric"] = float(metric)
        save_checkpoint(path, self.model, self.optimizer, metadata=values)
        self.paths.append(path)
        if metric is not None and (
            self.best_metric is None
            or (self.mode == "min" and metric < self.best_metric)
            or (self.mode == "max" and metric > self.best_metric)
        ):
            self.best_metric = float(metric)
            self.best_path = path
        self._prune()
        return path

    def _prune(self) -> None:
        keep = set(self.paths[-self.keep_last :])
        if self.best_path is not None:
            keep.add(self.best_path)
        for path in self.paths:
            if path not in keep and path.exists():
                path.unlink()
        self.paths = [path for path in self.paths if path.exists()]

    def load_best(self) -> dict[str, Any]:
        from .serialization import load_checkpoint

        if self.best_path is None:
            raise RuntimeError("no best checkpoint has been saved")
        return load_checkpoint(self.best_path, self.model, self.optimizer)

    @property
    def latest_path(self) -> Any | None:
        return self.paths[-1] if self.paths else None


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
        grad_scaler: Any = None,
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
        self.grad_scaler = grad_scaler
        self.max_steps = max_steps
        self.epoch = 0
        self.global_step = 0
        self.history: list[dict[str, float]] = []

    def _apply_step(self, loss: Tensor) -> None:
        parameters = self.model.parameters()
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_gradients(parameters)
        if self.grad_clip is not None:
            clip_grad_norm_(parameters, self.grad_clip)
        if self.grad_scaler is not None:
            self.grad_scaler.step(self.optimizer, parameters)
        else:
            self.optimizer.step()
        if self.scheduler is not None:
            step_scheduler(self.scheduler, float(loss.item()))
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
            backward_loss = (
                self.grad_scaler.scale_loss(loss) if self.grad_scaler is not None else loss
            )
            (backward_loss / self.gradient_accumulation_steps).backward()
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
