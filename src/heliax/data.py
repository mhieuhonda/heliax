"""Data utilities and a tiny training loop."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .tensor import Tensor


def seed_everything(seed: int) -> np.random.Generator:
    """Seed NumPy and return a local generator for repeatable experiments."""

    np.random.seed(seed)
    return np.random.default_rng(seed)


class TensorDataset:
    def __init__(self, *tensors: Tensor | np.ndarray | Sequence[Any]) -> None:
        if not tensors:
            raise ValueError("TensorDataset requires at least one tensor")
        self.tensors = [
            item if isinstance(item, Tensor) else Tensor(np.asarray(item)) for item in tensors
        ]
        lengths = {item.shape[0] for item in self.tensors}
        if len(lengths) != 1:
            raise ValueError("all dataset tensors must have the same first dimension")

    def __len__(self) -> int:
        return self.tensors[0].shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        return tuple(item[index] for item in self.tensors)


@dataclass
class DataLoader:
    dataset: TensorDataset
    batch_size: int
    shuffle: bool = False
    drop_last: bool = False
    generator: np.random.Generator | None = None
    sampler: Iterable[int] | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(self.dataset, TensorDataset):
            raise TypeError("DataLoader currently expects a TensorDataset")
        if self.sampler is not None and self.shuffle:
            raise ValueError("DataLoader cannot combine an explicit sampler with shuffle=True")

    def __iter__(self) -> Iterator[tuple[Tensor, ...]]:
        if self.sampler is not None:
            indices = np.asarray(list(self.sampler), dtype=np.int64)
        else:
            indices = np.arange(len(self.dataset))
            if self.shuffle:
                generator = self.generator or np.random.default_rng()
                generator.shuffle(indices)
        length = len(indices)
        limit = length - (length % self.batch_size) if self.drop_last else length
        for start in range(0, limit, self.batch_size):
            selected = indices[start : start + self.batch_size]
            yield tuple(Tensor(item.numpy()[selected]) for item in self.dataset.tensors)

    def __len__(self) -> int:
        length = len(self.sampler) if self.sampler is not None else len(self.dataset)
        if self.drop_last:
            return length // self.batch_size
        return (length + self.batch_size - 1) // self.batch_size


def fit(
    model: Any,
    loader: Iterable[tuple[Tensor, ...]],
    optimizer: Any,
    loss_fn: Any,
    *,
    epochs: int = 1,
    grad_clip: float | None = None,
    gradient_accumulation_steps: int = 1,
    scheduler: Any = None,
    max_steps: int | None = None,
    on_epoch_end: Any = None,
) -> list[float]:
    """Run a compact training loop for a model exposing ``__call__``.

    Gradient accumulation, schedulers, clipping, and a global ``max_steps``
    bound are optional so the default remains a small readable reference loop.
    """

    from .optim import clip_grad_norm_

    if epochs <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("epochs and gradient_accumulation_steps must be positive")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when provided")
    parameters = getattr(model, "parameters", list)()
    history: list[float] = []
    global_steps = 0
    stop = False
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        examples = 0
        optimizer.zero_grad()
        pending_batches = 0
        for batch in loader:
            if not batch:
                continue
            inputs, targets = batch[0], batch[1]
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            (loss / gradient_accumulation_steps).backward()
            pending_batches += 1
            batch_size = inputs.shape[0] if inputs.shape else 1
            total_loss += float(loss.item()) * batch_size
            examples += batch_size
            if pending_batches == gradient_accumulation_steps:
                if grad_clip is not None:
                    clip_grad_norm_(parameters, grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                pending_batches = 0
                global_steps += 1
                if scheduler is not None:
                    scheduler.step()
                if max_steps is not None and global_steps >= max_steps:
                    stop = True
                    break
        if pending_batches and not stop:
            if grad_clip is not None:
                clip_grad_norm_(parameters, grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            global_steps += 1
        average = total_loss / max(examples, 1)
        history.append(average)
        if on_epoch_end:
            on_epoch_end(epoch, average, model)
        if stop:
            break
    return history
