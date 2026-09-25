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

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(self.dataset, TensorDataset):
            raise TypeError("DataLoader currently expects a TensorDataset")

    def __iter__(self) -> Iterator[tuple[Tensor, ...]]:
        length = len(self.dataset)
        indices = np.arange(length)
        if self.shuffle:
            generator = self.generator or np.random.default_rng()
            generator.shuffle(indices)
        limit = length - (length % self.batch_size) if self.drop_last else length
        for start in range(0, limit, self.batch_size):
            selected = indices[start : start + self.batch_size]
            yield tuple(Tensor(item.numpy()[selected]) for item in self.dataset.tensors)

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def fit(
    model: Any,
    loader: Iterable[tuple[Tensor, ...]],
    optimizer: Any,
    loss_fn: Any,
    *,
    epochs: int = 1,
    grad_clip: float | None = None,
    on_epoch_end: Any = None,
) -> list[float]:
    """Run a compact training loop for a model exposing ``__call__``."""

    from .optim import clip_grad_norm_

    history: list[float] = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        examples = 0
        for batch in loader:
            if not batch:
                continue
            inputs, targets = batch[0], batch[1]
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()
            if grad_clip is not None:
                clip_grad_norm_(getattr(model, "parameters", list)(), grad_clip)
            optimizer.step()
            batch_size = inputs.shape[0] if inputs.shape else 1
            total_loss += float(loss.item()) * batch_size
            examples += batch_size
        average = total_loss / max(examples, 1)
        history.append(average)
        if on_epoch_end:
            on_epoch_end(epoch, average, model)
    return history
