"""Small, backend-neutral data-parallel helpers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from .tensor import Parameter


class DistributedSampler:
    """Deterministic index sampler for one rank of a data-parallel job."""

    def __init__(
        self,
        dataset_length: int,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if dataset_length < 0 or num_replicas <= 0 or rank < 0 or rank >= num_replicas:
            raise ValueError("invalid distributed sampler configuration")
        self.dataset_length = int(dataset_length)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return self.dataset_length // self.num_replicas
        return (self.dataset_length + self.num_replicas - 1) // self.num_replicas

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            indices = np.random.default_rng(self.seed + self.epoch).permutation(self.dataset_length)
        else:
            indices = np.arange(self.dataset_length)
        if self.drop_last:
            usable = (self.dataset_length // self.num_replicas) * self.num_replicas
            indices = indices[:usable]
        indices = indices[self.rank :: self.num_replicas]
        return iter(int(index) for index in indices)


def reduce_gradients(parameters: Iterable[Parameter], world_size: int = 1) -> None:
    """Average gradients in place for a single-process simulation of DDP."""

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if world_size == 1:
        return
    parameters = list(parameters)
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad._data /= world_size
