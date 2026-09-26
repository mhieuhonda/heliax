"""Reusable temporary-buffer pools for layout and kernel hot paths."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class WorkspaceStats:
    hits: int = 0
    misses: int = 0
    allocated_bytes: int = 0
    released_bytes: int = 0
    live_buffers: int = 0


class Workspace:
    """Bounded pool of shape/dtype keyed NumPy scratch buffers."""

    def __init__(self, max_buffers_per_key: int = 4) -> None:
        if max_buffers_per_key <= 0:
            raise ValueError("max_buffers_per_key must be positive")
        self.max_buffers_per_key = int(max_buffers_per_key)
        self._buffers: dict[tuple[tuple[int, ...], np.dtype], list[np.ndarray]] = {}
        self.stats = WorkspaceStats()

    def acquire(self, shape: Sequence[int], dtype: Any = np.float32) -> np.ndarray:
        key = (tuple(int(size) for size in shape), np.dtype(dtype))
        pool = self._buffers.get(key, [])
        if pool:
            buffer = pool.pop()
            self.stats.hits += 1
            self.stats.live_buffers += 1
            return buffer
        buffer = np.empty(key[0], dtype=key[1])
        self.stats.misses += 1
        self.stats.allocated_bytes += int(buffer.nbytes)
        self.stats.live_buffers += 1
        return buffer

    def release(self, buffer: np.ndarray) -> None:
        if not isinstance(buffer, np.ndarray):
            raise TypeError("workspace release expects a NumPy array")
        key = (tuple(int(size) for size in buffer.shape), buffer.dtype)
        pool = self._buffers.setdefault(key, [])
        if len(pool) < self.max_buffers_per_key:
            pool.append(buffer)
            self.stats.released_bytes += int(buffer.nbytes)
        self.stats.live_buffers = max(self.stats.live_buffers - 1, 0)

    def clear(self) -> None:
        self._buffers.clear()
        self.stats.live_buffers = 0

    def report(self) -> dict[str, int]:
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "allocated_bytes": self.stats.allocated_bytes,
            "released_bytes": self.stats.released_bytes,
            "live_buffers": self.stats.live_buffers,
        }


@contextmanager
def workspace(max_buffers_per_key: int = 4) -> Iterator[Workspace]:
    pool = Workspace(max_buffers_per_key=max_buffers_per_key)
    try:
        yield pool
    finally:
        pool.clear()
