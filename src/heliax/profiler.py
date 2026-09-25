"""Lightweight timing and parameter utilities."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class ProfileResult:
    name: str
    elapsed_ms: float
    metadata: dict[str, object] = field(default_factory=dict)


@contextmanager
def profile(
    name: str = "block", results: list[ProfileResult] | None = None
) -> Iterator[ProfileResult]:
    record = ProfileResult(name=name, elapsed_ms=0.0)
    started = perf_counter()
    try:
        yield record
    finally:
        record.elapsed_ms = (perf_counter() - started) * 1000.0
        if results is not None:
            results.append(record)


def count_parameters(module: object) -> int:
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        raise TypeError("count_parameters expects a Module with parameters()")
    return sum(parameter.size for parameter in parameters())
