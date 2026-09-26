"""Small, repeatable microbenchmark helpers for backend selection."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from statistics import median
from time import perf_counter
from typing import Any


def time_callable(function: Callable[[], Any], *, warmup: int = 2, repeats: int = 7) -> float:
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(repeats):
        started = perf_counter()
        function()
        samples.append((perf_counter() - started) * 1000.0)
    return float(median(samples))


def select_fastest(
    candidates: Mapping[str, Callable[[], Any]], *, warmup: int = 2, repeats: int = 7
) -> tuple[str, dict[str, float]]:
    if not candidates:
        raise ValueError("at least one candidate is required")
    timings = {
        name: time_callable(function, warmup=warmup, repeats=repeats)
        for name, function in candidates.items()
    }
    return min(timings, key=timings.get), timings


def autotune_report(
    candidates: Mapping[str, Callable[[], Any]], *, warmup: int = 2, repeats: int = 7
) -> dict[str, Any]:
    winner, timings = select_fastest(candidates, warmup=warmup, repeats=repeats)
    return {"winner": winner, "median_ms": timings, "repeats": repeats, "warmup": warmup}
