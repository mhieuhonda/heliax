"""Lightweight timing, graph, and memory utilities."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter

from .tensor import Tensor


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


def memory_bytes(tensor: Tensor) -> int:
    """Return the raw storage size of a tensor in bytes."""

    if not isinstance(tensor, Tensor):
        raise TypeError("memory_bytes expects a Tensor")
    return int(tensor.numpy().nbytes)


def graph_summary(tensor: Tensor) -> dict[str, int]:
    """Count nodes, edges, and retained array storage in a graph."""

    if not isinstance(tensor, Tensor):
        raise TypeError("graph_summary expects a Tensor")
    visited: set[int] = set()
    stack = [tensor]
    nodes = 0
    edges = 0
    storage = 0
    while stack:
        current = stack.pop()
        identity = id(current)
        if identity in visited:
            continue
        visited.add(identity)
        nodes += 1
        storage += memory_bytes(current)
        edges += len(current._prev)
        stack.extend(current._prev)
    return {"nodes": nodes, "edges": edges, "storage_bytes": storage}


def gradient_memory_report(module: object) -> dict[str, int]:
    """Account for gradient buffers and missing gradients on a module."""

    named_parameters = getattr(module, "named_parameters", None)
    if not callable(named_parameters):
        raise TypeError("gradient_memory_report expects a Module with named_parameters()")
    gradient_bytes = 0
    present = 0
    for _, parameter in named_parameters():
        if parameter.grad is not None:
            present += 1
            gradient_bytes += memory_bytes(parameter.grad)
    total = sum(memory_bytes(parameter) for _, parameter in named_parameters())
    return {
        "gradients": present,
        "gradient_bytes": gradient_bytes,
        "missing_gradients": max(len(list(named_parameters())) - present, 0),
        "parameter_plus_gradient_bytes": total + gradient_bytes,
    }


def memory_report(module: object) -> dict[str, int]:
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        raise TypeError("memory_report expects a Module with parameters()")
    parameter_bytes = sum(memory_bytes(parameter) for parameter in parameters())
    buffer_bytes = 0
    for _, buffer in getattr(module, "named_buffers", list)():
        buffer_bytes += int(buffer.nbytes)
    report = {
        "parameters": count_parameters(module),
        "parameter_bytes": parameter_bytes,
        "buffer_bytes": buffer_bytes,
        "total_bytes": parameter_bytes + buffer_bytes,
    }
    report.update(gradient_memory_report(module))
    return report


def training_memory_report(module: object, loss: Tensor) -> dict[str, int]:
    """Report retained graph storage plus parameter and gradient storage."""

    if not isinstance(loss, Tensor):
        raise TypeError("training_memory_report expects a loss Tensor")
    graph = graph_summary(loss)
    report = memory_report(module)
    report.update(
        {
            "graph_nodes": graph["nodes"],
            "graph_edges": graph["edges"],
            "graph_storage_bytes": graph["storage_bytes"],
            "total_training_bytes": graph["storage_bytes"]
            + report["parameter_plus_gradient_bytes"],
        }
    )
    return report


def model_summary(module: object) -> list[dict[str, object]]:
    """Return a compact per-parameter summary for reports and logs."""

    parameters = getattr(module, "named_parameters", None)
    if not callable(parameters):
        raise TypeError("model_summary expects a Module with named_parameters()")
    return [
        {
            "name": name,
            "shape": tuple(parameter.shape),
            "dtype": str(parameter.dtype),
            "elements": parameter.size,
        }
        for name, parameter in parameters()
    ]


def op_histogram(tensor: Tensor) -> dict[str, int]:
    """Return counts of operation names reachable from ``tensor``."""

    counts: dict[str, int] = {}
    stack = [tensor]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if current._op:
            counts[current._op] = counts.get(current._op, 0) + 1
        stack.extend(current._prev)
    return counts
