"""Portable NPZ checkpoints and state-dict helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .nn import Module
from .optim import Optimizer

FORMAT_VERSION = 1


def save_state_dict(path: str | Path, state: dict[str, np.ndarray]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, **state)
    return target


def load_state_dict(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def save_checkpoint(
    path: str | Path,
    model: Module,
    optimizer: Optimizer | None = None,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    state: dict[str, np.ndarray] = {
        "format_version": np.asarray(FORMAT_VERSION, dtype=np.int64),
        **{f"model.{key}": value for key, value in model.state_dict().items()},
    }
    if optimizer is not None:
        for index, values in optimizer.state.items():
            for key, value in values.items():
                if isinstance(value, np.ndarray):
                    state[f"optimizer.{index}.{key}"] = value
        state["optimizer.defaults.n"] = np.asarray(len(optimizer.parameters), dtype=np.int64)
    for key, value in (metadata or {}).items():
        if isinstance(value, (str, int, float, bool)):
            state[f"meta.{key}"] = np.asarray(value)
    return save_state_dict(path, state)


def load_checkpoint(
    path: str | Path, model: Module, optimizer: Optimizer | None = None
) -> dict[str, Any]:
    archive = load_state_dict(path)
    version = int(np.asarray(archive.get("format_version", FORMAT_VERSION)).item())
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported Heliax checkpoint format: {version}")
    model_state = {
        key.removeprefix("model."): archive[key] for key in archive if key.startswith("model.")
    }
    model.load_state_dict(model_state)
    metadata: dict[str, Any] = {}
    for key, value in archive.items():
        if key.startswith("meta."):
            metadata[key.removeprefix("meta.")] = (
                value.item() if np.asarray(value).ndim == 0 else value
            )
    if optimizer is not None:
        for index, values in optimizer.state.items():
            for key in list(values):
                key_name = f"optimizer.{index}.{key}"
                if key_name in archive:
                    values[key] = archive[key_name]
    return metadata
