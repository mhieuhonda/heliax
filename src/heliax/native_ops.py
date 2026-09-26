"""Optional native CPU kernels loaded through ctypes.

The library is never required. Build it explicitly with:

    python scripts/build_native.py

Set ``HELIAX_DISABLE_NATIVE=1`` to force the portable NumPy path in tests or
benchmarks.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

import numpy as np

_LIB: ctypes.CDLL | None = None
_LOAD_ERROR: str | None = None


def _library_path() -> Path:
    override = os.environ.get("HELIAX_NATIVE_LIB")
    if override:
        return Path(override)
    return Path(__file__).with_name("_native.so")


def _load() -> ctypes.CDLL | None:
    global _LIB, _LOAD_ERROR
    if _LIB is not None:
        return _LIB
    if os.environ.get("HELIAX_DISABLE_NATIVE", "").lower() in {"1", "true", "yes"}:
        _LOAD_ERROR = "disabled by HELIAX_DISABLE_NATIVE"
        return None
    path = _library_path()
    if not path.is_file():
        _LOAD_ERROR = f"native library not found at {path}"
        return None
    try:
        library = ctypes.CDLL(str(path))
        library.hx_add_relu.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        library.hx_gelu.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        library.hx_softmax_lastdim.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        library.hx_huber.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_float,
        ]
        library.hx_mse.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        library.hx_cross_entropy_lastdim.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        library.hx_adamw.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
        ]
        library.hx_layernorm_lastdim.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
        ]
        for name in (
            "hx_add_relu",
            "hx_gelu",
            "hx_softmax_lastdim",
            "hx_huber",
            "hx_mse",
            "hx_cross_entropy_lastdim",
            "hx_layernorm_lastdim",
            "hx_adamw",
        ):
            getattr(library, name).restype = None
    except OSError as error:  # pragma: no cover - platform-specific
        _LOAD_ERROR = str(error)
        return None
    _LIB = library
    _LOAD_ERROR = None
    return library


def reload_native() -> bool:
    """Retry loading after a local native build in the same interpreter."""

    global _LIB, _LOAD_ERROR
    _LIB = None
    _LOAD_ERROR = None
    return _load() is not None


def native_available() -> bool:
    return _load() is not None


def native_enabled() -> bool:
    """Return whether the native path is explicitly enabled for dispatch."""

    return os.environ.get("HELIAX_NATIVE", "").lower() in {"1", "true", "yes", "on"}


def native_info() -> dict[str, Any]:
    library = _load()
    return {
        "available": library is not None,
        "enabled": native_enabled(),
        "library": str(_library_path()) if library is not None else None,
        "error": _LOAD_ERROR,
        "kernels": [
            "add_relu",
            "gelu",
            "huber",
            "mse",
            "softmax_lastdim",
            "cross_entropy_lastdim",
            "layernorm_lastdim",
            "adamw",
        ],
    }


def _float32_view(array: np.ndarray) -> np.ndarray:
    if array.dtype != np.float32 or not array.flags.c_contiguous:
        return np.ascontiguousarray(array, dtype=np.float32)
    return array


def add_relu(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    library = _load()
    if library is None:
        raise RuntimeError(_LOAD_ERROR or "native backend unavailable")
    left_array = _float32_view(np.asarray(left))
    right_array = _float32_view(np.asarray(right))
    if left_array.shape != right_array.shape:
        raise ValueError("add_relu operands must have identical shapes")
    output = np.empty_like(left_array)
    library.hx_add_relu(
        left_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        right_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.size,
    )
    return output


def gelu(value: np.ndarray) -> np.ndarray:
    library = _load()
    if library is None:
        raise RuntimeError(_LOAD_ERROR or "native backend unavailable")
    array = _float32_view(np.asarray(value))
    output = np.empty_like(array)
    library.hx_gelu(
        array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.size,
    )
    return output


def softmax_lastdim(value: np.ndarray) -> np.ndarray:
    library = _load()
    if library is None:
        raise RuntimeError(_LOAD_ERROR or "native backend unavailable")
    array = _float32_view(np.asarray(value))
    if array.ndim < 1 or array.shape[-1] == 0:
        raise ValueError("softmax_lastdim needs a non-empty last dimension")
    output = np.empty_like(array)
    library.hx_softmax_lastdim(
        array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        array.size // array.shape[-1],
        array.shape[-1],
    )
    return output


def huber(
    prediction: np.ndarray, target: np.ndarray, delta: float = 1.0
) -> tuple[float, np.ndarray]:
    library = _load()
    if library is None:
        raise RuntimeError(_LOAD_ERROR or "native backend unavailable")
    pred_input = np.asarray(prediction)
    target_input = np.asarray(target)
    try:
        result_shape = np.broadcast_shapes(pred_input.shape, target_input.shape)
    except ValueError as error:
        raise ValueError("huber operands are not broadcastable") from error
    pred = _float32_view(np.broadcast_to(pred_input, result_shape))
    target_array = _float32_view(np.broadcast_to(target_input, result_shape))
    loss = np.zeros(1, dtype=np.float32)
    gradient = np.empty_like(pred)
    library.hx_huber(
        pred.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        target_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        loss.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        gradient.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        pred.size,
        ctypes.c_float(delta),
    )
    return float(loss[0]), gradient


def mse(prediction: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    library = _load()
    if library is None:
        raise RuntimeError(_LOAD_ERROR or "native backend unavailable")
    pred_input = np.asarray(prediction)
    target_input = np.asarray(target)
    try:
        result_shape = np.broadcast_shapes(pred_input.shape, target_input.shape)
    except ValueError as error:
        raise ValueError("mse operands are not broadcastable") from error
    pred = _float32_view(np.broadcast_to(pred_input, result_shape))
    target_array = _float32_view(np.broadcast_to(target_input, result_shape))
    loss = np.zeros(1, dtype=np.float32)
    gradient = np.empty_like(pred)
    library.hx_mse(
        pred.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        target_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        loss.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        gradient.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        pred.size,
    )
    return float(loss[0]), gradient


def cross_entropy_lastdim(logits: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
    library = _load()
    if library is None:
        raise RuntimeError(_LOAD_ERROR or "native backend unavailable")
    array = _float32_view(np.asarray(logits))
    target_array = np.ascontiguousarray(np.asarray(targets), dtype=np.int64)
    if array.ndim < 1 or array.shape[-1] == 0:
        raise ValueError("cross_entropy_lastdim needs a non-empty last dimension")
    rows = array.size // array.shape[-1]
    if target_array.size != rows or (
        target_array.size and (target_array.min() < 0 or target_array.max() >= array.shape[-1])
    ):
        raise ValueError("cross entropy targets must contain one valid class index per row")
    loss = np.zeros(1, dtype=np.float32)
    gradient = np.empty_like(array)
    library.hx_cross_entropy_lastdim(
        array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        target_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        loss.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        gradient.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        rows,
        array.shape[-1],
    )
    return float(loss[0]), gradient


def layernorm_lastdim(
    value: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float
) -> np.ndarray:
    library = _load()
    if library is None:
        raise RuntimeError(_LOAD_ERROR or "native backend unavailable")
    array = _float32_view(np.asarray(value))
    gamma = _float32_view(np.asarray(weight))
    beta = _float32_view(np.asarray(bias))
    if array.ndim < 1 or gamma.shape != (array.shape[-1],) or beta.shape != gamma.shape:
        raise ValueError("layernorm_lastdim expects matching trailing weight/bias shapes")
    output = np.empty_like(array)
    library.hx_layernorm_lastdim(
        array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        gamma.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        beta.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        array.size // array.shape[-1],
        array.shape[-1],
        ctypes.c_float(eps),
    )
    return output


def adamw(
    parameter: np.ndarray,
    gradient: np.ndarray,
    first_moment: np.ndarray,
    second_moment: np.ndarray,
    *,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    weight_decay: float,
    bias_correction1: float,
    bias_correction2: float,
) -> None:
    library = _load()
    if library is None:
        raise RuntimeError(_LOAD_ERROR or "native backend unavailable")
    arrays = [
        _float32_view(np.asarray(item))
        for item in (parameter, gradient, first_moment, second_moment)
    ]
    if any(item.shape != arrays[0].shape for item in arrays):
        raise ValueError("adamw operands must have identical shapes")
    param, grad, first, second = arrays
    library.hx_adamw(
        param.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        first.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        second.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        param.size,
        ctypes.c_float(learning_rate),
        ctypes.c_float(beta1),
        ctypes.c_float(beta2),
        ctypes.c_float(epsilon),
        ctypes.c_float(weight_decay),
        ctypes.c_float(bias_correction1),
        ctypes.c_float(bias_correction2),
    )
