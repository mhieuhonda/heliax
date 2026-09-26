"""Benchmark native Heliax kernels against portable NumPy paths.

The report is descriptive only. Native dispatch remains opt-in and this script
never changes global backend state.
"""

from __future__ import annotations

import json
from time import perf_counter

import numpy as np

import heliax as hx


def _time(function, *, warmup: int = 2, repeats: int = 7) -> float:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(repeats):
        started = perf_counter()
        function()
        samples.append((perf_counter() - started) * 1000.0)
    return float(np.median(samples))


def build_report(*, repeats: int = 7) -> dict[str, object]:
    generator = np.random.default_rng(0)
    left = generator.normal(size=(256, 256)).astype(np.float32)
    right = generator.normal(size=(256, 256)).astype(np.float32)
    logits = generator.normal(size=(128, 16)).astype(np.float32)
    targets = generator.integers(0, 16, size=128, dtype=np.int64)
    gamma = np.ones(16, dtype=np.float32)
    beta = np.zeros(16, dtype=np.float32)
    parameter = np.ones(4096, dtype=np.float32)
    gradient = np.linspace(-1.0, 1.0, 4096, dtype=np.float32)
    first = np.zeros_like(parameter)
    second = np.zeros_like(parameter)
    report: dict[str, object] = {
        "available": hx.native_available(),
        "enabled": hx.native_ops.native_enabled(),
        "repeats": repeats,
    }
    if not hx.native_available():
        return report
    prediction = generator.normal(size=(128, 16)).astype(np.float32)
    target_values = generator.normal(size=(128, 16)).astype(np.float32)
    bce_logits = generator.normal(size=(128, 16)).astype(np.float32)
    bce_targets = generator.integers(0, 2, size=(128, 16)).astype(np.float32)
    cases = {
        "mae": (
            lambda: hx.native_ops.mae(prediction, target_values),
            lambda: hx.functional.mean_absolute_error(hx.tensor(prediction), target_values).numpy(),
        ),
        "bce_with_logits": (
            lambda: hx.native_ops.bce_with_logits(bce_logits, bce_targets),
            lambda: hx.functional.binary_cross_entropy(hx.tensor(bce_logits), bce_targets).numpy(),
        ),
        "huber": (
            lambda: hx.native_ops.huber(prediction, target_values, delta=0.75),
            lambda: hx.functional.huber_loss(
                hx.tensor(prediction), target_values, delta=0.75
            ).numpy(),
        ),
        "mse": (
            lambda: hx.native_ops.mse(prediction, target_values),
            lambda: hx.functional.mean_squared_error(hx.tensor(prediction), target_values).numpy(),
        ),
        "add_relu": (
            lambda: hx.native_ops.add_relu(left, right),
            lambda: np.maximum(left + right, 0.0),
        ),
        "silu": (
            lambda: hx.native_ops.silu(left),
            lambda: hx.functional.silu(hx.tensor(left)).numpy(),
        ),
        "gelu": (
            lambda: hx.native_ops.gelu(left),
            lambda: hx.functional.gelu(hx.tensor(left)).numpy(),
        ),
        "log_softmax_lastdim": (
            lambda: hx.native_ops.log_softmax_lastdim(logits),
            lambda: hx.functional.log_softmax(hx.tensor(logits)).numpy(),
        ),
        "softmax_lastdim": (
            lambda: hx.native_ops.softmax_lastdim(logits),
            lambda: hx.functional.softmax(hx.tensor(logits)).numpy(),
        ),
        "layernorm_lastdim": (
            lambda: hx.native_ops.layernorm_lastdim(logits, gamma, beta, 1e-5),
            lambda: hx.nn.LayerNorm(16)(hx.tensor(logits)).numpy(),
        ),
        "cross_entropy_lastdim": (
            lambda: hx.native_ops.cross_entropy_lastdim(logits, targets),
            lambda: hx.functional.fused_cross_entropy(hx.tensor(logits), targets).numpy(),
        ),
    }
    timings: dict[str, dict[str, float]] = {}
    for name, (native, portable) in cases.items():
        native_ms = _time(native, repeats=repeats)
        portable_ms = _time(portable, repeats=repeats)
        timings[name] = {
            "native_ms": native_ms,
            "portable_ms": portable_ms,
            "speedup": portable_ms / native_ms if native_ms else float("inf"),
        }
    adamw_kwargs = {
        "learning_rate": 0.001,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "weight_decay": 0.01,
        "bias_correction1": 0.1,
        "bias_correction2": 0.001,
    }

    def portable_adamw() -> None:
        first[:] = adamw_kwargs["beta1"] * first + (1.0 - adamw_kwargs["beta1"]) * gradient
        second[:] = (
            adamw_kwargs["beta2"] * second + (1.0 - adamw_kwargs["beta2"]) * gradient * gradient
        )
        update = (first / adamw_kwargs["bias_correction1"]) / (
            np.sqrt(second / adamw_kwargs["bias_correction2"]) + adamw_kwargs["epsilon"]
        )
        parameter[:] -= adamw_kwargs["learning_rate"] * (
            update + adamw_kwargs["weight_decay"] * parameter
        )

    native_adamw = lambda: hx.native_ops.adamw(parameter, gradient, first, second, **adamw_kwargs)
    timings["adamw"] = {
        "native_ms": _time(native_adamw, repeats=repeats),
        "portable_ms": _time(portable_adamw, repeats=repeats),
    }
    timings["adamw"]["speedup"] = timings["adamw"]["portable_ms"] / timings["adamw"]["native_ms"]
    report["timings"] = timings
    return report


def main() -> None:
    print(json.dumps(build_report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
