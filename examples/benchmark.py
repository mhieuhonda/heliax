"""Measure Heliax tensor and autograd throughput on the active backend."""

from __future__ import annotations

from time import perf_counter

import numpy as np

import heliax as hx


def timed(label: str, function, repeats: int = 20) -> float:
    function()
    started = perf_counter()
    for _ in range(repeats):
        function()
    elapsed = (perf_counter() - started) * 1000.0 / repeats
    print(f"{label:24s} {elapsed:8.3f} ms/iter")
    return elapsed


def main() -> None:
    rng = np.random.default_rng(0)
    left = hx.tensor(rng.normal(size=(512, 512)).astype(np.float32), requires_grad=True)
    right = hx.tensor(rng.normal(size=(512, 512)).astype(np.float32), requires_grad=True)
    target = hx.tensor(rng.normal(size=(512,)).astype(np.float32))

    print("Heliax benchmark", dict(hx.backend_info()))
    timed("matmul forward", lambda: left @ right, repeats=10)
    timed("elementwise add", lambda: left + right, repeats=30)
    timed("mean", lambda: left.mean(), repeats=30)

    def train_step() -> None:
        prediction = (left @ right).mean(axis=1)
        loss = hx.mean_squared_error(prediction, target)
        loss.backward()
        left.zero_grad()
        right.zero_grad()

    timed("autograd step", train_step, repeats=5)


if __name__ == "__main__":
    main()
