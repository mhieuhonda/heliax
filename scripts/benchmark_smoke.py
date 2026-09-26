"""Small CI-safe performance smoke test for Heliax."""

from __future__ import annotations

from time import perf_counter

import numpy as np

import heliax as hx


def main() -> None:
    generator = np.random.default_rng(0)
    left = hx.tensor(generator.normal(size=(128, 128)).astype(np.float32), requires_grad=True)
    right = hx.tensor(generator.normal(size=(128, 128)).astype(np.float32), requires_grad=True)
    for _ in range(2):
        output = (left @ right).mean()
        output.backward()
        left.zero_grad()
        right.zero_grad()
    started = perf_counter()
    iterations = 5
    for _ in range(iterations):
        output = (left @ right).mean()
        output.backward()
        left.zero_grad()
        right.zero_grad()
    elapsed = perf_counter() - started
    assert np.isfinite(elapsed) and elapsed > 0
    print(f"heliax benchmark smoke: {iterations / elapsed:.2f} autograd steps/s")


if __name__ == "__main__":
    main()
