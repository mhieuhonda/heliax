"""Optional side-by-side Heliax/NumPy and PyTorch compatibility check."""

from __future__ import annotations

from time import perf_counter

import numpy as np

import heliax as hx


def timed(function, repeats: int = 10) -> float:
    function()
    started = perf_counter()
    for _ in range(repeats):
        function()
    return (perf_counter() - started) * 1000 / repeats


def main() -> None:
    if not hx.torch_available():
        print("PyTorch is not installed; install heliax[torch] to run this comparison.")
        return
    import torch

    rng = np.random.default_rng(0)
    left = rng.normal(size=(512, 512)).astype(np.float32)
    right = rng.normal(size=(512, 512)).astype(np.float32)
    torch_left = torch.from_numpy(left)
    torch_right = torch.from_numpy(right)
    heliax_left = hx.tensor(left)
    heliax_right = hx.tensor(right)
    print("Heliax backend:", dict(hx.backend_info()))
    print("matmul Heliax ms:", round(timed(lambda: heliax_left @ heliax_right), 3))
    print("matmul Torch ms:", round(timed(lambda: torch.matmul(torch_left, torch_right)), 3))
    logits = rng.normal(size=(1024, 16)).astype(np.float32)
    torch_logits = torch.from_numpy(logits)
    print("softmax Heliax ms:", round(timed(lambda: hx.softmax(hx.tensor(logits))), 3))
    print("softmax Torch ms:", round(timed(lambda: torch.softmax(torch_logits, dim=-1)), 3))


if __name__ == "__main__":
    main()
