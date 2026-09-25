"""Inspect a Heliax graph and verify its gradients."""

from __future__ import annotations

import heliax as hx


def main() -> None:
    a = hx.tensor([1.5, -0.5], requires_grad=True)
    b = hx.tensor([0.25, 2.0], requires_grad=True)
    output = ((a * b) + (a / b)).sum()
    output.backward()
    print("output:", output.item())
    print("a.grad:", a.grad)
    print("b.grad:", b.grad)
    print("check:", hx.gradcheck(lambda x, y: ((x * y) + (x / y)).sum(), [a, b]))


if __name__ == "__main__":
    main()
