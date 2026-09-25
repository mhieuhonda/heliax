"""Train a tiny Heliax classifier on synthetic blobs."""

from __future__ import annotations

import numpy as np

import heliax as hx


def make_data(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    x = generator.normal(size=(256, 4)).astype(np.float32)
    weights = np.array([1.4, -0.8, 0.5, 1.1], dtype=np.float32)
    logits = x @ weights + 0.15 * x[:, 0] * x[:, 1]
    y = (logits > 0).astype(np.int64)
    return x, y


def main() -> None:
    generator = np.random.default_rng(7)
    features, targets = make_data()
    dataset = hx.TensorDataset(hx.tensor(features), hx.tensor(targets))
    loader = hx.DataLoader(dataset, batch_size=32, shuffle=True, generator=generator)
    model = hx.nn.MLP(4, [32, 16], 2, rng=generator)
    optimizer = hx.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    history = hx.fit(model, loader, optimizer, hx.cross_entropy, epochs=8)
    print("loss:", " -> ".join(f"{value:.4f}" for value in history))
    print("backend:", hx.backend_info())
    print("parameters:", hx.count_parameters(model))


if __name__ == "__main__":
    main()
