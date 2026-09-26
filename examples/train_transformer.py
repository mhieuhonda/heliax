"""Train a tiny Heliax transformer classifier on synthetic sequences."""

from __future__ import annotations

import numpy as np

import heliax as hx


def make_data(seed: int = 12) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    x = generator.integers(0, 32, size=(128, 12), dtype=np.int64)
    y = (x.sum(axis=1) % 2).astype(np.int64)
    return x, y


def main() -> None:
    features, targets = make_data()
    dataset = hx.TensorDataset(hx.tensor(features), hx.tensor(targets))
    loader = hx.DataLoader(dataset, batch_size=16, shuffle=True, generator=np.random.default_rng(2))
    model = hx.nn.Sequential(
        hx.nn.Embedding(32, 16),
        hx.PositionalEncoding(16, max_length=12),
        hx.TransformerEncoder(
            hx.TransformerEncoderLayer(
                16, 4, dim_feedforward=32, dropout=0.0, rng=np.random.default_rng(3)
            ),
            num_layers=1,
        ),
    )
    # Pool token representations and classify the pooled state.
    classifier = hx.nn.Linear(16, 2, rng=np.random.default_rng(4))
    optimizer = hx.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=2e-3)
    loss_fn = hx.nn.CrossEntropyLoss()
    for epoch in range(6):
        total = 0.0
        for inputs, labels in loader:
            optimizer.zero_grad()
            features = model(inputs).mean(axis=1)
            loss = loss_fn(classifier(features), labels)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        print(f"epoch {epoch + 1}: {total / len(loader):.4f}")


if __name__ == "__main__":
    main()
