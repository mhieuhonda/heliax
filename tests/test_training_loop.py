import numpy as np

import heliax as hx


def test_fit_supports_accumulation_scheduler_and_max_steps():
    generator = np.random.default_rng(14)
    features = hx.tensor(generator.normal(size=(8, 2)).astype(np.float32))
    targets = hx.tensor(features.numpy() * 0.5)
    loader = hx.DataLoader(hx.TensorDataset(features, targets), batch_size=2, shuffle=False)
    model = hx.nn.Linear(2, 1, rng=generator)
    optimizer = hx.optim.SGD(model.parameters(), lr=0.05)
    scheduler = hx.optim.StepLR(optimizer, step_size=1, gamma=0.9)
    history = hx.fit(
        model,
        loader,
        optimizer,
        hx.mean_squared_error,
        epochs=2,
        gradient_accumulation_steps=2,
        scheduler=scheduler,
        max_steps=3,
    )
    assert len(history) == 2
    assert all(np.isfinite(value) for value in history)
