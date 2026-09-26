import numpy as np

import heliax as hx


def test_reduce_lr_on_plateau():
    model = hx.nn.Linear(2, 1, rng=np.random.default_rng(1))
    optimizer = hx.optim.SGD(model.parameters(), lr=0.1)
    scheduler = hx.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    assert scheduler.step(1.0) == 0.1
    assert scheduler.step(1.5) == 0.1
    assert scheduler.step(1.6) == 0.05
    assert np.isclose(optimizer.defaults["lr"], 0.05)

    features = hx.tensor(np.ones((4, 2), dtype=np.float32))
    targets = hx.zeros(4, 1)
    loader = hx.DataLoader(hx.TensorDataset(features, targets), batch_size=2)
    fit_model = hx.nn.Linear(2, 1, rng=np.random.default_rng(2))
    fit_optimizer = hx.optim.SGD(fit_model.parameters(), lr=0.1)
    hx.fit(
        fit_model,
        loader,
        fit_optimizer,
        hx.mean_squared_error,
        epochs=1,
        scheduler=hx.ReduceLROnPlateau(fit_optimizer, patience=0),
    )
