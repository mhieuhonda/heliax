import numpy as np

import heliax as hx


def make_regression_problem():
    generator = np.random.default_rng(42)
    x = hx.tensor(generator.normal(size=(32, 3)).astype(np.float32))
    target = hx.tensor(
        (x.numpy() * np.array([1.5, -0.5, 2.0], dtype=np.float32) + 0.2).astype(np.float32)
    )
    model = hx.nn.Linear(3, 1, rng=generator)
    optimizer = hx.optim.SGD(model.parameters(), lr=0.08)
    return x, target, model, optimizer


def test_sgd_reduces_loss():
    x, target, model, optimizer = make_regression_problem()
    initial = None
    for _ in range(80):
        optimizer.zero_grad()
        prediction = model(x)
        loss = hx.mean_squared_error(prediction, target)
        loss.backward()
        optimizer.step()
        initial = float(loss.item()) if initial is None else initial
    final = float(hx.mean_squared_error(model(x), target).item())
    assert final < initial


def test_adamw_and_clipping():
    parameter = hx.Parameter(np.ones((2, 2), dtype=np.float32))
    optimizer = hx.optim.AdamW([parameter], lr=0.1, weight_decay=0.01)
    parameter.grad = hx.tensor(np.ones_like(parameter.numpy()))
    norm = hx.clip_grad_norm_([parameter], max_norm=0.1)
    assert float(norm.item()) > 0
    optimizer.step()
    assert np.all(parameter.numpy() < 1.0)


def test_schedulers_and_zero_grad():
    parameter = hx.Parameter(np.ones(2, dtype=np.float32))
    optimizer = hx.optim.SGD([parameter], lr=0.1)
    scheduler = hx.optim.CosineAnnealingLR(optimizer, t_max=10)
    before = scheduler.get_lr()
    scheduler.step()
    scheduler.step()
    assert scheduler.get_lr() < before
    parameter.grad = hx.tensor(np.ones(2, dtype=np.float32))
    optimizer.zero_grad()
    assert parameter.grad is None
