import numpy as np

import heliax as hx


def test_autograd_grad_does_not_accumulate():
    generator = np.random.default_rng(1)
    features = hx.tensor(generator.normal(size=(3, 2)).astype(np.float32), requires_grad=True)
    model = hx.nn.Linear(2, 1, rng=generator)
    loss = (model(features) ** 2).mean()
    existing = hx.tensor(np.zeros((1, 1), dtype=np.float32), requires_grad=False)
    parameter_grad = hx.grad(loss, model.weight, retain_graph=True)
    feature_grad = hx.grad(loss, features, retain_graph=True)
    assert parameter_grad[0].shape == model.weight.shape
    assert feature_grad[0].shape == features.shape
    assert model.weight.grad is None
    assert features.grad is None
    assert existing.numpy().shape == (1, 1)
