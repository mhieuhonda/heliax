import numpy as np

import heliax as hx


def test_checkpoint_matches_standard_backward():
    generator = np.random.default_rng(3)
    x = hx.tensor(generator.normal(size=(3, 4)).astype(np.float32), requires_grad=True)
    model = hx.nn.Linear(4, 2, rng=generator)

    def block(value):
        return hx.nn.GELU()(model(value))

    normal = block(x)
    normal.sum().backward()
    normal_input = x.grad.numpy().copy()
    normal_weight = model.weight.grad.numpy().copy()
    x.zero_grad()
    model.zero_grad()
    saved = hx.checkpoint(block, x)
    saved.sum().backward()
    assert np.allclose(x.grad.numpy(), normal_input, atol=1e-5)
    assert np.allclose(model.weight.grad.numpy(), normal_weight, atol=1e-5)
