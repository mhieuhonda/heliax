import numpy as np

import heliax as hx


def test_batchnorm_training_and_running_buffers():
    norm = hx.nn.BatchNorm1d(3)
    value = hx.tensor(
        np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], dtype=np.float32), requires_grad=True
    )
    output = norm(value)
    assert output.shape == value.shape
    assert np.allclose(output.numpy().mean(axis=0), 0.0, atol=1e-5)
    output.sum().backward()
    assert value.grad is not None
    assert "running_mean" in norm.state_dict()
    before = norm.state_dict()["running_mean"].copy()
    norm.eval()
    norm(value)
    assert np.allclose(norm.state_dict()["running_mean"], before)


def test_batchnorm2d_gradcheck():
    norm = hx.nn.BatchNorm2d(2)
    value = hx.tensor(
        np.random.default_rng(3).normal(size=(2, 2, 3, 3)).astype(np.float32), requires_grad=True
    )
    result = hx.gradcheck(lambda a: norm(a).sum(), [value], eps=0.01, atol=0.04, rtol=0.04)
    assert result["passed"], result
