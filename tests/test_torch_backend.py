import numpy as np
import pytest

import heliax as hx


@pytest.mark.skipif(not hx.torch_backend_available(), reason="PyTorch extra is not installed")
def test_torch_backend_dispatches_core_kernels():
    previous = hx.backend._ACTIVE_BACKEND
    try:
        hx.set_backend("torch")
        value = hx.tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), requires_grad=True)
        output = hx.softmax(value)
        loss = output.square().mean()
        loss.backward()
        assert value.grad is not None
        assert np.isfinite(value.grad.numpy()).all()
        norm = hx.nn.LayerNorm(2)
        assert norm(value).shape == value.shape
    finally:
        hx.set_backend(previous)


@pytest.mark.skipif(not hx.torch_backend_available(), reason="PyTorch extra is not installed")
def test_torch_backend_device_selector():
    previous = hx.backend._ACTIVE_BACKEND
    try:
        hx.set_backend("torch:cpu")
        assert hx.backend.get_backend().device == "cpu"
    finally:
        hx.set_backend(previous)
