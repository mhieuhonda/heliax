import numpy as np
import pytest

import heliax as hx


def test_tensor_construction_and_repr():
    value = hx.tensor([[1, 2], [3, 4]], dtype=np.float32, requires_grad=True)
    assert value.shape == (2, 2)
    assert value.dtype == np.float32
    assert value.device == "cpu"
    assert "Tensor" in repr(value)
    assert np.array_equal(value.numpy(), np.array([[1, 2], [3, 4]], dtype=np.float32))


def test_broadcasting_and_reverse_operators():
    left = hx.tensor([[1.0], [2.0]], requires_grad=True)
    right = hx.tensor([3.0, 4.0], requires_grad=True)
    result = (2.0 - left) + (left * right) / right
    result.sum().backward()
    assert left.grad.shape == left.shape
    assert right.grad.shape == right.shape
    assert np.allclose(left.grad.numpy(), 0.0, atol=1e-6)
    assert np.allclose(right.grad.numpy(), 0.0, atol=1e-6)


def test_shapes_and_views_backward():
    value = hx.tensor(np.arange(12, dtype=np.float32).reshape(3, 4) / 10, requires_grad=True)
    reshaped = value.reshape(2, 6)
    transposed = reshaped.transpose()
    transposed.sum().backward()
    assert value.grad.shape == value.shape
    assert np.allclose(value.grad.numpy(), 1.0)

    value.zero_grad()
    value[:, 1].sum().backward()
    assert np.array_equal(value.grad.numpy()[:, 1], np.ones(3, dtype=np.float32))
    assert np.array_equal(value.grad.numpy()[:, 0], np.zeros(3, dtype=np.float32))


def test_no_grad_and_detach():
    value = hx.tensor([1.0, 2.0], requires_grad=True)
    with hx.no_grad():
        result = value * 2
    assert not result.requires_grad
    detached = value.detach()
    detached.sum().backward() if detached.requires_grad else None
    assert value.grad is None


def test_clone_and_inplace_operations():
    value = hx.tensor([1.0, 2.0], requires_grad=True)
    clone = value.clone()
    clone.sum().backward()
    assert np.allclose(value.grad.numpy(), [1.0, 1.0])
    value.zero_grad()
    value.add_(1.0).mul_(2.0)
    assert np.allclose(value.numpy(), [4.0, 6.0])
    assert value.numel == 2
    assert value.is_contiguous


def test_error_on_non_scalar_backward():
    value = hx.tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(RuntimeError):
        (value * value).backward()
