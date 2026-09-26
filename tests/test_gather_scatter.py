import numpy as np

import heliax as hx


def test_gather_and_scatter_add_gradients():
    value = hx.tensor(np.arange(6, dtype=np.float32).reshape(2, 3), requires_grad=True)
    index = np.array([[0, 0, 2], [1, 1, 0]], dtype=np.int64)
    gathered = value.gather(1, index)
    assert np.allclose(gathered.numpy(), np.take_along_axis(value.numpy(), index, axis=1))
    gathered.sum().backward()
    expected = np.array([[2.0, 0.0, 1.0], [1.0, 2.0, 0.0]], dtype=np.float32)
    assert np.allclose(value.grad.numpy(), expected)

    base = hx.zeros(2, 3, requires_grad=True)
    source = hx.tensor(np.ones((2, 3), dtype=np.float32), requires_grad=True)
    scattered = base.scatter_add(1, index, source)
    assert np.allclose(scattered.numpy(), np.array([[2, 0, 1], [1, 2, 0]], dtype=np.float32))
    scattered.sum().backward()
    assert np.allclose(base.grad.numpy(), np.ones((2, 3), dtype=np.float32))
    assert np.allclose(source.grad.numpy(), np.ones((2, 3), dtype=np.float32))
