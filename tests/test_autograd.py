import numpy as np

import heliax as hx


def test_gradcheck_elementwise_and_matmul():
    left = hx.tensor([1.2, 2.1], requires_grad=True)
    right = hx.tensor([0.4, -0.7], requires_grad=True)
    result = hx.gradcheck(lambda a, b: ((a * b) / (a + b)).sum(), [left, right])
    assert result["passed"], result

    matrix_a = hx.tensor([[1.2, 2.1]], requires_grad=True)
    matrix_b = hx.tensor([[0.4], [-0.7]], requires_grad=True)
    result = hx.gradcheck(lambda a, b: (a @ b).sum(), [matrix_a, matrix_b])
    assert result["passed"], result


def test_gradcheck_softmax_and_cross_entropy():
    logits = hx.tensor([[1.2, 2.1], [0.4, -0.7]], requires_grad=True)
    assert hx.gradcheck(lambda a: hx.softmax(a).sum(), [logits])["passed"]
    assert hx.gradcheck(lambda a: hx.cross_entropy(a, np.array([0, 1])), [logits])["passed"]


def test_reduction_gradients_broadcast_to_input_shape():
    value = hx.tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), requires_grad=True)
    result = hx.gradcheck(lambda a: a.mean(axis=1).sum(), [value])
    assert result["passed"], result


def test_shared_graph_accumulates_once_per_edge():
    value = hx.tensor([2.0, 3.0], requires_grad=True)
    shared = value * value
    (shared + shared).sum().backward()
    assert np.allclose(value.grad.numpy(), [8.0, 12.0])
