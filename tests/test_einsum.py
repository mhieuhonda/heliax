import numpy as np

import heliax as hx


def test_einsum_matmul_and_trace_gradients():
    left = hx.tensor(np.arange(6, dtype=np.float32).reshape(2, 3) / 10, requires_grad=True)
    right = hx.tensor(np.arange(12, dtype=np.float32).reshape(3, 4) / 10, requires_grad=True)
    output = hx.einsum("ij,jk->ik", left, right)
    assert np.allclose(output.numpy(), left.numpy() @ right.numpy())
    result = hx.gradcheck(
        lambda a, b: hx.einsum("ij,jk->ik", a, b).sum(),
        [left, right],
        eps=0.01,
        atol=0.03,
        rtol=0.03,
    )
    assert result["passed"], result


def test_einsum_trace_and_transpose():
    value = hx.tensor(np.arange(9, dtype=np.float32).reshape(3, 3) / 10, requires_grad=True)
    assert np.allclose(hx.einsum("ii->", value).numpy(), np.trace(value.numpy()), atol=1e-6)
    assert np.allclose(hx.einsum("ij->ji", value).numpy(), value.numpy().T, atol=1e-6)
    hx.einsum("ij->ji", value).sum().backward()
    assert value.grad is not None
