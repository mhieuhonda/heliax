import numpy as np

import helianthus
import heliax as hx


def test_fused_linear_bias_matches_reference_and_gradcheck():
    value = hx.tensor(np.array([[1.0, -0.5], [0.2, 2.0]], dtype=np.float32), requires_grad=True)
    weight = hx.Parameter(np.array([[0.4, 0.7], [-1.0, 0.3]], dtype=np.float32))
    bias = hx.Parameter(np.array([0.1, -0.2], dtype=np.float32))
    fused = hx.fused_linear_bias(value, weight, bias)
    reference = value @ weight.transpose((1, 0)) + bias
    assert np.allclose(fused.numpy(), reference.numpy())
    assert hx.gradcheck(
        lambda a: hx.fused_linear_bias(a, weight, bias).sum(),
        [value],
        eps=0.01,
        atol=0.03,
        rtol=0.03,
    )["passed"]


def test_fused_linear_gelu_module():
    layer = hx.FusedLinearGELU(4, 6, rng=np.random.default_rng(2))
    value = hx.tensor(
        np.random.default_rng(3).normal(size=(3, 4)).astype(np.float32), requires_grad=True
    )
    output = layer(value)
    assert output.shape == (3, 6)
    output.sum().backward()
    assert value.grad is not None
    assert layer.weight.grad is not None


def test_helianthus_alias_exposes_core():
    assert helianthus.__version__ == hx.__version__
    assert helianthus.Tensor is hx.Tensor
    assert hasattr(helianthus, "fused_linear_bias")
