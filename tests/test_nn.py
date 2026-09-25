import numpy as np

import heliax as hx


def test_linear_and_mlp_forward_backward():
    model = hx.nn.MLP(4, [8, 6], 3, rng=np.random.default_rng(7))
    value = hx.tensor(
        np.random.default_rng(8).normal(size=(5, 4)).astype(np.float32), requires_grad=True
    )
    output = model(value)
    assert output.shape == (5, 3)
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert hx.count_parameters(model) == 4 * 8 + 8 + 8 * 6 + 6 + 6 * 3 + 3


def test_layernorm_and_embedding():
    normalized = hx.nn.LayerNorm(3)
    value = hx.tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32), requires_grad=True)
    output = normalized(value)
    assert output.shape == value.shape
    output.sum().backward()
    assert normalized.weight.grad is not None
    assert normalized.bias.grad is not None

    embedding = hx.nn.Embedding(8, 4, rng=np.random.default_rng(9))
    indices = hx.tensor(np.array([0, 3, 3, 7], dtype=np.int64))
    output = embedding(indices)
    assert output.shape == (4, 4)
    output.sum().backward()
    assert np.allclose(embedding.weight.grad.numpy()[3], 2.0)
    assert np.allclose(embedding.weight.grad.numpy()[0], 1.0)


def test_conv2d_and_gradcheck():
    convolution = hx.nn.Conv2d(1, 2, kernel_size=3, padding=1, rng=np.random.default_rng(10))
    value = hx.tensor(
        np.arange(1 * 1 * 5 * 5, dtype=np.float32).reshape(1, 1, 5, 5) / 25, requires_grad=True
    )
    output = convolution(value)
    assert output.shape == (1, 2, 5, 5)
    result = hx.gradcheck(lambda a: convolution(a).sum(), [value], eps=0.01, atol=0.03, rtol=0.03)
    assert result["passed"], result


def test_multihead_attention_shape_and_gradients():
    attention = hx.MultiheadAttention(embed_dim=8, num_heads=2, rng=np.random.default_rng(13))
    value = hx.tensor(
        np.random.default_rng(14).normal(size=(2, 4, 8)).astype(np.float32), requires_grad=True
    )
    output = attention(value)
    assert output.shape == (2, 4, 8)
    output.square().mean().backward()
    assert value.grad is not None
    assert attention.in_proj_weight.grad is not None

    model = hx.nn.Sequential(
        hx.nn.Linear(3, 2, rng=np.random.default_rng(11)),
        hx.nn.Dropout(0.5, rng=np.random.default_rng(12)),
    )
    state = model.state_dict()
    assert state
    restored = hx.nn.Sequential(
        hx.nn.Linear(3, 2, rng=np.random.default_rng(99)), hx.nn.Dropout(0.5)
    )
    restored.load_state_dict(state)
    assert np.allclose(restored.state_dict()["layers.0.weight"], state["layers.0.weight"])
    restored.eval()
    assert not restored.training
