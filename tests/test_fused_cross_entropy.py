import numpy as np

import heliax as hx


def test_fused_cross_entropy_matches_reference_and_gradcheck():
    logits = hx.tensor(
        np.array([[1.0, 2.0, 0.5], [-1.0, 0.2, 2.0]], dtype=np.float32), requires_grad=True
    )
    target = np.array([2, 0])
    reference = hx.cross_entropy(logits, target)
    fused = hx.fused_cross_entropy(logits, target)
    assert np.allclose(fused.item(), reference.item(), atol=1e-6)
    result = hx.gradcheck(
        lambda a: hx.fused_cross_entropy(a, target), [logits], eps=0.01, atol=0.03, rtol=0.03
    )
    assert result["passed"], result
    assert hx.nn.CrossEntropyLoss()(logits.detach(), target).shape == ()


def test_fused_cross_entropy_supports_arbitrary_class_axis():
    generator = np.random.default_rng(1)
    logits = hx.tensor(generator.normal(size=(2, 3, 4)).astype(np.float32), requires_grad=True)
    target = np.array([0, 2, 1, 1, 0, 2, 1, 0], dtype=np.int64).reshape(2, 4)
    fused = hx.fused_cross_entropy(logits, target, axis=1)
    reference = hx.cross_entropy(logits, target, axis=1)
    assert np.allclose(fused.numpy(), reference.numpy(), atol=1e-6)
    fused.backward()
    assert logits.grad is not None
    logits.zero_grad()
    one_hot = np.zeros((2, 3, 4), dtype=np.float32)
    row, column = np.indices(target.shape)
    one_hot[row, target, column] = 1.0
    assert np.allclose(
        hx.fused_cross_entropy(logits, one_hot, axis=1).numpy(), reference.numpy(), atol=1e-6
    )


def test_fused_cross_entropy_module_training():
    model = hx.nn.Linear(3, 2, rng=np.random.default_rng(4))
    loss_fn = hx.nn.CrossEntropyLoss()
    values = hx.tensor(np.random.default_rng(5).normal(size=(6, 3)).astype(np.float32))
    targets = hx.tensor(np.array([0, 1, 1, 0, 1, 0], dtype=np.int64))
    optimizer = hx.optim.AdamW(model.parameters(), lr=0.01)
    initial = float(loss_fn(model(values), targets).item())
    for _ in range(20):
        optimizer.zero_grad()
        loss = loss_fn(model(values), targets)
        loss.backward()
        optimizer.step()
    assert float(loss_fn(model(values), targets).item()) < initial
