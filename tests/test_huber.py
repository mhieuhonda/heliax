import heliax as hx


def test_huber_loss_gradcheck():
    value = hx.tensor([0.2, 1.5, -2.0], requires_grad=True)
    result = hx.gradcheck(
        lambda a: hx.huber_loss(a, [0.0, 0.0, 0.0], delta=1.0),
        [value],
        eps=0.01,
        atol=0.03,
        rtol=0.03,
    )
    assert result["passed"], result


def test_mean_absolute_and_binary_cross_entropy_gradients():
    value = hx.tensor([0.2, 1.5, -2.0], requires_grad=True)
    mae = hx.gradcheck(
        lambda a: hx.mean_absolute_error(a, [0.0, 0.0, 0.0]),
        [value],
        eps=0.01,
        atol=0.03,
        rtol=0.03,
    )
    assert mae["passed"], mae
    logits = hx.tensor([0.2, 1.5, -2.0], requires_grad=True)
    bce = hx.gradcheck(
        lambda a: hx.binary_cross_entropy(a, [1.0, 0.0, 1.0]),
        [logits],
        eps=0.01,
        atol=0.03,
        rtol=0.03,
    )
    assert bce["passed"], bce
