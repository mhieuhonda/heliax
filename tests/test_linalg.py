import numpy as np

import heliax as hx


def test_l2_norm_and_cosine_similarity_gradients():
    value = hx.tensor([3.0, 4.0], requires_grad=True)
    norm_check = hx.gradcheck(lambda a: hx.l2_norm(a), [value])
    assert norm_check["passed"], norm_check
    left = hx.tensor([1.0, 2.0], requires_grad=True)
    right = hx.tensor([2.0, 1.0], requires_grad=True)
    similarity = hx.cosine_similarity(left, right)
    assert np.allclose(similarity.numpy(), 0.8, atol=1e-5)
    result = hx.gradcheck(lambda a, b: hx.cosine_similarity(a, b).sum(), [left, right])
    assert result["passed"], result
