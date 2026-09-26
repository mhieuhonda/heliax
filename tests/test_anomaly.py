import numpy as np
import pytest

import heliax as hx


def test_anomaly_detection_and_graph_release():
    value = hx.tensor([1.0, 2.0], requires_grad=True)
    output = (value * value).sum()
    with hx.anomaly_detection():
        output.backward(retain_graph=False)
    assert hx.is_anomaly_detection_enabled() is False
    assert len(output._prev) == 0

    bad = hx.tensor([np.nan], requires_grad=True)
    with hx.anomaly_detection(), pytest.raises(FloatingPointError):
        (bad * bad).sum().backward()
