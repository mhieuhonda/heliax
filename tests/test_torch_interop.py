import numpy as np
import pytest

import heliax as hx


def test_torch_interop_is_optional():
    if hx.torch_interop.torch_available():
        pytest.skip("PyTorch is installed; optional interop smoke is covered below")
    with pytest.raises(RuntimeError):
        hx.to_torch(hx.tensor([1.0]))


def test_torch_conversion_round_trip_when_available():
    if not hx.torch_interop.torch_available():
        pytest.skip("optional PyTorch extra is not installed")
    source = hx.tensor(np.array([1.0, 2.0], dtype=np.float32))
    converted = hx.to_torch(source)
    restored = hx.from_torch(converted)
    assert np.allclose(restored.numpy(), source.numpy())
    accelerator = hx.torch_interop.TorchAccelerator()
    assert accelerator.info()["backend"] == "pytorch-interop"
