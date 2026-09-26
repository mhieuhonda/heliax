import numpy as np
import pytest

import heliax as hx

pytestmark = pytest.mark.skipif(
    not hx.native_available(), reason="native library has not been built"
)


def numpy_gelu(value: np.ndarray) -> np.ndarray:
    coefficient = np.sqrt(np.asarray(2.0 / np.pi, dtype=value.dtype))
    cubic = value * value * value
    return 0.5 * value * (1.0 + np.tanh(coefficient * (value + 0.044715 * cubic)))


def numpy_layernorm(value: np.ndarray) -> np.ndarray:
    mean = np.mean(value, axis=-1, keepdims=True, dtype=np.float32)
    variance = np.mean((value - mean) ** 2, axis=-1, keepdims=True, dtype=np.float32)
    return (value - mean) / np.sqrt(variance + 1e-5, dtype=np.float32)


def test_native_kernels_match_numpy_reference():
    rng = np.random.default_rng(1)
    array = rng.normal(size=(5, 7)).astype(np.float32)
    left = rng.normal(size=(5, 7)).astype(np.float32)
    right = rng.normal(size=(5, 7)).astype(np.float32)
    assert np.allclose(hx.native_ops.gelu(array), numpy_gelu(array), atol=1e-6)
    assert np.allclose(hx.native_ops.silu(array), array / (1.0 + np.exp(-array)), atol=1e-6)
    shifted = array - np.max(array, axis=-1, keepdims=True)
    reference_softmax = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
    assert np.allclose(hx.native_ops.softmax_lastdim(array), reference_softmax, atol=1e-6)
    axis_zero_softmax = hx.functional.softmax(hx.tensor(array), axis=0)
    reference_axis_zero_softmax = np.exp(array - np.max(array, axis=0, keepdims=True))
    reference_axis_zero_softmax /= np.sum(reference_axis_zero_softmax, axis=0, keepdims=True)
    assert np.allclose(axis_zero_softmax.numpy(), reference_axis_zero_softmax, atol=1e-6)
    reference_log_softmax = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    assert np.allclose(hx.native_ops.log_softmax_lastdim(array), reference_log_softmax, atol=1e-6)
    axis_zero = hx.functional.log_softmax(hx.tensor(array), axis=0)
    reference_axis_zero = (
        array
        - np.max(array, axis=0, keepdims=True)
        - np.log(
            np.sum(np.exp(array - np.max(array, axis=0, keepdims=True)), axis=0, keepdims=True)
        )
    )
    assert np.allclose(axis_zero.numpy(), reference_axis_zero, atol=1e-6)
    assert np.allclose(hx.native_ops.add_relu(left, right), np.maximum(left + right, 0), atol=1e-6)
    assert np.allclose(
        hx.native_ops.layernorm_lastdim(
            array, np.ones(7, dtype=np.float32), np.zeros(7, dtype=np.float32), 1e-5
        ),
        numpy_layernorm(array),
        atol=1e-5,
    )
    parameter = np.ones(5, dtype=np.float32)
    gradient = np.arange(1, 6, dtype=np.float32)
    first = np.zeros(5, dtype=np.float32)
    second = np.zeros(5, dtype=np.float32)
    hx.native_ops.adamw(
        parameter,
        gradient,
        first,
        second,
        learning_rate=0.1,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.01,
        bias_correction1=0.1,
        bias_correction2=0.001,
    )
    assert np.all(parameter < 1.0)
    assert np.all(first > 0.0) and np.all(second > 0.0)
    rms_parameter = np.ones(5, dtype=np.float32)
    rms_gradient = np.arange(1, 6, dtype=np.float32)
    rms_average = np.zeros(5, dtype=np.float32)
    hx.native_ops.rmsprop(
        rms_parameter,
        rms_gradient,
        rms_average,
        learning_rate=0.01,
        decay=0.9,
        epsilon=1e-8,
    )
    assert np.all(rms_parameter < 1.0)
    assert np.all(rms_average > 0.0)
    mae_prediction = rng.normal(size=(4, 5)).astype(np.float32)
    mae_target = rng.normal(size=(4, 5)).astype(np.float32)
    mae_loss, mae_gradient = hx.native_ops.mae(mae_prediction, mae_target)
    mae_difference = mae_prediction - mae_target
    assert np.allclose(mae_loss, float(np.abs(mae_difference).mean()), atol=1e-5)
    assert np.allclose(mae_gradient, np.sign(mae_difference) / 20, atol=1e-5)
    bce_logits = rng.normal(size=(4, 5)).astype(np.float32)
    bce_targets = rng.integers(0, 2, size=(4, 5)).astype(np.float32)
    bce_loss, bce_gradient = hx.native_ops.bce_with_logits(bce_logits, bce_targets)
    bce_reference = (
        np.maximum(bce_logits, 0) - bce_logits * bce_targets + np.log1p(np.exp(-np.abs(bce_logits)))
    )
    assert np.allclose(bce_loss, float(bce_reference.mean()), atol=1e-5)
    assert np.allclose(
        bce_gradient, (1.0 / (1.0 + np.exp(-bce_logits)) - bce_targets) / 20, atol=1e-5
    )
    prediction = rng.normal(size=(4, 5)).astype(np.float32)
    target_values = rng.normal(size=(4, 5)).astype(np.float32)
    huber_loss, huber_gradient = hx.native_ops.huber(prediction, target_values, delta=0.75)
    difference_huber = prediction - target_values
    absolute_huber = np.abs(difference_huber)
    expected_huber = np.where(
        absolute_huber <= 0.75,
        0.5 * absolute_huber**2,
        0.75 * (absolute_huber - 0.375),
    )
    assert np.allclose(huber_loss, float(expected_huber.mean()), atol=1e-5)
    expected_huber_gradient = np.clip(difference_huber, -0.75, 0.75) / 20
    assert np.allclose(huber_gradient, expected_huber_gradient, atol=1e-5)
    mse_loss, mse_gradient = hx.native_ops.mse(prediction, target_values)
    difference = prediction - target_values
    assert np.allclose(mse_loss, float(np.mean(difference**2)), atol=1e-5)
    assert np.allclose(mse_gradient, 2.0 * difference / 20, atol=1e-5)
    targets = np.array([0, 2, 4, 1, 3], dtype=np.int64)
    logits = rng.normal(size=(5, 5)).astype(np.float32)
    loss, gradient = hx.native_ops.cross_entropy_lastdim(logits, targets)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    reference_probabilities = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
    reference_loss = float(-np.log(reference_probabilities[np.arange(5), targets]).mean())
    assert np.allclose(loss, reference_loss, atol=1e-5)
    assert np.allclose(gradient, (reference_probabilities - np.eye(5)[targets]) / 5, atol=1e-5)


def test_native_add_relu_gradcheck():
    left = hx.tensor([1.0, -2.0, 3.0], requires_grad=True)
    right = hx.tensor([0.5, 1.0, -4.0], requires_grad=True)
    result = hx.gradcheck(
        lambda a, b: hx.add_relu(a, b).sum(), [left, right], eps=0.01, atol=0.03, rtol=0.03
    )
    assert result["passed"], result
