"""CPU checks for matching transforms and dynamic activation preparation."""

import numpy as np
import pytest
from scipy.linalg import hadamard

from embedded_jev.activation import quantize_a8_per_group, rotate_signed_hadamard


@pytest.mark.parametrize("block_size", [8, 128])
def test_signed_rotation_matches_explicit_matrix_and_dense_linear(block_size):
    generator = np.random.default_rng(block_size)
    width = block_size * 2
    signs = generator.choice([-1, 1], size=width)
    inputs = generator.normal(size=(3, 2, width)).astype(np.float32)
    weights = generator.normal(size=(5, width)).astype(np.float32)
    bias = generator.normal(size=5).astype(np.float32)
    rotation = np.zeros((width, width))
    for start in range(0, width, block_size):
        rotation[start : start + block_size, start : start + block_size] = (
            signs[start : start + block_size, None] * hadamard(block_size)
            / np.sqrt(block_size)
        )

    rotated_inputs = rotate_signed_hadamard(inputs, signs, block_size)
    rotated_weights = rotate_signed_hadamard(weights, signs, block_size)
    np.testing.assert_allclose(rotated_inputs, inputs @ rotation, rtol=1e-5, atol=2e-5)
    np.testing.assert_allclose(rotated_weights, weights @ rotation, rtol=1e-5, atol=2e-5)
    np.testing.assert_allclose(
        rotated_inputs @ rotated_weights.T + bias,
        inputs @ weights.T + bias, rtol=1e-5, atol=6e-5,
    )


@pytest.mark.parametrize(
    ("values", "signs", "block_size"),
    [
        ([1.0, 2.0, 3.0], [1, 1, 1], 2),
        ([1.0] * 8, [1] * 8, 3),
        ([1.0] * 8, [1] * 7, 8),
        ([1.0] * 8, [1] * 7 + [0], 8),
        ([1.0, float("nan")], [1, -1], 2),
    ],
)
def test_rotation_rejects_invalid_width_signs_or_values(values, signs, block_size):
    with pytest.raises(ValueError, match="invalid signed Hadamard"):
        rotate_signed_hadamard(values, signs, block_size)


def test_dynamic_a8_scales_each_token_group_and_rounds_even():
    inputs = np.zeros((2, 256), dtype=np.float32)
    inputs[0, :5] = [127, -127, 0.5, 1.5, -0.5]
    inputs[0, 128:130] = [63.5, -63.5]
    inputs[1, :3] = [0.25, -0.25, 0.125]
    codes, scales = quantize_a8_per_group(inputs)
    assert codes.dtype == np.int8 and scales.dtype == np.float32
    np.testing.assert_array_equal(
        scales, np.array([[1, 0.5], [0.25 / 127, 1]], dtype=np.float32)
    )
    np.testing.assert_array_equal(codes[0, :5], [127, -127, 0, 2, 0])
    np.testing.assert_array_equal(codes[0, 128:130], [127, -127])
    assert np.all(codes[1, 128:] == 0)
    restored = codes.reshape(2, 2, 128) * scales[..., None]
    assert np.all(
        np.abs(restored - inputs.reshape(2, 2, 128))
        < scales[..., None] / 2 + 1e-5
    )


@pytest.mark.parametrize("inputs", [np.ones(129), [1.0, np.inf], [np.nan] * 128, 1.0])
def test_dynamic_a8_rejects_tails_nonfinite_and_scalars(inputs):
    with pytest.raises(ValueError, match="invalid A8"):
        quantize_a8_per_group(inputs)