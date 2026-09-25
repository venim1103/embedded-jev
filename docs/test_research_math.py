"""Executable checks of the audit's algebra, not model-quality benchmarks."""

import math

import numpy as np
from scipy.linalg import hadamard
from scipy.special import softmax


def test_matching_rotation_preserves_linear_output():
    generator = np.random.default_rng(27)
    inputs = generator.normal(size=(17, 8))
    weights = generator.normal(size=(5, 8))
    signs = generator.choice([-1.0, 1.0], size=8)
    rotation = signs[:, None] * hadamard(8) / math.sqrt(8)
    expected = inputs @ weights.T
    rotated = (inputs @ rotation) @ (weights @ rotation).T
    np.testing.assert_allclose(rotated, expected, atol=1e-12)
    assert not np.allclose(inputs @ (weights @ rotation).T, expected)


def test_rotation_preserves_hessian_spectrum_and_condition_number():
    generator = np.random.default_rng(158)
    inputs = generator.normal(size=(64, 8)) * np.geomspace(1.0, 100.0, 8)
    rotation = hadamard(8) / math.sqrt(8)
    curvature = 2.0 * inputs.T @ inputs / len(inputs)
    rotated_inputs = inputs @ rotation
    rotated_curvature = 2.0 * rotated_inputs.T @ rotated_inputs / len(inputs)
    np.testing.assert_allclose(
        rotated_curvature, rotation.T @ curvature @ rotation, atol=1e-10
    )
    np.testing.assert_allclose(
        np.linalg.eigvalsh(rotated_curvature),
        np.linalg.eigvalsh(curvature),
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        np.linalg.cond(rotated_curvature), np.linalg.cond(curvature), rtol=1e-9
    )


def test_repeated_activations_do_not_add_rank_or_covariance_information():
    inputs = np.array([[1.0, 2.0, 3.0, 4.0], [3.0, 1.0, 0.0, 5.0]])
    repeated = np.tile(inputs, (64, 1))
    assert np.linalg.matrix_rank(inputs) == np.linalg.matrix_rank(repeated) == 2
    np.testing.assert_allclose(
        inputs.T @ inputs / len(inputs), repeated.T @ repeated / len(repeated)
    )


def test_group_scales_cannot_be_replaced_by_one_tensor_scale():
    codes = np.tile(np.array([-1.0, 0.0, 1.0, 0.0]), 64)
    scales = np.repeat([0.25, 1.0], 128)
    weights = codes * scales
    reconstructed_with_one_scale = np.sign(weights) * np.max(np.abs(weights))
    assert not np.array_equal(weights, reconstructed_with_one_scale)
    np.testing.assert_array_equal(weights, codes * scales)


def test_format_bit_rates_include_scales():
    assert math.ceil(128 * math.log2(3)) == 203
    assert 24 * 5 + 2 * 4 == 128
    assert (24 + 2 + 2) * 8 / 128 == 1.75
    assert (32 + 2) * 8 / 128 == 2.125
    assert (16 + 2) * 8 / 64 == 2.25
    assert (48 + 4 + 2) * 8 / 256 == 1.6875


def test_int16_cannot_hold_worst_case_1024_point_fwht():
    worst_case = 1024 * 127
    assert worst_case == 130048
    assert np.iinfo(np.int16).max < worst_case < np.iinfo(np.int32).max
    values = np.full(1024, 127, dtype=np.int8)
    assert int(values.sum(dtype=np.int32)) == worst_case
    assert int(values.sum(dtype=np.int16)) != worst_case


def test_mimo_vocabulary_matrix_memory():
    parameters = 248320 * 4096
    assert parameters == 1017118720
    assert parameters * 2 == 2034237440
    assert parameters * 2 * 2 == 4068474880
    assert parameters // 32 * 34 == 1080688640
    assert 16 * 4096 * 2 == 128 * 1024


def test_cache_curvature_and_activation_accounting():
    kv_bytes_per_token = 8 * 2 * 4 * 256 * 2
    recurrent_bytes = 24 * 32 * 128 * 128 * 4
    assert kv_bytes_per_token == 32768
    assert kv_bytes_per_token * 4096 == 128 * 1024**2
    assert recurrent_bytes == 48 * 1024**2
    assert 12288**2 * 4 == 576 * 1024**2
    assert 524288 * 4096 * 2 == 4 * 1024**3


def test_selected_linear_head_matches_conditional_full_head():
    generator = np.random.default_rng(9)
    hidden = generator.normal(size=32)
    output_head = generator.normal(size=(128, 32))
    label_ids = np.array([7, 23, 41, 65])
    full_logits = output_head @ hidden
    selected_logits = output_head[label_ids] @ hidden
    np.testing.assert_allclose(selected_logits, full_logits[label_ids], atol=1e-12)
    full_probabilities = softmax(full_logits)
    conditional = full_probabilities[label_ids] / full_probabilities[label_ids].sum()
    np.testing.assert_allclose(softmax(selected_logits), conditional, atol=1e-12)