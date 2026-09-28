"""Offline code/scale identity checks for the deterministic ternary baseline."""

import numpy as np
import pytest

from embedded_jev.ternary import quantize_ternary_rtn, reconstruct_ternary


def test_fp16_group_scales_are_used_to_assign_and_reconstruct_codes():
    weights = np.zeros((2, 256), dtype=np.float32)
    weights[0, :3] = [1.0004, 0.5001, -0.75]
    weights[1, 0] = 2.25
    weights[1, 128] = -3.5

    codes, scales = quantize_ternary_rtn(weights)
    assert codes.dtype == np.int8 and scales.dtype == np.float16
    np.testing.assert_array_equal(scales, [[1.0, 0.0], [2.25, 3.5]])
    np.testing.assert_array_equal(codes[0, :3], [1, 1, -1])
    assert np.all(codes[0, 128:] == 0)
    assert codes[1, 0] == 1 and codes[1, 128] == -1
    restored = reconstruct_ternary(codes, scales)
    assert restored.dtype == np.float32
    np.testing.assert_array_equal(
        restored.reshape(2, 2, 128),
        codes.reshape(2, 2, 128).astype(np.float32) * scales.astype(np.float32)[..., None],
    )
    assert np.array_equal(codes, quantize_ternary_rtn(weights)[0])


@pytest.mark.parametrize(
    "weights",
    [np.zeros((2, 129)), np.full((1, 128), 70000.0),
     np.full((1, 128), 1e-9), np.full((1, 128), np.nan), np.zeros((0, 128))],
)
def test_quantizer_rejects_tails_overflow_underflow_and_nonfinite(weights):
    with pytest.raises(ValueError, match="invalid ternary|not representable"):
        quantize_ternary_rtn(weights)


@pytest.mark.parametrize(
    ("codes", "scales"),
    [
        (np.full((1, 128), 2, dtype=np.int8), np.ones((1, 1), dtype=np.float16)),
        (np.zeros((1, 128), dtype=np.int16), np.ones((1, 1), dtype=np.float16)),
        (np.zeros((1, 128), dtype=np.int8), np.ones((1, 1), dtype=np.float32)),
        (np.zeros((1, 129), dtype=np.int8), np.ones((1, 2), dtype=np.float16)),
        (np.zeros((1, 128), dtype=np.int8), np.array([[-1]], dtype=np.float16)),
        (np.zeros((1, 0), dtype=np.int8), np.ones((1, 1), dtype=np.float16)),
        (np.ones((1, 128), dtype=np.int8), np.zeros((1, 1), dtype=np.float16)),
    ],
)
def test_reconstruction_refuses_invalid_stored_artifacts(codes, scales):
    with pytest.raises(ValueError, match="invalid stored ternary"):
        reconstruct_ternary(codes, scales)