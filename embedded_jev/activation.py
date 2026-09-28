"""Matching FP32 input-axis transforms for small CPU quantization fixtures."""

import numpy as np


def rotate_signed_hadamard(values, signs, block_size: int) -> np.ndarray:
    """Apply x @ (diag(signs) H / sqrt(block_size)) in contiguous blocks."""
    inputs = np.asarray(values, dtype=np.float32)
    sign_vector = np.asarray(signs)
    if (
        inputs.ndim == 0
        or inputs.shape[-1] == 0
        or block_size < 1
        or block_size & (block_size - 1)
        or inputs.shape[-1] % block_size
        or sign_vector.shape != (inputs.shape[-1],)
        or not np.isin(sign_vector, (-1, 1)).all()
        or not np.isfinite(inputs).all()
    ):
        raise ValueError("invalid signed Hadamard transform inputs")

    width = inputs.shape[-1]
    rotated = (inputs * sign_vector.astype(np.float32)).reshape(
        -1, width // block_size, block_size
    )
    for stride in (1 << power for power in range(block_size.bit_length() - 1)):
        butterflies = rotated.reshape(-1, width // block_size, block_size // (2 * stride), 2, stride)
        lower = butterflies[..., 0, :].copy()
        upper = butterflies[..., 1, :].copy()
        butterflies[..., 0, :] = lower + upper
        butterflies[..., 1, :] = lower - upper
    return rotated.reshape(inputs.shape) / np.float32(np.sqrt(block_size))


def quantize_a8_per_group(values, group_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric A8 with one FP32 scale per token and contiguous input group."""
    inputs = np.asarray(values, dtype=np.float32)
    if (
        inputs.ndim < 1
        or inputs.shape[-1] == 0
        or group_size < 1
        or inputs.shape[-1] % group_size
        or not np.isfinite(inputs).all()
    ):
        raise ValueError("invalid A8 activation shape or values")

    groups = inputs.reshape(*inputs.shape[:-1], -1, group_size)
    maxima = np.max(np.abs(groups), axis=-1)
    scales = np.where(maxima == 0, np.float32(1), maxima / np.float32(127))
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("A8 group scale is not representable")
    codes = np.clip(np.rint(groups / scales[..., None]), -127, 127).astype(np.int8)
    return codes.reshape(inputs.shape), scales