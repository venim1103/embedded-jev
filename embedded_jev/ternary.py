"""Deterministic row/group ternary baseline with retained FP16 scales."""

import numpy as np


def quantize_ternary_rtn(weights, group_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Assign -1/0/+1 codes using each group's representable max-abs scale."""
    matrix = np.asarray(weights, dtype=np.float32)
    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[1] == 0
        or group_size < 1
        or matrix.shape[1] % group_size
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("invalid ternary matrix or group width")

    groups = matrix.reshape(matrix.shape[0], -1, group_size)
    maxima = np.max(np.abs(groups), axis=-1)
    with np.errstate(over="ignore", under="ignore"):
        scales = maxima.astype(np.float16)
    if not np.isfinite(scales).all() or np.any((maxima > 0) & (scales == 0)):
        raise ValueError("ternary group scale is not representable in FP16")

    divisors = np.where(scales == 0, np.float32(1), scales.astype(np.float32))
    codes = np.clip(np.rint(groups / divisors[..., None]), -1, 1).astype(np.int8)
    return codes.reshape(matrix.shape), scales


def reconstruct_ternary(codes, scales) -> np.ndarray:
    """Rebuild weights from stored codes and stored FP16 row/group scales."""
    ternary = np.asarray(codes)
    saved_scales = np.asarray(scales)
    if (
        ternary.ndim != 2
        or ternary.dtype != np.int8
        or 0 in ternary.shape
        or not np.isin(ternary, (-1, 0, 1)).all()
        or saved_scales.ndim != 2
        or saved_scales.dtype != np.float16
        or saved_scales.shape[0] != ternary.shape[0]
        or 0 in saved_scales.shape
        or ternary.shape[1] % saved_scales.shape[1]
        or not np.isfinite(saved_scales).all()
        or np.any(saved_scales < 0)
    ):
        raise ValueError("invalid stored ternary codes or scales")
    group_size = ternary.shape[1] // saved_scales.shape[1]
    groups = ternary.reshape(ternary.shape[0], saved_scales.shape[1], group_size)
    if np.any((saved_scales == 0) & np.any(groups != 0, axis=-1)):
        raise ValueError("invalid stored ternary codes or scales")
    return (groups.astype(np.float32) * saved_scales.astype(np.float32)[..., None]).reshape(
        ternary.shape
    )