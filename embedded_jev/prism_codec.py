"""Exact ternary-subset PQ2_0 byte conversion for the pinned Prism format."""

import numpy as np


def pack_ternary_pq2_0(codes, scales) -> np.ndarray:
    """Pack adjacent low-bit-first codes with one little-endian FP16 scale per 128 values."""
    values = np.asarray(codes)
    saved_scales = np.asarray(scales)
    if (
        values.ndim != 3 or values.dtype != np.int8 or values.shape[2] != 128
        or not 1 <= values.shape[0] <= 4096 or not 1 <= values.shape[1] <= 96
        or not np.isin(values, (-1, 0, 1)).all()
        or saved_scales.dtype != np.float16 or saved_scales.shape != values.shape[:2]
        or not np.isfinite(saved_scales).all() or np.any(saved_scales < 0)
        or np.any((saved_scales == 0) & np.any(values != 0, axis=-1))
    ):
        raise ValueError("invalid bounded ternary PQ2_0 codes or scales")
    rows, groups, _ = values.shape
    blocks = np.zeros((rows, groups, 34), dtype=np.uint8)
    blocks[:, :, :2] = np.ascontiguousarray(saved_scales, dtype="<f2").view(np.uint8).reshape(rows, groups, 2)
    lanes = (values + 1).astype(np.uint8).reshape(rows, groups, 32, 4)
    for lane in range(4):
        blocks[:, :, 2:] |= lanes[..., lane] << (2 * lane)
    return blocks


def unpack_ternary_pq2_0(blocks) -> tuple[np.ndarray, np.ndarray]:
    """Decode only {-1,0,+1}; native PQ2_0's additional +2 code is not ternary."""
    packed = np.asarray(blocks)
    if (
        packed.ndim != 3 or packed.dtype != np.uint8 or packed.shape[2] != 34
        or not 1 <= packed.shape[0] <= 4096 or not 1 <= packed.shape[1] <= 96
    ):
        raise ValueError("invalid bounded PQ2_0 block array")
    rows, groups, _ = packed.shape
    scales = np.ascontiguousarray(packed[:, :, :2]).view("<f2").reshape(rows, groups)
    if not np.isfinite(scales).all() or np.any(scales < 0):
        raise ValueError("invalid PQ2_0 FP16 scales")
    codes = np.empty((rows, groups, 32, 4), dtype=np.int8)
    for lane in range(4):
        values = (packed[:, :, 2:] >> (2 * lane)) & 3
        if np.any(values == 3):
            raise ValueError("PQ2_0 +2 code is outside the ternary subset")
        codes[..., lane] = values.astype(np.int8) - 1
    codes = codes.reshape(rows, groups, 128)
    if np.any((scales == 0) & np.any(codes != 0, axis=-1)):
        raise ValueError("zero PQ2_0 scale with nonzero ternary codes")
    return codes, scales