"""Deterministic toy ternary quantizers with retained FP16 row/group scales.

The compensated traversal adapts the update ordering from IST-DASLab/gptq
commit 2d65066eeb06a5c9ff5184d8cebdf33662c67faf under the repo's
Apache-2.0 license; it is not a drop-in implementation of upstream GPTQ.
"""

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class CompensationFactors:
    sample_shape: tuple[int, int]
    sample_sha256: str
    damping_ratio: float
    max_damping_attempts: int
    damping: float
    damping_attempts: int
    upper: np.ndarray


def prepare_compensation_factors(
    activations, *, damping_ratio: float = 0.01, max_damping_attempts: int = 3,
) -> CompensationFactors:
    """Prepare one bounded, read-only curvature factor tied to calibration inputs."""
    samples = np.asarray(activations, dtype=np.float64)
    if (
        samples.ndim != 2 or samples.shape[0] == 0 or not 1 <= samples.shape[1] <= 256
        or not np.isfinite(samples).all() or not np.isfinite(damping_ratio) or damping_ratio <= 0
        or type(max_damping_attempts) is not int or not 1 <= max_damping_attempts <= 4
    ):
        raise ValueError("invalid compensation factor inputs or bounds")
    width = samples.shape[1]
    curvature = 2 * samples.T @ samples / samples.shape[0]
    baseline_damp = damping_ratio * np.mean(np.diag(curvature))
    if not np.isfinite(curvature).all() or baseline_damp <= 0:
        raise ValueError("nonfinite or zero activation curvature")
    for attempt in range(max_damping_attempts):
        damping = baseline_damp * 10**attempt
        damped = curvature + np.eye(width) * damping
        try:
            np.linalg.cholesky(damped)
            inverse = np.linalg.solve(damped, np.eye(width))
            inverse = (inverse + inverse.T) / 2
            upper = np.linalg.cholesky(inverse).T
            break
        except np.linalg.LinAlgError:
            continue
    else:
        raise ValueError("curvature factorization failed after bounded damping")
    if not np.isfinite(upper).all() or not np.isfinite(damping):
        raise ValueError("nonfinite compensation factors")
    upper.setflags(write=False)
    return CompensationFactors(
        sample_shape=tuple(samples.shape),
        sample_sha256=hashlib.sha256(np.ascontiguousarray(samples).tobytes()).hexdigest(),
        damping_ratio=float(damping_ratio), max_damping_attempts=max_damping_attempts,
        damping=float(damping), damping_attempts=attempt + 1, upper=upper,
    )


def _group_scales(groups: np.ndarray, scale_search: bool) -> np.ndarray:
    maxima = np.max(np.abs(groups), axis=-1)
    with np.errstate(over="ignore", under="ignore"):
        baseline = maxima.astype(np.float16)
    if not np.isfinite(baseline).all() or np.any((maxima > 0) & (baseline == 0)):
        raise ValueError("ternary group scale is not representable in FP16")
    if not scale_search:
        return baseline

    best = baseline.copy()
    best_error = np.full(maxima.shape, np.inf)
    for ratio in np.linspace(0.5, 1.0, 11):
        with np.errstate(over="ignore", under="ignore"):
            candidate = (maxima * ratio).astype(np.float16)
        valid = (maxima == 0) | (candidate > 0)
        divisor = np.where(candidate == 0, 1.0, candidate.astype(np.float64))
        codes = np.clip(np.rint(groups / divisor[..., None]), -1, 1)
        error = np.sum(
            (groups - codes * candidate.astype(np.float64)[..., None]) ** 2,
            axis=-1,
        )
        better = valid & (error < best_error)
        best[better] = candidate[better]
        best_error[better] = error[better]
    return best


def quantize_ternary_rtn(
    weights, group_size: int = 128, *, scale_search: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Assign -1/0/+1 codes using each group's representable max-abs scale."""
    matrix = np.asarray(weights, dtype=np.float32)
    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[1] == 0
        or group_size < 1
        or matrix.shape[1] % group_size
        or type(scale_search) is not bool
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("invalid ternary matrix or group width")

    groups = matrix.reshape(matrix.shape[0], -1, group_size)
    scales = _group_scales(groups, scale_search)

    divisors = np.where(scales == 0, np.float32(1), scales.astype(np.float32))
    codes = np.clip(np.rint(groups / divisors[..., None]), -1, 1).astype(np.int8)
    return codes.reshape(matrix.shape), scales


def pack_group128_codes(codes: np.ndarray) -> np.ndarray:
    """Pack -1/0/+1 groups for the isolated BitNet-derived AVX2 kernel."""
    ternary = np.asarray(codes)
    if (
        ternary.ndim != 3 or ternary.dtype != np.int8 or ternary.shape[2] != 128
        or 0 in ternary.shape or not np.isin(ternary, (-1, 0, 1)).all()
    ):
        raise ValueError("group-128 ternary codes required for native packing")
    rows, groups, _ = ternary.shape
    lanes = (ternary + 1).astype(np.uint8).reshape(rows, groups, 4, 32)
    packed = np.zeros((rows, groups, 32), dtype=np.uint8)
    for lane in range(4):
        packed |= lanes[:, :, lane, :] << (6 - 2 * lane)
    return packed


def unpack_group128_codes(packed: np.ndarray) -> np.ndarray:
    """Decode the isolated AVX2 2-bit groups back to signed ternary codes."""
    blocks = np.asarray(packed)
    if blocks.ndim != 3 or blocks.dtype != np.uint8 or 0 in blocks.shape or blocks.shape[2] != 32:
        raise ValueError("packed group-128 ternary blocks required")
    rows, groups, _ = blocks.shape
    codes = np.empty((rows, groups, 4, 32), dtype=np.int8)
    for lane in range(4):
        value = (blocks >> (6 - 2 * lane)) & 3
        if np.any(value == 3):
            raise ValueError("packed ternary block contains invalid code")
        codes[:, :, lane, :] = value.astype(np.int8) - 1
    return codes.reshape(rows, groups, 128)


def quantize_ternary_compensated(
    weights, activations, group_size: int = 128, *, damping_ratio: float = 0.01,
    max_damping_attempts: int = 3, processing_block_size: int | None = None,
    scale_search: bool = False,
    prepared: CompensationFactors | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Toy GPTQ-style traversal with fixed representable row/group scales."""
    matrix = np.asarray(weights, dtype=np.float32)
    samples = np.asarray(activations, dtype=np.float64)
    if processing_block_size is None and matrix.ndim == 2:
        processing_block_size = matrix.shape[1]
    if (
        matrix.ndim != 2 or 0 in matrix.shape or matrix.shape[1] > 256
        or group_size < 1 or matrix.shape[1] % group_size
        or type(processing_block_size) is not int
        or processing_block_size < group_size
        or processing_block_size > matrix.shape[1]
        or processing_block_size % group_size
        or type(scale_search) is not bool
        or samples.ndim != 2 or samples.shape[0] == 0
        or samples.shape[1] != matrix.shape[1]
        or not np.isfinite(matrix).all() or not np.isfinite(samples).all()
        or not np.isfinite(damping_ratio) or damping_ratio <= 0
        or type(max_damping_attempts) is not int
        or not 1 <= max_damping_attempts <= 4
    ):
        raise ValueError("invalid compensated ternary inputs or bounds")

    width = matrix.shape[1]
    if prepared is None:
        prepared = prepare_compensation_factors(
            samples, damping_ratio=damping_ratio, max_damping_attempts=max_damping_attempts,
        )
    if (
        not isinstance(prepared, CompensationFactors)
        or prepared.sample_shape != tuple(samples.shape)
        or prepared.sample_sha256 != hashlib.sha256(np.ascontiguousarray(samples).tobytes()).hexdigest()
        or prepared.damping_ratio != damping_ratio
        or prepared.max_damping_attempts != max_damping_attempts
        or not isinstance(prepared.upper, np.ndarray)
        or prepared.upper.shape != (width, width) or prepared.upper.dtype != np.float64
        or prepared.upper.flags.writeable or not np.isfinite(prepared.upper).all()
        or np.any(np.diag(prepared.upper) <= 0) or np.any(np.tril(prepared.upper, k=-1) != 0)
        or not np.isfinite(prepared.damping) or prepared.damping <= 0
        or type(prepared.damping_attempts) is not int or not 1 <= prepared.damping_attempts <= max_damping_attempts
    ):
        raise ValueError("prepared compensation factors disagree with calibration inputs or settings")
    upper = prepared.upper

    working = matrix.astype(np.float64)
    codes = np.zeros(matrix.shape, dtype=np.int8)
    scales = np.empty((matrix.shape[0], width // group_size), dtype=np.float16)
    for block_start in range(0, width, processing_block_size):
        block_end = min(block_start + processing_block_size, width)
        block_weights = working[:, block_start:block_end].copy()
        block_errors = np.zeros_like(block_weights)
        for local_column in range(block_end - block_start):
            column = block_start + local_column
            group = column // group_size
            if column % group_size == 0:
                group_weights = block_weights[:, local_column : local_column + group_size]
                scales[:, group] = _group_scales(group_weights, scale_search)

            group_scales = scales[:, group].astype(np.float64)
            divisors = np.where(group_scales == 0, 1.0, group_scales)
            current = block_weights[:, local_column]
            codes[:, column] = np.clip(np.rint(current / divisors), -1, 1)
            reconstructed = codes[:, column] * group_scales
            error = (current - reconstructed) / upper[column, column]
            block_weights[:, local_column:] -= (
                error[:, None] * upper[column, column:block_end][None, :]
            )
            block_errors[:, local_column] = error
        if block_end < width:
            working[:, block_end:] -= block_errors @ upper[block_start:block_end, block_end:]

    return codes, scales, {"damping": prepared.damping, "damping_attempts": prepared.damping_attempts}


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