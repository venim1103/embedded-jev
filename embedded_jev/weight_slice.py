"""Read only tiny BF16 projection slices from the pinned MiMo revision."""

import argparse
import hashlib
import json

import numpy as np

from embedded_jev.inventory import (
    InventoryError,
    _json_object,
    _open_bounded,
    build_inventory,
    fetch_pinned_headers,
)
from embedded_jev.ternary import (
    quantize_ternary_compensated,
    quantize_ternary_rtn,
    reconstruct_ternary,
)


MAX_ROWS = 4
MAX_GROUPS = 2
GROUP_SIZE = 128
MAX_PAYLOAD_BYTES = MAX_ROWS * MAX_GROUPS * GROUP_SIZE * 2
DEFAULT_TENSOR = "model.language_model.layers.3.mlp.down_proj.weight"


def fetch_bf16_projection_slice(
    metadata_files, shard_headers, *, name=DEFAULT_TENSOR,
    start_row: int = 0, rows: int = 4, start_column: int = 0, groups: int = 2,
) -> tuple[np.ndarray, dict]:
    """Read bounded row/group ranges, never an entire tensor or weight shard."""
    if (
        type(start_row) is not int or start_row < 0
        or type(rows) is not int or not 1 <= rows <= MAX_ROWS
        or type(start_column) is not int or start_column < 0
        or start_column % GROUP_SIZE
        or type(groups) is not int or not 1 <= groups <= MAX_GROUPS
    ):
        raise InventoryError("unsupported BF16 slice request bounds")

    report = build_inventory(metadata_files, shard_headers)
    tensor = next((item for item in report["tensors"] if item["name"] == name), None)
    if tensor is None or tensor["dtype"] != "BF16" or not tensor["quantization_eligible"]:
        raise InventoryError("BF16 slice requires an indexed eligible projection")
    total_rows, total_columns = tensor["shape"]
    columns = groups * GROUP_SIZE
    if start_row + rows > total_rows or start_column + columns > total_columns:
        raise InventoryError("BF16 slice extends outside the tensor")
    shard = tensor["shard"]
    header, file_bytes = shard_headers[shard]
    index = _json_object(header[8:], shard)
    offset = index[name]["data_offsets"][0]
    payload_bytes = rows * columns * 2
    if payload_bytes > MAX_PAYLOAD_BYTES:
        raise InventoryError("BF16 slice exceeds transfer budget")
    values = np.empty((rows, columns), dtype=np.float32)
    digests = []
    for row in range(rows):
        absolute_start = (
            len(header) + offset +
            ((start_row + row) * total_columns + start_column) * 2
        )
        if absolute_start < len(header) or absolute_start + columns * 2 > file_bytes:
            raise InventoryError("BF16 slice offset exceeds shard")
        data, reported_bytes = _open_bounded(shard, start=absolute_start, length=columns * 2)
        if reported_bytes != file_bytes:
            raise InventoryError("BF16 shard size changed during range fetch")
        bits = np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16
        values[row] = bits.view("<f4").astype(np.float32)
        digests.append(hashlib.sha256(data).hexdigest())
    if not np.isfinite(values).all():
        raise InventoryError("BF16 slice contains nonfinite values")
    return values, {
        "model": report["source"]["model"],
        "revision": report["source"]["revision"],
        "tensor": name,
        "shard": shard,
        "start_row": start_row,
        "rows": rows,
        "start_column": start_column,
        "columns": columns,
        "payload_bytes": payload_bytes,
        "row_sha256": digests,
        "full_weight_hash": "not_checked_bounded_slice_only",
    }


def screen_synthetic_reconstruction(weights) -> dict:
    """Compare fixed toy policies on disjoint synthetic activation sets."""
    matrix = np.asarray(weights, dtype=np.float32)
    if matrix.ndim != 2 or 0 in matrix.shape or matrix.shape[0] > MAX_ROWS or matrix.shape[1] != 256:
        raise ValueError("toy screen requires at most four rows of 256 weights")
    calibration = np.random.default_rng(902).normal(size=(512, 256)).astype(np.float32)
    evaluation = np.random.default_rng(903).normal(size=(64, 256)).astype(np.float32)
    dense = evaluation @ matrix.T

    def mse(artifact):
        reconstructed = reconstruct_ternary(*artifact)
        return float(np.mean((evaluation @ reconstructed.T - dense) ** 2))

    return {
        "purpose": "synthetic_local_reconstruction_not_model_quality",
        "calibration": "512_synthetic_gaussian_rows_seed_902",
        "evaluation": "64_disjoint_synthetic_gaussian_rows_seed_903",
        "local_mse": {
            "rtn_maxabs": mse(quantize_ternary_rtn(matrix)),
            "rtn_grid": mse(quantize_ternary_rtn(matrix, scale_search=True)),
            "compensated_maxabs": mse(quantize_ternary_compensated(
                matrix, calibration, processing_block_size=128
            )[:2]),
            "compensated_grid": mse(quantize_ternary_compensated(
                matrix, calibration, processing_block_size=128, scale_search=True
            )[:2]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read a bounded pinned BF16 projection slice")
    parser.add_argument("--screen-toy", action="store_true", help="run synthetic local MSE only")
    args = parser.parse_args()
    metadata, headers = fetch_pinned_headers()
    values, provenance = fetch_bf16_projection_slice(metadata, headers)
    provenance["min_value"] = float(values.min())
    provenance["max_value"] = float(values.max())
    if args.screen_toy:
        provenance["screen"] = screen_synthetic_reconstruction(values)
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()