"""Read only tiny BF16 projection slices from the pinned MiMo revision."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from embedded_jev.activation import rotate_signed_hadamard
from embedded_jev.inventory import (
    InventoryError,
    _json_object,
    _open_bounded,
    build_inventory,
    fetch_pinned_headers,
    read_local_headers,
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
MAX_LOCAL_PAYLOAD_BYTES = 128 * 1024
MAX_STREAM_TENSOR_BYTES = 128 * 1024 * 1024
STREAM_ROWS = 64
DEFAULT_TENSOR = "model.language_model.layers.3.mlp.down_proj.weight"


def read_local_bf16_projection_rows(
    directory: Path, *, name=DEFAULT_TENSOR, start_row: int = 0, rows: int = 4,
) -> tuple[np.ndarray, dict]:
    """Read at most four complete projection rows from a local pinned snapshot."""
    if type(start_row) is not int or start_row < 0 or type(rows) is not int or not 1 <= rows <= MAX_ROWS:
        raise InventoryError("unsupported local BF16 row request")

    metadata, shard_headers = read_local_headers(directory)
    report = build_inventory(metadata, shard_headers)
    tensor = next((entry for entry in report["tensors"] if entry["name"] == name), None)
    if tensor is None or tensor["dtype"] != "BF16" or not tensor["quantization_eligible"]:
        raise InventoryError("local BF16 rows require an indexed eligible projection")
    total_rows, columns = tensor["shape"]
    row_bytes = columns * 2
    if start_row + rows > total_rows or rows * row_bytes > MAX_LOCAL_PAYLOAD_BYTES:
        raise InventoryError("local BF16 rows exceed bounds or payload budget")

    shard = tensor["shard"]
    header, file_bytes = shard_headers[shard]
    offsets = _json_object(header[8:], shard)[name]["data_offsets"]
    values = np.empty((rows, columns), dtype=np.float32)
    digests = []
    try:
        with (directory / shard).open("rb") as source:
            for row in range(rows):
                position = len(header) + offsets[0] + (start_row + row) * row_bytes
                if position < len(header) or position + row_bytes > len(header) + offsets[1]:
                    raise InventoryError("local BF16 row exceeds tensor bounds")
                source.seek(position)
                data = source.read(row_bytes)
                if len(data) != row_bytes or position + row_bytes > file_bytes:
                    raise InventoryError("short local BF16 row")
                bits = np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16
                values[row] = bits.view("<f4")
                digests.append(hashlib.sha256(data).hexdigest())
    except OSError as exc:
        raise InventoryError(f"unable to read local BF16 rows: {exc}") from exc
    if not np.isfinite(values).all():
        raise InventoryError("local BF16 rows contain nonfinite values")
    return values, {
        "model": report["source"]["model"],
        "revision": report["source"]["revision"],
        "tensor": name,
        "shard": shard,
        "start_row": start_row,
        "rows": rows,
        "columns": columns,
        "payload_bytes": rows * row_bytes,
        "row_sha256": digests,
        "full_weight_hash": "not_checked_local_rows_only",
    }


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


def screen_calibration_reconstruction(weights, activations, *, validation_activations=None) -> dict:
    """Compare four bounded policies on calibration-only slice reconstructions."""
    matrix = np.asarray(weights, dtype=np.float32)
    samples = np.asarray(activations, dtype=np.float32)
    if (
        matrix.ndim != 2 or not 1 <= matrix.shape[0] <= MAX_ROWS or matrix.shape[1] != 256
        or samples.ndim != 2 or not 1 <= samples.shape[0] <= 128 or samples.shape[1] != 256
        or not np.isfinite(matrix).all() or not np.isfinite(samples).all()
    ):
        raise ValueError("calibration screen requires bounded four-row, 256-column inputs")
    if not np.any(samples):
        raise ValueError("calibration screen requires nonzero activation curvature")
    validation = None
    if validation_activations is not None:
        validation = np.asarray(validation_activations, dtype=np.float32)
        if (
            validation.ndim != 2 or not 1 <= validation.shape[0] <= 128 or validation.shape[1] != 256
            or not np.isfinite(validation).all()
        ):
            raise ValueError("validation screen requires bounded 256-column inputs")
    reference = samples.astype(np.float64) @ matrix.astype(np.float64).T
    reference_energy = float(np.mean(reference ** 2))
    policies = {}
    for label, compensated, search in (
        ("rtn_maxabs", False, False), ("rtn_grid", False, True),
        ("compensated_maxabs", True, False), ("compensated_grid", True, True),
    ):
        if compensated:
            codes, scales, details = quantize_ternary_compensated(
                matrix, samples, processing_block_size=128, scale_search=search,
            )
        else:
            codes, scales = quantize_ternary_rtn(matrix, scale_search=search)
            details = {}
        reconstruction = reconstruct_ternary(codes, scales).astype(np.float64)
        error = samples.astype(np.float64) @ reconstruction.T - reference
        output_mse = float(np.mean(error ** 2))
        weight_mse = float(np.mean((reconstruction - matrix) ** 2))
        if not np.isfinite(output_mse) or not np.isfinite(weight_mse):
            raise ValueError("nonfinite calibrated slice error")
        policies[label] = {
            "output_mse": output_mse, "weight_mse": weight_mse,
            "relative_output_rmse": float(np.sqrt(output_mse / reference_energy)) if reference_energy else None,
            "scale_dtype": str(scales.dtype), **details,
            "codes_sha256": hashlib.sha256(codes.tobytes()).hexdigest(),
            "scales_sha256": hashlib.sha256(scales.tobytes()).hexdigest(),
        }
        if validation is not None:
            validation_reference = validation.astype(np.float64) @ matrix.astype(np.float64).T
            validation_error = validation.astype(np.float64) @ reconstruction.T - validation_reference
            validation_mse = float(np.mean(validation_error ** 2))
            validation_energy = float(np.mean(validation_reference ** 2))
            if not np.isfinite(validation_mse):
                raise ValueError("nonfinite validation slice error")
            policies[label]["validation_output_mse"] = validation_mse
            policies[label]["validation_relative_output_rmse"] = (
                float(np.sqrt(validation_mse / validation_energy)) if validation_energy else None
            )
    return {
        "purpose": (
            "calibration_fit_validation_slice_not_model_quality" if validation is not None
            else "calibration_slice_reconstruction_not_validation_or_model_quality"
        ),
        "rows": matrix.shape[0], "columns": matrix.shape[1], "calibration_rows": samples.shape[0],
        "transform": "identity", "activation_quantized": False, "policies": policies,
        "validation_rows": validation.shape[0] if validation is not None else 0,
        "validation_used_for_fitting": False,
    }


def screen_local_calibrated_slice(
    directory: Path, capture_dir: Path, *, dataset_path: Path | None = None,
    validation_case_id: str | None = None,
) -> dict:
    """Use only a hash-checked calibration capture for a local two-group trial."""
    from embedded_jev.decision_dataset import load_calibration_capture, load_decision_dataset

    activations, manifest = load_calibration_capture(capture_dir)
    if (dataset_path is None) != (validation_case_id is None):
        raise InventoryError("live validation requires a dataset and an explicit validation case id")
    weights, provenance = read_local_bf16_projection_rows(directory)
    if manifest["tensor"] != provenance["tensor"] or activations.shape[1] != weights.shape[1]:
        raise InventoryError("calibration capture disagrees with local projection input width")
    validation = None
    validation_report = None
    if dataset_path is not None:
        from embedded_jev.streamed_text import run_streamed_text

        _, digest = load_decision_dataset(dataset_path)
        if digest != manifest["dataset"]["sha256"]:
            raise InventoryError("calibration capture and validation dataset digest disagree")
        observed = []
        validation_report = run_streamed_text(
            directory, layers=4, prompt="", dataset_path=dataset_path,
            split="validation", case_id=validation_case_id, activation_observer=observed.append,
        )
        if len(observed) != 1 or validation_report["dataset"]["sha256"] != digest:
            raise InventoryError("missing or changed live validation activation source")
        validation = observed[0][:, :256]
    result = {
        "source": provenance, "capture_dataset": manifest["dataset"],
        "capture_array_sha256": manifest["array"]["sha256"],
        "scope": "first_four_rows_first_two_groups_not_full_projection_or_quality",
        "screen": screen_calibration_reconstruction(
            weights[:, :256], activations[:, :256], validation_activations=validation,
        ),
        "candidate_saved": False,
    }
    if validation_report is not None:
        result["validation_source"] = validation_report["dataset"]
        result["validation_input_sha256"] = validation_report["ffn_down_input_sha256"]
        result["validation_capture_saved"] = False
    return result


def screen_full_width_synthetic(weights) -> dict:
    """Compare bounded RTN options on identical synthetic full-width inputs."""
    matrix = np.asarray(weights, dtype=np.float32)
    if (
        matrix.ndim != 2 or 0 in matrix.shape or matrix.shape[0] > MAX_ROWS
        or matrix.shape[1] % GROUP_SIZE or matrix.nbytes > 2 * MAX_LOCAL_PAYLOAD_BYTES
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("full-width screen requires at most four bounded projection rows")
    inputs = np.random.default_rng(903).normal(size=(32, matrix.shape[1])).astype(np.float32)
    reference = inputs @ matrix.T
    signs = np.random.default_rng(773).choice([-1, 1], size=matrix.shape[1])
    rotated_weights = rotate_signed_hadamard(matrix, signs, GROUP_SIZE)
    rotated_inputs = rotate_signed_hadamard(inputs, signs, GROUP_SIZE)
    if not np.allclose(rotated_inputs @ rotated_weights.T, reference, rtol=1e-5, atol=1e-4):
        raise ValueError("signed rotation changed dense projection")
    reference_energy = float(np.mean(reference ** 2))
    results = {}
    for label, source, activations, search in (
        ("maxabs", matrix, inputs, False),
        ("searched_fp16", matrix, inputs, True),
        ("signed_hadamard_128_searched_fp16", rotated_weights, rotated_inputs, True),
    ):
        codes, scales = quantize_ternary_rtn(source, scale_search=search)
        reconstruction = reconstruct_ternary(codes, scales)
        weight_mse = float(np.mean((reconstruction - source) ** 2))
        output_mse = float(np.mean((activations @ reconstruction.T - reference) ** 2))
        if not np.isfinite(weight_mse) or not np.isfinite(output_mse):
            raise ValueError("nonfinite synthetic reconstruction error")
        results[label] = {
            "weight_mse": weight_mse,
            "output_mse": output_mse,
            "relative_output_rmse": (
                float(np.sqrt(output_mse / reference_energy)) if reference_energy else None
            ),
        }
    return {
        "purpose": "four_full_width_rows_synthetic_inputs_not_model_quality",
        "evaluation": "32_synthetic_gaussian_rows_seed_903",
        "signs": "signed_hadamard_128_seed_773",
        "policies": results,
    }


def screen_local_projection(directory: Path) -> dict:
    """Stream one pinned projection through in-memory RTN reconstruction checks."""
    metadata, shard_headers = read_local_headers(directory)
    report = build_inventory(metadata, shard_headers)
    tensor = next((entry for entry in report["tensors"] if entry["name"] == DEFAULT_TENSOR), None)
    if tensor is None or tensor["dtype"] != "BF16" or not tensor["quantization_eligible"]:
        raise InventoryError("stream screen requires the pinned eligible BF16 projection")
    rows, columns = tensor["shape"]
    if (
        rows > 4096 or columns > 12288 or columns % GROUP_SIZE
        or tensor["storage_bytes"] > MAX_STREAM_TENSOR_BYTES
    ):
        raise InventoryError("projection exceeds stream screen bounds")
    shard = tensor["shard"]
    header, file_bytes = shard_headers[shard]
    offsets = _json_object(header[8:], shard)[DEFAULT_TENSOR]["data_offsets"]
    if offsets[1] - offsets[0] != tensor["storage_bytes"] or len(header) + offsets[1] > file_bytes:
        raise InventoryError("local projection byte span mismatch")

    signs = np.random.default_rng(773).choice([-1, 1], size=columns)
    results = {
        label: {"squared_error": 0.0, "nonzero_codes": 0, "row_mse": []}
        for label in ("maxabs", "searched_fp16", "signed_hadamard_128_searched_fp16")
    }
    weight_energy = 0.0
    try:
        with (directory / shard).open("rb") as source:
            source.seek(len(header) + offsets[0])
            for start_row in range(0, rows, STREAM_ROWS):
                batch_rows = min(STREAM_ROWS, rows - start_row)
                data = source.read(batch_rows * columns * 2)
                if len(data) != batch_rows * columns * 2:
                    raise InventoryError("short streamed BF16 projection batch")
                weights = (np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16).view("<f4")
                weights = weights.reshape(batch_rows, columns)
                if not np.isfinite(weights).all():
                    raise InventoryError("streamed BF16 projection contains nonfinite values")
                weight_energy += float(np.sum(weights.astype(np.float64) ** 2))
                rotated = rotate_signed_hadamard(weights, signs, GROUP_SIZE)
                for label, matrix, search in (
                    ("maxabs", weights, False),
                    ("searched_fp16", weights, True),
                    ("signed_hadamard_128_searched_fp16", rotated, True),
                ):
                    codes, scales = quantize_ternary_rtn(matrix, scale_search=search)
                    difference = matrix.astype(np.float64) - reconstruct_ternary(codes, scales)
                    result = results[label]
                    result["squared_error"] += float(np.sum(difference ** 2))
                    result["nonzero_codes"] += int(np.count_nonzero(codes))
                    result["row_mse"].extend(np.mean(difference ** 2, axis=1).tolist())
    except OSError as exc:
        raise InventoryError(f"unable to stream local projection: {exc}") from exc
    if not np.isfinite(weight_energy) or any(
        not np.isfinite(result["squared_error"]) for result in results.values()
    ):
        raise InventoryError("nonfinite streamed projection error")
    for result in results.values():
        squared_error = result.pop("squared_error")
        result["weight_mse"] = squared_error / tensor["parameters"]
        result["relative_weight_rmse"] = (
            float(np.sqrt(squared_error / weight_energy)) if weight_energy else None
        )
        result["nonzero_fraction"] = result.pop("nonzero_codes") / tensor["parameters"]
        result["row_mse_quantiles_0_50_95_99_100"] = np.percentile(
            result.pop("row_mse"), [0, 50, 95, 99, 100]
        ).tolist()
    return {
        "model": report["source"]["model"],
        "revision": report["source"]["revision"],
        "tensor": DEFAULT_TENSOR,
        "rows": rows,
        "columns": columns,
        "payload_bytes": tensor["storage_bytes"],
        "rows_per_batch": STREAM_ROWS,
        "purpose": "whole_projection_weight_reconstruction_not_model_quality",
        "full_weight_hash": "not_checked_stream_only",
        "policies": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read a bounded pinned BF16 projection slice")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--screen-toy", action="store_true", help="run synthetic local MSE only")
    modes.add_argument("--screen-full-rows", action="store_true", help="screen four local full-width rows")
    modes.add_argument("--screen-projection", action="store_true", help="stream one full local projection")
    modes.add_argument("--screen-calibrated-slice", action="store_true", help="compare bounded policies on calibration inputs only")
    parser.add_argument("--local-dir", type=Path, help="verified local model snapshot for full-width rows")
    parser.add_argument("--calibration-capture", type=Path)
    parser.add_argument("--validation-dataset", type=Path)
    parser.add_argument("--validation-case-id")
    args = parser.parse_args()
    if args.screen_calibrated_slice:
        if args.local_dir is None or args.calibration_capture is None:
            parser.error("--screen-calibrated-slice requires --local-dir and --calibration-capture")
        print(json.dumps(screen_local_calibrated_slice(
            args.local_dir, args.calibration_capture, dataset_path=args.validation_dataset,
            validation_case_id=args.validation_case_id,
        ), indent=2, sort_keys=True))
        return
    if args.calibration_capture is not None:
        parser.error("--calibration-capture requires --screen-calibrated-slice")
    if args.validation_dataset is not None or args.validation_case_id is not None:
        parser.error("live validation requires --screen-calibrated-slice")
    if args.screen_projection:
        if args.local_dir is None:
            parser.error("--screen-projection requires --local-dir")
        print(json.dumps(screen_local_projection(args.local_dir), indent=2, sort_keys=True))
        return
    if args.screen_full_rows:
        if args.local_dir is None:
            parser.error("--screen-full-rows requires --local-dir")
        values, provenance = read_local_bf16_projection_rows(args.local_dir)
        provenance["screen"] = screen_full_width_synthetic(values)
    else:
        if args.local_dir is not None:
            parser.error("--local-dir requires --screen-full-rows or --screen-projection")
        metadata, headers = fetch_pinned_headers()
        values, provenance = fetch_bf16_projection_slice(metadata, headers)
        if args.screen_toy:
            provenance["screen"] = screen_synthetic_reconstruction(values)
    provenance["min_value"] = float(values.min())
    provenance["max_value"] = float(values.max())
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()