"""Exercise a tiny CUDA matmul with an optional GPU-enabled PyTorch install."""

import json
from importlib import import_module


class TorchProbeError(RuntimeError):
    """The optional GPU Torch environment cannot run the smoke operation."""


def probe_torch_cuda(torch_module=None) -> dict:
    if torch_module is None:
        try:
            torch_module = import_module("torch")
        except ImportError as exc:
            raise TorchProbeError("CUDA-enabled PyTorch is not installed") from exc
    if torch_module.version.cuda is None:
        raise TorchProbeError("installed PyTorch has no CUDA support")
    if not torch_module.cuda.is_available():
        raise TorchProbeError("PyTorch cannot access a CUDA device")

    device = torch_module.device("cuda:0")
    properties = torch_module.cuda.get_device_properties(device)
    values = torch_module.arange(16, dtype=torch_module.float32, device=device).reshape(4, 4)
    identity = torch_module.eye(4, dtype=torch_module.float32, device=device)
    product = values @ identity
    torch_module.cuda.synchronize(device)
    if not torch_module.equal(product, values):
        raise TorchProbeError("CUDA matrix multiplication produced incorrect values")
    return {
        "torch_version": torch_module.__version__,
        "torch_cuda_version": torch_module.version.cuda,
        "device": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": properties.total_memory,
        "operation": "4x4_fp32_cuda_matmul",
        "result": "passed",
        "model_downloaded": False,
    }


def main() -> None:
    try:
        result = probe_torch_cuda()
    except TorchProbeError as exc:
        raise SystemExit(f"CUDA Torch probe failed: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()