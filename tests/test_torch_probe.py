"""Offline contract checks for the optional GPU PyTorch smoke."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from embedded_jev import torch_probe


def fake_torch(*, cuda_version="12.8", available=True, correct=True):
    values = MagicMock()
    values.reshape.return_value = values
    product = object()
    values.__matmul__.return_value = product
    return SimpleNamespace(
        __version__="2.10.0+cu128",
        version=SimpleNamespace(cuda=cuda_version),
        cuda=SimpleNamespace(
            is_available=lambda: available,
            get_device_properties=lambda device: SimpleNamespace(
                name="Fake CUDA GPU", major=8, minor=6, total_memory=12 * 1024**3
            ),
            synchronize=lambda device: None,
        ),
        device=lambda name: name,
        float32="float32",
        arange=lambda length, dtype, device: values,
        eye=lambda size, dtype, device: object(),
        equal=lambda result, expected: correct and result is product and expected is values,
    )


def test_cuda_matmul_probe_reports_version_device_and_synchronization():
    torch = fake_torch()
    report = torch_probe.probe_torch_cuda(torch)
    assert report == {
        "torch_version": "2.10.0+cu128",
        "torch_cuda_version": "12.8",
        "device": "Fake CUDA GPU",
        "compute_capability": "8.6",
        "total_memory_bytes": 12 * 1024**3,
        "operation": "4x4_fp32_cuda_matmul",
        "result": "passed",
        "model_downloaded": False,
    }
    torch.cuda.synchronize = MagicMock()
    torch_probe.probe_torch_cuda(torch)
    torch.cuda.synchronize.assert_called_once_with("cuda:0")


def test_cuda_probe_refuses_missing_cpu_only_unavailable_or_wrong_results(monkeypatch):
    def no_torch(name):
        raise ImportError(name)

    monkeypatch.setattr(torch_probe, "import_module", no_torch)
    with pytest.raises(torch_probe.TorchProbeError, match="not installed"):
        torch_probe.probe_torch_cuda()
    with pytest.raises(torch_probe.TorchProbeError, match="no CUDA support"):
        torch_probe.probe_torch_cuda(fake_torch(cuda_version=None))
    with pytest.raises(torch_probe.TorchProbeError, match="cannot access"):
        torch_probe.probe_torch_cuda(fake_torch(available=False))
    with pytest.raises(torch_probe.TorchProbeError, match="incorrect values"):
        torch_probe.probe_torch_cuda(fake_torch(correct=False))