"""Offline CUDA driver probe contracts; no GPU is required."""

import ctypes
from types import SimpleNamespace

import pytest

from embedded_jev.gpu_probe import GPUProbeError, probe_cuda_driver


def fake_driver():
    events = []
    device_memory = bytearray(4)

    def initialize(flags):
        assert flags == 0
        return 0

    def device_count(result):
        ctypes.cast(result, ctypes.POINTER(ctypes.c_int))[0] = 1
        return 0

    def device_get(result, index):
        assert index == 0
        ctypes.cast(result, ctypes.POINTER(ctypes.c_int))[0] = 3
        return 0

    def capability(major, minor, device):
        assert device == 3
        ctypes.cast(major, ctypes.POINTER(ctypes.c_int))[0] = 8
        ctypes.cast(minor, ctypes.POINTER(ctypes.c_int))[0] = 6
        return 0

    def memory(result, device):
        assert device == 3
        ctypes.cast(result, ctypes.POINTER(ctypes.c_size_t))[0] = 12 * 1024**3
        return 0

    def device_name(result, length, device):
        assert length == 128 and device == 3
        ctypes.memmove(result, b"Fake CUDA GPU\0", len(b"Fake CUDA GPU\0"))
        return 0

    def create_context(result, flags, device):
        assert flags == 0 and device == 3
        ctypes.cast(result, ctypes.POINTER(ctypes.c_void_p))[0] = 0x1234
        events.append("context")
        return 0

    def allocate(result, size):
        assert size == 4
        ctypes.cast(result, ctypes.POINTER(ctypes.c_uint64))[0] = 0x5678
        events.append("allocate")
        return 0

    def copy_to_device(memory, source, size):
        assert memory == 0x5678 and size == 4
        device_memory[:] = ctypes.string_at(source, size)
        events.append("to_device")
        return 0

    def copy_to_host(destination, memory, size):
        assert memory == 0x5678 and size == 4
        ctypes.memmove(destination, bytes(device_memory), size)
        events.append("to_host")
        return 0

    def free(memory):
        assert memory == 0x5678
        events.append("free")
        return 0

    def destroy(context):
        assert context == 0x1234
        events.append("destroy")
        return 0

    return SimpleNamespace(
        events=events,
        cuInit=initialize,
        cuDeviceGetCount=device_count,
        cuDeviceGet=device_get,
        cuDeviceComputeCapability=capability,
        cuDeviceTotalMem_v2=memory,
        cuDeviceGetName=device_name,
        cuCtxCreate_v2=create_context,
        cuMemAlloc_v2=allocate,
        cuMemcpyHtoD_v2=copy_to_device,
        cuMemcpyDtoH_v2=copy_to_host,
        cuMemFree_v2=free,
        cuCtxDestroy_v2=destroy,
    )


def test_probe_reports_driver_device_shape_and_memory():
    report = probe_cuda_driver(loader=lambda name: fake_driver())
    assert report == {
        "driver_api": "CUDA",
        "device_count": 1,
        "devices": [{
            "name": "Fake CUDA GPU", "compute_capability": "8.6",
            "total_memory_bytes": 12 * 1024**3,
        }],
    }


def test_memory_round_trip_checks_bytes_and_releases_resources():
    driver = fake_driver()
    report = probe_cuda_driver(loader=lambda name: driver, memory_round_trip=True)
    assert report["memory_round_trip_bytes"] == 4
    assert driver.events == [
        "context", "allocate", "to_device", "to_host", "free", "destroy"
    ]

    driver = fake_driver()
    driver.cuMemcpyHtoD_v2 = lambda *args: 700
    with pytest.raises(GPUProbeError, match="cuMemcpyHtoD_v2 failed with status 700"):
        probe_cuda_driver(loader=lambda name: driver, memory_round_trip=True)
    assert driver.events == ["context", "allocate", "free", "destroy"]


def test_probe_rejects_missing_driver_and_errors():
    def unavailable(name):
        raise OSError("no libcuda")

    with pytest.raises(GPUProbeError, match="libcuda.so.1 is unavailable"):
        probe_cuda_driver(loader=unavailable)

    driver = fake_driver()
    driver.cuInit = lambda flags: 100
    with pytest.raises(GPUProbeError, match="cuInit failed with status 100"):
        probe_cuda_driver(loader=lambda name: driver)

    driver = fake_driver()
    driver.cuDeviceGetCount = lambda result: 0
    with pytest.raises(GPUProbeError, match="no devices"):
        probe_cuda_driver(loader=lambda name: driver)