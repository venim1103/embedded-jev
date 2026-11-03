"""Offline CUDA driver probe contracts; no GPU is required."""

import ctypes
import json
import subprocess
from types import SimpleNamespace

import pytest

from embedded_jev import gpu_probe
from embedded_jev.gpu_probe import GPUProbeError, probe_cuda_driver, probe_host_cache


def host_runner(volumes, drives):
    commands = []

    def run(command, **kwargs):
        assert kwargs == {"check": True, "capture_output": True, "text": True, "timeout": 10}
        commands.append(command)
        if command[0] == "podman":
            assert command == [
                "podman", "volume", "ls", "--filter", "name=embedded-jev-cache", "--format", "json",
            ]
            result = volumes
        else:
            assert command == [
                "powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command",
                "ConvertTo-Json -Compress -InputObject @(Get-PSDrive -PSProvider FileSystem "
                "| Select-Object Name,Used,Free)",
            ]
            result = drives
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(stdout=json.dumps(result))

    return run, commands


def test_host_preflight_lists_metadata_without_selecting_or_verifying_cache():
    run, commands = host_runner(
        [{"Name": "embedded-jev-cache-existing", "Mountpoint": "never inspected"}],
        [{"Name": "C", "Used": 1024, "Free": 4096}],
    )
    assert probe_host_cache(run) == {
        "operation": "read_only_host_cache_preflight",
        "cache_volumes": {"status": "listed", "names": ["embedded-jev-cache-existing"]},
        "windows_filesystems": {
            "status": "reported", "drives": [{"name": "C", "used_bytes": 1024, "free_bytes": 4096}],
        },
        "cache_contents_verified": False,
        "vhdx_drive_identified": False,
        "model_execution_authorized": False,
    }
    assert len(commands) == 2


def test_host_preflight_keeps_unavailable_tools_distinct_from_empty_results():
    run, commands = host_runner(FileNotFoundError(), FileNotFoundError())
    report = probe_host_cache(run)
    assert report["cache_volumes"] == {"status": "unavailable", "names": []}
    assert report["windows_filesystems"] == {"status": "unavailable", "drives": []}
    assert len(commands) == 2
    run, _ = host_runner([], [])
    report = probe_host_cache(run)
    assert report["cache_volumes"]["status"] == "listed"
    assert report["windows_filesystems"]["status"] == "reported"


@pytest.mark.parametrize("volumes,drives", [
    ({}, []), ([{}], []), ([{"Name": "another-project"}], []),
    ([{"Name": "embedded-jev-cache-"}], []),
    ([{"Name": "embedded-jev-cache-../escape"}], []),
    ([{"Name": "embedded-jev-cache-same"}] * 2, []),
    ([{"Name": "embedded-jev-cache-many"}] * 129, []),
    ([], {}), ([], [None]), ([], [{"Name": "C", "Used": 1, "Free": True}]),
    ([], [{"Name": "C", "Used": 1, "Free": -1}]),
    ([], [{"Name": "C", "Used": 1, "Free": 2}] * 2),
])
def test_host_preflight_refuses_malformed_or_ambiguous_metadata(volumes, drives):
    run, _ = host_runner(volumes, drives)
    with pytest.raises(GPUProbeError):
        probe_host_cache(run)


@pytest.mark.parametrize("failure", [
    subprocess.TimeoutExpired("podman", 10), subprocess.CalledProcessError(1, "podman"),
])
def test_host_preflight_refuses_command_failure_without_exposing_output(failure):
    run, commands = host_runner(failure, [])
    with pytest.raises(GPUProbeError, match="host preflight command"):
        probe_host_cache(run)
    assert len(commands) == 1


@pytest.mark.parametrize("output,error", [
    ("not JSON", "invalid JSON"), (" " * 32769, "exceeds 32 KiB"),
])
def test_host_preflight_refuses_invalid_or_oversized_json(output, error):
    with pytest.raises(GPUProbeError, match=error):
        probe_host_cache(lambda *args, **kwargs: SimpleNamespace(stdout=output))


def test_host_preflight_cli_never_loads_cuda(monkeypatch, capsys):
    run, _ = host_runner([], [])
    monkeypatch.setattr(gpu_probe.subprocess, "run", run)
    monkeypatch.setattr("sys.argv", ["gpu_probe", "--host-cache-preflight"])
    monkeypatch.setattr(gpu_probe, "probe_cuda_driver", lambda **kwargs: pytest.fail("CUDA accessed"))
    gpu_probe.main()
    assert json.loads(capsys.readouterr().out)["operation"] == "read_only_host_cache_preflight"


def test_host_preflight_cli_refuses_combination_with_gpu_memory_mode(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["gpu_probe", "--host-cache-preflight", "--memory-round-trip"])
    with pytest.raises(SystemExit) as error:
        gpu_probe.main()
    assert error.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


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


@pytest.mark.parametrize("symbol", ["cuMemFree_v2", "cuCtxDestroy_v2"])
def test_memory_round_trip_refuses_missing_release_symbols_before_context_creation(symbol):
    driver = fake_driver()
    delattr(driver, symbol)
    assert probe_cuda_driver(loader=lambda name: driver)["device_count"] == 1
    assert driver.events == []
    with pytest.raises(GPUProbeError, match=f"CUDA driver symbol is missing: {symbol}"):
        probe_cuda_driver(loader=lambda name: driver, memory_round_trip=True)
    assert driver.events == []


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