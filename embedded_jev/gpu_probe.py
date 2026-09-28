"""Inspect the CUDA driver API without Torch, a toolkit, or model weights."""

import argparse
import ctypes
import json


class GPUProbeError(RuntimeError):
    """CUDA driver access or device enumeration failed."""


def probe_cuda_driver(loader=ctypes.CDLL, *, memory_round_trip: bool = False) -> dict:
    try:
        driver = loader("libcuda.so.1")
    except OSError as exc:
        raise GPUProbeError("CUDA driver library libcuda.so.1 is unavailable") from exc

    def call(name, argtypes, *args):
        try:
            function = getattr(driver, name)
        except AttributeError as exc:
            raise GPUProbeError(f"CUDA driver symbol is missing: {name}") from exc
        function.argtypes = argtypes
        function.restype = ctypes.c_int
        status = function(*args)
        if status != 0:
            raise GPUProbeError(f"CUDA driver {name} failed with status {status}")

    pointer_int = ctypes.POINTER(ctypes.c_int)
    call("cuInit", [ctypes.c_uint], 0)
    count = ctypes.c_int()
    call("cuDeviceGetCount", [pointer_int], ctypes.byref(count))
    if count.value < 1:
        raise GPUProbeError("CUDA driver reports no devices")

    devices = []
    for index in range(count.value):
        device = ctypes.c_int()
        call("cuDeviceGet", [pointer_int, ctypes.c_int], ctypes.byref(device), index)
        major = ctypes.c_int()
        minor = ctypes.c_int()
        call(
            "cuDeviceComputeCapability", [pointer_int, pointer_int, ctypes.c_int],
            ctypes.byref(major), ctypes.byref(minor), device.value,
        )
        total_bytes = ctypes.c_size_t()
        call(
            "cuDeviceTotalMem_v2", [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int],
            ctypes.byref(total_bytes), device.value,
        )
        name = ctypes.create_string_buffer(128)
        call(
            "cuDeviceGetName", [ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int],
            name, len(name), device.value,
        )
        devices.append({
            "name": name.value.decode("utf-8"),
            "compute_capability": f"{major.value}.{minor.value}",
            "total_memory_bytes": total_bytes.value,
        })
        if memory_round_trip and index == 0:
            context = ctypes.c_void_p()
            call(
                "cuCtxCreate_v2", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int],
                ctypes.byref(context), 0, device.value,
            )
            try:
                memory = ctypes.c_uint64()
                call(
                    "cuMemAlloc_v2", [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t],
                    ctypes.byref(memory), ctypes.sizeof(ctypes.c_uint32),
                )
                try:
                    original = ctypes.c_uint32(0x12345678)
                    restored = ctypes.c_uint32()
                    call(
                        "cuMemcpyHtoD_v2", [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t],
                        memory.value, ctypes.byref(original), ctypes.sizeof(original),
                    )
                    call(
                        "cuMemcpyDtoH_v2", [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t],
                        ctypes.byref(restored), memory.value, ctypes.sizeof(restored),
                    )
                    if restored.value != original.value:
                        raise GPUProbeError("CUDA memory round trip returned different bytes")
                finally:
                    call("cuMemFree_v2", [ctypes.c_uint64], memory.value)
            finally:
                call("cuCtxDestroy_v2", [ctypes.c_void_p], context.value)
    result = {"driver_api": "CUDA", "device_count": count.value, "devices": devices}
    if memory_round_trip:
        result["memory_round_trip_bytes"] = ctypes.sizeof(ctypes.c_uint32)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Check CUDA driver access without Torch")
    parser.add_argument("--memory-round-trip", action="store_true")
    args = parser.parse_args()
    try:
        result = probe_cuda_driver(memory_round_trip=args.memory_round_trip)
    except GPUProbeError as exc:
        raise SystemExit(f"CUDA driver probe failed: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()