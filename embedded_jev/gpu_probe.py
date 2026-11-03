"""Inspect the CUDA driver API without Torch, a toolkit, or model weights."""

import argparse
import ctypes
import json
import subprocess


class GPUProbeError(RuntimeError):
    """CUDA driver access or host resource preflight failed."""


def probe_host_cache(run=None) -> dict:
    """Read host metadata without opening cache files or selecting a volume."""
    if run is None:
        run = subprocess.run

    def query(command):
        try:
            result = run(
                command, check=True, capture_output=True, text=True, timeout=10,
            )
        except FileNotFoundError:
            return None
        except subprocess.TimeoutExpired as exc:
            raise GPUProbeError(f"host preflight command timed out: {command[0]}") from exc
        except subprocess.CalledProcessError as exc:
            raise GPUProbeError(f"host preflight command failed: {command[0]}") from exc
        if len(result.stdout.encode("utf-8")) > 32768:
            raise GPUProbeError("host preflight metadata exceeds 32 KiB")
        try:
            records = json.loads(result.stdout)
        except ValueError as exc:
            raise GPUProbeError("host preflight command returned invalid JSON") from exc
        if not isinstance(records, list) or len(records) > 128:
            raise GPUProbeError("host preflight requires at most 128 metadata records")
        return records

    volumes = query([
        "podman", "volume", "ls", "--filter", "name=embedded-jev-cache", "--format", "json",
    ])
    names = []
    for record in volumes or []:
        name = record.get("Name") if isinstance(record, dict) else None
        if (
            not isinstance(name, str) or not name.startswith("embedded-jev-cache-")
            or len(name) <= len("embedded-jev-cache-") or len(name) > 255 or name in names
            or not name.isascii()
            or any(not (character.isalnum() or character in "_.-") for character in name)
        ):
            raise GPUProbeError("host preflight received an invalid cache volume name")
        names.append(name)

    drives = query([
        "powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command",
        "ConvertTo-Json -Compress -InputObject @(Get-PSDrive -PSProvider FileSystem "
        "| Select-Object Name,Used,Free)",
    ])
    filesystems = []
    drive_names = set()
    for record in drives or []:
        if not isinstance(record, dict):
            raise GPUProbeError("host preflight received an invalid Windows drive record")
        name = record.get("Name")
        if not isinstance(name, str) or not name or len(name) > 64 or name in drive_names:
            raise GPUProbeError("host preflight received an invalid Windows drive name")
        if any(type(record.get(field)) is not int or record[field] < 0 for field in ("Used", "Free")):
            raise GPUProbeError("host preflight requires nonnegative integer drive byte counts")
        drive_names.add(name)
        filesystems.append({"name": name, "used_bytes": record["Used"], "free_bytes": record["Free"]})

    return {
        "operation": "read_only_host_cache_preflight",
        "cache_volumes": {"status": "unavailable" if volumes is None else "listed", "names": sorted(names)},
        "windows_filesystems": {
            "status": "unavailable" if drives is None else "reported", "drives": filesystems,
        },
        "cache_contents_verified": False,
        "vhdx_drive_identified": False,
        "model_execution_authorized": False,
    }


def probe_cuda_driver(loader=ctypes.CDLL, *, memory_round_trip: bool = False) -> dict:
    try:
        driver = loader("libcuda.so.1")
    except OSError as exc:
        raise GPUProbeError("CUDA driver library libcuda.so.1 is unavailable") from exc

    def resolve(name, argtypes):
        try:
            function = getattr(driver, name)
        except AttributeError as exc:
            raise GPUProbeError(f"CUDA driver symbol is missing: {name}") from exc
        function.argtypes = argtypes
        function.restype = ctypes.c_int
        return function

    def call(name, argtypes, *args):
        status = resolve(name, argtypes)(*args)
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
            resolve("cuCtxDestroy_v2", [ctypes.c_void_p])
            resolve("cuMemFree_v2", [ctypes.c_uint64])
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
    parser = argparse.ArgumentParser(description="Check CUDA driver access or read-only host cache metadata")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--memory-round-trip", action="store_true")
    modes.add_argument("--host-cache-preflight", action="store_true")
    args = parser.parse_args()
    try:
        if args.host_cache_preflight:
            result = probe_host_cache()
        else:
            result = probe_cuda_driver(memory_round_trip=args.memory_round_trip)
    except GPUProbeError as exc:
        prefix = "Host cache preflight" if args.host_cache_preflight else "CUDA driver probe"
        raise SystemExit(f"{prefix} failed: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()