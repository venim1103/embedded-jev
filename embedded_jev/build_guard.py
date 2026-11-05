"""Bounded observation for separately authorized external native builds."""

import argparse
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import select
import signal
import sys
import time


@dataclass
class BuildObservation:
    host_swap_baseline_bytes: int
    minimum_host_available_bytes: int
    observed_host_swap_bytes: int = field(init=False)
    peak_group_rss_bytes: int = 0
    peak_group_high_water_bytes: int = 0
    peak_group_swap_bytes: int = 0
    observed_processes: int = 0
    live_processes: int = 0
    failure: str | None = None

    def __post_init__(self):
        self.observed_host_swap_bytes = self.host_swap_baseline_bytes

    def sample(self, *, rss, high_water, swap, count, host_available, host_swap):
        self.peak_group_rss_bytes = max(self.peak_group_rss_bytes, rss)
        self.peak_group_high_water_bytes = max(self.peak_group_high_water_bytes, high_water)
        self.peak_group_swap_bytes = max(self.peak_group_swap_bytes, swap)
        self.observed_processes = max(self.observed_processes, count)
        self.live_processes = count
        self.minimum_host_available_bytes = min(self.minimum_host_available_bytes, host_available)
        self.observed_host_swap_bytes = max(self.observed_host_swap_bytes, host_swap)
        if max(rss, high_water) > 1024**3 or swap or host_swap > self.host_swap_baseline_bytes:
            self.failure = "build RSS or swap budget exceeded"
        elif host_available < 2 * 1024**3:
            self.failure = "build host available-memory reserve exceeded"
        elif count > 32:
            self.failure = "build process count exceeds bounds"
        if self.failure:
            raise ValueError(self.failure)

    def verify_finished(self):
        if self.failure:
            raise ValueError(self.failure)
        if not self.observed_processes:
            raise ValueError("build process was never observed")
        if self.live_processes:
            raise ValueError("build processes still running")


@dataclass(frozen=True)
class ProcessSample:
    pid: int
    parent: int
    start_ticks: int
    arguments: tuple[str, ...]
    directory: Path
    rss: int
    high_water: int
    swap: int
    arguments_complete: bool = True

    @property
    def identity(self):
        return self.pid, self.start_ticks


def _read_process(pid):
    root = Path("/proc") / str(pid)
    try:
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        with (root / "cmdline").open("rb") as opened:
            encoded = opened.read(8193)
        arguments = tuple(os.fsdecode(value) for value in encoded[:8192].split(b"\0") if value)
        status = dict(line.split(":", 1) for line in (root / "status").read_text().splitlines() if ":" in line)
        memory = [int(status.get(name, "0 kB").split()[0]) * 1024 for name in ("VmRSS", "VmHWM", "VmSwap")]
        return ProcessSample(pid, int(fields[1]), int(fields[19]), arguments, (root / "cwd").resolve(strict=True),
            *memory, arguments_complete=len(encoded) <= 8192)
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def _process_snapshot():
    entries = [entry for entry in Path("/proc").iterdir() if entry.name.isdecimal()]
    if len(entries) > 4096:
        raise ValueError("build process inventory exceeds bounds")
    return {sample.pid: sample for entry in entries if (sample := _read_process(int(entry.name))) is not None}


def _matches_build(process, directory, target):
    arguments = process.arguments
    if not arguments:
        return False
    executable = Path(arguments[0]).name
    if executable == "cmake":
        if "--build" not in arguments or "--target" not in arguments:
            return False
        build_index, target_index = arguments.index("--build") + 1, arguments.index("--target") + 1
        if build_index >= len(arguments) or target_index >= len(arguments):
            return False
        if Path(arguments[build_index]).resolve() != directory or arguments[target_index] != target:
            return False
        if arguments.count("--target") != 1 or arguments.count("--build") != 1 or arguments.count("--config") > 1:
            raise ValueError("duplicate build selection flags are not authorized")
        if target_index + 1 < len(arguments) and not arguments[target_index + 1].startswith("-"):
            raise ValueError("multiple build targets are not authorized")
        if "--config" in arguments:
            index = arguments.index("--config") + 1
            if index >= len(arguments) or arguments[index] != "Release":
                raise ValueError("build variant is not Release")
    elif executable == "ninja":
        if process.directory != directory or target not in arguments:
            return False
        if any(argument in ("all", "clean", "install") for argument in arguments[1:]):
            raise ValueError("additional build targets are not authorized")
        positional = []
        index = 1
        while index < len(arguments):
            argument = arguments[index]
            if argument == "-j":
                index += 2
                continue
            if not argument.startswith("-"):
                positional.append(argument)
            index += 1
        if positional != [target]:
            raise ValueError("additional build targets are not authorized")
    else:
        return False
    jobs = []
    for index, argument in enumerate(arguments[1:], 1):
        if argument in ("-j", "--parallel"):
            jobs.append(arguments[index + 1] if index + 1 < len(arguments) else "")
        elif argument.startswith("-j"):
            jobs.append(argument[2:])
        elif argument.startswith("--parallel="):
            jobs.append(argument.partition("=")[2])
    if len(jobs) != 1 or not jobs[0].isdecimal() or not 1 <= int(jobs[0]) <= 2:
        raise ValueError("build requires an explicit one- or two-job limit")
    return True


def _build_members(snapshot, identities, directory, target, *, discover_roots=True):
    members = {pid for pid, process in snapshot.items()
        if process.identity in identities or (discover_roots and _matches_build(process, directory, target))}
    for _ in range(32):
        expanded = members | {pid for pid, process in snapshot.items() if process.parent in members}
        if expanded == members:
            break
        members = expanded
    if len(members) > 32:
        raise ValueError("build process count exceeds bounds")
    return [snapshot[pid] for pid in sorted(members)]


def _host_memory():
    values = {name: int(value.split()[0]) * 1024 for name, value in
        (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}
    return values["MemAvailable"], values["SwapTotal"] - values["SwapFree"]


def _terminate_build(identities, deadline):
    while time.monotonic() < deadline:
        members = _build_members(_process_snapshot(), identities, None, None, discover_roots=False)
        identities.update(sample.identity for sample in members)
        if not members:
            return True
        parents = {sample.parent for sample in members}
        for sample in members:
            if sample.pid not in parents:
                current = _read_process(sample.pid)
                if current is not None and current.identity == sample.identity:
                    try:
                        os.kill(sample.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        select.select([], [], [], min(0.01, max(0, deadline - time.monotonic())))
    return False


def watch_build(directory, target, sources, *, maximum_seconds=60, emit):
    if type(maximum_seconds) not in (int, float) or not 0 < maximum_seconds <= 60:
        raise ValueError("invalid build deadline")
    directory = Path(directory).resolve(strict=True)
    sources = frozenset(Path(source).resolve(strict=True) for source in sources)
    if not target or len(target) > 128 or not sources or len(sources) > 4:
        raise ValueError("invalid build target or source allowlist")
    with (directory / "CMakeCache.txt").open("rb") as opened:
        cache = opened.read(256 * 1024 + 1)
    if len(cache) > 256 * 1024 or b"CMAKE_BUILD_TYPE:STRING=Release\n" not in cache:
        raise ValueError("build requires a bounded existing Release cache")
    available, swap = _host_memory()
    state = BuildObservation(swap, available)
    state.sample(rss=0, high_water=0, swap=0, count=0, host_available=available, host_swap=swap)
    if _build_members(_process_snapshot(), set(), directory, target):
        raise ValueError("build already running before watcher readiness")
    identities = set()
    observed_announced = False
    started = time.monotonic()
    cleanup_seconds = min(2, maximum_seconds / 4)
    deadline = started + maximum_seconds - cleanup_seconds
    emit({"event": "ready", "build_directory": str(directory), "target": target,
        "host_swap_baseline_bytes": swap})
    try:
        while True:
            if time.monotonic() >= deadline:
                raise ValueError("build deadline exceeded" if identities else "build process was never observed before deadline")
            with (directory / "CMakeCache.txt").open("rb") as opened:
                if opened.read(256 * 1024 + 1) != cache:
                    raise ValueError("build cache changed during observation")
            members = _build_members(_process_snapshot(), identities, directory, target)
            identities.update(sample.identity for sample in members)
            available, swap = _host_memory()
            state.sample(rss=sum(sample.rss for sample in members), high_water=sum(sample.high_water for sample in members),
                swap=sum(sample.swap for sample in members), count=len(members), host_available=available, host_swap=swap)
            for sample in members:
                if not sample.arguments_complete:
                    raise ValueError("build command line exceeds bounds")
                for argument in sample.arguments[1:]:
                    if not argument.startswith("-") and Path(argument).suffix in (".c", ".cc", ".cpp", ".cxx"):
                        if (sample.directory / argument).resolve() not in sources:
                            raise ValueError("unexpected dependency compilation")
            if identities and state.observed_processes and not members:
                state.verify_finished()
                receipt = {"event": "finished", **asdict(state), "elapsed_seconds": time.monotonic() - started,
                    "resource_observation_only": True, "build_success_verified": False}
                emit(receipt)
                return receipt
            if members and not observed_announced:
                emit({"event": "observed", "processes": len(members)})
                observed_announced = True
            select.select([], [], [], min(0.01, max(0, deadline - time.monotonic())))
    except (ValueError, OSError) as error:
        cleanup = _terminate_build(identities, min(started + maximum_seconds, time.monotonic() + cleanup_seconds))
        emit({"event": "failed", **asdict(state), "reason": str(error), "tracked_processes_stopped": cleanup,
            "elapsed_seconds": time.monotonic() - started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-directory", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--maximum-seconds", type=float, default=60)
    arguments = parser.parse_args()
    try:
        watch_build(arguments.build_directory, arguments.target, arguments.source,
            maximum_seconds=arguments.maximum_seconds, emit=lambda event: print(json.dumps(event, sort_keys=True), flush=True))
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
