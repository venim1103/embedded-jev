# Development Environment

The first environment is CPU-first so source research, mathematical tests, Python
development, and C++ kernel work do not require an NVIDIA runtime. It is a
development image, not the eventual minimal edge deployment image.

## What Is Included

- Ubuntu 24.04 devcontainer base and non-root `vscode` development user.
- Python 3.12 in `/opt/venv`; pip 25.2 and uv 0.8.22.
- Hash-locked NumPy 2.2.6, SciPy 1.15.3, pytest 8.4.2, Ruff 0.13.1, and dependencies.
- Clang/LLD 18, GCC build tools, CMake, Ninja, OpenMP, OpenBLAS, ccache, and GDB.
- Git/Git LFS, curl, jq, ShellCheck, numactl, and GNU time.
- Python and C++ VS Code extensions configured for container installation.
- Persistent workspace-specific cache mounted at `/home/vscode/.cache`.

No model downloads, runtime clones, CUDA installs, external services, privileged
mode, host Docker socket, camera access, or inbound ports are configured.
The cache contains model data only after an explicit later download.

## Open in VS Code

1. Use a Linux container engine supported by Dev Containers. Docker is the usual
   default; this initial setup was also built and started with rootless Podman.
2. Install the VS Code Dev Containers extension on the host.
3. Open this repository and run **Dev Containers: Reopen in Container**.
4. With Podman, set the host's `dev.containers.dockerPath` to `podman`.
5. Wait for the post-create smoke check to report `"status": "ok"`.

The container uses 2 GiB shared memory. Increase it only when a measured workload
requires it. The image is intended for native Linux x86-64 and ARM64 development;
only x86-64 has been validated here. RISC-V targets need a separate native or
cross-compilation workflow and are not implicitly supported by this image.

Do not reopen the editor automatically during a running agent task; this can
interrupt the session. The CLI can validate the container independently.

## CLI Workflow

Run on the host from the repository root:

```bash
devcontainer build --workspace-folder . --docker-path podman --image-name embedded-jev-research:dev
devcontainer up --workspace-folder . --docker-path podman
devcontainer exec --workspace-folder . --docker-path podman /opt/venv/bin/python .devcontainer/smoke.py
devcontainer exec --workspace-folder . --docker-path podman env PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python -m pytest -q -p no:cacheprovider docs/test_research_math.py
devcontainer exec --workspace-folder . --docker-path podman /opt/venv/bin/ruff check --no-cache .devcontainer/smoke.py docs/test_research_math.py
```

Replace `podman` with `docker`, or omit `--docker-path`, on a Docker host. These
commands do not move the current editor into the container. `up` returns a
container ID that can be stopped later with the chosen engine's `stop` command.
The stopped container and persistent cache can be reused; do not prune them
indiscriminately.

Inside VS Code after reopening:

```bash
python .devcontainer/smoke.py
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider docs/test_research_math.py
ruff check --no-cache .devcontainer/smoke.py docs/test_research_math.py
```

The smoke check runs a Cholesky solve, compiles and executes a C++17/OpenMP
program in temporary storage, verifies tool availability and writable paths,
and runs `pip check`. It does not exercise a model, GPU, or SIMD ternary kernel.
The current nine tests support the audit's deductions, not whole-model quality.

## Dependencies and Reproducibility

Direct research dependencies are in
[.devcontainer/requirements.in](../.devcontainer/requirements.in).
The image installs the generated
[.devcontainer/requirements.lock](../.devcontainer/requirements.lock)
with `--require-hashes`.

To update deliberately, edit the direct pins, then run inside the container:

```bash
uv pip compile --universal --python-version 3.12 --generate-hashes --no-annotate --output-file .devcontainer/requirements.lock .devcontainer/requirements.in
```

Review the resolved change, rebuild, and rerun the smoke and audit tests. A universal
lock includes platform wheels, but resolution alone is not proof that every target
platform installs or works.

The base image uses a versioned Ubuntu tag, not an immutable digest, and apt uses
current Ubuntu package repositories. Python research dependencies are locked;
the complete OS image is **not bit-reproducible**. Before publishing benchmarks,
record the actual image digest and system package versions or freeze a tested
image. Record native compiler flags and submodule commits separately.

Keep third-party runtimes and generated build outputs outside tracked source, for
example under `$HOME/.cache/embedded-jev`. Do not add large models, private data,
credentials, or generated checkpoints to Git. This initial phase leaves the
user's existing ignore rules unchanged.

## ML and GPU Work

Install the quantization ML stack only after host inventory. Use a dedicated
environment and lock, not ad hoc packages in the research base. The inspected
SemIf revision pins Torch 2.10.0 and Transformers 5.17.0; MiMo's configuration
records Transformers 5.12.1. These are useful compatibility anchors, not proof
that all three projects share one working environment.

Maintain separate environments for:

1. The quantization/reference stack: Torch, a validated Qwen3.5 Transformers
   implementation, datasets, safetensors, and the exact tokenizer/processor.
2. SemIf reference tooling, using its own pinned requirements.
3. Each native runtime's converter and matching gguf package/binding.

For NVIDIA quantization, first establish GPU model/VRAM, driver and runtime
compatibility on the host. A future opt-in profile should request GPU access and
install matching Torch wheels. Compiling CUDA extensions additionally requires
a matching CUDA toolkit and `nvcc`; `--gpus all` alone does not provide them.
Docker and Podman GPU passthrough differ. GPU validation is not claimed here.

Quantize on the workstation and infer on the edge. Full-model BF16 weights,
activation banks, factors, and temporary exports require more memory/storage than
the final packed model. Use the inventory to size these before downloads.

## Native Build Discipline

Clang 18 satisfies BitNet's documented compiler prerequisite. Do not assume a
successful C++ smoke test implies BitNet itself has been built. Native inference
and model-loading smoke tests are later roadmap gates.

Pin parent repositories and all submodules. BitNet's inspected top-level options
are `BITNET_ARM_TL1` and `BITNET_X86_TL2`; the PDF's generic TL flags are not the
verified interface. Kernel generation and format compatibility are also required.
Do not load an arbitrary system `libllama` or presume a pip wheel uses a fork.

Use native ISA autodetection only for a binary that will run on the build machine.
Cross-compiled builds need explicit target ISA and runtime feature checks. AVX2
does not imply AVX-512 or VNNI. ARM NEON does not by itself imply DOTPROD. RVV
version and vendor extensions must be established on the actual RISC-V target.

## Initial Verification Record

On 2026-09-25, using Dev Containers CLI 0.87.0 and rootless Podman on Linux x86-64:

| Check | Result |
| --- | --- |
| Configuration resolution | Passed |
| Image build | Passed |
| Container startup and post-create | Passed |
| Python 3.12, Clang 18, CMake/Ninja availability | Passed |
| C++17/OpenMP compilation and execution | Passed |
| Numerical libraries and dependency consistency | Passed |
| Non-root workspace/cache writes | Passed |
| Research math suite | 9 passed |
| Ruff on smoke/test code | Passed |
| GPU, ARM64, RISC-V, actual model/runtime | Not tested |