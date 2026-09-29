# Engineering Handover

Prepared: 2026-09-25. Updated: 2026-09-28. Phase: Milestone 1 feasibility checks in progress.

This document is intended to let a new developer or coding-agent session continue
after reopening the repository inside the devcontainer, without access to the
original conversation. Read this first, then follow the links for detail.

**The environment and pinned metadata inventory work. The model pipeline does not
exist yet.** Do not confuse passing mathematical/header tests with a successfully
quantized or deployed model.

## 1. User Intent

The user wants an **embedded Jev-like decision engine**, built from this chain:

`MiMo-V2.6-Distill-Qwen-9B -> ternary quantization -> BitNet-capable CPU inference -> SemIf-style decisions`

The original motivation is to use a knowledgeable, agentically fine-tuned model
on an edge computer, eventually with camera/sensor evidence. Outputs should be
typed decisions and probabilities without a generated answer sequence. Potential
targets are an Intel N100-class machine, an ARM board, or RISC-V hardware.

The user supplied a long conversation with another AI containing a proposed
implementation. The user explicitly asked us to double-check its physics,
mathematics, and research rather than trust its claims. Many critical statements
and scripts in that conversation are wrong or incomplete.

The user also requested:

- A devcontainer before starting the model implementation.
- Detailed checked-in research, design, and planning documentation.
- This detailed handover, because the next session may lose chat history.
- Local commits for the work so the user can push it themselves.

BitNet is a real requirement. Do not deliver ordinary llama.cpp inference and
call the BitNet part complete. Conversely, do not pretend that ternary MiMo is a
model trained natively with the BitNet b1.58 recipe.

## 2. Read These Documents

| Document | Why it matters |
| --- | --- |
| [README.md](../README.md) | Project entry point, current status, commands, and key corrections |
| [docs/research-audit.md](research-audit.md) | Technical audit, equations, actual formats, native API and cache risks, memory accounting |
| [docs/design.md](design.md) | Proposed artifact contracts, quantizer, genuine BitNet integration, decision semantics, caching, and vision |
| [docs/roadmap.md](roadmap.md) | Ordered implementation milestones, proposed modules, tests, and benchmark protocol |
| [docs/development.md](development.md) | Environment contents, Docker/Podman usage, dependency updates, and unverified platforms |
| [docs/sources.md](sources.md) | Primary source URLs, immutable revisions where resolved, and limits of the evidence |

These documents supersede unsupported claims from the supplied AI conversation.
They are not substitutes for reading the selected runtime implementation before
adapting its converter, kernels, graph, or bindings.

## 3. Repository State

The repository originally contained a one-line README and a license. A
user-created ignore rule and the conversation PDF were also present locally.
No existing quantizer, training code, inference service, package scaffold, or
native runtime needed to be preserved or extended.

Delivered files:

| File | Implemented role |
| --- | --- |
| [.devcontainer/devcontainer.json](../.devcontainer/devcontainer.json) | CPU devcontainer, cache volume, extensions, post-create smoke command |
| [.devcontainer/Dockerfile](../.devcontainer/Dockerfile) | Ubuntu/Python/C++ toolchain and installation of locked numerical dependencies |
| [.devcontainer/requirements.in](../.devcontainer/requirements.in) | Four direct numerical/test/lint package pins |
| [.devcontainer/requirements.lock](../.devcontainer/requirements.lock) | Generated transitive lock with package hashes |
| [.devcontainer/smoke.py](../.devcontainer/smoke.py) | Numerical solve, C++17/OpenMP compile/run, tools and permissions check |
| [docs/test_research_math.py](test_research_math.py) | Nine small executable checks of audit assumptions |
| [embedded_jev/inventory.py](../embedded_jev/inventory.py) | Bounded pinned metadata/header retrieval, deterministic inventory and memory estimator |
| [tests/test_inventory.py](../tests/test_inventory.py) | Offline fixtures for valid, incomplete, inconsistent, unsupported, and range-ignoring sources |
| [embedded_jev/label_probe.py](../embedded_jev/label_probe.py) | Pinned text-only MiMo template and A-P token-boundary check without model weights |
| [tests/test_label_probe.py](../tests/test_label_probe.py) | Offline tokenizer, processor, and fixture contract cases |
| [tests/fixtures/agent_tool_smoke.json](../tests/fixtures/agent_tool_smoke.json) | Five synthetic option-mapping cases, never calibration or benchmark data |
| [.gitignore](../.gitignore) | Existing PDF exclusion preserved; generated Python bytecode caches ignored |

The documentation above is also delivered. The existing [LICENSE](../LICENSE)
was not modified. Third-party model, data, and code licenses remain separate.

The original PDF is deliberately not part of the commits. Do not force-add it.
All essential project decisions are captured in tracked Markdown instead.
No legacy devcontainer backup was edited. If backups appear later, leave them
alone unless the user explicitly requests otherwise.

## 4. Entering the Environment

The original host workspace is `/home/vscode/AI/embedded-jev`. The observed CLI
mount inside the container is `/workspaces/embedded-jev`. Use the current workspace
root rather than assuming that host-specific absolute path in scripts.

In VS Code, use **Dev Containers: Reopen in Container**. The host needs the Dev
Containers extension and a working container engine. With Podman, configure the
host setting `dev.containers.dockerPath` as `podman`.

The bootstrap session built and started the container through the CLI without
reopening the editor. A previous container/image may therefore already exist.
Reuse or rebuild through Dev Containers; do not delete unrelated containers or
prune caches. A hard-coded historical container ID is not a durable interface.

Host-side commands, from this repository:

```bash
devcontainer read-configuration --workspace-folder . --docker-path podman --log-level info
devcontainer build --workspace-folder . --docker-path podman --image-name embedded-jev-research:dev
devcontainer up --workspace-folder . --docker-path podman
```

Use `docker` instead of `podman`, or omit `--docker-path`, on a Docker host.
These are host commands. The development image intentionally does not mount a
host container socket or provide nested container management.

Inside the container, run:

```bash
pwd
python --version
python .devcontainer/smoke.py
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider docs/test_research_math.py
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_inventory.py
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_label_probe.py
ruff check --no-cache .devcontainer/smoke.py docs/test_research_math.py
ruff check --no-cache embedded_jev tests
git status --short
```

Expected Python interpreter: `/opt/venv/bin/python`. The smoke check must print
`"status": "ok"`; the research suite must report nine passing tests. No model
downloads or credentials are needed for these checks.

From outside, use `devcontainer exec --workspace-folder . --docker-path podman`
before an inside-container command. Do not try to fix a missing host NumPy by
installing the research dependencies globally; use the container.

The opt-in [Podman/WSL NVIDIA profile](../.devcontainer/gpu/devcontainer.json)
reuses the CPU image and adds `nvidia.com/gpu=all`, `/dev/dxg`, and the WSL
driver-library mount from the user's working container. The default config
remains CPU-only. Its post-create command checks `/dev/dxg` and `nvidia-smi`;
see [docs/development.md](development.md) for host-side launch commands and
prerequisites. A WSL/Podman host build and post-create check succeeded on
2026-09-28, reporting an RTX A3000 Laptop GPU (12,288 MiB, driver 595.95)
inside the container. The base image contains no GPU Torch; a separate isolated
GPU venv passed a tiny CUDA matmul (below). Model weights and native BitNet
remain future work.
An optional [CUDA driver probe](../embedded_jev/gpu_probe.py) can now run from
the already-started GPU container without a rebuild or Torch installation;
it successfully enumerated one RTX A3000 Laptop GPU at compute capability 8.6
and 12,884,377,600 driver-reported memory bytes. Its opt-in four-byte
`--memory-round-trip` mode passed on the same WSL/Podman container, confirming
context creation, allocation, host/device copies, and cleanup. No GPU kernel
was executed by this driver-only probe.
An optional [GPU Torch compute probe](../embedded_jev/torch_probe.py) has offline
fail-closed tests and a 4x4 FP32 CUDA matmul. It passed on the RTX A3000 with
isolated `torch==2.10.0+cu128` (CUDA 12.8); the base research venv remains
unchanged. NumPy was not installed in the isolated GPU venv, producing a warning
but not affecting this pure-Torch smoke. Model loading, quantization quality,
and BitNet CPU execution remain untested.
The compatible `torch==2.10.0+cu128` CPython 3.12 Linux wheel is 916,856,347
bytes by HTTP header, plus dependencies. [docs/development.md](development.md)
has the opt-in isolated install and run commands. The first GPU-container uv
attempt failed before installation: top-level `.cache` is root-owned, so uv's
default cache directory was not writable. Use the documented writable volume
child and explicit `UV_CACHE_DIR`; do not change ownership or rebuild for this.
WSL `df` does not establish physical free space on the Windows drive containing
the virtual disk. No Torch package was added to the lightweight research base;
the separate GPU venv and cache are not a locked full-model environment.

## 5. What Was Actually Verified

The initial checks ran on Linux x86-64 with rootless Podman, Dev Containers CLI
0.87.0, and Node.js 24.19.0 on the host. This is not evidence for an ARM or RISC-V
build, nor a benchmark of any proposed edge board.

Observed versions inside the container:

| Component | Version |
| --- | --- |
| Python | 3.12.3 |
| Clang | 18.1.3 |
| CMake | 3.28.3 |
| Ninja | 1.11.1 |
| Git | 2.49.0 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| pytest | 8.4.2 |
| Ruff | 0.13.1 |
| uv | 0.8.22 |

The following succeeded:

- Devcontainer configuration resolution, image build, and startup.
- Post-create smoke check as the configured non-root development user.
- A numerical Cholesky solve using NumPy/SciPy.
- Compilation and execution of a small C++17/OpenMP program.
- Writable workspace and Hugging Face cache paths.
- `pip check`, with no broken package requirements.
- Nine mathematical/accounting checks and Ruff on the two Python files.
- Eight offline inventory tests, Ruff/editor diagnostics, and live pinned-header
  reconciliation without downloading weight payloads.
- Opt-in WSL/Podman GPU profile image build and post-create CPU smoke; inside
  the container `nvidia-smi` reported RTX A3000 Laptop GPU, 12,288 MiB VRAM,
  driver 595.95. No CUDA kernels or model inference were exercised.
- Local Markdown link and code-fence checks during documentation preparation.

The nine tests cover matching rotations, Hessian spectrum/condition-number
invariance, repeated-activation rank, inability to replace group scales with one
tensor scale, format bit rates, INT16 FWHT overflow, vocabulary matrix memory,
cache/Hessian/activation accounting, and selected-head conditional probabilities.

They do **not** cover a GPTQ implementation, full-model loading, tokenizer parity,
native codecs, BitNet kernel dispatch, camera processing, or real decision quality.
The inventory tests also do not test those behaviors.

## 6. What Is Not Installed or Built

The base contains compiler tools and a lightweight numerical Python environment.
It deliberately does not install Torch, Transformers, datasets, safetensors,
gguf, SemIf, llama-cpp-python, BitNet, Prism, or CUDA. No inference server runs,
no ports are forwarded, and no camera or actuator is exposed.

The header-only pinned inventory fetched 172,461 metadata/header response-body
bytes. Later, a separate opt-in reader fetched 2,048 BF16 weight bytes from four
rows and 256 columns of the pinned `layers.3.mlp.down_proj.weight`. No full
MiMo tensor/shard was downloaded; its full shard payload hash is not verified.
Separately, an MIT-licensed 1,187,801,280-byte BitNet control GGUF was
downloaded, SHA-256 verified, and loaded in the pinned native fork. See
[docs/development.md](development.md) for both independent tracks. No MiMo
inference runtime or whole model has been validated, and no owned MiMo quality,
perplexity, latency, energy, or RSS result exists. The host GPU
is an RTX A3000 Laptop GPU with 12,288 MiB VRAM. On 2026-09-28, WSL showed
29 GiB RAM, 21 GiB available, and 8 GiB swap; the user reported 48.3 GB free
on the Windows drive backing the virtual disk. These point-in-time readings do
not establish space or RAM for full-model conversion: 18,819,627,488 BF16
weight bytes alone occupy about 17.53 GiB of memory before caches and scratch.
Use bounded streaming/offloading and account for disk copies before downloading.

Do not run the full-model commands from the original PDF: their modules do not
exist here, and several use incompatible layouts or nonexistent APIs.

## 7. Immutable Source Anchors

Use these as the audit's starting points, not as automatically compatible
dependencies:

| Component | Inspected revision |
| --- | --- |
| MiMo HF checkpoint | `2367e865d009c13ac81713a2878291d33ab28177` |
| Microsoft BitNet parent | `0b341e582afbf9e1011f24744b554c96a3477eb5` |
| BitNet llama.cpp gitlink | `390c307752ab78fd8189f359d6954c9ba1be74af` in `isHuangXin/llama.cpp` |
| SemIf | `23cf1f39fc9534fe81437200959b6dfc7106e45a`, branch `master` |
| Prism source | Release tag `prism-b10735-842b188`; resolve full commit and binary hash before execution |

Some broader documentation was read on moving branches. The source register
distinguishes those observations from immutable pins. Pin all selected runtime
submodules, converter packages, and datasets for actual experiments.

SemIf's inspected dependency metadata pins Torch 2.10.0, Transformers 5.17.0,
and llama-cpp-python 0.3.35 for its optional native backend. MiMo's configuration
records Transformers 5.12.1. Do not assume these and BitNet's build dependencies
should be installed into one environment.

## 8. Critical Findings to Preserve

### The Requested Checkpoint Is Real, but the Proposed Footprint Is Not

The pinned model is `Qwen3_5ForConditionalGeneration`, with 32 text layers,
24 linear-attention and eight full-attention blocks, hidden width 4,096,
intermediate width 12,288, and vocabulary size 248,320. Its embeddings are untied.
HF reports 9,409,813,744 BF16 parameters. The pinned header inventory independently
counts 760 tensors, 9,409,813,744 parameters, and 18,819,627,488 tensor bytes.
It attributes 912,020,960 bytes to vision; the config declares one optional MTP
layer but no MTP tensors appear in the weight index or headers.

Each 248,320-by-4,096 BF16 vocabulary matrix is 2,034,237,440 bytes. Input embedding
plus output projection therefore requires about 4.068 GB before the transformer.
The conversation's 150 MB output head and complete under-3-GB BF16-preserved model
are not possible. Q8_0 would be about 1.081 GB per such matrix, subject to actual
converter support and quality validation.

The configured SSM dtype is FP32. The inspected Prism hybrid-cache constructor
also uses FP32 recurrent state. Preserving a few recurrent *parameters* in BF16
does not establish the dtype or size of runtime recurrent *state*.

### Ternary Values Are Not a Storage or Execution Standard

- `PQ2_0`: 128 values, 34 bytes, 2.125 bpw.
- `PTQ1_0`: 128 values, 28 bytes, 1.75 bpw; dense trits and a scale.
- `Q2_0`: 64 values, 18 bytes, 2.25 bpw.
- `TQ1_0`: 256 values, 54 bytes, 1.6875 bpw; cannot merge arbitrary two-group scales.

Those rates are for the respective blocks, not total resident model memory.
I2_S, TL1, TL2, codebook IQ formats, and Prism formats have different contracts.
Never relabel bytes as another ggml type or write them as `I8` and claim native
ternary inference.

### Genuine BitNet Integration Is Necessary Work

The inspected I2_S quantizer uses a tensor-level FP32 scale and architecture/build
dependent packed ordering. It is not the PDF's per-group FP16 layout. A MiMo
group-scaled artifact cannot pass through it unchanged without losing scales.
Its MAD kernels use integer multiply-accumulate; LUT kernels are a separate path.
BitNet does not mean that every transformer operation becomes addition-only.

An isolated [group-scale AVX2 fixture](../native/bitnet_group_scale.cpp) now
executes a pinned BitNet I2_S-style packed-code MAD block on x86-64. The 128
codes occupy 32 bytes; one separate FP32 scale per output row/group adds four
bytes (2.25 bpw for that weight representation, not PTQ1_0's 1.75 bpw).
Because BitNet packs ternary codes as 0/1/2, the kernel subtracts each group's
signed A8 activation sum before applying row/group and activation scales.
Seven golden tests compare compiled outputs against an independent scalar
reference for zero weights, negative/extreme activations, unequal scales, and
serial three-token batches at 256/4,096/12,288 input widths. The
[BitNet MIT notice](../native/BitNet-LICENSE.txt) is retained separately.
The test-only [activation preparation](../embedded_jev/activation.py) applies
the same signed normalized Hadamard transform to activations and weight inputs,
then rounds rotated activations to symmetric A8 with one FP32 scale per
token/input group (all-zero groups use scale 1). Toy integrated tests at 128-
and 1,024-point transforms compare the compiled multi-token output with an
independent quantized reference and bound its difference from rotated dense.
This does **not** execute the Microsoft fork's GGML graph, run transforms/A8
inside the runtime, process an actual model block, run fused GEMM, or prove
runtime dispatch, quality, ARM/RISC-V support, or a speedup. Those remain gates.

Separately, the pinned Microsoft BitNet parent and llama.cpp gitlink were
checked out **outside the workspace** (about 208 MB source). Clang 18 built
only the `ggml-cpu` library after setting `CCACHE_DIR` to a writable path.
The build graph includes `ggml-cpu-i2s.c`, fork `quants.c`, and BitNet's LUT
source, but not `ggml-bitnet-mad.cpp` under these flags. The compiled fork
exports `ggml_vec_dot_i2_i8_s`; optional
[native-control tests](../tests/test_bitnet_native_control.py) call this real
symbol. On AVX2, an all-zero ternary group encoded as 1 returns the signed
activation sum, so the adapter subtracts it **per group** before applying
row/group and activation scales. Golden vectors pass. The pinned fork builds
with warnings including enum-initializer overrides; these model-free tests
alone did not establish GGUF loading or model inference. A separate
[model-free GGML graph smoke](../native/bitnet_ggml_graph_smoke.cpp) did execute
its I2_S path with four packed rows, yielding [0, 128, -128, 0] for an all-ones
input. This verifies that **one tiny graph** dispatches, not that any Qwen3.5
projection or group-scaled MiMo model does.
A second mode reuses that GGML graph serially for two group-128 blocks, then
combines their partial outputs with different per-row/group scales outside the
graph. It returns [32, -320, -256, 64] on a fixed four-row fixture. This
proves group scales can survive execution through genuine fork I2_S graphs in
a toy serial bridge; it is not a native packed group-scale type, fused GEMM,
MiMo model loader, or measured speedup.
The same graph also passed two independent token columns with +1.0 and -2.0
F32 inputs, verifying different per-token dynamic A8 scales in this fixture.
No hybrid sequence state or real model activations were exercised.

Later, the separately pinned official
[BitNet control GGUF](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T-gguf/tree/a1f2f1c765812aa8af3f6eda4a313707064bba15)
(1,187,801,280 bytes; SHA-256
`4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`)
loaded via the pinned fork's `llama` API. Its 22-token CPU prompt returned
128,256 finite final-position logits with **zero sampled/generated tokens**.
A batch debugger stopped at `llamafile_sgemm_i2s` from
`ggml_compute_forward_mul_mat`, establishing actual I2_S execution for that
control model. One load and prefill took about 410/439 ms on this machine;
these are not latency benchmarks. The loader also reported CPU_REPACK buffer
fallback, which needs profiling before any optimized-path or speed claim.
The control prompt's A-C labels each extend the exact prefix by one distinct,
non-special token (IDs 32/33/34). Direct final-logit softmax selected A with
conditional probability about 0.553, but the allowed labels carried only
about 0.000070 of the full-vocabulary mass and calibration status is unknown.
This is a structural **uncalibrated** SemIf-style readout on a different model,
not MiMo tokenizer parity, MiMo graph dispatch, group-scale preservation in
MiMo, SemIf backend integration, or decision correctness.

Use a native BitNet checkpoint as a tooling control. The preferred proposed MiMo
direction is to integrate a real BitNet-derived kernel into a Qwen3.5-capable
runtime, preserving group scales, rotations, activations, and hybrid operations.
This is a hypothesis awaiting a small integration spike. Extending BitNet's own
fork is an alternative to compare, not an already ruled-out possibility.

The inspected top-level build options are `BITNET_ARM_TL1` and
`BITNET_X86_TL2`. Kernel generation, model shapes, layout, and dispatch must still
match. RVV support for these paths is not established by generic ggml RVV flags.

### Rotations Must Preserve the Function

For row-vector activations and weights with input features on the last axis:

$$
XW^T=(XR)(WR)^T,\quad RR^T=I.
$$

Both operands need the same complete transform. Explicit signs can make a
Hadamard-based rotation nonsymmetric. Do not interchange inverse and forward
order. Preserve bias and do not move rotations through gates/nonlinearities
without proving equivalence.

Curvature is collected in that same basis:

$$
G=2X^TX/N,\quad G_R=R^TGR.
$$

Orthogonal similarity preserves eigenvalues, rank, and spectral condition number.
It changes coordinate incoherence, not conditioning by 10-100x as claimed in the
PDF. Damping changes eigenvalues. More tokens do not guarantee full rank, and
duplicating activation samples does not add new covariance information.

### The Provided GPTQ Loop Is Not Correct

The PDF computes a scale across output rows per column, changes it while walking
the group, and saves only the last scale. That is not one retained scale per
output-row/input-group. Its fallback from inverse-Cholesky to `pinv` changes the
meaning of the update. Its replay drops actual decoder arguments and rotates
weights without the matching activations.

Start from a reviewed implementation and small reconstruction tests. Store codes
and exact representable scales explicitly; never infer them again from BF16
dequantized tensors. Keep processing-block size separate from scale-group size.
Act-order permutations require matching runtime/group mapping and are initially
best left disabled.

The toy [ternary RTN baseline](../embedded_jev/ternary.py) now saves one FP16
max-abs scale and -1/0/+1 codes per output row and contiguous group. Code
assignment uses the **stored representable scale**, handles zero groups, and
rejects nonfinite, overflowing, underflowing, or tail groups. Reconstruction
uses only those saved arrays. Toy golden tests also cast the saved scales to
FP32 for the isolated BitNet-derived AVX2 kernel and compare its output with
the saved-artifact reconstruction at 256 and 4,096 input values. This is not
GPTQ, an exporter/codec, a model-weight quantization result, or native graph
integration; quality and format gates remain open.

The same module also has a separate **256-column-bounded GPTQ-style toy**:
token-normalized curvature, positive bounded damping retries, inverse-Cholesky
error updates, and FP16 row/group scales fixed at group entry. Processing blocks
are separate from scale groups and must be group-aligned; tested smaller blocks
reproduce the full-width toy's saved codes and scales. A correlated
four-input toy improves its local reconstruction error relative to RTN, while
diagonal-curvature, zero-group, factorization-failure, and native saved-artifact
tests pass. It follows the algorithmic update in
[pinned IST-DASLab GPTQ](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py)
under this repository's Apache-2.0 license. No pseudoinverse or act-order
fallback is used. It is not reviewed full-model GPTQ, a proof of model quality,
or ready for multi-thousand-width factors or the MiMo hybrid graph.

The [toy artifact container](../embedded_jev/ternary_artifact.py) uses bounded
uncompressed ZIP entries containing non-pickle NumPy codes/scales and a versioned
JSON manifest with hashes. Identity and explicit signed-Hadamard transforms are
validated, including sign width, block size, and sign-before-Hadamard order.
Tests reject changed hashes, object arrays, malformed transform metadata, and
oversized data; loaded artifacts feed the native AVX2 kernel in toy cases.
The archive is capped at 1 MiB and marked as toy/no-model-weights, not PTQ1_0,
PQ2_0, GGUF, or a complete source/calibration/runtime provenance artifact.
A separate toy Prism v1 metadata check requires full logical input widths,
rejects unsupported foldable names, and refuses conflicting block sizes or
sign vectors for the same input width. A conservative candidate mapping now
derives widths from the pinned MiMo inventory for exact HF projection paths;
layer-3 FFN down is 12,288 inputs and FFN gate/attention Q/K are 4,096.
It rejects assigning the 256-wide slice to the full FFN down projection.
The pinned Prism Qwen3.5 name map returns no direct match for MiMo's exact
`model.language_model.layers.*` names; selected FFN/attention names map only
after explicit prefix normalization. A model-free opt-in test verifies this,
but no converter run or loader parity authorizes GGUF export. The inspected
Prism release tag is fully pinned at
`842b1880415d6f508f03b789e5ce70194def7bfd`; no converter or Prism model
loader was exercised by this check.
An opt-in native test also processes the bounded 2,048-byte real MiMo projection
slice through toy signed rotation, searched FP16 code/scale storage, reload,
synthetic dynamic A8, and the standalone AVX2 BitNet-derived batch kernel.
The original and rotated dense outputs agree; native output matches the saved
artifact reference. This is only four weight rows with synthetic activations,
not a full attention/recurrent block, actual graph execution, or quality proof.

### Prism Has a Concrete Transform Schema

The inspected `prism.hadamard.*` metadata specifies version, one block size,
transform identity, input axis, tensor names, and identity/explicit signs.
Explicit sign vectors are keyed by input width, not independently by tensor.
Optional `gdn_v_grouped` changes the value-head ordering at `ssm_out`.

The generic matmul helper applies optional permutation, signs, Hadamard, matmul,
and scale in that order. Inverse embedding lookup uses Hadamard then signs.
Version 2 is tied-output mode; our MiMo checkpoint is untied. Begin with compatible
version-1 semantics and unrotated embeddings/head.

A richer internal manifest must not silently export incompatible per-tensor
block sizes or signs into this schema. The pinned Prism fork at
`842b1880415d6f508f03b789e5ce70194def7bfd` built CPU `llama`/GGML libraries.
An opt-in [native FWHT test](../tests/test_prism_native_control.py) executed a
two-token signed 128-point Hadamard graph with zero matrix data and agreed with
independent dense math (max error about 7.2e-7). A separate toy now schedules
the BitNet-derived AVX2 grouped dot as Prism's `MAP_CUSTOM2` node after FWHT
and dynamic A8 (max transform/output errors about 3.8e-7/9.5e-7).
Two consecutive graph evaluations also passed when both input and sign leaves
were restored before each run; the toy graph allocator may overwrite those
buffers during compute. This is a fixture-owned callback, **not** a loadable
group-scale GGUF type, Qwen3.5 model hook, safe sequence-state implementation,
codec test, other CPU/ARM backend, or speedup.

### Integer Arithmetic Needs Bounds

A 1,024-point unnormalized FWHT of INT8 inputs can reach 130,048, exceeding INT16.
Use FP32 rotation first, then dynamic A8 and INT32 accumulation. Integer FWHT is
a later measured optimization requiring proven intermediate scaling or width.
Quantizing before rotation changes the error relative to quantizing after it.

Group-scaled ternary execution must retain
$y_r=\sum_g s_{r,g}a_g\sum_{i\in g}t_{r,i}z_i$.
Row-dependent group scales cannot generally be folded into a single shared input
scale. Measure preprocessing, lookup construction, unpacking, and rescaling.

### Jev Is an Interface Inspiration, Not a Recovered Architecture

TypeSafe describes its own architecture, parallel sampler, and RLCD training.
SemIf explicitly reproduces an open interface pattern with frozen models, not
that undisclosed stack. Direct logits do not provide generated chain-of-thought
reasoning for free and do not eliminate semantic errors.

Public Jev exposes Choice, Score, and Noul. Choice selects a categorical option;
Score is an expectation over ordered descriptive levels; Noul is probability of
yes, not a degree measurement. Confidence is a separate distribution statistic.
Our initial scope is SemIf's 2-16 fixed-label choices, with later typed adapters.
Use `max_option_probability` and explicit calibration status rather than pretend
to duplicate TypeSafe's confidence computation.

### The Existing SemIf Backend Is Useful

SemIf already includes a llama.cpp CPU backend. It uses a pinned HF tokenizer,
verifies GGUF tokenization, marks the final batch position for logits, and reads
`llama_get_logits_ith`. It snapshots/restores the entire hybrid sequence state.
Its CPU shared path iterates restored suffixes serially; it does not evaluate
arbitrarily many questions in constant time.

Map semantic option IDs/descriptions to context-verified one-token labels such
as A-P. Never take the last token of an arbitrary multi-token option string.
Preserve exact prompt prefix IDs and isolate KV, recurrent, convolution, and
position state. Fit decision calibration separately after quantization is frozen.

### There Is a Source-Disclosed Recurrent Cache Risk

The pinned Prism `build_rs_cache_view` comments describe possible overwrite of a
main state row during extra-row relocation in multi-sequence execution. This was
not reproduced or fixed here. Start with `n_seq_max=1`; concurrency requires a
tested fix or verified conservative path. `GGML_GDN_STATE_GATHER=1` selects the
legacy gather path in the inspected graph, but still needs parity/performance tests.

### The Native API Must Match the Build

BitNet's pinned llama header contains real `llama_*` lifecycle, batch, logits,
tokenizer, and state APIs, not the PDF's `bitnet_eval_prompt` shim. Calling
`llama_decode` on prompt tokens is appropriate for prefill even if no token is
sampled. Names are not execution semantics.

An arbitrary llama-cpp-python wheel does not automatically link to a separately
built fork. Use a verified compatible binding or compile a small native adapter
against the exact headers. A versioned JSONL executable is the first proposed
boundary when fork ABIs differ. Copy transient logits, check errors, and release
contexts/models explicitly. Do not load incompatible native forks into one process.

## 9. Useful Optimization, Not Yet Implemented

For a linear output head, retain only rows corresponding to A-P and compute their
logits from the exact final normalized hidden state. Sixteen BF16 rows of width
4,096 occupy 128 KiB. Conditional softmax over those logits is algebraically equal
to conditioning the full vocabulary distribution on the same labels.

This could remove most output-head storage/work in a decision-only product.
It does not remove input embeddings. It requires deliberate loader/graph support;
simply omitting the output tensor can activate tied-embedding fallback and change
the model. It also removes full-vocabulary perplexity, allowed-label-mass
diagnostics, and general generation from that artifact. Keep a full-head reference
and compare actual native outputs before adopting the optimization.

## 10. Calibration and Measurement Discipline

There are three distinct calibration activities: PTQ activation statistics,
A8 clipping/scales, and decision probability calibration. Use separate manifests
and disjoint final evaluation data. Split by conversation/repository/session
before chunking. Do not train or calibrate on a benchmark's held-out answers and
later report that benchmark as untouched.

The MiMo template renders real `reasoning_content` and structured `tool_calls`
itself. Use it directly. Do not invent reasoning traces, substitute Qwen2.5's
tokenizer, silently drop tool schemas, or repeat two records until the token
counter looks large. For deployment, include representative short non-thinking
decision prompts; for vision, include actual image-derived activations.

No damping value, data mixture, target perplexity gap, or temperature in the
supplied conversation is a proven universal optimum. Local MSE is a screening
metric. Evaluate whole-model language and decision behavior on held-out data.

Report cold load, warm prefill, suffix processing, complete-state copying, vision
encoding, score readout, p50/p95 latency, peak RSS, and joules/decision separately.
Zero generated answer tokens does not mean zero prompt compute or data movement.
Never infer prefill latency from a decode memory-bandwidth bound. Keep original
7B/3B T-MAC and BitNet results attributed to their models and hardware.

Physical safety constraints belong outside the model. Initial sensor integration
is advisory/log-only, with freshness checks, watchdogs, bounded queues, abstention,
and deterministic limits. Typed output does not prove safe actuator behavior.

## 11. Milestone 1: Inventory Delivered, Model Work Pending

Run `PYTHONDONTWRITEBYTECODE=1 python -m embedded_jev.inventory` to reproduce
the full sorted JSON inventory and estimates. A live run reconciles the pinned
index to all four safetensors headers: 760 tensors, 9,409,813,744 BF16 parameters,
18,819,627,488 tensor bytes, and 93,360 header-overhead bytes. Retrieved metadata
and headers (not payloads) total 172,461 response-body bytes; metadata and header
SHA-256 hashes, revision, Python/tool versions, and per-tensor policy reasons
appear in the report. Eight offline tests cover incomplete/mismatched inputs,
unsupported metadata, and refusal of full-body shard responses.

The policy initially allows only geometrically compatible FFN and full-attention
projections: 5,301,600,256 candidate parameters. It retains embeddings/head,
vision, sensitive/recurrent weights, and fused/linear-attention projections.
Estimated on-disk weights (not BitNet I2_S or resident allocations) are
9,376,152,032 bytes for PTQ1_0 group-128 with BF16 vocabulary, or
7,469,054,432 bytes if both vocabulary matrices are **hypothetically** Q8_0.
A selected-16 head estimate requires a custom loader; it is not implemented.
The report keeps FP16 KV and FP32 recurrent-state examples separate from weight
payload. It cannot detect identical payloads stored in different shards.

The inventory command covers these metadata-only requirements:

1. Read the model config, tokenizer/processor metadata, weight index, and bounded
   safetensors headers for the pinned revision. Use an established HF/safetensors
   metadata facility or HTTP range requests with explicit byte limits. If the
   server ignores a range, stop rather than downloading an entire shard.
2. Produce exact tensor names, shapes, storage dtypes, parameter counts, byte
   counts, shard locations, and categories: language projections, sensitive
   parameters, input embedding, output head, vision, and optional MTP.
3. Detect tied/shared data and duplicate accounting. Reconcile sums against the
   index/metadata and explain any mismatch instead of hiding it.
4. Produce an explicit initial quantization allowlist and exclusion reasons.
   Check actual dimensions for candidate group/rotation sizes.
5. Estimate separate byte totals for retained BF16/FP32, Q8 embeddings/head,
   group-128 ternary blocks, vision, and an optional selected-label head.
   Keep runtime cache/scratch estimates separate from stored weights.
6. Record source revision, retrieved-file hashes, tool versions, assumptions,
   and transfer bytes. Do not deserialize arbitrary pickle or enable remote
   model code by default.
7. Add offline fixtures for valid, missing, inconsistent, and unsupported
   metadata. Unit tests should need neither network nor GPU.

Acceptance: deterministic reconciled inventory, explicit unsupported cases, no
bulk weight download, and a credible memory budget. The current HF API total is
not a substitute for this implementation.

Next, verify real processor inputs and native tokenizer parity, freeze an owned
decision fixture, and establish the dense reference. Ask about host memory and
download budgets before fetching model payloads. Then take one actual block
through rotated dense equivalence and ternary reconstruction, alongside a small
BitNet group-scale kernel proof. Do not create the entire proposed module tree as
empty scaffolding or copy the nine numbered scripts from the PDF.

## 12. Questions Before Large Jobs

Ask the user for these once they become necessary:

- Which machine performs quantization? CPU/architecture, RAM, GPU model/VRAM,
  host driver, storage location/free space, and whether the GPU is accessible
  from containers.
- Which device is the first deployment target? N100, a specific ARM board, or a
  specific RISC-V SoC; RAM, OS, cooling, and power budget matter.
- Is the first deployment text/sensor-only or does it need raw camera frames?
  What input token/image resolution and criteria-per-state budgets are realistic?
- What decision accuracy, false-positive/negative cost, abstention rate, p95
  latency, and memory ceilings are acceptable?
- Are selective higher precision, QAT recovery, and a decision-only output head
  acceptable, or must arbitrary text generation remain available?
- What downloads and long-running compute are authorized? Avoid assuming the
  current lightweight container is permission to fetch tens of gigabytes.

No hardware purchase was recommended or priced during this phase. Old pricing
tables and expected tokens/s from the PDF are not verified evidence. For Finland,
obtain dated delivered quotes with taxes, shipping, and required accessories
only after a workload/target is chosen.

## 13. Environment Pitfalls Already Encountered

- Two WSL/Podman GPU `devcontainer up` attempts failed at `apt-get update`, first
  with a `noble-backports/multiverse` Hash Sum mismatch and then with one in
  `noble-security/restricted`. The backports-only workaround was disproven and
  replaced by HTTPS for the existing Ubuntu sources, preserving all pockets
  and APT integrity checks. Both failing by-hash URLs returned the expected
  SHA-256 over HTTPS from the CPU container. A later WSL/Podman `devcontainer up`
  built successfully and passed post-create `nvidia-smi` with driver 595.95.
  Do not bypass TLS or checksum verification if a future mirror issue recurs.
- Dev Containers CLI 0.87.0 accepts `info`, `debug`, and `trace` log levels,
  not `error`. The initial unsupported flag was corrected.
- Direct rootless Podman bind mounts used for one-off tooling needed
  `--userns=keep-id` and the host UID/GID to preserve workspace permissions.
  Do not compensate by recursively changing the repository owner.
- The scratch uv image lacked Python/libc discovery tools and a useful writable
  environment for this resolution step. The successful bootstrap resolver image
  was `ghcr.io/astral-sh/uv:0.8.22-python3.12-bookworm-slim`. Future lock updates
  can simply use the uv already installed in the devcontainer.
- Podman printed harmless unconsumed BuildKit-argument/OCI-shell warnings while
  the image build and startup still succeeded. Judge the exit status and smoke
  results, not those warnings alone.
- The SemIf repository's branch is `master`; fetching its README from `main`
  returned 404. The repository itself is available.
- Verify research paper titles, not remembered arXiv numbers. The checked
  QuaRot identifier is `2404.00456` and QuIP# is `2402.04396`.
- The user prefers avoiding `rg` commands unless explicitly requested.

The Python research lock is hash-checked, but Ubuntu apt packages and the tagged
base image can change. This is not a bit-reproducible OS build. Record image
digests and full build/package provenance before publishing performance results.

## 14. Git and Delivery

The bootstrap began on `main` at `3e846bb` (`Initial commit`). The user requested
local commits, not a push or a new branch. The intended organization is:

1. `chore: add validated CPU research devcontainer`, including the unchanged
   user-created PDF ignore rule.
2. `docs: add research audit, design, and engineering handover`, including the
   executable mathematical audit and README.

Use `git log` to discover the resulting commit IDs; the handover cannot embed its
own containing commit hash. No push is performed by the bootstrap task. Before
the user pushes, verify:

```bash
git status --short
git log -2 --oneline
git diff --check
```

The clean-worktree expectation described the earlier bootstrap commits. This
Milestone 1 continuation leaves its code/documentation edits uncommitted; it
did not push or change branches. The ignored conversation PDF stays local.
Container images, running containers, and cache volumes are not stored by Git
and must be recreated or reused through the documented environment workflow.

## 15. Pinned Text-Only Label Boundary

The [label probe](../embedded_jev/label_probe.py) checks the actual MiMo template
with `enable_thinking=False`, matches rendered and tokenized prompt IDs, and
requires each A-P label to add one distinct non-special token. The sample has
46 prompt tokens and label IDs A-P = 32-47. The pinned inventory reconciles
standalone and nested image/video processor metadata. With approved isolated
CPU-only Torch 2.10.0, Torchvision 0.25.0, and Pillow 12.1.1, real
`Qwen3VLProcessor` text inputs matched the tokenizer IDs and mask, with all-zero
multimodal token types. The external cache totals about 979 MB; no model weights
have been downloaded. [docs/development.md](development.md) has the commands.

The [synthetic fixture](../tests/fixtures/agent_tool_smoke.json) has five
agent/tool cases: inspect-first with an irrelevant-context perturbation,
authorized README edits with reordered options, and missing upload permission.
Their expected semantic IDs map to A/A/B/C/C; the pinned text-only processor
passed all five prompt/label boundary checks. Six offline label/processor tests
pass. The CLI records hashes and versions, not model predictions. This is **not**
calibration data, a held-out quality benchmark, image processing, native backend
tokenization parity, or a trained decision service.

## 16. Restart Brief

For a fresh coding session, the entire immediate objective is:

> Read this handover and the linked research/design documents. Confirm the CPU
> smoke check, nine numerical tests, eight offline inventory tests, and six
> label/processor tests. Re-run bounded inventory and synthetic fixture checks
> when appropriate; then obtain a real held-out agent/tool decision fixture,
> verify native tokenizer parity, and clarify host budgets before model weights.
> Preserve the genuine BitNet execution requirement and SemIf fixed-label
> contract. Treat unmeasured quality, hardware performance, and proprietary model
> claims as unknown. Make the smallest testable implementation step.