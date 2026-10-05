# Engineering Handover

Prepared: 2026-09-25. Updated: 2026-10-05. Phase: bounded text/native integration; full-model deployment remains open.

This document is intended to let a new developer or coding-agent session continue
after reopening the repository inside the devcontainer, without access to the
original conversation. Read this first, then follow the links for detail.

**The 32-layer BF16 text reference and one-projection BitNet-derived native
substitution work. A complete model-loadable ternary/BitNet runtime does not.**
The current checkpoint below supersedes the bootstrap-era status statements.
Later sections retain the scope and results of individual historical probes;
do not read a probe's limitations as the current status of every later path.

## Current Checkpoint (2026-10-05)

The last tested implementation commit is `27cfd63` (2026-10-05),
`fix: reject invalid native control arguments`, following BitNet model prefill at `50a943b`.
This continuation started clean at `445c8d7`, with `main` and `origin/main`
matching after the user's push. Local commits `d95ffa4` (repeated graph),
`e2378e4` (owned handles), `4efa59b` (streamed backend), and `cb4fe74` followed,
with documentation checkpoints at `79608a8` and `03b183d`. The latest
continuation began clean at the user-pushed `03b183d`; `b1f306a` (versioned CPU
buffer) and `9a1fbfa` (explicit loader route) followed. The assistant made no push
or branch change. The resource-plan continuation started clean at `c390c0d`,
adding `ecfcff5` (header-only native budgets) and `3c4fa81` (full runtime build
and no-weight tokenizer preflight). Recheck Git state in each session.
After reboot, the caches and packaged runtime remained intact. This continuation
started at the user-pushed `1a7b08a`, preserving the untracked user `.vscode/`
directory. `64c5b1f` added synthetic native prefill and cached converter-index
validation; `50a943b` added nonzero BitNet FFN execution inside that model.
`27cfd63` rejects unknown, missing, and extra control arguments before discovery,
instead of silently returning the unrelated default control's successful report.

Current gates: **159 default tests passed, 26 optional tests skipped; all nine
pinned Prism controls passed** with full-size PQ2, reused weight/graph, owned
handle, two-forward module, tagged toy GGUF import, and isolated versioned CPU
discovery, mixed concurrent graphs, and actual pinned loader selection/upload
enabled, including the sole full-size frozen projection, packaged dependency
provenance, guarded vocabulary-only native tokenizer parity, and synthetic
dense/BitNet hybrid model prefill (35.30 s).
The isolated full llama build also passes two CTests.
The 32-layer direct/callback/registered, reordered synthetic typed-option, and
signed-Hadamard full gate passed at `4efa59b` (138.36 s); these unchanged model
controls were not rerun after the later native file/runtime additions. Ruff,
editor diagnostics, and `git diff --check` passed. GCC ASan/UBSan and leak checks
passed for the bridge runtime controls; cached GGML is not instrumented. See the
development guide for flags.

- The complete pinned BF16 snapshot is verified and cached outside Git. The
  streamed text-only reference executes all 32 decoder layers and scores
  context-verified selected head rows with **zero generated answer tokens**.
- Exactly one hash-checked searched-FP16 layer-3 FFN-down fixture is retained:
  4,096 x 12,288, group size 128, about 13 MB. Saved loading bypasses that BF16
  projection. Direct BitNet-derived AVX2, actual Prism `MAP_CUSTOM2`, and
  native-A8 callback paths reproduce the same final hidden hash and scores.
  The new `prism_ggml_registered` backend also reproduces them through a native
  weight tensor/`MUL_MAT` trait, with exactly one counted grouped kernel call,
  one upload/repack, native A8, and explicit handle release after the layer.
- Matching signed-Hadamard weight/input rotation (seed 773) passes dense and
  native-reference controls. It is a reversible in-memory experiment, refuses
  the saved identity artifact, and has no saved candidate or quality promotion.
- Separate PQ2_0 conversion preserves every frozen code/FP16 scale exactly;
  the actual native decoder and PQ2 tensor/`MUL_MAT` controlled outputs pass.
  This is **Prism PQ2 dispatch, not BitNet dispatch or GGUF loader proof**.
  Native PQ2 uses Q8_K when input width is divisible by 256, otherwise Q8_0
  with FP16 activation-scale rounding; our callback uses group-128 A8 instead.
- A separate scoped `JEV_BITNET_GROUP128` CPU buffer/tensor trait now registers
  a PQ2-storage weight tensor and dispatches `MUL_MAT` through the BitNet-derived
  grouped kernel, with native group-128 A8 and one counted call per successful
  evaluation. One graph/weight tensor now handles two changed input batches
  with one weight repack; late failures leave the entire caller output intact.
  Reusable native create/compute/free handles own copied weight bytes, graph,
  repacked codes, and scales between caller invocations. Two live toy handles,
  failed-input recovery, fixed token shape, and registry cleanup pass.
  Explicit adjacent/low-bit-first PQ2 bytes are repacked to BitNet's separated,
  high-bit-first lanes; independent FP16 row/group scales are expanded exactly.
  One-/two-token controls and the sole full-size frozen projection match the
  direct kernel exactly. Invalid +2 codes, negative/nonfinite scales, nonfinite
  inputs, and excess tokens are rejected without changing caller output.
  This is a pinned internal-ABI, single-threaded tensor proof, not a new GGUF
  type, loader integration, persistent registration, or model quality evidence.
- Registered and direct execution give identical typed IDs, labels, conditional
  scores, and final hidden hash on the existing reordered synthetic option case,
  with zero generation. This is an engineering smoke, not calibrated confidence
  or representative quality. The frozen evaluator still excludes the new mode.
- Native `prism_bitnet_registered_projection_create_from_gguf` now imports one
  explicitly tagged, identity-only PQ2 ternary weight through the pinned GGUF
  parser into the same owned handle. A tiny generated fixture has exact counted
  BitNet parity after its source file is deleted. Missing/wrong execution tags,
  transforms, name/type/count/geometry mismatches, malformed payloads, truncation,
  and excess file size are refused. This is standalone one-weight file import,
  not Prism model-loader selection or a MiMo GGUF load. No actual projection
  was initially converted to a file; the later full-size loader control below
  uses a transient exact encoding, without retaining another candidate.
- `prism_bitnet_cpu_runtime_init_v1` exposes a stable `JEV_BITNET_LOADER_V1`
  buffer through actual early CPU discovery, checks the declared source/ABI and
  storage layout, is idempotent, and refuses initialization after discovery is
  cached. Its owned tensor has exact BitNet parity at 1, 2, and 128 tokens with
  one repack, chunked upload checks, invalid-input recovery, immutable packing,
  and invalid-payload refusal. Default loader dummy probes are intentionally
  rejected: ordinary PQ2 must not be automatically reassigned. Library lifetime
  must cover all CPU users; initialization must precede discovery on one thread.
  Two independent BitNet backend graphs sharing a weight also pass concurrently
  with ordinary PQ2 execution, with eleven total calls and one repack. This is
  initialized, read-only steady-state use, not registry mutation/lifecycle race
  safety. This new path is not yet used by the streamed module.
- `prism_bitnet_cpu_loader_override_from_gguf_v1` validates the bounded tagged
  file and ternary payload before returning exactly one anchored public
  `tensor_buft_overrides` rule. A temporary executable compiled from the actual
  pinned loader/support/registry sources verifies `create_tensor`, allocation,
  `load_all_data`, and two native BitNet `MUL_MAT` calls with one repack. The
  full 4,096 x 12,288 frozen projection passes this route via a temporary GGUF
  encoding of its unchanged codes/scales, deleted in `finally`. Untagged PQ2
  remains ordinary CPU even when this buffer is a candidate. This is genuine
  one-tensor loader selection/upload/dispatch, not a complete MiMo GGUF/model
  load or 32-layer native hosting. Factory callers must load the same unchanged
  validated file and keep the library alive; no full-model policy is authorized.
- Header-only `native_reference_plan` reconciles 427 text tensors with
  17,907,606,528 source bytes (16.678 GiB), excluding vision/MTP and retaining
  the full embedding/head. The largest source+FP32 staging basis is 5.684 GiB;
  encoded output and runtime allocations are extra. The development guide
  records actual RAM/disk observations and the staged conversion plan, not a
  native-fit or converter-peak guarantee.
- The versioned [native build wrapper](../native/CMakeLists.txt) compiles the
  full pinned llama library in a separate approximately 15.3 MB cache directory,
  importing unchanged GGML CPU/base dependencies. Existing build/dispatch
  controls pass. A guarded temporary vocabulary-only GGUF has zero weight
  tensors and loads via real native `vocab_only` API; exact non-thinking prompt
  and A/B/C token IDs match HF with zero generation and source-weight access
  forbidden. The existing dense environment supplies the full converter CLI;
  no installs, source duplication, or cached library overwrite followed.
- A temporary four-layer Qwen3.5 model now exercises actual public native model
  loading, CPU context construction, and prompt-only `llama_decode`. Two recurrent
  and two full-attention layers have zero attention outputs; a nonzero layer-3
  FFN has a 32 x 256 down projection with two independent FP16 groups per row.
  Dense FP32 and packed BitNet/A8 final logits match independent NumPy references
  within 2e-5. Fixed synthetic typed IDs and numeric option-token IDs remain
  correctly paired after reordering, with stable conditional softmax and zero
  generated answer tokens. Full/chunked/reset executions check actual hybrid
  memory positions, one BitNet call per decode, and one repack per model load.
  Illegal +2 codes, negative/nonfinite scales, nonfinite head weights, and unknown
  modes return no score report. Files are removed in `finally`.
  The BitNet route is a test-only explicit override, capped at 1 MiB and exact
  four-layer/32 x 256 geometry; it does not call or expand the public one-tensor
  factory. This proves synthetic architecture/dispatch compatibility, not real
  MiMo weight loading, meaningful tokenizer labels, nonzero recurrent arithmetic,
  native MiMo parity, model quality, or a retained candidate. See the
  [synthetic native gate](development.md#bounded-synthetic-native-prefill).
- Split-aware datasets, calibration-only hashed captures, and frozen paired
  evaluation are implemented. The attributed CC-BY-3.0 CLINC150 four-choice
  proxy has four cases per split and four training captures. Held-out proxy
  inference remains untouched; it is not representative agent/tool quality.
- Independent-block compensation is not full GPTQ and has negative wider
  validation evidence: relative error 0.426 versus RTN 0.406 with one training
  context, and 0.456 with four balanced contexts. No compensated candidate,
  bulk fit, or policy promotion followed.
- Full MiMo ternary GGUF loading, production registry lifecycle safety,
  whole-model native hosting, calibrated decision quality,
  vision, edge performance, and ARM/RISC-V validation remain open.

### Reusable Local Paths

All paths below are outside Git under `$HOME/.cache/huggingface/embedded-jev`.
Check their existence after container/cache changes; do not redownload or
rebuild the environments merely because the chat history changed.

| Cache-relative path | Purpose |
| --- | --- |
| `models/mimo-2367e865d009c13ac81713a2878291d33ab28177` | Complete pinned BF16 source; 17 verified files |
| `quantized/layer3-ffn-down-rtn-searched-fp16` | The sole retained packed projection fixture |
| `dense-venv/bin/python` | CPU Torch 2.10.0, Transformers 5.12.1, safetensors 0.7.0 |
| `native/prism-source` | Prism commit `842b1880415d6f508f03b789e5ce70194def7bfd` |
| `native/prism-build/bin` | Pinned GGML CPU/base libraries |
| `native/jev-prism-runtime-v1-build/bin` | Separate versioned full llama/registry/bridge build and control; no model weights |
| `native/bitnet-source` | Pinned Microsoft BitNet checkout and llama.cpp submodule |
| `native/converter-venv/bin/python` | Isolated CPU converter environment; use the pinned `gguf-py` path |
| `datasets/clinc150-828f809-proxy-seed902` | Attributed public proxy, provenance, and four training captures |

The base interpreter is `/opt/venv/bin/python`; use the terminal for environment
work, as requested by the user. Set `OMP_NUM_THREADS=4` for dense model probes;
forcing `OPENBLAS_NUM_THREADS=1` or `MKL_NUM_THREADS=1` previously made them
dramatically slower. The cached `native/prism_group_scale_probe.so` includes
the signed-Hadamard path, but its PQ2 and registered-tensor exports have not been
refreshed: these controls compiled a fresh bridge in pytest temporary storage.
Rebuild only if an intended caller needs those exports, using the development
guide's command, including the two pinned internal-header include paths.

### Resume Contract

The user authorized routine bounded experiments, the existing full source
download, quantization scratch space, and periodic **local commits**. Keep only
one saved quantized candidate, do not push or create branches, preserve user
changes, and do not launch subagents without explicit authorization. Do not
bulk-quantize from these diagnostics or consume held-out data while tuning.
Do not stop solely because representative domain data is absent if bounded
runtime work can still proceed. Section 16 gives the next implementation task.
The registered-tensor step did not fit a policy, create another saved candidate,
run validation/held-out inference, or promote compensation.

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
were not exercised by that GPU smoke; later CPU checks used cached weights.
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
but not affecting this pure-Torch smoke. This GPU-only check does not validate
model loading, quantization quality, or BitNet CPU execution; later CPU evidence
is recorded separately below.
The compatible `torch==2.10.0+cu128` CPython 3.12 Linux wheel is 916,856,347
bytes by HTTP header, plus dependencies. [docs/development.md](development.md)
has the opt-in isolated install and run commands. The first GPU-container uv
attempt failed before installation: top-level `.cache` is root-owned, so uv's
default cache directory was not writable. Use the documented writable volume
child and explicit `UV_CACHE_DIR`; do not change ownership or rebuild for this.
WSL `df` does not establish physical free space on the Windows drive containing
the virtual disk. No Torch package was added to the lightweight research base;
the separate GPU venv and cache are not a locked full-model environment.

## 5. Initial Environment Validation

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

## 6. Base Environment Scope and Implementation History

The base contains compiler tools and a lightweight numerical Python environment.
It deliberately does not install Torch, Transformers, datasets, safetensors,
gguf, SemIf, llama-cpp-python, BitNet, Prism, or CUDA. No inference server runs,
no ports are forwarded, and no camera or actuator is exposed.

The header-only pinned inventory fetched 172,461 metadata/header response-body
bytes. Later, a separate opt-in reader fetched 2,048 BF16 weight bytes from four
rows and 256 columns of the pinned `layers.3.mlp.down_proj.weight`. On
2026-09-29 the user authorized and downloaded the complete pinned BF16 snapshot
into `$HOME/.cache/huggingface/embedded-jev/models/mimo-2367e865d009c13ac81713a2878291d33ab28177`.
All 17 files passed pinned Hub size/content checks, and all four safetensors
shards passed LFS SHA-256 verification. Offline header/index reconciliation
counts 760 tensors and 18,819,627,488 logical weight bytes; the four shard
files occupy 18,819,720,848 bytes including headers. A local 2 KiB sample
matched the prior pinned remote ranges and the opt-in native toy slice test
passed. A second bounded reader checks four complete layer-3 FFN-down rows
(98,304 BF16 bytes) from the local shard. On fixed synthetic Gaussian inputs,
max-abs, searched-scale, and signed-Hadamard searched-scale ternary RTN had
relative output RMSE of 0.685, 0.426, and 0.446, respectively. The signed
rotation itself preserves the dense output; rounding does not. Do not select
a full-model quantization policy from four rows or synthetic inputs. No
quantized model copy was kept; see [docs/development.md](development.md) for
the reproducible command. A second CLI streams the entire 4,096 x 12,288
FFN-down tensor in 64-row batches: relative **weight** RMSE is 0.770 for
max-abs RTN, 0.461 for searched FP16 RTN, and 0.454 for signed-Hadamard
searched RTN. This is one full projection, not representative activations or
a saved/loadable ternary model. Do not bulk-convert MiMo from these results.
An opt-in native test streams and packs this entire projection in memory,
then executes a BitNet-derived AVX2 group-scaled matvec: all 4,096 outputs
match the portable integer/group-scale reference (max error 1.67e-6).
Relative error versus dense BF16 output for one synthetic A8 input is 0.467.
This is neither the stock I2_S model path nor a MiMo loader/quality proof.
A separate CPU-only Transformers 5.12.1 environment now materializes the
verified embedding and first four text layers plus norm (55 BF16 tensors,
3,764,136,064 bytes) under a 4 GiB source-weight budget. The 13-token
text-prefix forward reaches the layer-3 FFN-down input. On those actual
model-path activations, unrotated searched-scale ternary RTN has 0.423 relative
output RMSE versus the BF16 FFN output, without A8. This one prompt is neither
calibration nor decision quality; the probe produces no final model logits,
no generated tokens, and no retained quantized candidate. The same captured
last-token activation, after dynamic A8, fed the isolated BitNet-derived
AVX2 group-scale kernel: 4,096 outputs matched portable integer arithmetic
(maximum difference 3.23e-8), while relative error versus dense BF16 was
0.433. This is a single projection with a real model-path activation, not a
registered Qwen3.5/BitNet tensor type or full-model decision result.
An opt-in non-thinking chat-template variant validates distinct A-P label
continuations and runs a 22-token prompt with zero generated tokens; the
last-token native/reference max difference is 1.10e-7 and its BF16-relative
error after A8 is 0.422. This is still synthetic engineering smoke, not
representative calibration.
The pinned BF16 text decoder can now execute all 32 layers sequentially,
materializing at most one 436,814,208-byte layer at a time after the
2,034,237,440-byte embedding. The four-layer activation and final-norm hashes
match the previous full-prefix control exactly. A bounded untied LM-head reader
then scored only A/B (16,384 BF16 bytes): FP32-accumulated logits 19.481 and
20.478, conditional probabilities 0.269/0.731, zero generated tokens.
Independent safetensors BF16 row slices and Torch BF16 rounding passed parity
(max rounding gap 0.022). The full-vocabulary label mass is unknown, and this
single engineering prompt is not calibrated confidence or a decision benchmark.
The five versioned synthetic agent/tool fixture cases (80-87 tokens) now pass
through the same streamed text scorer and return typed option IDs, descriptions,
labels, and A-C conditional scores with zero generated tokens. All five selected
IDs matched fixture expectations; reordered labels and a missing-permission
case are gated in an opt-in test. This is engineering smoke only, not an owned
held-out benchmark, representative calibration, or a confidence claim.
A subsequent Python-hosted staged text run replaces only layer 3's real
FFN-down matmul with the BitNet-derived searched-FP16 ternary/A8 grouped AVX2
kernel. The BF16 and substituted runs share the same pre-FFN activation hash;
native/portable last-token outputs agree to 1.10e-7. Final A/B conditional
scores on the 22-token prompt move from 0.2695/0.7305 to 0.2709/0.7291.
No quantized candidate was saved. This is a proof of one in-model Python
adapter, not evidence of acceptable quality, a registered GGUF type, or a
loadable BitNet/Qwen3.5 runtime.
Subsequently **one** searched-FP16 layer-3 FFN-down projection was saved outside
Git as hash-checked packed codes, FP16 row/group scales, and a JSON manifest
(about 13 MB) after verifying the pinned source shard SHA-256. Reloading it
through the same native adapter reproduces the fresh-RTN 32-layer final hash
and A/B scores exactly. [docs/development.md](development.md) records its path,
hashes, and opt-in test. This is not GGUF or an approved model-loadable format;
it does not address the large ternary error or missing calibration/holdout data.
An opt-in full-vocabulary normalizer streams the 2,034,237,440-byte BF16
output head in bounded row batches. On the 22-token smoke, selected A/B mass
was 0.873 for BF16 and 0.882 after the one-layer native substitution; B was
the top token both times. This clarifies the earlier conditional-only score
for this prompt, but is not confidence calibration or a decision-quality gate.
The 2026-09-30 post-restart review reverified all source and candidate hashes.
It fixed actual-size-before-hash checks and ambiguous projection manifests,
FP32 numerator/denominator inconsistency in full-vocabulary mass, and missing
native dimension/output-byte overflow guards. The staged-prefix regression
now compares directly with the normal Transformers forward. The cached
candidate's bytes and its unrotated searched-scale policy were not changed.
The saved-projection path now bypasses BF16 FFN-down tensor materialization;
an instrumented full-model regression rejects any attempt to load that source
tensor and confirms identical saved/fresh final hidden hashes and option scores.
A separate dataset contract now supports explicit calibration, validation, and
held-out splits with declared source/license provenance. Case IDs, groups,
and whitespace/option-order-equivalent prompts cannot cross splits. The
streamed scorer requires an explicit split and retains its digest and purpose;
it never relabels the existing synthetic fixture as representative data.
The dataset-to-32-layer smoke passes using temporary synthetic split data.
Representative agent/tool calibration/holdout cases are still absent; scoring
performs no tuning.
The dataset path can now save the actual layer-3 FFN-down input in a bounded
calibration-only array plus digest/provenance manifest. Evaluation splits and
native substitutions cannot request captures; the writer checks the original
dataset digest again before saving. A real-prefix temporary capture reproduces
the model's activation hash exactly, but retains its synthetic-smoke purpose.
No representative activations or additional quantized candidate were created.
A bounded paired evaluator now compares explicit cases with BF16 and the
single frozen saved native projection. It records immutable data/candidate/
kernel/source/runtime bindings, choices, conditional-score deltas, and expected
matches without fitting. Changed inputs or inconsistent paired reports abort
the comparison. Real BF16/native paired execution passes on temporary synthetic
split data; this is evaluation infrastructure, not representative quality proof.
A small public CLINC150 four-choice proxy is now cached outside Git, pinned at
`828f8093932c8fe6ca7936c3d2e52903b1c523de`, with the original CC BY 3.0
license and attribution README. Its 4/4/4 source train/validation/test subsets
retain those roles and use deterministic distractor shortlists, excluding OOS.
One validation case matched the gold intent in BF16/native runs (conditional
total variation 0.00354), and one training FFN capture reloads correctly.
Proxy held-out data remains unscored and no new candidate was fitted. This is
not official CLINC150 performance, agent/tool representativeness, or proof
against training contamination; see the development guide for provenance.
The 256-column compensated toy was fit using 75 tokens from the public train
capture and scored on 70 live validation tokens. For four rows/two groups,
searched RTN relative error is 0.417 calibration / 0.423 validation, versus
0.212 / 0.401 for searched compensation; max-abs compensation gives 0.970
on validation. Code/scale hashes cannot depend on validation inputs, and
compensated native A8 arithmetic parity passes. No candidate was saved or
replaced, held-out data was not observed, and these local errors do not prove
full-width GPTQ or final-model quality.
The bounded compensation factorization is reusable across row batches, with
read-only factors tied to calibration digest/shape and damping settings.
Fresh/cached codes and scales match exactly; reuse with changed inputs or
settings is refused. Its original 256-column width bound remains intact.
A separate independent-256-column-block approximation was tested on four
complete 12,288-input rows. It discards cross-block curvature/error propagation
and is not full GPTQ. Searched compensation has 0.194 calibration / 0.426
validation relative error, versus 0.390 / 0.406 for searched RTN on the same
public proxy pair. Native parity passes for all 96 groups, but this validation
result is worse; no wider fit, policy promotion, or second artifact was saved.
The four public training FFN captures are now available. A bounded loader
balances 32 evenly spaced tokens from each into a read-only 128-token sample,
retaining source indices/hashes and refusing mixed datasets or repeated IDs.
On the same validation case, full-width four-row searched compensation becomes
worse (0.456 versus the unchanged RTN 0.406), although native parity passes.
No merged capture or compensated candidate is saved, and held-out inference
remains untouched. Further policy promotion needs a different justified
approximation and representative quality evidence, not more training-score wins.
The original frozen RTN projection now also executes through a real pinned
Prism CPU GGML `MAP_CUSTOM2` node. Its full 32-layer hidden hash and selected
scores match direct native execution exactly. The bounded graph bridge checks
one callback and releases its graph/backend buffers after each call. Actual
GGML dependency paths/hashes are recorded and frozen by paired evaluation.
This is one native graph-scheduled projection with Python-hosted model/A8
preparation and callback-owned packed weights, not a registered GGUF format or
whole-model native runtime. No new candidate or speed claim follows.
A third `prism_ggml_f32` backend now performs production-batch A8 in the native
GGML callback itself. Its codes/scales and final 32-layer selected scores
match the direct and Python-prepared graph paths exactly. An instrumented
test rejects Python preparation of that batch; only the one-token reference
check remains Python-side. Frozen artifact/model hosting and format limits
are unchanged, and no speed or quality claim follows.
An additional reversible `prism_ggml_hadamard128` experiment rotates weights
in 64-row batches with signs (seed 773) and matching native input signs/FWHT/A8.
Identity artifacts are refused on that path. A four-row dense FP32 probe
passes before rounding, and the 32-layer/public validation smoke remains
finite, zero-generation, and native-reference consistent. The public case
still selects `calories`; no quality promotion or rotated artifact followed.
A separate native codec control now proves that the frozen signed codes and
FP16 scales convert exactly to pinned PQ2_0 bytes (different packing order
from BitNet). Every full-projection native-decoded weight matches exactly,
and a real PQ2 tensor/MUL_MAT matches controlled activation oracles. Native
Q8_0/Q8_K preparation differs from our group-128 A8 contract. No conversion
file or second candidate was saved, and this is Prism PQ2 dispatch, not
BitNet dispatch or full GGUF/model-loader compatibility.
Separately, an MIT-licensed 1,187,801,280-byte BitNet control GGUF was
downloaded, SHA-256 verified, and loaded in the pinned native fork. See
the development guide for both bounded tracks. No vision path or registered
MiMo ternary/BitNet model runtime has been validated, and no owned MiMo quality,
perplexity, latency, energy, or RSS result exists. The host GPU
is an RTX A3000 Laptop GPU with 12,288 MiB VRAM. On 2026-09-28, WSL showed
29 GiB RAM, 21 GiB available, and 8 GiB swap; the user reported 48.3 GB free
on the Windows drive backing the virtual disk. These point-in-time readings do
not establish RAM for full-model conversion: 18,819,627,488 BF16
weight bytes alone occupy about 17.53 GiB of memory before caches and scratch.
On 2026-09-29, the cache filesystem had 319 GB free after the download; this
does not establish working RAM or a safe conversion scratch budget. Keep only
one quantized candidate at a time and measure/offload before full-model work.

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
| Prism source | `842b1880415d6f508f03b789e5ce70194def7bfd` (`prism-b10735-842b188`); build-specific library hashes are in the development guide |

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
it still rejects `ssm_out`: the inventory does not carry the key/value-head
counts needed to validate Prism's grouped-V loader geometry.
layer-3 FFN down is 12,288 inputs and FFN gate/attention Q/K are 4,096.
It rejects assigning the 256-wide slice to the full FFN down projection.
The pinned Prism Qwen3.5 name map returns no direct match for MiMo's exact
`model.language_model.layers.*` names, but the converter's shared filter removes
`language_model.` before mapping. A model-free opt-in check of all 760 pinned
index names found 333 vision-side names filtered from the text pass and all
427 retained text names mapped after that filter, including 24 `.dt_bias`
names renamed to `.dt_proj.bias` by the pinned Qwen converter source. Name
coverage alone does not verify tensor values. A separate pinned source-method
test with isolated CPU Torch 2.10.0 executed small QKV/Z/alpha/conv1d value-head
row reorders against independent NumPy ordering, and checked A-log `-exp` and
dt-bias reorder/rename. Unrotated `out_proj` permutes value-head columns;
Hadamard-folded `out_proj` keeps training order and sets the grouped-V runtime
permutation flag. Executing the pinned converter's `add_hadamard_metadata`
method on a toy folded `ssm_out` manifest also records a typed
`prism.hadamard.gdn_v_grouped=true` writer call only when that flag is set.
The pinned GGUF writer/reader round-trips that bool and folded weight name in
a no-tensor metadata file. This does **not** cover MiMo-sized tensors, model
tensor or vision-projector export, or native loader parity. The inspected
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
The pinned MiMo `config.json` is 2,784 bytes and declares 16 linear key heads,
32 linear value heads, and 128 dimensions per value head, so Prism's declared
repetition count is two. This metadata alone does not prove folded `ssm_out`
tensor shape/value parity or safely extend our reject-only toy exporter.

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
An optional grouped-V mode first permutes a two-key-head/two-repetition,
64-wide tiled activation to grouped feature order, then applies grouped signs,
128-point FWHT, A8, and that same dot. Independent scalar parity on both
tokens passed (max transform/output errors about 5.7e-7/7.2e-7).
Two consecutive graph evaluations also passed when both input and sign leaves
were restored before each run; the toy graph allocator may overwrite those
buffers during compute. This is a fixture-owned callback, **not** a loadable
group-scale GGUF type, Qwen3.5 model hook, proof of MiMo head geometry or
folded-weight export, safe sequence-state implementation,
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

## 9. Selected-Label Head: Bounded Reader Implemented

The Python-hosted streamed reference now reads only the head rows required by
the context-verified option labels and computes their logits from the exact
final normalized hidden state. Sixteen BF16 rows of width 4,096 occupy 128 KiB.
Conditional softmax over those logits is algebraically equal to conditioning
the full vocabulary distribution on the same labels. The optional full-head
normalizer remains available as a bounded reference diagnostic.

This could remove most output-head storage/work in a decision-only product.
It does not remove input embeddings. A native selected-head artifact still
requires deliberate loader/graph support;
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

## 11. Delivered Inventory and Remaining Native Gate

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

The processor checks, full pinned BF16 download, streamed dense text reference,
one-projection native substitution, and bounded rotation/codec controls are now
complete at the scopes described above. Native MiMo tokenizer/loader parity and
representative owned decision data remain separate open gates. Continue from
Section 16 rather than repeating bootstrap or downloading the model again.
Measure memory and bound temporary copies before full-model conversion; do not
create an empty module tree or copy the numbered scripts from the original PDF.

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
periodic local commits, not a push or a new branch. Implementation through
`6d74ae8` is committed, and the user has independently pushed checkpoints.
Use `git log` to identify any later commits; this document cannot embed its own
containing commit hash. Before committing or handing work back, verify:

```bash
git status --short
git log -2 --oneline
git diff --check
```

Do not revert unrelated user work or treat an earlier clean-worktree observation
as a guarantee about a later session. The ignored conversation PDF stays local.
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
multimodal token types. That initial metadata-only cache was about 979 MB;
the complete model snapshot was downloaded later on 2026-09-29.
[docs/development.md](development.md) has the commands.

The [synthetic fixture](../tests/fixtures/agent_tool_smoke.json) has five
agent/tool cases: inspect-first with an irrelevant-context perturbation,
authorized README edits with reordered options, and missing upload permission.
Their expected semantic IDs map to A/A/B/C/C; the pinned text-only processor
passed all five prompt/label boundary checks. Six offline label/processor tests
pass. The CLI records hashes and versions, not model predictions. This is **not**
calibration data, a held-out quality benchmark, image processing, native backend
tokenization parity, or a trained decision service.

## 16. Restart Brief

For a fresh coding session:

1. Read the current checkpoint above and the native projection/PQ2 sections of
  [docs/development.md](development.md), then consult design, roadmap, and
  source pins as needed. Check Git status and existing caches before acting.
2. Preserve genuine BitNet-derived CPU execution, independent FP16 scales per
  output row/input group, and typed conditional option scores with zero
  generated answer tokens. PQ2 storage/dispatch alone does not meet BitNet.
3. Start with the native PQ2 tensor control in
  [native/prism_group_scale.cpp](../native/prism_group_scale.cpp) and its
  [tests](../tests/test_prism_native_control.py), now especially
  `prism_bitnet_registered_tensor_matmul`. The scoped buffer/tensor registration
  now has reusable `prism_bitnet_registered_projection_create/compute/free`
  ownership, tagged `create_from_gguf` toy file import, and a Python-hosted
  `prism_ggml_registered` backend. Keep the exact
  one-/two-evaluation, rejection/recovery, full-text, and typed synthetic gates.
  Versioned early CPU discovery, per-buffer ownership, and an exact tagged
  one-tensor public loader override now pass, including real loader upload and
  full-size frozen-projection dispatch. Preserve default dummy-probe refusal,
  file identity/lifetime requirements, and the isolated steady-state concurrent
  graph gate. The separate full-runtime build and guarded vocabulary-only native
  tokenization now pass. The separate bounded synthetic hybrid model also passes
  genuine nonzero BitNet FFN prefill, typed order, chunk/reset, and rejection gates;
  start its next local control in
  [native/prism_bitnet_loader_control.cpp](../native/prism_bitnet_loader_control.cpp)
  and the existing optional test. Its test-only override must not become an
  unvalidated full-model loader policy. Read the staged
  [native hosting plan](development.md#native-hosting-preflight-and-conversion-plan)
  before further work: bulk text-only BF16 conversion needs separate approval,
  followed by bounded real-architecture prefill and native error propagation.
  The legacy scoped
  bridge must not run concurrently with arbitrary Prism graphs or registry
  mutation. Full-model conversion/loading remains outside the current scope.
  A production loader change
  needs a versioned runtime/build contract; do not overwrite the immutable
  cached source/library pins or bulk-convert MiMo. Do not claim loader parity.
4. Keep fitting on calibration only, diagnostics on validation, and held-out
  inference untouched. Do not promote the negative independent-block
  compensation approximation or save another candidate. No full-model
  conversion, benchmark, speed, or calibrated confidence claim follows.
5. Validate each edit locally, update these documents, and make periodic local
  commits. Continue autonomously within the authorized bounded scope; ask
  only for a genuine blocker or permission to expand scope. Use the terminal
  for environments, no subagents, no push, and no branch change.

The default regression command is
`PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider`.
The development guide gives the pinned optional native gate and rebuild
commands. Run relevant gates when implementation changes, not every expensive
model experiment merely to reconstruct the history.