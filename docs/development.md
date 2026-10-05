# Development Environment

The first environment is CPU-first so source research, mathematical tests, Python
development, and C++ kernel work do not require an NVIDIA runtime. It is a
development image, not the eventual minimal edge deployment image.

## Current Checkpoint and Gates

The last tested implementation checkpoint is `27cfd63` (2026-10-05), following
the isolated runtime/vocabulary preflight and metadata-only resource planner.
The [current handover](handover.md#current-checkpoint-2026-10-05) is the
authoritative resume summary, including external cache paths and the next
native integration task. Do not recreate environments or download another
source/model copy just to start a new chat.

Current results: 159 default tests passed, 26 optional tests skipped, and all
nine pinned Prism controls passed with full-size PQ2, repeated/owned native
weights, two-forward module reuse, tagged toy GGUF import, and versioned CPU
discovery, mixed concurrent graphs, and real pinned loader selection/upload,
including the full-size frozen projection and guarded vocabulary-only native
tokenizer preflight and dense/BitNet synthetic model prefill (35.30 s).
The separate full runtime build passes two CTests.
The 32-layer direct/callback/registered, reordered synthetic typed-decision,
and signed-Hadamard full gate passed at `4efa59b` (138.36 s), before the isolated
native file/runtime additions; unchanged model inference was not rerun afterwards.
Ruff, editor diagnostics, and whitespace checks passed. GCC ASan/UBSan and leak
checks passed for bridge runtime controls; cached GGML is not instrumented. These
are correctness/scope gates, not whole-model quantization acceptance, a
benchmark, representative quality, or calibrated confidence.

Default regression and lint commands, from the workspace root:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
ruff check --no-cache embedded_jev tests
git diff --check
```

To reproduce the earlier full-text gate together with current native controls,
reuse the existing cache and run:

```bash
cache="$HOME/.cache/huggingface/embedded-jev"
PRISM_SOURCE_DIR="$cache/native/prism-source" \
PRISM_GGML_CPU_LIBRARY="$cache/native/prism-build/bin/libggml-cpu.so" \
BITNET_SOURCE_DIR="$cache/native/bitnet-source" \
PRISM_CONVERTER_PYTHON="$cache/native/converter-venv/bin/python" \
MIMO_DENSE_PYTHON="$cache/dense-venv/bin/python" \
MIMO_LOCAL_DIR="$cache/models/mimo-2367e865d009c13ac81713a2878291d33ab28177" \
MIMO_PROJECTION_ARTIFACT="$cache/quantized/layer3-ffn-down-rtn-searched-fp16" \
MIMO_PRISM_GRAPH_TEST=1 MIMO_ROTATED_GRAPH_TEST=1 MIMO_PQ2_CODEC_TEST=1 \
MIMO_REGISTERED_MODULE_TEST=1 MIMO_REGISTERED_TYPED_TEST=1 \
MIMO_NATIVE_VOCAB_TEST=1 \
MIMO_PRISM_RUNTIME_BUILD="$cache/native/jev-prism-runtime-v1-build" \
OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
python -m pytest -q -p no:cacheprovider tests/test_prism_native_control.py
```

This compiles a temporary bridge for the tests; it does not save another
quantization or replace the cached shared bridge used by CLI probes. The
later GGML-scheduled projection section gives that separate rebuild command.
The full-size PQ2 `MUL_MAT` control uses two exactly representable input
columns, not arbitrary model activations or a GGUF-loaded model. The PQ2
operator is Prism's implementation, not the BitNet-derived callback.

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
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_inventory.py
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_label_probe.py
ruff check --no-cache .devcontainer/smoke.py docs/test_research_math.py
ruff check --no-cache embedded_jev tests
```

The smoke check runs a Cholesky solve, compiles and executes a C++17/OpenMP
program in temporary storage, verifies tool availability and writable paths,
and runs `pip check`. It does not exercise a model, GPU, or SIMD ternary kernel.
The current nine tests support the audit's deductions, not whole-model quality.

## Pinned Metadata Inventory

Run `PYTHONDONTWRITEBYTECODE=1 python -m embedded_jev.inventory` from the
workspace root for deterministic JSON including all tensor names, shapes,
dtypes, shard locations, categories, counts, byte costs, and initial policy
reasons. The report also records metadata-file hashes, header hashes, source
revision, tool/Python versions, index reconciliation, and transferred body bytes.
For a short summary, pipe the command to `jq '{totals, memory_estimates, accounting}'`.

Only seven small metadata files and two HTTP byte ranges per indexed shard are
requested. Metadata and each header are capped at 1 MiB; no more than 16 shards
are accepted. HTTP `200` for a shard range is refused before reading its body.
The command never opens a full weight payload, tokenizer vocabulary, or remote
model code. Offline fixtures in [tests/test_inventory.py](../tests/test_inventory.py)
cover valid, missing, inconsistent, unsupported, and range-ignoring responses.

For the pinned revision, 760 tensors total 18,819,627,488 bytes (plus 93,360
bytes of shard headers); a live inventory transferred 172,461 response-body
bytes. An initial group-128 PTQ1_0 estimate with both vocabulary matrices in
Q8_0 is 7,469,054,432 **weight payload bytes**. This is a format estimate, not
a converted artifact, a BitNet I2_S layout, or measured resident RAM. Vision
stays in BF16 in this estimate. FP16 KV and FP32 recurrent-state examples are
reported separately; convolution state, scratch, transforms, and allocator
overhead require later measurement. No hardware or GPU is needed for inventory.

## Bounded Real-Weight Slice

An opt-in [slice reader](../embedded_jev/weight_slice.py) reuses the pinned
index/header checks but can fetch at most four BF16 projection rows and two
contiguous 128-column groups per row (2,048 payload bytes maximum). It refuses
out-of-range requests and HTTP responses that ignore `Range`; unlike the
inventory command, it reads actual tensor values without downloading a full
shard. Reproduce the selected layer-3 FFN sample and exploratory
synthetic-activation comparison with:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m embedded_jev.weight_slice --screen-toy
```

The JSON includes the pinned revision, shard, four per-row SHA-256 digests,
bytes transferred, and separately labeled local MSE. With 512 synthetic
Gaussian calibration rows (seed 902) and 64 independent synthetic evaluation
rows (seed 903), this 4x256 slice gave MSE: max-abs RTN 0.0181, searched-scale
RTN 0.0070, max-abs compensated 0.0315, searched-scale compensated 0.0096.
The deterministic 11-candidate FP16 grid optimizes each weight group's local
squared error; it was not tuned on held-out model tasks. These comparisons
cannot establish MiMo quality, genuine activation statistics, or a useful
whole-model compression ratio. The slice reader alone does not hash a full shard.
For a separate toy native round trip using those same real weight bytes:

```bash
MIMO_BF16_SLICE_TEST=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_bitnet_group_scale.py -k bounded_mimo_slice
```

This opt-in test checks signed dense rotation parity and stored-code/FP16-scale
output parity after synthetic dynamic A8 preparation through the standalone
AVX2 kernel. It does not run the Qwen3.5 graph or real model activations.

## Pinned BF16 Snapshot

On 2026-09-29, the user authorized the complete pinned MiMo source download.
One snapshot, not a Hugging Face cache duplicate or a converted model, is at:

```bash
snapshot="$HOME/.cache/huggingface/embedded-jev/models/mimo-2367e865d009c13ac81713a2878291d33ab28177"
PYTHONDONTWRITEBYTECODE=1 python -m embedded_jev.inventory --local-dir "$snapshot" \
   | jq '{totals, accounting}'
```

All 17 pinned files were checked against Hub API byte sizes and Git blob IDs
or LFS SHA-256 (for LFS content, including `tokenizer.json`). The four
safetensors shards have these pinned content SHA-256 digests, in shard order:

```text
aab052180118aee34abc3029b54eaa49096aac606b97d703866b420dceb703c3
7a0486565f06d25ac4628e9dba470dc3f604353471d240d5a0bf7128f64df396
6c73207563d1879bfd6c143a028cc70be68edff56280458c71a43df4240300f4
1379a7cf8c0b8555a39ab65a47e830e0eb45e776b46045734da3afd73c09eea2
```

The local inventory reconciles 760 tensors, 9,409,813,744 parameters,
18,819,627,488 logical BF16 weight bytes, and 93,360 shard-header bytes.
It reads at most 1 MiB per metadata file and shard header; it does not verify
full-file hashes, which were checked separately with `sha256sum`. The four
shard files total 18,819,720,848 bytes; about 319 GB remained on the cache
filesystem after download. No quantized model or second source copy was kept.
A local 2 KiB row sample matched the pinned remote ranges, and the existing
opt-in native slice test passed; its synthetic MSE is not decision quality.
Do not load all BF16 tensors into the 29 GiB host RAM just to test the cache.

For a bounded full-width trial, read only four rows of the layer-3 FFN-down
projection (4 x 12,288 BF16 values; 98,304 bytes):

```bash
PYTHONDONTWRITEBYTECODE=1 python -m embedded_jev.weight_slice \
   --local-dir "$snapshot" --screen-full-rows
```

The local reader checks the pinned index/header, caps payloads at 128 KiB,
and records a SHA-256 for each complete row. On 32 fixed synthetic Gaussian
inputs (seed 903), relative output RMSE versus those four dense rows was
0.685 for max-abs RTN, 0.426 for searched FP16 row/group-128 scales, and
0.446 for signed 128-point Hadamard followed by searched RTN (sign seed 773).
Dense signed-rotation parity passed before quantization. These are warnings
about this sample, not representative activations, calibrated quality, or
evidence for a whole-model policy. All three trials ran in memory; no
quantized checkpoint was stored.

The same shard also supports one capped, streaming projection check:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m embedded_jev.weight_slice \
   --local-dir "$snapshot" --screen-projection
```

This reads the full 4,096 x 12,288 layer-3 FFN-down tensor (100,663,296 BF16
bytes) in 64-row batches, with a 128 MiB tensor limit. In-memory group-128 RTN
on all rows gave relative **weight** RMSE of 0.770 (max-abs scales), 0.461
(searched FP16 scales), and 0.454 (signed 128-point Hadamard and searched
scales, sign seed 773). This checks reconstruction, not real activation output,
perplexity, or decision quality. The command does not check the full shard hash
again or write quantized weights; the snapshot was separately SHA-256 verified.
These errors are too large to promote a whole-model ternary candidate without
representative activations and block-level quality checks.

For a full-projection **native arithmetic** check (no retained candidate):

```bash
MIMO_LOCAL_DIR="$snapshot" MIMO_FULL_PROJECTION_TEST=1 \
   PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -s -p no:cacheprovider \
   tests/test_bitnet_group_scale.py -k full_mimo_projection_matches_bitnet_derived_native_dot
```

The opt-in test streams the same 4,096 rows through searched FP16 group-128
RTN, packs 12,582,912 ternary-code bytes in memory, prepares one synthetic A8
token, and executes the BitNet-derived AVX2 grouped dot. All outputs agreed
with an independent integer/group-scale reference (maximum absolute error
1.67e-6). Relative error versus the dense BF16 projection on that one
synthetic input was 0.467, including A8 rounding. This is not stock BitNet
I2_S or a model-loadable MiMo operator, and says nothing about decision quality.

## Partial Dense Text Prefix

An isolated CPU-only environment under
`$HOME/.cache/huggingface/embedded-jev/dense-venv` has Torch 2.10.0+cpu,
Transformers 5.12.1 (the pinned MiMo config version), safetensors 0.7.0,
Accelerate 1.12.0, and NumPy 2.2.6. It is separate from the Prism converter
environment. To recreate it if absent, use `uv` with a writable cache:

```bash
cache="$HOME/.cache/huggingface/embedded-jev"
UV_CACHE_DIR="$cache/native/uv" uv venv "$cache/dense-venv" --python /opt/venv/bin/python
UV_CACHE_DIR="$cache/native/uv" uv pip install --python "$cache/dense-venv/bin/python" \
   --index https://download.pytorch.org/whl/cpu 'torch==2.10.0+cpu'
UV_CACHE_DIR="$cache/native/uv" uv pip install --python "$cache/dense-venv/bin/python" \
   'transformers==5.12.1' 'safetensors==0.7.0' 'accelerate==1.12.0' \
   'numpy==2.2.6' 'jinja2==3.1.6'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
   "$cache/dense-venv/bin/python" -m embedded_jev.dense_probe \
   --local-dir "$snapshot" --layers 4 --compare-ternary
```

The probe constructs a `Qwen3_5TextModel` on `meta`, requires exact names and
BF16 shapes for the 55 pinned embedding/first-four-layer/final-norm tensors,
and materializes only 3,764,136,064 BF16 weight bytes. On its 13-token local
text prompt the Torch fallback produced finite outputs and captured the real
layer-3 FFN-down input (1 x 13 x 12,288). Searched FP16 group-128 RTN for that
one complete projection had relative output RMSE 0.423 against the BF16
layer output, **without** activation A8. The probe computes no final model
logits or answer tokens, and this one prompt is not representative calibration
or a decision-quality result. It stores no activations or quantized weights.
Run the optional end-to-end prefix smoke with `MIMO_DENSE_PREFIX_TEST=1`,
`MIMO_DENSE_PYTHON="$cache/dense-venv/bin/python"`, and `MIMO_LOCAL_DIR="$snapshot"`
against `tests/test_dense_probe.py`.

For an opt-in native run on the **captured last-token activation**, compile
the existing BitNet-derived AVX2 kernel outside the repo and pass its library
to the same probe:

```bash
clang++-18 -std=c++17 -O2 -mavx2 -shared -fPIC native/bitnet_group_scale.cpp \
   -o "$cache/native/bitnet_group_scale_probe.so"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
   "$cache/dense-venv/bin/python" -m embedded_jev.dense_probe \
   --local-dir "$snapshot" --layers 4 --compare-ternary \
   --native-library "$cache/native/bitnet_group_scale_probe.so"
```

The 4,096 native outputs agreed with portable integer/group-scale arithmetic
to max absolute error 3.23e-8 after dynamic per-group A8. Relative error
against that last token's dense BF16 FFN output was 0.433; this combines A8
and ternary rounding. Set `MIMO_NATIVE_ACTIVATION_TEST=1` alongside the three
prefix-smoke environment variables above to make the optional test compile
a temporary library and check this path. No quantized model was kept, and the
kernel is not the stock I2_S type or a Qwen3.5 model loader/decision engine.
Add `--chat-template` to the probe command (or `MIMO_CHAT_TEMPLATE_TEST=1`
to the opt-in test) to use the pinned non-thinking chat template and recheck
the A-P one-token boundary before forwarding. The short prompt becomes 22
tokens with zero generated tokens. On that template-faithful input, the
last-token native/reference maximum difference was 1.10e-7 and relative
error versus BF16 was 0.422 after A8. This is still one synthetic prompt,
not calibrated option scores or a decision-quality measurement.

## Streamed Text-Only Scores

The same pinned CPU environment now streams all 32 text decoder layers, loading
one at a time rather than retaining the 18.8 GB BF16 checkpoint in RAM. Its
four-layer run exactly matches the full-prefix FFN input and final-norm hashes
above. With the pinned non-thinking 22-token prompt, it completed 32 layers,
applied the text final norm, and read only two BF16 rows (16,384 bytes) from
the untied LM head:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
   "$cache/dense-venv/bin/python" -m embedded_jev.streamed_text \
   --local-dir "$snapshot" --layers 32 --label-count 2
MIMO_STREAMED_TEXT_TEST=1 MIMO_DENSE_PYTHON="$cache/dense-venv/bin/python" \
   MIMO_LOCAL_DIR="$snapshot" PYTHONDONTWRITEBYTECODE=1 \
   python -m pytest -q -p no:cacheprovider tests/test_dense_probe.py \
   -k streamed_full_text_scores_only_selected_labels
```

The A/B FP32-accumulated logits from BF16 head rows were 19.481/20.478,
conditional probabilities 0.269/0.731. Independent safetensors row slices
matched the byte hashes; Torch BF16 head rounding differed by at most 0.022.
No answer tokens were generated. The head was never fully loaded and the
full-vocabulary mass was **not** computed. These scores are conditional among
A/B on one engineering prompt, not calibrated confidence, task accuracy, a
vision path, or a ternary/BitNet model dispatch result.

For a typed option-mapping smoke, select one case from the repository's
versioned synthetic fixture. Its messages use the same formatter and A-P
continuation check as the tokenizer probe:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
   "$cache/dense-venv/bin/python" -m embedded_jev.streamed_text \
   --local-dir "$snapshot" --layers 32 \
   --fixture tests/fixtures/agent_tool_smoke.json --case-id edit-reordered-options
MIMO_TYPED_FIXTURE_TEST=1 MIMO_DENSE_PYTHON="$cache/dense-venv/bin/python" \
   MIMO_LOCAL_DIR="$snapshot" PYTHONDONTWRITEBYTECODE=1 \
   python -m pytest -q -p no:cacheprovider tests/test_dense_probe.py \
   -k streamed_text_maps_synthetic_options_to_typed_scores
```

The five 80-87-token engineering cases were tried one at a time; each returned
typed option IDs/descriptions and conditional A-C scores without generating
tokens, and its selected ID matched the fixture's expected ID. Option reorder
and missing-permission cases have a gated regression check. This tiny synthetic
set is **not** a held-out benchmark or calibration set; full-vocabulary mass
and real decision quality remain unknown. No quantized model copies were kept.

## Split-Aware Decision Data

The streamed scorer also accepts a bounded dataset with explicit calibration,
validation, and held-out splits. This does not change the existing synthetic
fixture schema or supply representative data. The dataset is capped at 1 MiB
and has these exact top-level fields:

- `schema_version`: integer `1`.
- `purpose`: `user_labeled_text_decisions`, `synthetic_split_contract_smoke`,
  or `public_intent_proxy`.
- `provenance`: nonempty `source` and `license` strings; these are declared
   provenance, not automatic verification of ownership or representativeness.
- `splits`: exactly `calibration`, `validation`, and `held_out`, each containing
   1-32 cases with the existing `id`, `group`, `state`, `question`, `options`,
   and `expected_option_id` fields.

Case IDs must be unique across splits. A group may not cross splits, and the
loader rejects equivalent state/question/option descriptions across splits
even after whitespace changes or option reordering. These checks do not detect
every paraphrase or establish statistical independence.

Set `dataset` to the actual local dataset path and select the split explicitly:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
   "$cache/dense-venv/bin/python" -m embedded_jev.streamed_text \
   --local-dir "$snapshot" --layers 32 --dataset "$dataset" \
   --split held_out --case-id case-to-evaluate
```

Results retain dataset purpose, SHA-256, source/license, and split alongside
typed conditional option scores. Scoring does not tune weights or scales.
The gated `MIMO_DATASET_TEST=1` test exercises this path using a temporary
split of the existing synthetic cases; it is not held-out quality evidence.

For a calibration-only FFN input capture, use an unused output directory whose
parent exists:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
   "$cache/dense-venv/bin/python" -m embedded_jev.streamed_text \
   --local-dir "$snapshot" --layers 4 --dataset "$dataset" \
   --split calibration --case-id calibration-case \
   --calibration-output "$capture"
```

Only a BF16 run with at least four layers may capture inputs. Validation and
held-out splits are rejected before model imports; the writer also resolves
the case strictly from calibration. At most 128 x 12,288 float32 values are
saved in a non-pickle NumPy array, accompanied by a hashed provenance manifest.
The dataset digest must still match the pre-inference digest before writing.
Reload is read-only and checks byte sizes, hash, dtype, shape, finiteness, and
calibration role. Existing captures are not overwritten. Synthetic captures
retain their synthetic purpose and cannot establish representativeness merely
because model weights produced their activations. The opt-in dataset test
checks exact activation-byte parity using a temporary synthetic capture.

## Frozen-Candidate Comparisons

The paired evaluator runs BF16 and the existing saved native projection on
1-4 explicitly selected cases, without fitting or replacing that candidate:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
   "$cache/dense-venv/bin/python" -m embedded_jev.evaluation \
   --local-dir "$snapshot" --dataset "$dataset" --split validation \
   --case-ids case-to-compare --candidate "$candidate" \
   --native-library "$cache/native/bitnet_group_scale_probe.so"
```

Reports bind the dataset digest/split, saved candidate metadata and manifest
digest, kernel digest, scoring-source hashes, and runtime versions. Inputs
must remain unchanged between runs. Prompt, token-count, pre-FFN activation,
case IDs, option/head-row mappings, logits, and conditional-score normalization
must agree with their respective contracts; no generated tokens are allowed.
The output records both choices, expected-label matches, per-option logit and
probability deltas, conditional total variation, and changed-choice counts.
It does not convert conditional probabilities into calibrated confidence.

Use validation for candidate diagnostics; reserve held-out data for the frozen
final evaluation rather than repeatedly selecting policies from it. The
`MIMO_PAIRED_EVAL_TEST=1` smoke (with the existing MiMo interpreter/snapshot
and `MIMO_PROJECTION_ARTIFACT` variables) uses temporary synthetic split data
and checks the full BF16/native path. Its expected-choice agreement is not
representative task accuracy or a whole-model quantization quality gate.

## Attributed Public Intent Proxy

An optional importer prepares a tiny diagnostic proxy from CLINC150 at
revision `828f8093932c8fe6ca7936c3d2e52903b1c523de`, under CC BY 3.0. It
verifies the original data, license, and attribution README against pinned Git
blob IDs and the data SHA-256
`36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0`.
The successful bundle transfer is 2,517,525 body bytes. Original source and
attribution files stay outside Git alongside the derived decisions:

```bash
proxy="$cache/datasets/clinc150-828f809-proxy-seed902"
PYTHONDONTWRITEBYTECODE=1 python -m embedded_jev.public_data \
   --output "$proxy" --cases-per-split 4 --seed 902
```

Run only when the bundle is absent; existing outputs are refused. The source
JSON has a separate explicit 3 MiB decoding bound. Ordinary model metadata
keeps its 1 MiB default and the inventory's network range limits are unchanged.
Four source training cases map to calibration, four source validation cases to
validation, and four source test cases to held-out. Repeated normalized
utterances are excluded across these subsets. Every case has the gold intent
plus three deterministic distractors in shuffled order. Out-of-scope rows are
excluded. Source, authors/paper, license, transformations, and seed are retained
in provenance, along with the original license and README.

**This is not the official 150-intent/OOS benchmark**, a representative
agent/tool corpus, or evidence of freedom from model-training contamination.
The reduced shortlist deliberately makes a different task. The derived dataset
SHA-256 is `0efccfd5c6b3d0a1759c1a2c0bf34c8f17f5fa5612ecd5f29870cbbb4c14733f`.
All 12 non-thinking prompts fit within 66-80 tokens. One frozen-candidate
validation observation (`clinc150_val_1561`) selected `calories` in BF16 and
native runs with conditional total variation 0.00354 and zero generation or
fitting. One training input capture (`clinc150_train_127`) has shape
75 x 12,288 float32 and reloads read-only with its public-proxy provenance.
No proxy held-out case has been scored or used for fitting.

Set `MIMO_PUBLIC_PROXY_TEST=1` and `MIMO_PUBLIC_DATASET="$proxy/decisions.json"`
alongside the interpreter/snapshot/candidate variables for the cached-only
public integration test. It verifies original source hashes, one validation
comparison, and a temporary training capture without network access.

## Calibration-Fitted Slice Check

The existing 256-column compensated toy can now consume a hash-checked
calibration capture, with a live validation prefix kept read-only and never
persisted or passed into fitting:

```bash
capture="$proxy/calibration-clinc150_train_127"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
   "$cache/dense-venv/bin/python" -m embedded_jev.weight_slice \
   --local-dir "$snapshot" --screen-calibrated-slice \
   --calibration-capture "$capture" --validation-dataset "$proxy/decisions.json" \
   --validation-case-id clinc150_val_1561
```

This examines only four weight rows and the first two 128-column groups.
Fitting uses 75 training-capture tokens; scoring uses a separate 70-token
validation prefix. Searched FP16 RTN has relative reconstruction error 0.417
on calibration and 0.423 on validation. Searched compensated fitting gives
0.212 and 0.401, respectively, while max-abs compensation worsens validation
error to 0.970. The large training benefit does not generalize proportionally,
and the compensated weight error itself is higher than searched RTN's.

Reports retain source/capture hashes, code/scale hashes, damping, and separate
calibration/validation metrics. Tests require identical fitted code/scale
hashes when validation inputs change. Held-out activation observation is
refused. `MIMO_CALIBRATED_SLICE_TEST=1` with the interpreter/snapshot/public
dataset and `MIMO_CALIBRATION_CAPTURE` variables verifies compensated codes,
FP16 scales, and dynamic A8 through the native grouped kernel on live
validation inputs. No candidate is saved and the sole full-projection RTN
artifact remains unchanged. These are local slice numerics, not full-projection
GPTQ, final-model scores, or representative task quality.
Bounded compensation curvature can now be prepared once and reused across
weight-row batches. Factors are read-only and bound to the float64 calibration
sample digest, shape, damping ratio, and retry bound. Changed samples/settings
and invalid triangular factors are rejected. Cached and fresh paths produce
identical codes/scales/damping, and row batching is invariant. This avoids
re-solving the same bounded system; it does not expand the 256-column limit
or establish a full-width Hessian approximation.

An explicitly separate block-diagonal diagnostic extends **only four weight
rows** across the full 12,288 inputs:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
   "$cache/dense-venv/bin/python" -m embedded_jev.weight_slice \
   --local-dir "$snapshot" --screen-block-diagonal \
   --calibration-capture "$capture" --validation-dataset "$proxy/decisions.json" \
   --validation-case-id clinc150_val_1561
```

It prepares 48 independent 256-column factors, reuses them across row batches,
and intentionally omits cross-block curvature and error propagation. It is
**not full GPTQ**. The original compensated routine remains capped at 256
columns; the separate approximation accepts at most 64 weight rows per call
and 128 calibration tokens. Fresh/cached and independent-block row batching
match exactly, with calibration/settings digest guards.

On the same single train/validation proxy pair, searched block compensation
has relative error 0.194 on calibration but 0.426 on validation; searched RTN
gives 0.390 and 0.406. Thus the narrow-slice benefit does not carry over here.
No candidate was saved, replaced, or promoted, and no full 4,096-row fit or
final-model score was evaluated. The opt-in calibrated native test also checks
all 96 groups for these four full-width rows against portable integer/A8
arithmetic on live validation features. Arithmetic parity is not a quality
acceptance criterion.

The diagnostic can balance up to four hash-checked training captures with
`--additional-calibration-captures`. All must share the dataset digest/purpose,
and case IDs must be distinct. At most 128 tokens are selected with equal
case quotas and evenly spaced source indices; those indices and source-array
hashes are retained in the report. No combined capture is written.
For the four public training cases, 32 tokens per case produced a 128-token
sample (raw float32 SHA-256
`c0936357d1dd63ab92cca8e5d2727a43b0aa3265f08eda99ed6aecf6c6c3bfe3`).
Full-width four-row searched compensation then gave validation relative error
0.456, worse than the one-context 0.426 and searched RTN's unchanged 0.406.
Native parity still passes. Broader training coverage alone did not rescue
this independent-block approximation; do not promote or save it from these
results. The source proxy, its four calibration-only captures, and the sole
frozen RTN candidate remain separate. Held-out inference still has not run.

## In-Memory Native FFN Substitution

The streamed BF16 text path can replace **only** layer 3's FFN-down matmul
with an in-memory searched-FP16 group-128 ternary/A8 adapter. It uses the
BitNet-derived AVX2 grouped kernel built above, keeps every other layer in
BF16, and releases the packed candidate after the process exits:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
   "$cache/dense-venv/bin/python" -m embedded_jev.streamed_text \
   --local-dir "$snapshot" --layers 32 --label-count 2 \
   --native-ffn-library "$cache/native/bitnet_group_scale_probe.so"
MIMO_IN_MODEL_NATIVE_TEST=1 MIMO_DENSE_PYTHON="$cache/dense-venv/bin/python" \
   MIMO_LOCAL_DIR="$snapshot" PYTHONDONTWRITEBYTECODE=1 \
   python -m pytest -q -p no:cacheprovider tests/test_dense_probe.py \
   -k streamed_text_substitutes_one_bitnet_derived_ffn
```

The 22-token non-thinking prompt reaches the same pre-FFN activation hash
in BF16 and substituted runs. The adapter is called once for the whole token
batch and agrees with portable integer/group-scale arithmetic on its last
token (max difference 1.10e-7). The final A/B conditional scores move from
0.2695/0.7305 to 0.2709/0.7291. This single smoke cannot establish quality
or safety, and a small final-score change does not erase the substantial
local FFN approximation error. This is a Python-hosted, in-memory substitution,
**not** a registered model-loadable ternary GGUF/operator or stock BitNet I2_S.

To diagnose how much probability mass the selected A/B labels receive, add
`--full-vocabulary-mass` to either streamed-text command. This optional pass
reads the 2,034,237,440-byte BF16 untied head in 256-row batches and computes
a stable full-vocabulary log-sum-exp without generating tokens or keeping the
head in memory. Set `MIMO_FULL_HEAD_TEST=1` alongside
`MIMO_IN_MODEL_NATIVE_TEST=1` for the gated BF16/native comparison. On the same
22-token prompt, A/B mass was 0.873 in BF16 and 0.882 after the one-layer
native substitution; B was the top token in both. These are measurements for
one deliberately constrained prompt, **not** calibrated confidence, a quality
benchmark, or evidence that bulk ternary quantization is safe.

## GGML-Scheduled Frozen Projection

The frozen RTN projection can also run through a real CPU GGML `MAP_CUSTOM2`
node using the pinned Prism library instead of invoking the grouped kernel
directly. Build the bounded shared bridge outside Git:

```bash
native="$cache/native"
clang++-18 -std=c++17 -O2 -mavx2 -shared -fPIC \
   -I "$native/prism-source/ggml/include" \
   -I "$native/prism-source/ggml/src" \
   -I "$native/prism-source/ggml/src/ggml-cpu" \
   -I "$native/prism-source/include" \
   native/prism_group_scale.cpp native/bitnet_group_scale.cpp \
   -L "$native/prism-build/bin" -Wl,-rpath,"$native/prism-build/bin" \
   -lggml-cpu -lggml-base -o "$native/prism_group_scale_probe.so"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
   "$cache/dense-venv/bin/python" -m embedded_jev.streamed_text \
   --local-dir "$snapshot" --layers 32 --projection-artifact "$candidate" \
   --native-ffn-library "$native/prism_group_scale_probe.so" \
   --native-ffn-backend prism_ggml
```

The graph bridge caps 128 tokens, 4,096 rows, and 96 groups, requires exactly
one callback, and releases its graph/backend buffers on return. Repeated graph
calls are exactly equal to the direct kernel on golden fixtures. The real
frozen 4,096 x 12,288 projection, within all 32 BF16 text layers, gives exactly
the same final hidden hash and selected A/B logits/scores as direct native
execution, with zero generation and no BF16 source-projection materialization.

Set `MIMO_PRISM_GRAPH_TEST=1` alongside the Prism source/library and MiMo
interpreter/snapshot/candidate variables to extend
`tests/test_prism_native_control.py -k shared_group_scale_graph_matches_direct_kernel`
to the real full-text parity check. The frozen evaluator accepts
`--native-backend prism_ggml` with this bridge. Reports obtain the actual
loaded GGML paths via `dladdr`, record CPU/base library hashes, and refuse
changed dependencies or mismatched execution reports. The CPU hash is
`adaaacaf406df3700fb5f05cb5749990e9b17d410d91e13d0c58263ae971a150`;
the base hash is `57a9b6060d3bc902c30fd97fb271ae7f027054d4a9f8b50e27dd6fe91951847d`.
The public validation observation remains identical to direct execution
(conditional total variation 0.00354 relative to BF16).

This is **native graph scheduling of one frozen projection**, not a native
whole-model runtime or GGUF tensor registration. A8 preparation remains in
the Python caller; packed weights/scales are callback-owned, not model-loaded
GGML weight tensors. No speedup, ARM support, calibrated confidence, or quality
acceptance is established.

A separate `--native-ffn-backend prism_ggml_f32` path now uploads FP32 FFN
inputs and performs dynamic group-128 A8 **inside the native GGML callback**
before invoking the BitNet-derived dot. Rebuild the bridge with the same
command above. The evaluator also accepts `--native-backend prism_ggml_f32`.
Native codes/scales match Python's nearest-even quantizer exactly on zero,
half-way, and random inputs. Nonfinite and underflowing-scale inputs are
rejected rather than assigned codes. Native preparation
requires round-to-nearest mode and rejects unsupported dimensions or
unrepresentable scales.

Direct, Python-prepared GGML, and native-A8 GGML paths have identical real
32-layer hidden hashes and selected scores for the frozen projection. The
opt-in graph test forbids Python A8 preparation of the production token batch;
only a one-token portable diagnostic reference remains Python-side. This
extends the one-projection native boundary without moving model hosting,
tokenization, or saved-weight ownership into a registered GGUF runtime.

An experimental `--native-ffn-backend prism_ggml_hadamard128` path applies
matching explicit signs and normalized 128-point Hadamard to weights and
inputs before ternary/A8 rounding. It rotates/quantizes weight rows in 64-row
batches in memory and executes input signs/FWHT/A8 natively. It must run
without `--projection-artifact`: identity artifacts are refused, not
reinterpreted. The fixed sign seed is 773 and float32 sign-vector SHA-256 is
`9b53024527e03670a7f12b6b8eeab741a6f41f13fdbfe26dff9a0c7b2e3258c6`.

Dense FP32 equivalence is checked on four probe weight rows before native
rounding. The public validation observation remains `calories`, with zero
generation; the dense transform and native reference errors were 2.61e-8
and 8.38e-8. These are transform/arithmetic checks on one projection, not
full-model quantization acceptance or a calibrated confidence improvement.
`MIMO_ROTATED_GRAPH_TEST=1` alongside the existing graph-test variables adds
the in-memory 32-layer transform smoke. The frozen paired evaluator deliberately
does not accept this freshly built rotated policy as the saved identity
candidate. No rotated artifact was retained or promoted; the one saved RTN
fixture is unchanged.

## Native PQ2 Codec Control

A separate ternary-subset codec converts signed codes and FP16 group scales
to the pinned Prism `PQ2_0` layout: 128 values per 34-byte block, two bytes
of little-endian scale followed by adjacent low-bit-first 2-bit codes.
This is **not** the existing BitNet fixture's high-bit-first separated-lane
packing. Codes retain {-1,0,+1}; native PQ2's additional +2 code is rejected
by this subset converter rather than silently treated as ternary.

Golden bytes and scale/code roundtrips pass. The actual pinned
`dequantize_row_pq2_0` reconstructs every value of the frozen 4,096 x 12,288
projection exactly after in-memory conversion. A genuine `GGML_TYPE_PQ2_0`
tensor and `GGML_OP_MUL_MAT` also match the portable reconstructed-weight
control for two exactly representable input columns across all 4,096 rows.
For non-256-divisible widths, the native operator uses Q8_0 activation blocks
with FP16 scale rounding; other widths use Q8_K. The 256/384-input fixtures
therefore compare against actual native activation quantizer/decoder oracles,
not unrounded inputs. This is a different activation contract from our
per-token/group-128 A8 callback.

Set `MIMO_PQ2_CODEC_TEST=1` and `MIMO_PROJECTION_ARTIFACT` alongside the pinned
Prism source/library variables to extend the optional tests to the real
projection decoder and controlled native tensor operation. No converted
representation is written, no full GGUF or loader parity is established,
and this control uses Prism's PQ2 implementation, **not BitNet dispatch**.
It is a storage/operator compatibility step, not the ternary decision engine.

## Registered BitNet Weight Tensor Control

The separate `prism_bitnet_registered_tensor_matmul` export registers a scoped
`JEV_BITNET_GROUP128` CPU extra-buffer type. Its weight buffer initializes
`tensor->extra` with the BitNet tensor trait; the CPU backend recognizes the
buffer and dispatches a real `GGML_OP_MUL_MAT` through that trait, not
`MAP_CUSTOM2` or the ordinary Prism PQ2 dot. A dispatch counter must report
exactly one invocation of `bitnet_group_scale_matmul_avx2` on success.

Storage is deliberately still `GGML_TYPE_PQ2_0`: each row/input-group block has
a little-endian FP16 scale and 32 adjacent, low-bit-first code bytes. Only
codes 0/1/2 (ternary -1/0/+1) are accepted; PQ2's +2 code, negative scales,
and nonfinite scales are rejected. Native execution explicitly repacks to
BitNet's four separated high-bit-first lanes and expands each independent
FP16 scale exactly to FP32. No scale is refitted. FP32 inputs use native
per-token/group-128 A8, nearest-even rounding, clipping to [-127,127], FP32
max-abs/127 scales, and scale 1 for zero groups. This is not Q8_0 or Q8_K.

The export takes PQ2 bytes, FP32 inputs, token/row/group counts, FP32 output,
and a `size_t*` dispatch counter. Bounds remain 128 tokens, 4,096 output rows,
and 96 input groups. It returns 0 on success; bad arguments return 1,
allocation/nonfinite-input failures 2, graph failures 3, dispatch mismatches 4,
and invalid weight payloads 5. Unsupported A8 rounding also returns 1.
Failures do not copy graph output to the caller, and rejected payloads/inputs
make no grouped kernel call. Both arrays use token-major output/input layout.

`prism_bitnet_registered_tensor_matmul_repeated` adds an evaluation count
(1 or 2) and a weight-repack counter. It uploads and validates the weights
once, reuses one graph across changed inputs, and repacks only once. Inputs
are `[evaluations, tokens, groups*128]`, outputs `[evaluations, tokens, rows]`.
Output publication is transactional: a rejected second input leaves all caller
outputs unchanged, while counters report the one completed kernel call.
`prism_bitnet_registered_tensor_registry_size` checks registry restoration.

Reusable native ownership is exposed separately:

- `prism_bitnet_registered_projection_create(pq2, tokens, rows, groups, void**)`
   copies the storage bytes into its native weight tensor, validates/repackages
   them once, and returns an opaque handle (null on failure).
- `prism_bitnet_registered_projection_compute(handle, inputs, output, calls*,
   repacks*)` reuses that graph/weight tensor at its fixed token/row/group shape.
   Counters are cumulative; each success adds one kernel call and repacks stay 1.
   Rejected nonfinite input leaves output unchanged and the handle reusable.
- `prism_bitnet_registered_projection_free(handle)` releases all owned state;
   null is allowed. Free each successful handle exactly once using its creating
   library. Foreign, freed, or concurrently freed pointers are invalid C callers.

Registration is scoped to initialization/compute, not handle lifetime. Idle
handles leave no global registry entry. Native byte ownership, two live toy
handles with different weights, rejection/recovery, and cleanup pass, including
full-size frozen-projection reuse. No weight mutation API is exposed.

One- and two-token controls at widths 256/384, repeated calls, zero groups,
rejection cases, and the sole saved 4,096 x 12,288 projection match the direct
BitNet/group-128 A8 reference exactly. To run all seven pinned native controls
with full-size coverage but without another streamed-model forward:

```bash
cache="$HOME/.cache/huggingface/embedded-jev"
PRISM_SOURCE_DIR="$cache/native/prism-source" \
PRISM_GGML_CPU_LIBRARY="$cache/native/prism-build/bin/libggml-cpu.so" \
BITNET_SOURCE_DIR="$cache/native/bitnet-source" \
PRISM_CONVERTER_PYTHON="$cache/native/converter-venv/bin/python" \
MIMO_PROJECTION_ARTIFACT="$cache/quantized/layer3-ffn-down-rtn-searched-fp16" \
MIMO_DENSE_PYTHON="$cache/dense-venv/bin/python" \
MIMO_PQ2_CODEC_TEST=1 MIMO_REGISTERED_MODULE_TEST=1 \
OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
python -m pytest -q -p no:cacheprovider tests/test_prism_native_control.py
```

This compiles a temporary bridge; the cached CLI bridge is unchanged. Rebuilding
it now needs the internal include paths in the build command above and the
exact pinned C++ ABI. Registration is removed on every return and these export
calls serialize with each other, but the global registry is not synchronized
with arbitrary Prism graphs or external mutation. Use only an isolated,
single-threaded probe. Device buffer discovery, production registry lifetime,
GGUF loader selection, full-model hosting, and target-platform parity remain
unproven. Standard PQ2 bytes, repacked weights, expanded FP32 weight scales,
and A8 scratch coexist during this control; no compact resident-memory or
performance claim follows. No new candidate or dataset inference is saved.

The `--native-ffn-backend prism_ggml_registered` streamed backend now uses these
owned handles for the sole frozen layer-3 FFN-down substitution. It performs
production-batch A8 in the tensor trait, records native dispatch/packing counts,
and closes the handle explicitly after the decoder layer (also on failure).
The module supports repeated forwards only at its initial token shape, rejects
use after close, and has a weakref cleanup fallback. Two-forward output parity
passes with Python production A8 forbidden. The frozen paired evaluator does
not accept this experimental backend yet.

All 32 Python-hosted text layers now yield exactly the same final hidden hash
and selected logits/conditional scores for direct, both callback paths, and
registered execution. The existing reordered synthetic A-C case also yields
identical typed option IDs/labels/scores, with zero generated answer tokens.
It is an engineering observation only; no validation or held-out dataset was
scored, no compensation promoted, and no second candidate saved.

To try the registered streamed backend, first rebuild the cached bridge using
the earlier shared-bridge command with both internal include paths, then run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
"$cache/dense-venv/bin/python" -m embedded_jev.streamed_text \
   --local-dir "$cache/models/mimo-2367e865d009c13ac81713a2878291d33ab28177" \
   --layers 32 --projection-artifact "$cache/quantized/layer3-ffn-down-rtn-searched-fp16" \
   --native-ffn-library "$cache/native/prism_group_scale_probe.so" \
   --native-ffn-backend prism_ggml_registered
```

### Tagged Single-Tensor GGUF Import

`prism_bitnet_registered_projection_create_from_gguf(path, tokens, void**)`
uses the pinned native `gguf_init_from_file_ptr` metadata parser on a single
open file, then creates the existing owned registered projection from its
validated payload. It is a standalone file-consumption control, not an automatic
Prism model-loader hook or a complete MiMo artifact. Its explicit contract is:

- GGUF v3, at most 14 MiB, exactly one `GGML_TYPE_PQ2_0` tensor named
   `blk.3.ffn_down.weight`, with contiguous logical `[width, rows, 1, 1]` geometry.
- Width is a positive multiple of 128, at most 12,288; rows are 1 through 4,096.
   Payload sizes/offsets are checked before bounded weight-byte allocation.
- String metadata `jev.bitnet.execution` must equal
   `group128-a8-fp32-nearest-even-identity-v1`. This explicitly selects the
   ternary/FP16 row-group and native-A8 identity contract, not generic PQ2/Q8_K.
   Missing, differently typed, or conflicting tags are refused. Any
   `prism.hadamard.*` metadata is refused rather than silently ignored.
- Existing PQ2 packing and payload checks still apply. Invalid codes/scales
   return 5; file/metadata/geometry/version/size/read failures return 6; invalid
   API arguments return 1 and allocation failures 2. Failed creation clears the
   output handle. Successful creation uses the ordinary compute/free API.

The pinned `gguf-py` writer generates only tiny pytest fixtures. Its uint8 input
must be shaped `[rows, groups*34]`, not the codec's `[rows, groups, 34]` array:
the writer expands the last byte axis to a logical quantized width. Native
inspection caught and corrected an initial `[128,3,5,1]` layout; the correct
fixture is `[384,5,1,1]` without changing any payload bytes.

Tests verify two counted kernel calls with one repack after deleting the source
file, exact direct-kernel outputs, registry cleanup, and rejection of contract,
transform, tensor, payload, truncated-file, and file-size mismatches. They run
in the shared group-scale control when `PRISM_CONVERTER_PYTHON` is set, including
both gate commands above. The standalone import fixtures remain toy-size. No
source or cached runtime is modified, and these file controls perform no
model-quality inference. The
later real-loader gate uses a separate transient full-size encoding, deleted
after parity; no second quantization policy/candidate is retained. This
explicit adapter does not make an ordinary Prism PQ2 model dispatch BitNet.

### Versioned Opt-In CPU Buffer

[native/prism_bitnet_runtime.h](../native/prism_bitnet_runtime.h) declares ABI v1
against the pinned Prism source revision. `prism_bitnet_cpu_runtime_init_v1`
returns the process-stable `JEV_BITNET_LOADER_V1` buffer and checks actual CPU
extra-buffer discovery. Wrong declared revision/ABI/storage geometry returns 1,
allocation failure 2, and already-cached discovery without this buffer 7.
Repeated successful initialization returns the same pointer exactly once.
Initialize on one thread before any CPU discovery and retain the bridge library
until all CPU users are finished. A caller revision string is not binary hash
attestation; the pinned dependency provenance checks still matter.

The buffer owns each tensor's copied weight bytes, native trait, repacked lanes,
and exact FP16-derived scales. Only the bounded named layer-3 PQ2 weight is
accepted. Sequential uploads are validated and packed once when complete;
rewriting packed weights is refused. Ordinary `MUL_MAT` performs native A8 for
1 through 128 tokens, allowing the token count to change between graphs.
`prism_bitnet_cpu_tensor_status_v1` reports cumulative kernel calls/repacks and
status 8 for incomplete or invalidated uploads. Graph compute status alone is
insufficient: failed native execution fills output with NaNs, and the tensor
status must be checked. Nonfinite input rejection can recover on a later batch.

The isolated [loader control](../native/prism_bitnet_loader_control.cpp) tests
early/late initialization in fresh processes, chunked loading, 1/2/128-token
exact direct-kernel parity, one repack, input rejection/recovery, and immutable
or invalid payload refusal. It runs in the existing shared native control.
Zero-size loader dummy probes are intentionally refused so this discoverable
buffer cannot silently take over ordinary PQ2. Two initialized independent
backend graphs sharing its weight pass alongside ordinary PQ2 graphs, with
eleven total counted BitNet calls and one repack. This exercises read-only
steady-state use, not concurrent registration, loading, freeing, or unloading;
no ThreadSanitizer claim follows. Do not interleave
the legacy scoped bridge registrations with arbitrary external CPU graphs or
discovery. No production registry concurrency or complete MiMo load follows.

### Explicit Real-Loader Route

`prism_bitnet_cpu_loader_override_from_gguf_v1(path, revision, abi, overrides**)`
shares the bounded GGUF reader with standalone import and validates ternary
codes/scales before CPU registration. Success returns a library-owned, stable,
null-terminated public `llama_model_tensor_buft_override` array containing only
`^blk\\.3\\.ffn_down\\.weight$`. Missing/conflicting identity tags, transforms,
file/layout bounds, and invalid payloads retain the existing statuses; late
discovery returns 7. Every failure clears the output pointer. Call before CPU
discovery, use the same unchanged validated file, and retain the library for
all buffer/CPU users. The factory does not bind or attest a subsequently changed
or different model file. It still accepts only a single-weight file, not a full
MiMo artifact. It does not request a new fit or convert the model.

When `PRISM_CONVERTER_PYTHON` is set, the test compiles the actual pinned
`llama-model-loader.cpp`, support units, and backend registry into a temporary
loader-only executable linked to the existing CPU/base libraries. Explicit
source-revision and temporary build-version checks are retained. Real
`create_tensor`, context allocation, `init_mappings`, and `load_all_data` run
with `LLAMA_LOAD_MODE_NONE`; `MUL_MAT` then matches the direct grouped kernel
at one and two tokens with exactly two calls and one repack. An untagged file
selects ordinary CPU despite the custom buffer being first in its candidate
list. Exact regex tests reject other layers, prefix/suffix matches, and wildcard
punctuation. This is actual one-weight loader selection/upload/dispatch, not
construction of a valid full Qwen3.5 model or architecture prefill.

With `MIMO_PQ2_CODEC_TEST=1`, the same executable also tests a temporary GGUF
encoding of the hash-checked 4,096 x 12,288 frozen projection. Existing codes
and FP16 row/group scales are preserved exactly; the test file is deleted in
`finally`. No new retained candidate, full-model conversion, or held-out
inference follows. The streamed module still uses its existing owned-handle
backend, not this loader route. Complete native architecture hosting and native
error propagation remain separate gates; always inspect tensor status before
using graph output. The cached CLI bridge and upstream source/build are unchanged.

The bridge runtime controls passed GCC address/undefined-behavior/leak sanitizer
checks, including mixed graph execution and teardown. Clang 18's sanitizer
runtime is absent in this container; existing GCC was reused without installs.
These sanitizer controls do not instrument the cached GGML libraries or the
temporary real-loader units and do not establish lifecycle race safety.

## Native Hosting Preflight and Conversion Plan

The resource planner in [embedded_jev/inventory.py](../embedded_jev/inventory.py)
adds `native_reference_plan` to the existing reconciled inventory. It reads only
bounded metadata/headers, excludes vision and optional MTP, and retains both
full vocabulary matrices. It is not conversion authorization, a measured
converter peak, or a complete GGUF size prediction. Reproduce it with:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m embedded_jev.inventory \
   --local-dir "$cache/models/mimo-2367e865d009c13ac81713a2878291d33ab28177" \
   | jq '.native_reference_plan'
```

Observed on 2026-10-01, before the vocabulary-only preflight:

| Quantity | Bytes | Interpretation |
| --- | ---: | --- |
| 427 text source tensors | 17,907,606,528 | 16.678 GiB; source-dtype payload floor, not final GGUF/RSS |
| Excluded vision | 912,020,960 | No vision conversion or inference in this stage |
| Optional MTP payload | 0 | Configured MTP does not imply stored tensors |
| Full input/output vocabulary matrices | 4,068,474,880 | Both retained; no selected-row loader assumption |
| Largest source tensor | 2,034,237,440 | A vocabulary matrix, 1.895 GiB |
| Largest FP32 tensor | 4,068,474,880 | 3.789 GiB conversion staging basis |
| Largest source plus FP32 copy | 6,102,712,320 | 5.684 GiB, before encoded output and other allocations |
| Above plus BF16 encoded output | 8,136,949,760 | 7.578 GiB illustrative concurrent-copy budget, not measured peak |
| All text values in FP32 | 35,815,213,056 | Do not assume an eager/all-FP32 conversion fits |
| Host available RAM | 22,345,261,056 | 20.811 GiB, time-sensitive; not a target-device budget |
| Host swap / currently used | 8,589,934,592 / 0 | Swap is not working-memory acceptance evidence |
| Cache filesystem available | 336,891,891,712 | 313.755 GiB, time-sensitive; recheck before conversion |

The cgroup memory limit read `max`; host availability still applies. Pinned
converter `prepare_tensors` routes source BF16 through FP32 and forces vectors,
norms, and SSM convolution weights to F32. `--outtype bf16` is therefore not a
uniform dtype promise. Qwen3.5 value transforms and generated metadata/alignment
must be audited before final payload accounting. The inventory's FP16 KV and
FP32 recurrent example is 134,217,728 and 50,331,648 bytes at 4,096 tokens/one
sequence; convolution, graph/scratch, output logits, and allocator costs are
excluded. Initial native controls should use at most 128 tokens, one sequence,
and one CPU thread for the bounded BitNet trait; this is not an edge benchmark.

### Isolated Full Runtime Build

[native/CMakeLists.txt](../native/CMakeLists.txt) builds the complete pinned
llama library, GGML registry/dynamic helper, bridge, and existing control against
the immutable cached CPU/base libraries. It refuses a different/dirty tracked
source revision, native non-AVX2 hosts, and build directories inside the pinned
source/build. It resolves one root-relative upstream CMake helper include in
memory; the actual cached helper/source files are not edited. No new source
copy, environment install, GGML CPU rebuild, or model conversion is required.

```bash
runtime="$cache/native/jev-prism-runtime-v1-build"
cmake -S native -B "$runtime" -G Ninja \
   -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++-18 \
   -DPRISM_SOURCE_DIR="$cache/native/prism-source" \
   -DPRISM_GGML_LIBRARY_DIR="$cache/native/prism-build/bin"
cmake --build "$runtime" --parallel 4
ctest --test-dir "$runtime" --output-on-failure
```

The build occupied approximately 15.3 MB, without model data. Its two CTests
pass actual native-A8/BitNet arithmetic, mixed graph concurrency, early discovery,
and late refusal while linked to the full llama library. This is build/runtime
compatibility, not complete MiMo weight loading or a native decoder prefill.

### Guarded Vocabulary Preflight

The lightweight `native/converter-venv` lacks Transformers for the full CLI.
The already-installed `dense-venv` can run the pinned converter; no packages
were installed. With offline flags, `--vocab-only --outtype bf16 --no-nextn`
creates only a temporary vocabulary/metadata GGUF. The test wraps safetensors
to reject all `get_tensor`/`get_slice` source-weight access, confirms zero GGUF
weight tensors, invokes the full public native load API with `vocab_only=true`,
and verifies exact HF/native IDs for the non-thinking rendered prompt and
single-token A/B/C labels. No context decode or answer generation occurs. The
test deletes the file in `finally` and compares packaged-bridge cached dependency
paths/SHA-256 hashes with the established bridge provenance.

Add these flags to the existing optional native gate after building:

```bash
MIMO_NATIVE_VOCAB_TEST=1 \
MIMO_PRISM_RUNTIME_BUILD="$cache/native/jev-prism-runtime-v1-build" \
MIMO_DENSE_PYTHON="$cache/dense-venv/bin/python" \
MIMO_LOCAL_DIR="$cache/models/mimo-2367e865d009c13ac81713a2878291d33ab28177"
```

### Bounded Synthetic Native Prefill

The existing optional native test writes one temporary four-layer Qwen3.5 GGUF
using the cached lightweight GGUF environment. It has two recurrent and two
full-attention layers, width 32, FFN width 256, and a 64-entry toy BPE vocabulary.
Attention outputs are zero; layer 3 has nonzero gate/up/down arithmetic. The
dense case stores the down projection in F32; the BitNet case stores its exact
ternary codes and two independent FP16 scales per output row as PQ2 bytes.
This is synthetic test data, not another fitted/retained candidate or source
MiMo conversion. No source model weights are opened for this fixture.

`--prefill-control` and test-only `--prefill-bitnet-control` load the model through
the real public API, use one CPU thread and at most 128 batch/ubatch tokens, and
prefill numeric tokens `[3,5,7]` without sampling or an answer decode. The pinned
context rounds its 128 requested context slots to 256. The BitNet control uses an
explicit exact layer-3 override, a 1 MiB file cap, and four-layer/32 x 256 geometry
checks. It does not use or relax the public one-tensor metadata-gated factory.
Its internal model-tensor accessor is a pinned test dependency, not a public API.

Independent NumPy RMS/SiLU/FFN/head math checks final logits; the BitNet reference
uses nearest-even group-128 A8 and integer group partials with exact stored FP16
scales. Typed IDs `inspect/edit/ask` map to synthetic numeric slots `11/17/23`.
These are not genuine MiMo label tokens or task-quality evidence. Selected-label
softmax is conditional on these options, not calibrated confidence.

Each fixture checks full, split 1+2-token, reset-after-another-prompt, and reordered
option execution. BitNet has one counted kernel dispatch per successful decode
and one weight repack per model load, including context reuse. Hybrid position
bounds report the intersection of KV/recurrent retained ranges: `[2,2]` here,
and an empty range immediately after reset. Malformed +2 codes, negative/NaN
FP16 scales, NaN head weights, and unknown modes must fail without stdout scores.
Unknown command flags and missing/extra arguments are also rejected before CPU
discovery, rather than falling through to an unrelated successful default report.
Model/context/backend cleanup is scoped; files are removed even on failure.
The existing converter-name gate also uses bounded local headers when
`MIMO_LOCAL_DIR` is set, avoiding unnecessary HTTPS after restart.

After the versioned build above, reproduce just this bounded control with:

```bash
PRISM_SOURCE_DIR="$cache/native/prism-source" \
PRISM_CONVERTER_PYTHON="$cache/native/converter-venv/bin/python" \
MIMO_PRISM_RUNTIME_BUILD="$cache/native/jev-prism-runtime-v1-build" \
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
   tests/test_prism_native_control.py -k tiny_qwen35_native_prefill
```

Both parameterized cases pass, including eight successful native invocations and
fourteen refusal checks. No nonzero recurrent/attention-reference parity, native MiMo
weight loading, full-model file policy, quality, or target benchmark follows.

### Staged Approval Gates

1. **Completed, no weight conversion:** reconciled header budgets, isolated full
   runtime build, dependency provenance, guarded vocabulary-only conversion,
   real native/HF prompt and label parity, and existing one-tensor BitNet proofs.
   Separate synthetic native hybrid prefill, nonzero BitNet FFN arithmetic,
   chunk/reset state, typed order, and no-score rejection controls also pass.
2. **Requires separate bulk-conversion approval:** create one text-only BF16
   native reference from the existing snapshot, retaining full embedding/head,
   excluding vision/MTP, leaving the sole ternary candidate unchanged. Use lazy
   conversion, not `--no-lazy`; plan a temporary output/cleanup path, refresh
   RAM/disk, measure actual peak/dtypes/bytes, and keep a filesystem/RAM reserve.
   The lazy CLI and `--use-temp-file` are available, but their combination,
   split policy, and peak resource behavior are not certified by this preflight.
3. **After reference acceptance:** load the real architecture, create a bounded
   native context, prefill without sampling/generation, and compare against the
   existing engineering reference. Diagnose backend precision/graph differences
   before claiming exact parity or selecting numerical tolerances. Record memory,
   finite logits, recurrent/cache state, label mapping, and conditional scores.
4. **Then one projection only:** extend the versioned file policy to a complete
   model while binding the exact layer-3 identity payload and preserving every
   stored code/FP16 scale. The current factory deliberately requires a one-tensor
   file and cannot be applied unchanged to a 427-tensor model. Integrate native
   error propagation, single-thread/batch bounds, library lifetime, and counted
   BitNet dispatch; do not substitute stock PQ2/Q8_K as BitNet evidence.
5. **Still not authorized:** full-model ternary conversion, compensation
   promotion, another fitted/retained quantized candidate, held-out inference,
   calibrated confidence, vision, or deployment benchmarks. Broader quantization
   requires separate resource, arithmetic, runtime, and representative quality
   gates. A successful dense native reference is not ternary quality evidence.

## Single-Projection Native Fixture

One searched-FP16 group-128 layer-3 FFN-down candidate is retained under
`$cache/quantized/layer3-ffn-down-rtn-searched-fp16`. The pinned source shard
was rehashed before quantization (SHA-256
`7a0486565f06d25ac4628e9dba470dc3f604353471d240d5a0bf7128f64df396`).
The non-pickle NumPy arrays are 12,583,040 bytes of BitNet-derived packed
codes (SHA-256 `34ddd9e802c9024a3d62688a90a3929d189a66502df41c48b06a6a53fe32670d`)
and 786,560 bytes of FP16 row/group scales (SHA-256
`25f919c92d05a09220970519ce06dbe991efbc3f9ba16d49240da54c01517913`),
plus a 697-byte JSON manifest. This is **one** experimental projection,
not stock I2_S, GGUF, or a full MiMo checkpoint. No second quantization was kept.

The writer refuses an existing output directory. To run its saved-candidate
parity check (rather than recomputing the same ternary weights), use:

```bash
candidate="$cache/quantized/layer3-ffn-down-rtn-searched-fp16"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
   "$cache/dense-venv/bin/python" -m embedded_jev.streamed_text \
   --local-dir "$snapshot" --layers 32 --label-count 2 \
   --native-ffn-library "$cache/native/bitnet_group_scale_probe.so" \
   --projection-artifact "$candidate"
MIMO_IN_MODEL_NATIVE_TEST=1 MIMO_PROJECTION_ARTIFACT="$candidate" \
   MIMO_DENSE_PYTHON="$cache/dense-venv/bin/python" MIMO_LOCAL_DIR="$snapshot" \
   PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
   tests/test_dense_probe.py -k streamed_text_substitutes_one_bitnet_derived_ffn
```

The full 32-layer saved-candidate output hash and A/B scores exactly match
fresh in-memory RTN, with the same native/integer reference parity. The
manifest and arrays are size/hash checked; invalid packed trits are rejected
even if array hashes are recomputed. A saved-candidate run omits the
100,663,296-byte BF16 FFN-down tensor from materialization and builds that
module directly from packed codes/scales. Its optional regression test forbids
the corresponding safetensors `get_tensor` call and still requires exact
saved/fresh hidden hashes and selected scores. Other model weights and
source-header checks remain unchanged. No approved group-scale GGUF codec or
model loader can consume this fixture directly, and its one-prompt score
parity is not a quantization quality gate.

## Post-Restart Review

On 2026-09-30 all four cached MiMo shards and both retained projection arrays
again matched their recorded SHA-256 values. The CPU environments and the
single-candidate cache survived the reboot; no model files were fetched again.

The review corrected three implementation boundaries. The projection loader
now rejects actual array-size mismatches before hashing and uses bounded,
duplicate-key-rejecting JSON with strict integer schema fields. Full-vocabulary
label mass now obtains its numerator from the same batched logits as the
denominator: separately accumulated FP32 label dots could previously yield
mass above one for valid, dominating labels. The native matvec now rejects
overflowing dimensions, and the batch entry point checks output bytes rather
than just float-element counts. The raw C API still requires caller-owned
buffers large enough for the accepted dimensions.

The opt-in streamed-text test also compares the four-layer activation and final
hidden hashes against the normal Transformers forward, rather than comparing
only staged runs. These checks strengthen arithmetic and reload evidence;
they do not provide calibration data, decision-quality validation, or a
registered ternary model format.

## Tokenizer-Only Label Probe

The [label probe](../embedded_jev/label_probe.py) verifies that the pinned MiMo
template's non-thinking assistant prefix has the same token IDs when rendered
or tokenized by Transformers, then checks that each A-P label extends that
exact prompt by one distinct, non-special token. It does not load weights or
generate an answer. Its offline tests run in the base container, but the live
probe needs a separate tokenizer-only environment:

```bash
export UV_CACHE_DIR="$HOME/embedded-jev-cache/uv"
uv venv "$HOME/embedded-jev-cache/tokenizer-probe" --python /opt/venv/bin/python
uv pip install --python "$HOME/embedded-jev-cache/tokenizer-probe/bin/python" 'transformers==5.12.1' 'jinja2==3.1.6'
export HF_HOME="$HOME/embedded-jev-cache/huggingface"
"$HOME/embedded-jev-cache/tokenizer-probe/bin/python" -m embedded_jev.label_probe --fetch
HF_HUB_OFFLINE=1 "$HOME/embedded-jev-cache/tokenizer-probe/bin/python" -m embedded_jev.label_probe
```

The opt-in `--fetch` mode checks file sizes before downloading only the pinned
`tokenizer.json` (at most 25 MiB), `tokenizer_config.json`, `config.json`, and
`chat_template.jinja` (each at most 1 MiB). The default mode requires those
files to be cached and does not fetch anything. Both modes reject unsupported
templates, label merges, duplicate/special label IDs, and missing provenance;
the JSON report hashes the four files and records library versions. The isolated
environment is a probe, not the hash-locked research environment or a validated
ML/runtime dependency lock. The verified text-only example yields label IDs
32-47 (A-P) under Transformers 5.12.1. Recheck every new prompt and native
tokenizer independently; no processor/vision or native inference parity follows
from this result.

The pinned [inventory](../embedded_jev/inventory.py) also compares the image
processor settings with their standalone config and requires the standalone
video settings to agree with their nested counterparts (which include additional
defaults). To exercise the actual text-only processor under the agreed budget,
add CPU-only dependencies to the isolated environment:

```bash
uv pip install --python "$HOME/embedded-jev-cache/tokenizer-probe/bin/python" 'pillow==12.1.1'
uv pip install --python "$HOME/embedded-jev-cache/tokenizer-probe/bin/python" --index https://download.pytorch.org/whl/cpu 'torch==2.10.0+cpu' 'torchvision==0.25.0+cpu'
"$HOME/embedded-jev-cache/tokenizer-probe/bin/python" -m embedded_jev.label_probe --fetch --processor
HF_HUB_OFFLINE=1 "$HOME/embedded-jev-cache/tokenizer-probe/bin/python" -m embedded_jev.label_probe --processor --fixture tests/fixtures/agent_tool_smoke.json
```

Processor mode allows only three additional pinned JSON files, each capped at
1 MiB. On the checked text-only prompts, `Qwen3VLProcessor` produces exactly
the tokenizer's prompt IDs, a matching attention mask, and all-zero
`mm_token_type_ids`; A-P each append one distinct token. The external environment
and caches totaled about 979 MB after installation; no weights were fetched.
The five labeled [fixture cases](../tests/fixtures/agent_tool_smoke.json) cover
an irrelevant-context perturbation, option order, and missing upload permission.
They are synthetic engineering checks, excluded from calibration and benchmark
claims. The CLI hashes the bounded fixture and reports each case's prompt hash,
expected option ID/label, and token IDs. Actual image inputs, native backend
tokenization, model judgments, and decision calibration remain unverified.

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

On 2026-09-28, WSL/Podman GPU builds encountered APT `Hash Sum mismatch`
errors for both `noble-backports/multiverse` and `noble-security/restricted` over
HTTP. Removing backports did not resolve the broader issue. The shared image
now uses HTTPS for both official Ubuntu APT sources while retaining all suites,
components, and APT signature/hash verification. HTTPS endpoint reachability was
checked in the CPU container; both previously failing by-hash index URLs also
returned content with the expected SHA-256 over HTTPS there. A subsequent
WSL/Podman build succeeded with the HTTPS sources and all pockets enabled.
Do not disable APT integrity checks to work around stale mirror data.

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
compatibility on the host. The opt-in
[Podman/WSL GPU profile](../.devcontainer/gpu/devcontainer.json) reuses the
CPU image and requests the same `nvidia.com/gpu=all` CDI device, `/dev/dxg`,
`/usr/lib/wsl`, and `label=disable` options as the user's working container.
It does not mount WSLg, forward ports, clone llama.cpp, or replace the CPU
default. Only choose it on a host with those WSL paths and Podman NVIDIA CDI
configured; it is not a generic Docker or native-Linux GPU profile.

On that host, from the workspace root, validate and launch explicitly:

```bash
devcontainer read-configuration --workspace-folder . --config "$PWD/.devcontainer/gpu/devcontainer.json" --docker-path podman
devcontainer up --workspace-folder . --config "$PWD/.devcontainer/gpu/devcontainer.json" --docker-path podman
devcontainer exec --workspace-folder . --config "$PWD/.devcontainer/gpu/devcontainer.json" --docker-path podman /usr/lib/wsl/lib/nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

With that GPU container running, check the CUDA Driver API without rebuilding
or installing Torch (the repository is bind-mounted into the container):

```bash
devcontainer exec --workspace-folder . --config "$PWD/.devcontainer/gpu/devcontainer.json" --docker-path podman env PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python -m embedded_jev.gpu_probe
```

The [driver probe](../embedded_jev/gpu_probe.py) calls `cuInit`, enumerates
devices, and reports each device's compute capability and total driver-reported
memory. On the WSL/Podman GPU container it reported one RTX A3000 Laptop GPU,
compute capability 8.6, and 12,884,377,600 bytes of driver-reported memory.
For a separate, opt-in four-byte host/device memory round trip, append
`--memory-round-trip` to that command. This mode creates and destroys a CUDA
context and frees its allocation; it passed on the WSL/Podman GPU container,
returning `"memory_round_trip_bytes": 4`. Offline fake-driver tests also cover
cleanup on copy failure. Neither check runs CUDA kernels or proves a
compatible GPU Torch wheel or usable full-model quantization.

For an actual GPU compute gate, use a separate environment in the GPU profile's
persistent cache. Check free disk space before installing; the official
CPython 3.12 Linux `torch==2.10.0+cu128` wheel is 916,856,347 bytes by HTTP
header, **excluding dependencies and cache duplication**. WSL/container `df`
shows the virtual Linux filesystem's reported free space, not necessarily the
physical free space on the Windows drive storing its VHDX and Podman data.
Check that Windows drive too. From the WSL host workspace root, with the GPU
container already running:

```bash
powershell.exe -NoProfile -Command 'Get-PSDrive -PSProvider FileSystem | Select-Object Name,Used,Free'
CACHE=/home/vscode/.cache/huggingface/embedded-jev/gpu
UV_CACHE_DIR="$CACHE/uv"
devcontainer exec --workspace-folder . --config "$PWD/.devcontainer/gpu/devcontainer.json" --docker-path podman df -h /home/vscode/.cache
devcontainer exec --workspace-folder . --config "$PWD/.devcontainer/gpu/devcontainer.json" --docker-path podman test -w /home/vscode/.cache/huggingface
devcontainer exec --workspace-folder . --config "$PWD/.devcontainer/gpu/devcontainer.json" --docker-path podman env UV_CACHE_DIR="$UV_CACHE_DIR" /opt/venv/bin/uv venv "$CACHE/torch-cu128" --python /opt/venv/bin/python
devcontainer exec --workspace-folder . --config "$PWD/.devcontainer/gpu/devcontainer.json" --docker-path podman env UV_CACHE_DIR="$UV_CACHE_DIR" /opt/venv/bin/uv pip install --python "$CACHE/torch-cu128/bin/python" --index https://download.pytorch.org/whl/cu128 'torch==2.10.0+cu128'
devcontainer exec --workspace-folder . --config "$PWD/.devcontainer/gpu/devcontainer.json" --docker-path podman env XDG_CACHE_HOME="$CACHE" PYTHONDONTWRITEBYTECODE=1 "$CACHE/torch-cu128/bin/python" -m embedded_jev.torch_probe
```

The top-level `/home/vscode/.cache` mount can be root-owned even when
`/home/vscode/.cache/huggingface` is writable by `vscode`. Explicit `UV_CACHE_DIR`
keeps uv from failing before installation; keeping the venv beneath that
writable child preserves it across container restarts without privilege changes.
If the write test fails, stop and inspect the mount permissions rather than
using `sudo` or changing ownership of unrelated caches.

The [Torch probe](../embedded_jev/torch_probe.py) requires a CUDA-enabled wheel
and performs one 4x4 FP32 matmul on `cuda:0`, synchronizes, and compares the
result against the input. On 2026-09-28 the WSL/Podman GPU container passed this
check with `torch==2.10.0+cu128`, CUDA 12.8, and compute capability 8.6. The
isolated venv printed an optional NumPy-missing warning on import; this pure
Torch operation still passed. Add a separately pinned NumPy dependency only
when a later workload needs it. This smoke does not download any model or prove
a full-model memory plan, ternary quality, or BitNet CPU execution. The research
venv remains hash-locked and untouched; the larger quantization environment
still needs a separate lock and validation.

The GPU profile runs the existing CPU smoke check and fails post-create if
`/dev/dxg` or `nvidia-smi` is unavailable. On 2026-09-28, the WSL/Podman host
completed the image build and post-create smoke check, then reported an
NVIDIA RTX A3000 12GB Laptop GPU, 12,288 MiB VRAM, and driver 595.95 via
`nvidia-smi` inside the container. Post-create verifies device visibility; the
separate Torch smoke above verifies a small CUDA computation, not a model or
quantizer. The shared CPU image does **not** install CUDA, GPU Torch, or `nvcc`;
device visibility alone does
not establish usable quantization, native BitNet execution, or GPU performance.
No weights, drivers, or GPU packages are downloaded by this profile.
Compiling CUDA extensions additionally requires a matching CUDA toolkit and
`nvcc`; `--gpus all` alone does not provide them.

Quantize on the workstation and infer on the edge. Full-model BF16 weights,
activation banks, factors, and temporary exports require more memory/storage than
the final packed model. Use the inventory to size these before downloads.
On 2026-09-28 the WSL host reported 29 GiB RAM, 21 GiB available, and 8 GiB
swap; the user reported 48.3 GB free on the Windows drive storing the WSL
virtual disk. These values can change, and WSL `df` is not a substitute for
Windows free space. The checkpoint's 18.82 GB BF16 weights alone occupy about
17.53 GiB in memory; loading them all at once leaves insufficient reliable
headroom for activations, factors, and runtime scratch. Plan a streamed or
offloaded reference and bound temporary disk copies before weight downloads.

## Native Build Discipline

Clang 18 satisfies BitNet's documented compiler prerequisite. Do not assume a
successful C++ smoke test implies BitNet itself has been built. Native inference
and model-loading smoke tests are later roadmap gates.

The [standalone AVX2 group-scale fixture](../native/bitnet_group_scale.cpp)
adapts BitNet's pinned 128-value packed-code integer dot for one row/group scale
at a time. Run its [golden tests](../tests/test_bitnet_group_scale.py) with:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_bitnet_group_scale.py
```

The tests compile in temporary storage with Clang 18 and require an x86-64 CPU
with AVX2. Packed weights use 32 bytes per group plus a separate FP32 scale;
this is neither PTQ1_0 nor a loadable stock BitNet I2_S model. Inputs are
already signed A8 with supplied activation scales. A serial multi-token wrapper
reuses the same matvec path and has independent three-token golden cases;
it is not fused GEMM. The separate [activation helper](../embedded_jev/activation.py)
applies explicit signed FP32 Hadamard blocks and symmetric per-token/group A8
in Python; toy tests feed that result to the native batch kernel. Run those
CPU checks with `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_activation.py`.
The separate [ternary RTN baseline](../embedded_jev/ternary.py) saves codes
and representable FP16 scales per output row/group, and its
[offline tests](../tests/test_ternary.py) verify reconstruction and reject
unsupported groups. Native golden tests check outputs using only the saved
codes and FP16 scales cast to FP32. This is an in-memory toy artifact, not a
serialized PTQ1_0/PQ2_0 codec or a GPTQ quality claim.
The same module has an opt-in 256-column toy GPTQ-style compensated traversal
based on the [pinned reviewed update](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py).
It uses token-normalized curvature and bounded damping without a pseudoinverse,
then retains FP16 row/group scales before updating subsequent columns. Its
optional processing block size must be a multiple of the scale-group size;
smaller aligned blocks reproduce the full-width toy's codes and scales. Offline
tests demonstrate one correlated reconstruction improvement and verify native
output parity from only its saved artifact; they do not establish model quality.
The [toy archive](../embedded_jev/ternary_artifact.py) stores those arrays with
`allow_pickle=False` and checks a versioned hashed manifest before loading.
It permits an explicit signed-Hadamard transform record or identity, refuses
malformed/oversized payloads, and is limited to 1 MiB. The
[offline artifact tests](../tests/test_ternary.py) and native fixture exercise a
saved-and-reloaded path. This is not a model converter or native packed format.
For candidate Prism v1 projection names, an allowlisted mapping now derives
expected input widths from the pinned MiMo header inventory rather than a
slice shape. The pinned converter's handling of the model's
`model.language_model` prefix is handled by its shared filter before the
Qwen3.5 tensor map: raw MiMo names return no match, but selected FFN/attention
names map after that filter. Across all 760 pinned weight-index names, the
text-side filter excludes 333 vision entries and all 427 retained text names
map after 24 source-derived `.dt_bias` to `.dt_proj.bias` renames. Run the
optional metadata-only pinned source
test with `PRISM_SOURCE_DIR` and `PRISM_CONVERTER_PYTHON` (an isolated Python
environment with NumPy 2.2.6, PyYAML 6.0.3 and CPU Torch 2.10.0). The same
opt-in test also runs pinned Qwen3.5 method bodies on tiny synthetic QKV/Z,
alpha/conv1d, A-log and dt-bias tensors, comparing row order against an
independent NumPy oracle. The folded `out_proj` case keeps grouped V columns
and sets its runtime permutation flag; the unrotated case reorders columns.
The pinned `add_hadamard_metadata` method also emits the grouped-V GGUF bool
writer call for a toy folded `ssm_out` manifest, not for the ungrouped case.
The pinned writer/reader round-trips the typed bool and folded weight name in
a no-tensor metadata-only GGUF.
Our toy metadata checker still rejects `ssm_out` pending verified head geometry.
Do not export or load a model GGUF from this name/sampled-value check without
full converter and native loader parity tests. The index-wide test transfers
only bounded JSON, not weight payloads.
No runtime-side rotation/A8, actual model block, loader/graph, ARM/RISC-V path,
or performance measurement is implemented by this proof.
The [upstream MIT notice](../native/BitNet-LICENSE.txt) is included with the
derived source.

For a direct model-free control, a shallow checkout of BitNet parent
`0b341e582afbf9e1011f24744b554c96a3477eb5` with llama.cpp gitlink
`390c307752ab78fd8189f359d6954c9ba1be74af` was made outside this repo
under `$HOME/embedded-jev-cache/bitnet-source` (about 208 MB). Clang 18
configured it with `-DGGML_NATIVE=OFF -DGGML_AVX2=ON` and built only the
`ggml-cpu` target. The initial build stopped because `/home/vscode/.cache/ccache`
was root-owned; retrying with `CCACHE_DIR=$HOME/embedded-jev-cache/ccache`
succeeded. No server or model weights were built/downloaded. To rerun the
optional [real-fork kernel controls](../tests/test_bitnet_native_control.py)
against that exact checkout:

```bash
BITNET_SOURCE_DIR="$HOME/embedded-jev-cache/bitnet-source" BITNET_GGML_CPU_LIBRARY="$HOME/embedded-jev-cache/bitnet-build/bin/libggml-cpu.so" LD_LIBRARY_PATH="$HOME/embedded-jev-cache/bitnet-build/bin" PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_bitnet_native_control.py
```

The test verifies both commit IDs, then calls the exported upstream
`ggml_vec_dot_i2_i8_s` symbol and compensates the activation sum separately
for each group; it skips in the ordinary offline suite without the source and
library environment variables. This particular CMake graph compiles fork
`quants.c` and BitNet's LUT source, **not** `ggml-bitnet-mad.cpp`. The build
emits upstream warnings, including type-enum initializer overrides; parser,
model inference, and runtime performance still need separate validation.
The optional test also compiles the
[tiny GGML graph fixture](../native/bitnet_ggml_graph_smoke.cpp) against the
same pinned libraries: zero/+1/-1/zero I2_S rows output [0, 128, -128, 0].
That proves this single model-free graph path works; it does not prove GGUF
loading, per-row/group scales inside the graph, or MiMo runtime dispatch.
Its opt-in `--groups` mode runs two separate group-128 graph evaluations and
applies distinct row/group scales to their partial sums outside GGML, yielding
[32, -320, -256, 64]. The optional native-control test verifies both modes.
This deliberately serial bridge preserves scales but is not an optimized
kernel or a Qwen3.5 graph integration.
Its `--batch` mode also evaluates two tokens (+1.0 and -2.0 F32 inputs),
confirming per-token dynamic A8 scale separation for this tiny graph. It
does not test KV/recurrent cache state or real model activations.

The pinned Prism release `prism-b10735-842b188` resolves to
`842b1880415d6f508f03b789e5ce70194def7bfd`. The original external
`$HOME/embedded-jev-cache` checkout disappeared on a container reopen. Use
the writable child of this CPU profile's persistent volume instead; different
devcontainer profiles may use different cache volumes. The pinned source and
model-free CPU library can be recreated without weights:

```bash
NATIVE_CACHE="$HOME/.cache/huggingface/embedded-jev/native"
mkdir -p "$NATIVE_CACHE"
git clone --depth 1 --branch prism-b10735-842b188 --single-branch --filter=blob:none https://github.com/PrismML-Eng/llama.cpp.git "$NATIVE_CACHE/prism-source"
git clone --depth 1 --filter=blob:none https://github.com/microsoft/BitNet.git "$NATIVE_CACHE/bitnet-source"
CCACHE_DIR="$NATIVE_CACHE/ccache" cmake -S "$NATIVE_CACHE/prism-source" -B "$NATIVE_CACHE/prism-build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18 -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF
CCACHE_DIR="$NATIVE_CACHE/ccache" cmake --build "$NATIVE_CACHE/prism-build" --target ggml-cpu --parallel 4
PRISM_SOURCE_DIR="$NATIVE_CACHE/prism-source" PRISM_GGML_CPU_LIBRARY="$NATIVE_CACHE/prism-build/bin/libggml-cpu.so" BITNET_SOURCE_DIR="$NATIVE_CACHE/bitnet-source" PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_prism_native_control.py
```

Clone only when those directories are absent and verify both `git rev-parse HEAD`
values against the pins above and in [docs/sources.md](sources.md). This CPU
build's `libggml-cpu.so` SHA-256 is
`adaaacaf406df3700fb5f05cb5749990e9b17d410d91e13d0c58263ae971a150`;
it need not match a differently configured build. One optional test verifies
signed F32 FWHT dispatch. The other runs the BitNet-derived grouped dot as a
toy Prism `MAP_CUSTOM2` graph node after native FWHT and A8. Two graph
evaluations pass when input and sign leaves are re-uploaded between runs;
allocator-managed graph leaves cannot be assumed intact after compute. Packed
codes/scales are fixture-owned, not a loadable GGUF type or MiMo model hook.
The same test runs `--grouped-v`: a 64-wide, two-key-head/two-repetition
permutation from tiled to grouped order before signed 128-point FWHT and A8,
with independent scalar parity and the same repeated-graph check.
No source was vendored or MiMo shard needed for this toy graph check. The earlier BitNet
control model/build paths above describe the old container; recreate their
pinned submodule/library/model only if rerunning those opt-in controls.

The `llama` target was later built from the same pins (using the writable
`CCACHE_DIR`). A supported MIT-licensed native BitNet GGUF at revision
`a1f2f1c765812aa8af3f6eda4a313707064bba15` was downloaded outside the
repo, with 1,187,801,280 bytes and verified SHA-256
`4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`.
The optional [prefill control](../native/bitnet_prefill_control.cpp) calls the
version-matched `llama` API, marks only the last prompt position for logits,
and never samples an answer. One run produced 128,256 finite logits for 22
prompt tokens; a debugger stopped inside `llamafile_sgemm_i2s` on prefill.
To rerun the opt-in pinned model tests after setting the source/library variables
shown above, also set `BITNET_CONTROL_GGUF` to the verified cached GGUF:

```bash
BITNET_SOURCE_DIR="$HOME/embedded-jev-cache/bitnet-source" BITNET_GGML_CPU_LIBRARY="$HOME/embedded-jev-cache/bitnet-build/bin/libggml-cpu.so" BITNET_CONTROL_GGUF="$HOME/embedded-jev-cache/bitnet-control-model/ggml-model-i2_s.gguf" LD_LIBRARY_PATH="$HOME/embedded-jev-cache/bitnet-build/bin" PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_bitnet_native_control.py
```

Five opt-in checks cover the direct dot, small GGML graphs, model logits, and
debugger-confirmed I2_S dispatch. Load/prefill were about 410/439 ms in one
run, **not a benchmark**. `CPU_REPACK` fallback was reported by the loader;
verify projection dispatch/fallback counts and run paired workloads before
any speed claims. No MiMo weight shards or BitNet alternative model weights
were downloaded.
The no-sampling control also checks A-C as exact distinct one-token labels
(IDs 32/33/34) in its own prompt and returns conditional probabilities,
selected label, `max_option_probability`, allowed-label mass, and an explicit
`uncalibrated` status. A sample result selected A at about 0.553 conditional
probability while allowed-label mass was only 0.000070; this is not a trusted
decision, MiMo tokenizer parity, or TypeSafe Jev confidence.

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

On 2026-09-28, the opt-in WSL/Podman GPU profile built successfully with HTTPS
Ubuntu sources, started, and passed its post-create CPU smoke check. Inside the
container, `nvidia-smi` reported an RTX A3000 Laptop GPU (12,288 MiB, driver
595.95). This establishes device visibility only; CUDA computation, GPU Torch,
model loading, ARM64, and RISC-V remain untested.