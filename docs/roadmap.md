# Implementation Roadmap

This is a research plan, not a claim that the listed pipeline already runs.
Every model-quality and speed result must reference a frozen artifact, workload,
runtime, and hardware configuration. No full-model job starts merely because a
toy mathematical test passes.

## Milestone 0: Research Environment

Status: complete on Linux x86-64 using rootless Podman and Dev Containers CLI.

- [x] Inspect the supplied conversation and primary upstream sources.
- [x] Build the CPU-first Python 3.12 / Clang 18 development image.
- [x] Install hash-locked numerical/test dependencies.
- [x] Pass post-create numerical, C++17/OpenMP, and writable-path checks.
- [x] Pass nine model-free mathematical/accounting tests.
- [x] Document architecture, compatibility risks, and staged validation.
- [ ] Validate the image natively on ARM64.
- [ ] Configure an optional CUDA quantization environment against the actual host.

Not installed automatically: PyTorch/CUDA, Transformers, datasets, gguf, SemIf,
BitNet, Prism, or model weights. Avoid mixing their dependency and ABI requirements
before their revisions and intended roles have been selected.

## Milestone 1: Quantization Feasibility

Status: in progress. The pinned header inventory and byte estimates reconcile;
the pinned tokenizer and text-only processor agree on prompt IDs and A-P labels.
A single complete BF16 source snapshot was downloaded and SHA-256 verified on
2026-09-29; its local headers also reconcile, but no quantized model was kept.
A synthetic five-case agent/tool fixture verifies option mapping, not model
quality. Vision inputs, native tokenization parity, a real held-out decision
fixture, and a dense reference remain open. The quantization host exposes an
RTX A3000 Laptop GPU with 12,288 MiB VRAM; CUDA Driver API initialization and
a four-byte memory round trip passed. An isolated CUDA Torch 2.10.0 environment
also passed one 4x4 FP32 matmul. WSL reported 29 GiB RAM (21 GiB available),
8 GiB swap, and the user reported 48.3 GB free on the Windows drive. These
readings are time-sensitive; full-model fit, conversion scratch, and
target-device resource budgets still require a plan.

1. Obtain host and target resource budgets. Inspect metadata and safetensors
   headers for the pinned MiMo checkpoint without downloading all weight shards.
2. Write a tensor policy and whole-model byte estimate, including embeddings,
   output, vision, MTP, transform metadata, and retained precision.
3. Verify MiMo template rendering, true single-token labels, and processor inputs.
4. Freeze a small domain-relevant decision fixture with labels, missing-evidence
   cases, and perturbations, isolated from all tuning datasets.
5. Load the dense reference in a separately pinned ML environment. Measure direct
   decision accuracy and, if resources allow, a Q8/Q4 runtime control. Keep these
   controls distinct from the ternary deliverable.

Gate: complete tensor inventory, no missing/mismatched required weights, exact
tokenization parity, finite reference logits, a defined quality target, and a
credible host-memory budget. If dense MiMo itself fails the task in non-thinking
mode, quantify that before attributing failure to quantization.

## Milestone 2: Ternary Reference

Toy linears now pass matching explicit-sign FP32 Hadamard rotation at block
sizes 128 and 1,024, followed by independent per-token/group-128 dynamic A8
preparation. A deterministic [ternary RTN baseline](../embedded_jev/ternary.py)
retains codes and FP16 row/group scales; toy reconstruction and native-kernel
outputs match those stored values. A bounded 256-column GPTQ-style compensated
traversal, informed by a pinned reviewed implementation, now improves one
correlated toy reconstruction and passes saved-artifact native parity. Processing
blocks aligned to scale groups preserve the full-width toy's stored codes/scales.
This has not been validated on a MiMo decoder block or calibration distribution.
A deterministic, 1 MiB-capped toy archive round-trips saved codes, FP16 scales,
and explicit signed-Hadamard metadata into the native CPU fixture. Full GPTQ
scaling, native artifact export, and model-quality steps remain open.
A metadata-only check against pinned Prism v1 transform rules rejects a toy
weight with the wrong full-model logical width or conflicting sign/block data;
an explicit candidate map derives selected logical widths from reconciled MiMo
headers. A pinned Prism Qwen3.5 name-map check returns no direct match for
MiMo's `model.language_model.layers.*` prefix, but the converter's shared
filter removes `language_model.` before mapping. Selected FFN/attention paths
map in a model-free source check. The same pinned filter and name map now cover
all 760 index entries: 333 vision-side names are excluded from the text pass,
and all 427 retained text names map after the converter's 24 `.dt_bias` renames.
This is text **name-path** coverage only; tensor-value conversion, vision export,
full GGUF export, and loader/tokenizer parity remain open. A separate pinned
CPU Torch method check now verifies QKV/Z value-head reorder and A-log/dt-bias
transformations on tiny synthetic tensors, plus alpha/conv1d row order and the
folded-versus-unrotated `out_proj` column/flag rule. This is not full MiMo
tensor geometry, complete SSM/vision value coverage, or a native graph test.
A bounded reader fetched only 2,048 BF16 bytes from four rows and two groups
of an eligible pinned MiMo projection. On disjoint synthetic Gaussian inputs,
an 11-candidate per-row/group FP16 scale grid lowered local MSE for both RTN
and compensation, but searched RTN still beat searched compensation on this
slice. These are exploratory numeric checks, not real activation or task quality.
A separate opt-in test now takes those four real weight rows through explicit
toy signed rotation, searched FP16 ternary scales, saved-artifact reload,
synthetic A8 preparation, and the standalone native AVX2 batch kernel. Dense
rotation and stored-artifact native parity pass; this is not a MiMo block or
Qwen3.5/BitNet runtime graph.

1. Implement matching weight/input rotations with per-tensor transform records.
2. Verify dense equivalence on toy linears, then an actual full-attention block
   and a linear-attention block before rounding weights.
3. Adapt a reviewed GPTQ reference for fixed per-row/group ternary scales.
4. Test all-zero groups, singular statistics, damping retries, FP16 scale rounding,
   tails, nonfinite inputs, deterministic traversal, and export reconstruction.
5. Quantize selected blocks using representative real activations, comparing
   unrotated RTN, rotated RTN, and rotated compensated ternary reconstruction.
6. Sweep a bounded set of group sizes, damping values, rotation placements, and
   calibration lengths. Use validation data, not the final test set, for selection.

Gate: exact code/scale reconstruction within declared scale precision, finite
factors, dense-rotation parity, and a useful measured block-quality improvement.
Do not start all layers while scales or activation transforms remain ambiguous.

The first real-code tests belong beside the modules they test. The current
[docs/test_research_math.py](test_research_math.py) is only an executable audit.

## Milestone 3: BitNet Runtime Proof

This milestone is mandatory, not an optional optimization after release. It can
progress alongside the later part of Milestone 2 after the container is ready.
An isolated [AVX2 group-scale fixture](../native/bitnet_group_scale.cpp) now
compiles and passes scalar-reference golden tests for group-128 ternary dots,
including serial three-token batches at 256, 4,096, and 12,288 input widths.
Toy Python FP32 rotation and dynamic A8 also feed the compiled batch path with
scalar parity at 128- and 1,024-point transform sizes. This does **not** satisfy
runtime transform/A8 execution, native model dispatch, optimized batch, or the
model-quality gate below. A model-free build of the pinned BitNet fork's
`ggml-cpu` target now succeeds, and optional tests call its **actual** I2_S
dot symbol with group-sum compensation. A separate four-row, 128-input GGML
graph smoke returned exact zero/positive/negative ternary outputs. No supported
BitNet checkpoint was downloaded or loaded at that stage, and MiMo dispatch
remains untested. Subsequently a pinned MIT-licensed 1.19 GB native BitNet
control GGUF loaded: 22 prompt tokens produced 128,256 finite final-position
logits with zero generated tokens. A debugger stopped inside the fork's
`llamafile_sgemm_i2s` during prefill. This proves control-model I2_S dispatch,
not group-scaled MiMo runtime integration or a performance result.
The control also verifies A-C as exact one-token continuations and reports
conditional option scores without generation. Allowed-label mass on one sample
was only about 0.000070, so its conditional maximum is not calibrated
confidence; MiMo/SemIf backend integration remains open.
Running that same graph serially for two 128-value groups and applying distinct
row/group scales outside the graph returned exact toy outputs. This is a
correctness bridge, not an integrated or optimized group-scale operator. The
same tiny graph also passed two token columns with different dynamic A8 scales
(+1.0 and -2.0 inputs); position/recurrent state is not involved.
The separately pinned Prism Qwen3.5-capable fork now builds on x86-64. Its
native CPU FWHT graph matches dense signed Hadamard on two tokens, and a
toy `MAP_CUSTOM2` graph node now schedules dynamic A8 and the BitNet-derived
group-scale kernel with scalar parity. Two repeat evaluations pass when graph
input and sign leaves are re-uploaded. An optional grouped-V mode also checks
the two-key-head/two-repetition tiled-to-grouped activation permutation before
signs/FWHT against independent scalar math. The node uses fixture-owned codes/scales,
not a registered loadable GGUF tensor or MiMo loader/dispatch parity.

1. Pin Microsoft BitNet and its submodules; build the documented native control
   with Clang 18. Use a supported model only after confirming download budget.
2. Expose prefill-only final logits through verified llama APIs and confirm no
   answer tokens are sampled. Confirm real I2_S dispatch with profiling/counters.
3. Freeze a small group-scaled ternary tensor fixture from Milestone 2.
4. Test a group-scale-preserving BitNet-kernel adaptation in a Qwen3.5-capable
   runtime. Compare every output with the portable reference for both one-token
   and multi-token batches, including activation preparation and rescaling.
5. Evaluate I2_S-style MAD and TL/T-MAC paths as distinct candidates. Include
   shapes 4,096 and 12,288 and the actual discovered attention widths.
6. Record format, architecture-specific packing, SIMD path, fallback counts,
   wall time, and resident bytes. Test misalignment and remainder handling.

Gate: demonstrable Microsoft BitNet-derived execution, no loss of group scales,
correct rotation/A8 ordering, acceptable numeric error, and no silent FP16
fallback. If a fork extension is required, name and version it honestly; do not
call its artifact stock I2_S.

## Milestone 4: Full Model and Export

1. Run the bounded-memory sequential quantizer with resumable per-block artifacts.
2. Separate weight-only quality loss from additional A8 loss using an ablation.
3. Export through the selected runtime's architecture/tokenizer converter.
4. Validate PQ2_0 and PTQ1_0 codecs against native reference decoding, including
   their exact scale representation and logical tensor shapes.
5. Check model-level hidden states and logits between the unpacked quantized
   reference and native packed runtime on identical inputs.
6. Evaluate held-out language loss and decisions. Use perplexity tools for
   perplexity; a coherent generation prompt is not a perplexity measurement.

Gate: parser, loader, numerical parity, and quality all pass. Report actual file
bytes, effective whole-model bpw, preserved-precision fraction, RSS, and scratch.
PTQ1_0 versus PQ2_0 is a storage/kernel tradeoff, not a new training recipe.

If quality fails after correctness is established, stop rollout and compare
selective higher precision, reconstruction fine-tuning, and QAT/distillation.
Changing the target model or dropping BitNet requires an explicit decision.

## Milestone 5: SemIf Integration

1. Reuse SemIf's validation, fixed-label contract, and prompt hashing.
2. Integrate the chosen native backend using a compatible binding or a small
   versioned native executable. Keep fork-native libraries isolated.
3. Implement direct scoring, then complete-state prefix snapshot/restore.
4. Test repeated calls, reversed branch order, cache misses, altered options,
   cancellation, max-context errors, and concurrent request isolation.
5. Fit decision temperature on held-out labeled calibration rows only after
   model/runtime parameters are frozen. Re-evaluate on untouched test rows.
6. Add advisory abstention and deterministic policy enforcement.

Gate: no generated tokens, verified answer-slot tokenization, equivalent fresh
and cached behavior within declared tolerance, auditable scores, and calibrated
metrics. No claim of reproducing TypeSafe's private Jev model.

## Milestone 6: Edge and Vision

1. Benchmark the integrated system natively on the first x86-64 target.
2. Validate ARM64 packing and kernels on an actual board, not QEMU timings.
3. Port the missing RISC-V paths only after specifying RVV version, vector length,
   compiler, OS ABI, and instruction availability. Generic ggml RVV support is
   not automatically support for this BitNet-derived format.
4. Add the actual MiMo vision processor and supported native projector path.
   Repeat quality and activation calibration with representative images.
5. Evaluate selected-label output-head pruning after full-head parity. Retain
   full-head artifacts for language evaluation and general generation.
6. Run sustained thermal/power tests and a log-only sensor loop with watchdogs.

Gate: target-specific latency/RAM/power/accuracy budgets met with raw evidence.
Only then make board recommendations or investigate NPU-specific lowering.

## Planned Code Inventory

The following are proposed modules, **not files delivered in this phase**. Use a
small Python package and subcommands rather than nine duplicated numbered scripts.
Do not create empty implementations merely to match this table.

| Proposed command/module | Responsibility | Key output |
| --- | --- | --- |
| `inspect-model` / `inventory` | Pinned metadata/header inventory and tensor policy | Inventory and byte budget |
| `prepare-calibration` / `calibration` | Real records, split checks, exact template, bounded token packing | Safe token tensors and provenance |
| `quantization.rotation` | Matching input/weight transforms and serialization | Transform specification |
| `quantization.ternary_gptq` | Curvature, damping, row/group scale search, error propagation | Codes, scales, diagnostics |
| `quantize` / `models.qwen35` | Actual hybrid block replay, device control, resumable processing | Sharded quantization artifact |
| `export` / `formats` | Runtime-specific mapping and codec parity | Valid native packed model |
| `export-vision` | Supported processor/projector conversion | Pinned vision artifact |
| `native/bitnet-adapter` | Group-scale-preserving integration of genuine BitNet kernels | Native executable/library and dispatch report |
| `score` / `decision` | SemIf contract, label checks, logits, safe state reuse | Auditable JSONL scores |
| `calibrate-decisions` / `evaluation` | Temperature fitting and reliability metrics | Calibration artifact |
| `benchmark` | Cold/warm/cached/vision timing, RSS, power | Raw samples and summary |

Reused upstream functionality should stay upstream or in a small adapter. New
code is justified only where existing tooling does not implement our contract.

## Verification Matrix

| Risk | Required check | Failure response |
| --- | --- | --- |
| Changed dense function | Dense vs rotated intermediate and final outputs | Fix transforms before PTQ |
| Wrong scale or packing | Python/native codec golden vectors and round trips | Block export |
| Changed graph semantics | Actual full and recurrent block replay | Fix architecture adapter |
| A8/accumulator errors | Zero/extreme/random activation cases and overflow bounds | Fix policy/kernel |
| Token mismatch | Exact rendered prompt and append-one-label token parity | Reject model/backend pair |
| Wrong final logits | Flagged final-position API, varied prompt lengths/batches | Fix native adapter |
| Cache contamination | Fresh/cached/reordered/repeated sequences | Disable reuse until corrected |
| Quantization quality loss | Paired dense/quantized decisions, NLL, perplexity where supported | Tune or recover, not deploy |
| Overconfidence | Held-out NLL/Brier, reliability, selective risk/coverage | Refit or abstain |
| False speedup | Same artifact/workload, actual dispatch, preprocessing included | Reject performance claim |
| Impossible memory target | Measured packed resident + runtime allocations | Change documented budget/design |

Set numerical tolerances per dtype before evaluating candidates. Initial FP64
toy equivalence can use tight tolerances; BF16 native inference must allow
declared numerical differences and report margin-sensitive decision flips.

## Benchmark Protocol

- Measure 32, 128, 512, and 2,048 input tokens where representative, at 1/4/16
  criteria per state. Record actual tokenizer counts and suffix lengths.
- Separate process startup/model load, cold page faults, warm direct prefill,
  state snapshot/restore, suffix compute, scoring, and complete wall time.
- Compare the same source checkpoint and decision fixture across dense, Q8/Q4,
  ternary floating-activation, and ternary A8 paths. Native BitNet is a separate
  model control, not a model-quality ablation.
- Start with at least 30 timed warm samples after declared warmups; retain all
  raw timings and state the protocol, p50/p95, and variability. Extend sustained
  runs to reveal thermal throttling rather than reporting only a burst.
- Record ISA, physical/logical cores, affinity, threads, RAM configuration,
  clock/power mode, compiler flags, runtime hashes, context settings, and cache mode.
- Measure peak RSS and native allocations, not just Python heap or file size.
  Report packed resident expansion, snapshot duplication, and swap/page faults.
- Use an external meter or supported energy counters for joules/decision.
  CPU TDP and nominal NPU TOPS are not measured energy or matching kernel throughput.
- Preserve accuracy, balanced accuracy/macro F1 where applicable, NLL, Brier,
  calibration error with binning stated, and risk/coverage for abstention.
  Bootstrap by source state/session when decisions are correlated.

## Decisions Needed Before Large Jobs

We still need the quantization host's GPU/VRAM and RAM, the initial device,
download/storage budget, deployment modality, and acceptable quality/latency
tradeoffs. The next implementation should inspect and estimate first, then request
approval for model downloads or compute that exceeds the agreed budget.