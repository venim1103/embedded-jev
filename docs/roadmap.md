# Implementation Roadmap

This is a research plan, not a claim that the listed pipeline already runs.
Every model-quality and speed result must reference a frozen artifact, workload,
runtime, and hardware configuration. No full-model job starts merely because a
toy mathematical test passes.

For the tested `cf16046` implementation checkpoint and the next bounded native
task, read the [current handover](handover.md#current-checkpoint-2026-10-05).
The numbered items below retain the milestone plan, not a list of entirely
unimplemented features; completed scoped probes are described alongside them.

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
2026-09-29; its local headers also reconcile, but no full quantized model was kept.
A synthetic five-case agent/tool fixture verifies option mapping, not model
quality. Vision inputs, native tokenization parity, a real held-out decision
fixture, representative calibration, and dense task accuracy remain open.
Full-vocabulary label mass was measured for the engineering prompt only.
A CPU-only BF16 text path now runs all 32 layers in sequence and reads only
selected untied-head rows: on one 22-token non-thinking engineering prompt,
A/B conditional scores were 0.269/0.731 with zero generated tokens. This is
not representative calibration or a quality gate.
The five synthetic agent/tool cases also yielded typed A-C conditional scores
and expected option IDs in an engineering smoke, including option reordering;
this must not be reported as dense task accuracy on held-out data.
Split-aware dataset loading now requires declared source/license, calibration,
validation, and held-out partitions, with case/group/equivalent-prompt leakage
checks and explicit split selection at scoring time. Its end-to-end test uses
synthetic cases; a representative dataset and quality target remain open.
The calibration split alone can export a bounded, hashed, unrotated FFN input
array; validation and held-out captures are refused before model loading.
Temporary synthetic capture parity passes, not representative calibration.
Frozen-candidate pairwise evaluation now records BF16/native choice changes
and conditional-score deltas with data, kernel, candidate, source, and runtime
bindings. Synthetic end-to-end parity passes; no policy fitting is performed.
An attributed pinned CLINC150 four-choice public proxy also exercises the
workflow with human-labeled source data, preserving original train/validation/
test roles. A frozen validation comparison and four training activation
captures pass. Calibration-only local compensation diagnostics have run,
but no new saved candidate or held-out scoring has followed. This reduced
in-scope proxy does not establish official CLINC150 or agent/tool quality.
Calibration-fitted compensation on a four-row/two-group real slice gives
0.401 validation relative error versus 0.423 for searched RTN, but its much
larger calibration benefit does not generalize proportionally. Native parity
passes; no full-width GPTQ, saved-candidate replacement, or quality gate
is established by this diagnostic.
The separately labeled independent-block approximation reaches four complete
input-width rows without a full Hessian. It gives worse validation error than
searched RTN (0.426 versus 0.406), despite better calibration error. Native
96-group arithmetic parity passes; do not promote or bulk-fit this policy
from that evidence.
Balancing four original-training contexts into a capped 128-token sample
also fails to improve this approximation (0.456 validation error versus
RTN 0.406). Source indices and dataset/array hashes are preserved; no
compensated candidate is retained or promoted.
The quantization host exposes an
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
This is not reviewed full-width GPTQ or proof of decoder-block/task quality.
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
A local reader now screens four complete layer-3 FFN-down rows (12,288 columns)
with max-abs, searched-scale, and signed-Hadamard searched-scale RTN in memory.
The best synthetic relative output RMSE on those four rows is still 0.426;
representative activation calibration and block-level parity are required
before committing to a whole-model ternary candidate.
A bounded streaming pass quantized the entire layer-3 FFN-down tensor in memory,
with max-abs, searched-scale, and signed-Hadamard searched-scale relative
**weight** RMSE of 0.770, 0.461, and 0.454. No candidate was retained; these
numbers are insufficient to choose a full-model quantization policy.
One searched-scale full-projection candidate was packed in memory and run
through the isolated BitNet-derived AVX2 grouped matvec. Its 4,096 outputs
matched portable group-scaled arithmetic (max absolute error 1.67e-6), but
the relative output error on one synthetic A8 input was 0.467 versus BF16.
No GGUF tensor format, Qwen3.5 dispatch, or quality result follows from this.
One full layer-3 FFN-down candidate is now retained as a hashed, non-pickle
native fixture. Saved and freshly quantized candidates match exactly through
the single-projection Python adapter; a complete registered GGUF/model format
and representative quality evidence remain open.
The isolated Transformers text prefix compares the same complete projection
against its actual BF16 output on one local prompt: searched-scale RTN has
relative output RMSE 0.423 without activation quantization. On the captured
last-token activation, dynamic A8 plus the BitNet-derived AVX2 grouped dot
matched a portable integer reference (max difference 3.23e-8), but BF16
relative output error was still 0.433. Expand to
representative activation sets and block-level parity before selecting policy.
The pinned non-thinking chat-template variant also passes A-P boundary checks
and real-prefix native parity on a 22-token smoke (BF16-relative error 0.422).
It does not provide final-model logits or owned decision labels.

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
through a registered loader remains untested. Subsequently a pinned
MIT-licensed 1.19 GB native BitNet control GGUF loaded: 22 prompt tokens
produced 128,256 finite final-position
logits with zero generated tokens. A debugger stopped inside the fork's
`llamafile_sgemm_i2s` during prefill. This proves control-model I2_S dispatch,
not group-scaled MiMo runtime integration or a performance result.
The control also verifies A-C as exact one-token continuations and reports
conditional option scores without generation. Allowed-label mass on one sample
was only about 0.000070, so its conditional maximum is not calibrated
confidence; MiMo/SemIf backend integration remains open.
A later Python-hosted BF16 text run substituted one actual MiMo FFN-down
projection with the BitNet-derived AVX2 group-scale kernel; the 22-token
prompt's A/B conditional scores changed from 0.2695/0.7305 to 0.2709/0.7291.
This is not a full-model ternary export or a quality result.
A pinned CPU GGML shared bridge now schedules that frozen projection through
`MAP_CUSTOM2`, with exact direct-versus-graph full-text score/hidden parity.
Loaded GGML dependency hashes are bound in frozen evaluation. A8 and model
hosting remain Python-side; GGUF tensor registration and native full-model
dispatch remain open.
The additional FP32 graph backend performs native dynamic A8 in the callback;
instrumented full-text parity proves Python prepares only the one-token
reference, not the production batch. The model is still Python-hosted and
its callback-owned packed weights are not registered GGUF tensors.
Matching native sign/FWHT/A8 and Python weight rotation also pass a reversible
in-memory one-projection smoke, with dense pre-rounding probe equivalence.
The identity saved fixture cannot be used under the rotated policy; no rotated
artifact or quality-policy promotion is established.
Separate pinned PQ2_0 codec conversion preserves every frozen ternary value
and FP16 scale exactly, and controlled native tensor/MUL_MAT parity passes.
Its Q8_0/Q8_K activation contract and Prism implementation are distinct from
BitNet-derived group-128 A8. A separate registered CPU extra-buffer/tensor
trait now dispatches `MUL_MAT` through the BitNet-derived grouped kernel with
native A8, explicit PQ2-to-BitNet repacking, and independent preserved FP16
row/group scales. Counted one-/two-token dispatch, invalid-payload rejection,
and exact full-size frozen-projection parity pass. Repeated graph evaluation
and owned create/compute/free handles now retain one upload/repack across
changed inputs, recover after rejected inputs, and restore the registry.
The Python-hosted `prism_ggml_registered` backend also has exact 32-layer final
hidden/score parity and reordered synthetic typed-option parity, with zero
generation and explicit handle release. Production registry lifecycle and
full-model native hosting remain open. A standalone tagged native GGUF import control
now feeds one toy identity weight into the same owned handle, with exact
counted dispatch after source-file deletion and comprehensive contract refusal.
It does not select the buffer automatically in Prism's model loader or convert
the actual saved candidate. Full GGUF/model loading remains open; no new
representative quality evidence or evaluator promotion follows.
Versioned early CPU discovery now exposes a stable owned buffer, refuses late
initialization, and preserves one native repack across 1/2/128-token graphs.
Default loader dummy probes refuse it, keeping ordinary PQ2 unmodified.
A metadata-gated exact `tensor_buft_overrides` hook now passes actual pinned
loader selection/upload and counted BitNet dispatch, including the frozen
full-size projection through a deleted temporary encoding. Untagged PQ2 keeps
ordinary CPU selection. Initialized read-only mixed graph concurrency passes;
registry mutation/lifecycle safety, full-model loading, architecture/error
integration, and native hosting remain open. The immutable cached build and
single retained candidate are unchanged.
A separate versioned build now compiles the full pinned llama library while
importing unchanged cached GGML dependencies. Its CTests and guarded no-weight
vocabulary-only native/HF prompt/label parity pass. Header-only planning derives
the 427-text-tensor 16.678 GiB source floor and explicit staging bases; it does
not certify native fit. The staged resource/conversion plan is in
[development](development.md#native-hosting-preflight-and-conversion-plan).
Bulk text-only BF16 reference conversion still needs separate approval, before
actual architecture prefill and one-projection integration; full-model ternary,
held-out inference, and quality promotion remain blocked.
A temporary four-layer synthetic Qwen3.5 model now covers native architecture
construction and zero-generation prefill with a nonzero BitNet FFN at layer 3.
Independent dense/A8 final-logit references, reordered typed options, chunked
prefill, context clearing/reuse, counted calls/one repack, and malformed-weight
refusal pass. The override is explicitly test-only and does not expand the
one-tensor production factory. This reduces bounded architecture/dispatch risk.
Nonzero fixed-gate recurrent and single-rotary-plane attention references now
also pass, with explicit FP16 KV caches, initialized shared-model context
isolation, and failed-kernel/no-score recovery. Real MiMo weights, general
gate/head/rotary coverage, registry/lifecycle race safety, full-model validated
file policy, and representative quality remain separate gates.
An earlier BitNet I2_S toy graph ran serially for two 128-value groups with
distinct row/group scales applied outside the graph, returning exact outputs. This is a
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