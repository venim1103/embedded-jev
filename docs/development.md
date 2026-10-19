# Development Environment

The first environment is CPU-first so source research, mathematical tests, Python
development, and C++ kernel work do not require an NVIDIA runtime. It is a
development image, not the eventual minimal edge deployment image.

## Model Scope and Existing Commands

The intended system supports multiple Qwen3.5 9B-class models.
MiMo-V2.6-Distill-Qwen-9B is the first test subject, not a permanent model
requirement. Broader compatibility is planned, not implemented by changing a
model path. On 2026-10-09 the user selected DavidAU's Defiant Fable safetensors
source as the first non-MiMo profile. Explicit offline inventory now supports its
verified metadata/headers, not model execution; its approved preflight ran the same day (see
the [preflight results](#preflight-results)).

The commands, `MIMO_*` environment variables, cached snapshot, fixed tensor
paths, token IDs, frozen projection and complete-model policy below describe
the existing MiMo controls. Keep these working examples and names unchanged;
do not rename them in documentation as if a generic CLI already exists.
Keep the current metadata, geometry and source guards. MiMo execution refuses
unmatched profiles; explicit offline packaging is a separate path (see the
identity boundary below).

Once implementation is resumed, onboard a different checkpoint through the
[model support gate](roadmap.md#model-scope-and-support-gate) and planned
[profile contract](design.md#model-profiles-and-portability). Validate its own
configuration, tokenizer/template, tensor mapping, precision/state behavior and
source-bound artifacts. Reuse compatible tools/environments after checking
their versions; model identity, calibration captures and results do not transfer.
Another download, conversion, fitted candidate or quality experiment still
requires the corresponding explicit approval.

**Identity boundary.** P0.2 now checks local metadata hashes and shard sizes/header
hashes against the [MiMo record](../embedded_jev/profiles/mimo.json) before model
loading or bound reporting. The tokenizer CLI also verifies cached file hashes;
real scorers check all nonweight files before importing model dependencies. Raw
header/tokenizer fixtures without a verified record have null model/revision,
not an assumed MiMo identity. Header checks do not authenticate weight payloads;
full shard hashing remains an onboarding/artifact gate. Existing CLI execution,
remote downloads and native policy are still MiMo-only. Never pass another
checkpoint to a MiMo scoring `--local-dir`, `MIMO_LOCAL_DIR` or the native MiMo
policy. The inventory-only profile command below is a separate offline path;
other profiles' runtime gates remain open.

### Offline Model Profile Records

P0.1 is implemented in [model_profile.py](../embedded_jev/model_profile.py).
The strict bounded v1 JSON record refuses unknown/duplicate fields, unsupported
source tiers, unsafe paths, invalid geometry and token declarations. Parsed
records and their nested values are immutable; a canonical record SHA-256 binds
the declarations independently of JSON formatting. This is provenance parsing,
not a compatibility or quality claim.

The [MiMo record](../embedded_jev/profiles/mimo.json) contains all 17 previously
verified files, full shard hash pins plus separate header identities, canonical
text geometry, packaging/dtype/template policies and pinned native/converter
contracts. `mimo_profile()` loads it. `verify_profile_headers()` checks an exact
metadata/shard set, metadata bytes/hashes and shard sizes/header hashes without
reading payloads. `verify_profile_files()` hashes bounded nonweight files and
refuses weight shards or files over 32 MiB. Full shard hashing remains an
onboarding/artifact-creation gate; a header check never authenticates payloads.

P0.2 wires the helpers into local/pinned header readers, inventory, tokenization,
real MiMo scoring, captures and paired evaluation. `build_inventory(...,
profile=record)` and the bounded local row-reader APIs can bind explicit offline
fixtures; they do not supply a generic model-execution CLI. Unmatched raw header
inputs remain unbound, while unmatched directories are refused. The original
MiMo Prism-width mapper remains MiMo-only; `profile_prism_expected_widths()`
requires an inventory bound to the supplied record.

The latest source checkpoint is `db002c1`: 234 default tests passed, 39 optional
tests skipped, with 139 focused inventory/label/data/evaluation cases. P0's cached
metadata/tokenizer/legacy-capture checks did not run model inference. P3 later
passed 11 selected compiled native controls and GCC ASan/UBSan/leak checks, as
recorded in [current gates](#current-checkpoint-and-gates). No new real complete
model was converted or inferred; full shard hashing was not rerun.

### Offline Packaging Adapters

P0.3 validates architecture/configuration paths, declared prefixes, exact
index-listed shards, metadata classes, text/MTP shapes and counts, per-class
float dtype and explicit eligible-projection policies. A declared stale index
total is reported as `index_total_difference_bytes`; undeclared mismatches still
refuse. MTP status is `absent`, `complete` or `incomplete`, with missing names
reported and `--no-nextn` recorded. Norm/convolution/SSM parameters stay outside
projection eligibility. Offline planners honor dtype/prefix policies but do not
enable another model's execution or change MiMo's BF16 numerical contracts.

The [Defiant Fable record](../embedded_jev/profiles/defiant-fable.json) binds
12 nonweight files, including the separately cached embedded template, and five
shard identities from the existing preflight.
Its full shard hashes are remote LFS pins; only headers/samples were read. The
record's header hashes include the safetensors eight-byte length, unlike the
preflight's raw-JSON header hashes in the source register. Its native policy is
null. Reproduce the metadata-only inventory without weights or HTTPS:

```bash
metadata="$HOME/.cache/huggingface/embedded-jev/models/defiant-fable-7af0a9c4-metadata"
PYTHONDONTWRITEBYTECODE=1 python -m embedded_jev.inventory \
   --local-dir "$metadata" --profile embedded_jev/profiles/defiant-fable.json \
   --header-dir "$metadata/headers" | jq '{source, accounting, totals, native_reference_plan}'
```

`--profile` and `--header-dir` require `--local-dir`; sidecars require an explicit
record and never enable a remote profile. Each raw JSON sidecar is named
`SHARD.safetensors.header.json`. The reader reconstructs/validates the complete
prefix and uses the record's verified remote size, reported as
`profile_pinned_remote_sizes`, not a local weight-file stat or payload hash.
Ordinary local snapshots still use real shard stats and bounded headers.

Current evidence: Defiant Fable has 775 tensors, including 427 text/333 vision/
15 complete MTP, 19,306,208,736 tensor bytes and a 67,108,864-byte index shortfall.
Its cached bound tokenizer yields 49 tokens for the default label prompt, A-P
32-47 and zero generation. MiMo's original 760-tensor accounting is unchanged.
These are inventory/tokenization gates, not inference, native compatibility or
quality. The P0.4 template policy below is implemented; native rendering remains
a separate gate.

### Template Control Policy

Profile-bound label probes call `validate_template_input()` before rendering.
Declared literal control prefixes are refused anywhere in bounded untrusted
dictionary keys/values, nested content and decision fields, including IDs and
option descriptions. Evidence is never silently stripped. Bounds are 4,096
nodes, depth 32 and 1 MiB UTF-8 text, with pre-expansion/encoding checks and
invalid-Unicode refusal. Errors return no score report. Mode arguments are
profile-owned, not derived from evidence or case metadata.

Defiant Fable conservatively declares `{REASON:` and pins
`enable_thinking=False`, `reasoning_effort="medium"` and
`add_generation_prompt=True`. This protects both source and embedded-template
probes even though the safetensors source's base template does not interpret
that marker. MiMo keeps its existing exact arguments and prompt bytes.

The Defiant Fable record additionally hashes the existing
`plusIQ-bf16-gguf-chat-template.jinja` separately and binds its originating GGUF
repository, pinned revision, file and remote whole-file SHA-256. The record now
has 12 nonweight files plus five shards. `probe_label_boundary(...,
template_source="gguf_embedded")` validates that template rather than treating
it as the source file; reports include both hashes, origin and render arguments.
The GGUF hash is a remote pin, not a local payload check or GGUF-source approval.
No generic GGUF weight adapter or native template support follows.

Both cached Defiant Fable templates render the same 49-token default prompt and
A-P IDs under the pinned mode. All five source-template fixture boundaries pass;
the cached embedded template refuses tested `ispoon`, `xhigh` and unknown markers
before rendering. MiMo's prompt hash is unchanged. Twelve focused policy/resource
tests pass; native rendering/tokenization, GGUF/source full-weight equality and
model execution remain unverified for this profile.

### MiMo Coupling Inventory

Originally audited at `bc8e616`; updated through `db002c1` on 2026-10-09.
"Effect" distinguishes explicit offline profiles and v2 owned projections from
the still-MiMo-only execution and v1 loader paths. The last column names the
implemented or remaining profile field or
[work item](roadmap.md#generalization-work-queue) that replaces the assumption.

| Location | MiMo-specific assumption | Effect on another checkpoint | Replacement |
| --- | --- | --- | --- |
| [inventory.py](../embedded_jev/inventory.py) `MODEL_ID`, `MODEL_REVISION`, `MODEL_URL` | Remote URL remains MiMo-only; reports match verified files to a profile | Unmatched directories refuse; unbound raw fixtures have null identity | P0.2/P0.3 offline identity and packaging implemented; generic execution remains open |
| inventory.py `_model_config`, `REQUIRED_METADATA` | Explicit Tier A profiles declare architecture/config path, metadata files/classes and untied embeddings; unbound legacy inputs retain MiMo rules | Text-only and missing-processor-config offline profiles validate as declared; custom-code wrappers remain unsupported | P0.3 implemented; wrapped-decoder adapter remains P6 |
| inventory.py `SHARD_NAME` | Unbound legacy inputs retain MiMo numbering; bound profiles use exact index-listed shard names | Explicit offline profiles accept alternate/unpadded/restored names; undeclared names refuse | P0.3 implemented |
| inventory.py index reconciliation | Bound profiles declare exact/recomputed totals and MTP status; raw legacy totals remain exact | Stale totals are recorded only as declared, MTP completeness validated, never used as identity | P0.3 implemented |
| inventory.py `_classify`, `_required_weights` | Bound prefixes, geometry, counts and per-class dtypes are profile-driven | Explicit text-only and multimodal fixtures validate; unknown names/shapes/dtypes refuse | P0.3 implemented |
| [label_probe.py](../embedded_jev/label_probe.py) `NON_THINKING_SUFFIX` | Unbound fixtures retain fallback suffix; explicit profiles pin templates/suffix/arguments/IDs/control prefixes; CLI fetch remains MiMo-only | Raw tokenizers stay unbound; marked evidence refuses before rendering; source/GGUF template origins remain separate | P0.2/P0.4 implemented; other-model native rendering remains open |
| [dense_probe.py](../embedded_jev/dense_probe.py) `plan_text_prefix`, `plan_streamed_text` | Bound offline plans use profile prefixes and dtype policy; real execution remains MiMo BF16 | F32 small-tensor plans validate without casting; another model's scoring remains refused | P0.3 planning implemented; P2 runtime gates remain |
| [streamed_text.py](../embedded_jev/streamed_text.py) | Real execution verifies MiMo; layer-3 target, 4,096 x 12,288 shapes, untied head, 32-layer full-vocabulary mass and seed-773 signs remain | Another checkpoint refuses before imports; metadata profile support does not enable its execution | Per-profile execution gates P2; target/depth generalization P4/P7 |
| [projection_artifact.py](../embedded_jev/projection_artifact.py), [evaluation.py](../embedded_jev/evaluation.py) | `PINNED_SHARD`, `PINNED_SHARD_SHA256`, MiMo identity and 32 layers | Artifacts and evaluations MiMo-only | Profile ID plus tensor payload SHA-256 |
| [decision_dataset.py](../embedded_jev/decision_dataset.py) captures | New v2 manifests bind a supplied profile or remain unbound; v1 MiMo remains read-only compatible; bounded layer-3 inputs | Cross-profile reuse and mixed capture banks refuse; raw arrays do not claim MiMo | P0.2 implemented; depth/target generalization still P7 |
| [weight_slice.py](../embedded_jev/weight_slice.py), [ternary_artifact.py](../embedded_jev/ternary_artifact.py) | Default target/fetch remains MiMo-only; bounded local APIs and width mapper can bind explicit profiles | Unmatched local sources refuse rather than acquiring MiMo identity; no generic remote fetch | P0.2 identity implemented; broader workflow/targets remain separate |
| [prism_codec.py](../embedded_jev/prism_codec.py), [native trait and handles](../native/prism_group_scale.cpp) | Legacy codec/v1 APIs retain 4,096-row bounds; v2 owned path and model policy are profile/manifest-gated | Larger projections, v2 policy/graphs and synthetic profile-to-BF16 prefill pass for all seven eligible roles; v1 real execution remains MiMo-only | Bounded P3/P4 synthetic prerequisites implemented; real caller/provenance/runtime gates need approval |
| Native v1 complete-model factory | MiMo revision and model tags, 427 tensors, 32/4,096/12,288, 248,320 vocabulary, at most 19 GiB, one layer-3 target | Every other file refused, including MTP-bearing, third-party and 48-layer GGUF files | Versioned profile-driven policy (P4, P5, P7) |
| Tests and documented commands | `MIMO_*` flags, cache paths, fixture prompts and expected IDs | MiMo evidence only | Keep names; add per-profile gates beside them |

Model-independent today: [activation.py](../embedded_jev/activation.py),
[ternary.py](../embedded_jev/ternary.py), PQ2 packing logic, the dataset schema
and split checks, [public_data.py](../embedded_jev/public_data.py), the trace
comparator, the [BitNet-derived kernel](../native/bitnet_group_scale.cpp) and
the [versioned runtime build](../native/CMakeLists.txt). The pinned Prism
converter registers `Qwen3_5TextModel` for both Qwen3.5 architectures. It removes
MTP tensors only with `--no-nextn`, which MiMo also needs because its configured
MTP layer has no tensors. No adapter exists yet for float-GGUF sources, wrapped
decoders or depth variants; see the
[support tiers](design.md#support-tiers-for-derivatives). Never execute code
shipped in a model repository to inspect it.

### Defiant Fable Preflight Procedure

On 2026-10-09 the user selected DavidAU's Defiant Fable safetensors source as the
first non-MiMo profile and approved one bounded metadata preflight, including the
`plusIQ` GGUF header-and-sample comparison. It first failed HTTPS certificate
verification (see the
[environment pitfall](handover.md#13-environment-pitfalls-already-encountered))
and ran after the container was rebuilt with [host certificates](#host-certificates);
the [results](#preflight-results) follow the steps. The approval covered only these
steps and this budget. A full snapshot (about 19.3 GB), conversion, inference or
retained weights need separate approval.

| Item | Source repository | GGUF repository |
| --- | --- | --- |
| Hub ID | `DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP` | `DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF` |
| Revision | `7af0a9c4e221e01b246b3c577fbb7110b79823e8` | `8b192a8e203440d6492a133f6cdfa4bf7bffac98` |
| In scope | Non-weight files, including `backup/`; headers of the four numbered shards and `model-mtp-restored.safetensors`; sampled tensor ranges | Header and sampled tensor ranges of `Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-plusIQ-NEO-MAX-MTP-bf16.gguf` |
| Verified identity | Shards of 4,942,706,128, 4,987,758,072, 4,954,219,624 and 4,354,511,264 bytes; 67,108,960-byte restored shard; 19,989,343-byte `tokenizer.json`; 6,722,759-byte `vocab.json`; no `merges.txt` or `processor_config.json`. Hashes are in the [source register](sources.md#defiant-fable-preflight-evidence) | 18,407,330,272 bytes; LFS SHA-256 `6a772118b107772fb83d111fec59e54b139c993fe6dc4906288a3bb81178705d` |

Run the steps in order with throwaway scripts outside the repository. Do not run
the MiMo `inventory`, `label_probe` or native tools against these files.

1. In a container rebuilt with the [host certificates](#host-certificates),
   confirm that `curl` returns HTTP 200 for the source revision's Hub API URL.
   Never pass `-k`, `verify=False` or any option that disables certificate or
   checksum verification.
2. Read both tree listings at the pinned revisions through the Hub API, including
   `backup/`, and record each path, size, Git blob ID and LFS SHA-256. Stop if the
   GGUF size or SHA-256 differs from the table.
3. Download every source file except weights and images into
   `$HOME/.cache/huggingface/embedded-jev/models/defiant-fable-7af0a9c4-metadata/`
   (about 47 MB), verifying each Git blob ID or LFS SHA-256.
4. Range-read each safetensors header: the 8-byte little-endian length (refuse
   above 16 MiB), then the JSON. Save each header with its SHA-256. Reconcile it
   with the index: names, BF16 dtypes, shapes, contiguous offsets and file sizes.
   Count text (427 expected), vision and MTP tensors, recompute total bytes and
   record the index `total_size` shortfall (67,108,864 bytes expected).
5. Probe the tokenizer and template from the downloaded files with the dense
   environment's Transformers 5.12.1 and `trust_remote_code=False`. Use MiMo's
   default label-probe prompt and the five
   [fixture cases](../tests/fixtures/agent_tool_smoke.json), rendered with
   `add_generation_prompt=True` and `enable_thinking=False`. Require the base
   suffix `<|im_start|>assistant\n<think>\n\n</think>\n\n`, equal templated and
   plain encodings, and A-P labels that each extend the prompt by one distinct
   non-special token. Record template and prompt hashes, token counts and IDs.
6. Range-read the GGUF header, starting with 32 MiB and refusing more than
   64 MiB, and parse it with a bounded GGUF v3 reader. Record the version,
   metadata keys with array lengths and hashes, block and `nextn` counts,
   layer-type keys, tensor names, types, shapes and offsets, data alignment, and
   the embedded template's hash and any in-band control syntax.
7. In disposable scratch, compare all of `blk.3.ffn_down.weight` with
   `model.language_model.layers.3.mlp.down_proj.weight` (100,663,296 bytes each);
   all of `blk.0.ssm_alpha.weight` and `blk.0.ssm_beta.weight` with layer 0
   `linear_attn.in_proj_a.weight` and `in_proj_b.weight` (262,144 bytes each),
   matching rows to identify tiled or grouped value-head order; and the first
   2 MiB of the token embedding and output matrices. BF16 2-D GGUF tensors keep
   the safetensors row-major bytes.
8. Write one JSON report under 1 MiB beside the metadata directory, delete the
   scratch bytes and verify the deletion.

Transfer stays below about 350 MB and retained files below about 60 MB. Matching
samples with tiled alpha/beta rows make `plusIQ` a likely template variant of the
source, unproven until every eligible tensor matches. Any mismatch leaves `plusIQ`
in Tier C behind [P5](roadmap.md#generalization-work-queue), and the source
profile proceeds alone. Record results in the handover and source register.

#### Preflight Results

The 2026-10-09 run transferred 291,402,916 bytes in 49 requests (48 s, 338 MB
peak RSS); every range request returned the requested bytes. It retained 47.2 MB:
the verified files, the five safetensors headers and the `plusIQ` template in
`$HOME/.cache/huggingface/embedded-jev/models/defiant-fable-7af0a9c4-metadata/`,
plus `defiant-fable-7af0a9c4-preflight-report-20261009.json` (93 KB) and the two
scripts in `defiant-fable-7af0a9c4-preflight-scripts-20261009/` beside it. Sampled
tensor bytes were compared in memory, never written, and the temporary working
directory was deleted.

- **Files:** 16 non-weight files (46,974,575 bytes) matched their Git blob IDs
  or LFS SHA-256. There is no `merges.txt` or `processor_config.json`; `backup/`
  repeats the root configuration, template and tokenizer files byte for byte.
- **Headers:** 775 BF16 tensors: 427 text, 333 vision and 15 MTP. The historical
   preflight summary said 774; P0.3's exact header recount corrected that total
   without changing category counts or bytes. Text names,
  shapes and dtypes equal MiMo's 427, and layer types match the configuration.
  Shards are contiguous and match their file sizes. Header tensor bytes exceed
  the index `total_size` by 67,108,864, exactly `mtp.fc.weight`, the only tensor
  in `model-mtp-restored.safetensors`.
- **Tokenizer:** Transformers 5.12.1 loads `TokenizersBackend` and renders the
  base suffix; templated and plain encodings agree. A-P are single tokens 32-47
  for the default prompt, and A/B/C are 32/33/34 in all five fixture cases, as
  for MiMo. Special IDs are `<|im_start|>` 248045, `<|im_end|>` 248046, `<think>`
  248068 and `</think>` 248069. `vocab.json` equals MiMo's; `tokenizer.json` is
  18 bytes longer than MiMo's.
- **GGUF header:** version 3, 35 keys and 442 tensors (258 BF16, 184 F32) that
  are aligned, non-overlapping and exactly fill the file after a 10,977,760-byte
  header. `qwen35.block_count` is 33 with `nextn_predict_layers` 1 (15 `blk.32`
  tensors, four of them `nextn`), `full_attention_interval` 4 and no per-layer
  array.
- **Samples:** `blk.3.ffn_down.weight` equals the source down projection
  (100,663,296 bytes), and the first 2 MiB of the embedding and output matrices
  are identical. Every `ssm_alpha` and `ssm_beta` row matches a source row in
  tiled order, the pinned converter's value-head convention.
- **Template:** the embedded `plusIQ` template (17,041 characters) differs from
  the source's, yet with `enable_thinking=False` and no marker it renders the same
  default prompt, token IDs and labels. A `{REASON:ispoon}` marker in user text
  disappears from the prompt and expands it from 202 to 4,219 characters.

The samples support, but do not prove, that the `plusIQ` BF16 file holds the
source weights with another template; the quantized `plusIQ` files are also only
6,688 bytes larger than the regular MTP files of the same type. Treat `plusIQ` as
a template variant of the Tier A profile only after every eligible tensor
matches, and keep its control syntax behind
[P0.4](roadmap.md#generalization-work-queue).

## Current Checkpoint and Gates

The latest tested source checkpoint is the 2026-10-13 reusable complete tensor
table API, following complete synthetic prefill integration at `8e11f7a`
and convolution file ranges at `7f21b1e`,
following convolution operands at `daf92f1`,
following BF16 matrix file ranges at `c783c2f`, exact matrix operands at `ef51266`, complete synthetic vectors
at `7326a14` and GGUF binding at `5240c26`
and conversion planning/vector operand parity at `b6a9f0d`
and runtime hashing at `c18e90f`,
following defining-symbol origins and optional six-module inference file pins
at `4973863`, following `1426bce`,
following verified-file owner `e6f6824`,
following bounded metadata correction `af62d7e`, the seven-role proof `fab3cc0`, `d653efc`, `c825cfc`, P0.1-P0.4
and bounded P3; resident compaction is `cedaeec`. The
[latest handover](handover.md#complete-table-api-checkpoint-2026-10-13) records
392 default passes/54 optional skips, twelve batch preflight cases, two pinned
operand/file controls and nine complete-stage/native cases:
189 dense ranges plus seven frozen payloads across seven supported roles.
The preceding convolution checkpoint retains 37 focused file/convolution cases
and two pinned-source controls with exact nonzero convolution operands/file ranges.
The preceding matrix checkpoint retains 37 focused matrix/file cases and
16 exact matrix/file ranges. The preceding complete-vector
checkpoint retains nine vector-enabled synthetic complete-model cases. The file-binding
checkpoint retains 22 focused plan/vector/file cases. The preceding runtime hashing checkpoint
retains 56 focused file/owner/runtime cases and nine synthetic caller cases
with verified full/parallel/recovery modes. The unchanged
native metadata checkpoint retains 23 compiled controls and GCC sanitizer
early/late plus four tiny v2 prefill cases. The large metadata
boundary controls were not sanitizer-replayed; the earlier standalone nine-mode
caller replay remains at `d653efc`. Python leak checks fail with the same
CPython/NumPy allocations in a no-prefill baseline. V1 factories
remain strict. No new CTest result or real complete-model inference is claimed.
The user resumed autonomous work on 2026-10-12. The synthetic caller boundary
covers all seven eligible roles, not real-profile execution approval.
Do not recreate environments or download another source/model copy to resume.

**Historical 2026-10-08 evidence:** `9408f3f` followed the MiMo complete-model
policy, precision traces and bounded recurrent controls. Those separately
approved conversions and measurements are not renewed authorization. At that
checkpoint, 163 default tests passed, 39 optional tests skipped, and all
seventeen pinned Prism controls passed with full-size PQ2, repeated/owned native
weights, two-forward module reuse, tagged toy GGUF import, and versioned CPU
discovery, mixed concurrent graphs, and real pinned loader selection/upload,
including the full-size frozen projection and guarded vocabulary-only native
tokenizer preflight, dense/BitNet synthetic model prefill, and complete-model
policy/CLI refusals plus trace equivalence, Q/K normalization, sixteen native
activated/raw-gate recurrence cases and eight pinned Torch recurrence/state
cases (51.67 s). Separately, the approved real one-BitNet prefill
regression passed in 221.62 s; its error/recovery cases passed in 733.45 s combined.
The five real-model cases were skipped in the seventeen-control invocation because
their separately approved files were not supplied there. The prior dense-only
regression passed at `462dc12`. The earlier dense and mixed temporary models were deleted;
the separately approved precision trace regression passed in 746.54 s at
`5b7e559`, and its model, spill and raw captures are also deleted. Details follow below.
At that historical checkpoint, the separate full runtime build passed two CTests.
The 32-layer direct/callback/registered, reordered synthetic typed-decision,
and signed-Hadamard full gate passed at `4efa59b` (138.36 s), before the isolated
native file/runtime additions; unchanged model inference was not rerun afterwards.
Ruff, Python editor diagnostics, explicit C++ compilation, and whitespace checks
passed. The C++ editor lacks the external GGML include path; no editor settings
were changed. GCC ASan/UBSan and leak
checks passed for bridge runtime controls; cached GGML is not instrumented. These
are correctness/scope gates, not whole-model quantization acceptance, a
benchmark, representative quality, or calibrated confidence.

Default regression and lint commands, from the workspace root:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
ruff check --no-cache embedded_jev tests
git diff --check
```

The expanded P3/P4 command below selects 25 compiled controls using
only pinned cached sources/environments, synthetic fixtures and the sole frozen
projection. It does not run a real complete model or convert/download one.
The latest role-only increment ran its 16 focused record/role checks, not this
expanded full command; the preceding 20-control result remains historical evidence.
Rebuild our isolated wrapper first only if its tracked native sources changed;
do not overwrite pinned GGML libraries or change user editor settings to fix
CMake Tools discovery. Run checks for a changed slice, not merely to repeat
unchanged successful gates:

```bash
cache="$HOME/.cache/huggingface/embedded-jev"
PRISM_SOURCE_DIR="$cache/native/prism-source" \
PRISM_GGML_CPU_LIBRARY="$cache/native/prism-build/bin/libggml-cpu.so" \
BITNET_SOURCE_DIR="$cache/native/bitnet-source" \
PRISM_CONVERTER_PYTHON="$cache/native/converter-venv/bin/python" \
MIMO_PRISM_RUNTIME_BUILD="$cache/native/jev-prism-runtime-v1-build" \
MIMO_LOCAL_DIR="$cache/models/mimo-2367e865d009c13ac81713a2878291d33ab28177" \
MIMO_PROJECTION_ARTIFACT="$cache/quantized/layer3-ffn-down-rtn-searched-fp16" \
MIMO_DENSE_PYTHON="$cache/dense-venv/bin/python" \
MIMO_PQ2_CODEC_TEST=1 MIMO_REGISTERED_MODULE_TEST=1 \
OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
python -m pytest -q -p no:cacheprovider tests/test_prism_native_control.py \
   -k 'shared_group_scale_graph_matches_direct_kernel or pq2_ternary_subset_matches_actual_decoder or tiny_qwen35_native_prefill_scores_without_generation or complete_model_policy_binds_exact_projection or profile_complete_model_policy or profile_manifest_native_prefill'
```

For the complete synthetic profile-to-prefill slice alone, this explicitly
enables all 28 tensor checks before native prefill for each of seven projection
roles; two unsupported profiles still refuse. The complete stage requires the
already cached Torch environment via `PRISM_PARITY_PYTHON`. Use
`PRISM_VECTOR_PYTHON` instead for the older ten-vector-only stage; with neither
variable set, native tests run without these provenance stages. Neither mode
authenticates real source payloads. The full stage hashes a producer-generated,
uncompressed <=16 MiB operand archive and checks all dense ranges plus the exact
frozen PQ2 payload. This sparse synthetic integration does not replace the
separate nonzero upstream-transform controls or source-file authentication.

```bash
cache="$HOME/.cache/huggingface/embedded-jev"
PRISM_SOURCE_DIR="$cache/native/prism-source" \
PRISM_CONVERTER_PYTHON="$cache/native/converter-venv/bin/python" \
PRISM_PARITY_PYTHON="$cache/dense-venv/bin/python" \
MIMO_PRISM_RUNTIME_BUILD="$cache/native/jev-prism-runtime-v1-build" \
OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
python -m pytest -q -p no:cacheprovider tests/test_prism_native_control.py \
   -k profile_manifest_native_prefill
```

The earlier full-text command below is historical reference, not the code-only
resume gate. It performs real MiMo streamed inference and must not be launched
implicitly by a documentation request or this next-chat preparation:

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
- Root certificates the host trusts, added at build time (see
  [host certificates](#host-certificates)), with `SSL_CERT_FILE`,
  `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE` and `NODE_EXTRA_CA_CERTS` pointing at
  the merged system bundle.
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

## Host Certificates

Networks with TLS inspection present server certificates issued by an
organisation root that the base image does not trust. Before each build, both
devcontainer profiles run [prepare-build.sh](../.devcontainer/prepare-build.sh)
on the host through `initializeCommand`, which needs Bash there (WSL, Linux,
macOS or Git Bash). It exports the macOS keychains or the Windows root stores,
including from WSL through `powershell.exe`, and otherwise copies the host's
Linux bundle, into `.devcontainer/host-ca-certificates.crt`. The
[Dockerfile](../.devcontainer/Dockerfile) splits that file, drops entries
OpenSSL cannot read, and runs `update-ca-certificates` before any download. An
empty export keeps the default trust store. Base-image pulls use the container
engine's own trust store on the host.

The generated file, and any image built with it, contain the host's trusted
roots, which can identify the organisation. The file is git-ignored: never
commit or quote it, and do not publish such an image. Verification is never
disabled. If HTTPS still fails after a rebuild, check the export output and the
host's trust store instead of weakening a client.

## CLI Workflow

Run on the host from the repository root:

```bash
bash .devcontainer/prepare-build.sh
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
indiscriminately. `up` and VS Code run the certificate export themselves; the
explicit first command covers a separate `build`.

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
New manifests use v2 `single_case_profile_calibration_activations` with a
`source` block carrying the profile ID/fingerprint. The scorer supplies its
verified MiMo profile. Calling `save_calibration_capture()` on raw arrays without
that explicit profile produces unbound data, which model-fitting consumers
refuse. `load_calibration_capture(..., profile=record)` checks the selected
identity; balanced banks may not mix profiles or bound/unbound captures.
Legacy v1 `single_case_mimo_calibration_activations` remains readable as a legacy
declaration, without rewriting its manifest or inventing a profile fingerprint.
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
exactly one invocation of the BitNet-derived grouped kernel on success.

Storage is deliberately still `GGML_TYPE_PQ2_0`: each row/input-group block has
a little-endian FP16 scale and 32 adjacent, low-bit-first code bytes. Only
codes 0/1/2 (ternary -1/0/+1) are accepted; PQ2's +2 code, negative scales,
and nonfinite scales are rejected. Since the P3 compaction checkpoint, owned
tensor storage repacks its code bytes in place to BitNet's four separated
high-bit-first lanes, retaining every independent FP16 scale bit. The new
`bitnet_group_scale_matmul_fp16_avx2` converts scales exactly at computation;
it does not retain an expanded FP32 array. Canonical PQ2 reads reconstruct
original code bytes on demand. File-policy validation never mutates canonical
reference bytes. No scale is refitted. FP32 inputs use native
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
- `prism_bitnet_registered_projection_storage_bytes_v1(handle, weight_bytes*,
   auxiliary_weight_bytes*)` reports the owned logical weight payload and extra
   weight-array allocation separately. Both are cleared on invalid arguments;
   activation/graph scratch, allocator alignment and process RSS are not included.

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
unproven. The former standard-PQ2/repacked-code/FP32-scale resident duplication
was removed by P3 compaction. The frozen weight payload is 13,369,344 bytes
(34 bytes per 128 weights, 2.125 bpw), with zero auxiliary weight-array bytes.
Graph/A8 scratch and allocation overhead remain separate; no whole-model fit
or performance claim follows. No new candidate or dataset inference is saved.

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

### Profile-Gated Owned Projections

P3 adds `prism_bitnet_registered_projection_create_v2` beside unchanged legacy
factories. [prism_bitnet_runtime.h](../native/prism_bitnet_runtime.h) declares
`jev_bitnet_projection_spec_v2`: ABI 2, exact GGUF projection name, logical
rows/columns, expected canonical PQ2 bytes and length. Creation validates
every payload byte against that reference, clears failed handles and binds
the trait to the exact name/shape. The caller keeps the reference valid during
creation; the resulting handle owns copied/repacked weight bytes. Use the
existing compute/free/storage diagnostics and keep the creating library alive.

V2 caps tokens at 128, rows/columns at 12,288, requires group-128 columns and
caps total logical weights at 50,331,648. Only `blk.0`-`blk.255` FFN gate/up/down
and attention Q/K/V/output names are accepted. Unknown/recurrent names and
reference mismatches return 6, invalid ABI/geometry/length/tokens return 1,
invalid ternary/scales retain 5 and allocation failure 2. The old v1 APIs still
reject >4,096 rows and retain all file/model policy refusals. This is an owned
projection API, not another discoverable loader buffer or complete-model policy.

`profile_projection_specification(profile, inventory, manifest, blocks)` in
[prism_codec.py](../embedded_jev/prism_codec.py) validates strict manifest fields:

- `schema_version=2`, `format="jev-profile-projection-v2"`.
- `profile_id`, `profile_sha256`, canonical HF `tensor` and `gguf_tensor`.
- `shape=[rows, columns]`, `group_size=128`, `transform="identity"`.
- `source_tensor_sha256`, `payload_bytes`, `payload_sha256`.

The source digest is declared provenance, not automatically rehashed or proof
of model quality. Geometry/dtype/eligibility must agree with the immutable
profile and its bound inventory, not just a caller-supplied report shape.
Contiguous uint8 blocks must have exact `[rows, columns/128, 34]` layout, size
and SHA-256; existing codec checks validate trits/scales in <=4,096-row slices.
`create_profile_projection(native, profile, inventory, manifest, blocks,
tokens=N)` performs those checks before native access and returns the owned
handle plus declared evidence. Free that handle with the existing native API.
This adapter writes no artifact and cannot create a full-model loader policy.

Golden controls cover 8,192 x 4,096 Q and 12,288 x 4,096 gate/up matrices,
exact direct-kernel parity, one repack, resident bytes, all v2 bound refusals and
continued v1 rejection. Setting existing `MIMO_LOCAL_DIR` and
`MIMO_PROJECTION_ARTIFACT` additionally takes the unchanged frozen projection
through this adapter. That test hashes only its existing BF16 source tensor in
1 MiB chunks, builds a transient manifest and saves no new candidate. The final
P3 gate passed 11 compiled controls (including v1 policies and tiny native
prefill), 234 default tests with 39 optional skips, and GCC ASan/UBSan/leak
checks on our bridge/control. Native model quality and broader loader/runtime
capabilities remain separate gates.

### Profile-Gated Complete-Model Policy

`profile_model_specification()` reuses the P3 manifest gate to derive the entire
text tensor table, BF16/F32 converter plan and 14 attention/SSM/rotary metadata
requirements. It supports the pinned Tier A safetensors adapter with `--no-nextn`,
regular attention patterns and <=32 layers; wrappers, float-GGUF sources, larger
depths and unpinned runtime/converter rules refuse. SSM convolutions and
one-dimensional tensors are explicitly F32; source dtypes are verified separately.

`create_profile_model_policy()` marshals ABI-2 specifications into
`prism_bitnet_cpu_model_override_from_gguf_v2`. It returns an owned policy handle,
anchored exact-name overrides and the checked specification, not a model or an
inference approval. Native tensor types use the pinned Prism enums (F32 0,
BF16 30, PQ2_0 142); metadata kinds are u32 scalar 1, F32 scalar 2 and four-element
i32 array 3. The native factory checks GGUF v3, every expected name/type/shape,
nonoverlapping file ranges, exact typed geometry metadata and canonical target
bytes, refusing any other PQ2, Hadamard transform or recurrent-layer override.
Its model tag is `profile-qwen35-single-projection-v2`; it also requires exact
`jev.model.profile_id`, `jev.model.profile_sha256`, `jev.model.source_revision`,
`jev.model.converter_revision` and `jev.bitnet.source_tensor_sha256` declarations,
with the unchanged identity execution contract. Tags do not authenticate source
weights. The Python adapter hashes the supplied payload, not source tensor bytes.

The policy-bound buffer checks upload bytes against its own canonical reference,
including chunked transfers after factory validation, and reuses the compact
FP16-block/BitNet-derived kernel lifecycle. `prism_bitnet_cpu_tensor_status_v2`
and `prism_bitnet_cpu_tensor_last_input_tokens_v2` require the policy handle;
v1 status APIs still refuse this distinct buffer.
The caller must retain the shared library and policy through every model/context/
buffer lifetime, then call `prism_bitnet_cpu_model_policy_free_v2`. Overrides and
their strings belong to the policy. Inputs/specifications are copied as needed
during creation; keep the model file unchanged and available for subsequent
loading. Policy creation/teardown is not general concurrent lifecycle safety.

`verified_profile_model_policy(native, model_path, profile, inventory, manifest,
pq2_blocks, expected_bytes=..., expected_sha256=...)` provides a separate
context-managed file-verification boundary. It defaults to a 16 MiB hashing
budget (`maximum_bytes`), bounded by the checked complete-model file maximum,
and a cooperative hashing deadline (`maximum_seconds`, default 30, at most 600).
The Linux-only API hashes a regular, non-symlink file in unbuffered 64 KiB chunks, verifies
descriptor identity, and exposes an anchored `/proc/self/fd/N` path while open.
Its owner retains the library, copied read-only blocks, native record, handle,
overrides and an immutable verification receipt. Use `owner.model_path`,
`owner.record`, `owner.handle` and `owner.overrides` only inside the context;
destroy all associated native users before exit. Cleanup frees the policy before
closing the descriptor and invalidates public path/record access. Changed files
refuse; no concurrent-writer or immutable-storage guarantee follows. Buffer
scores until context exit succeeds.

The receipt means SHA-256 matches **caller-supplied expected file identity**.
It does not authenticate the expected digest's origin, source weights, converted
tensor parity, loaded runtime binaries or tokenizer behavior, and explicitly
grants no runtime approval. Larger real-file hashing, conversion and execution
still require separately approved resource/provenance gates. The bare-handle
factory and all v1 policies remain unchanged; this owner is not a production
scoring API or permission to enlarge the compiled test hook.

Optional `expected_runtime_files` supplies exact `bytes`/`sha256` records for
`bridge`, `ggml_cpu` and `ggml_base`; `verify_model_runtime_files()` exposes the
same check separately. Records must total <=64 MiB (a smaller `maximum_bytes`
can be supplied to the standalone checker). Regular-file sizes are preflighted.
The checker resolves bridge/CPU/base defining symbols through
libc `dladdr` and cross-checks the CPU/base witness. `CDLL._name` is not used
as bridge identity. Each regular-file descriptor is opened nonblocking/no-follow
and hashed using unbuffered reads <=64 KiB, never beyond its initial extent or
declared pin size. Descriptor/path identity must match after hashing. One
cooperative deadline covers all files per check (`maximum_seconds`, default 30,
at most 600 for the standalone checker); the owner uses that default for each
entry/yield/exit runtime check. Paths/bytes/digests must match; pin dictionaries
are snapshotted. The owner
rechecks before yielding and at exit, with cleanup on refusal, and adds flat
runtime digests to its immutable receipt. Without pins it reports `not_checked`.

For the bounded prefill provider, standalone `scorer=...` or owner
`runtime_scorer=...` additionally requires `scorer`, `llama` and
`ggml_registry` byte/hash records. All six defining modules are located through
their known symbols, under the same 64 MiB aggregate ceiling. Passing a scorer
without pins refuses. `runtime_file_scope` distinguishes the three/six-module
cases; the library stays alive through cleanup. The test hook itself remains
unchanged and synthetic-only.

These are file matches against caller pins, not full runtime attestation:
loaded pages and transitive system libraries are not attested, and the
cooperative deadline is not a hard timeout on blocked I/O or native calls.
Keep runtime files immutable; concurrent writers are unsupported. No model
execution approval or source/converted-tensor provenance follows.

`profile_model_conversion_plan(profile, inventory, manifest, pq2_blocks)`
reuses the checked table and declares source names/shapes/dtypes, arithmetic
dtype, output geometry/type and ordered converter transforms. It distinguishes
norm +1 from unchanged recurrent norms, negative exponential of A_log, squeeze,
and grouped-to-tiled linear V-head row/column transforms. The selected PQ2 row
declares `frozen_projection_payload_not_dense_conversion`; codes/scales are
not regenerated. The plan is metadata, not source or converted-byte proof.

`verify_profile_converted_vector(..., source_name=..., source_values=...,
converted_values=..., expected_source_sha256=...)` checks one-dimensional F32
output arrays from BF16/F32 source arrays. BF16 operands are raw little-endian
uint16; F32 operands are little-endian float32. Shapes/layout/finite values and
the caller source digest must match. `maximum_bytes` defaults to and is capped
at 64 KiB, bounding `source.nbytes + 2 * converted.nbytes`; this is not peak RAM
and does not include the separately bounded frozen-payload validation. Matrices,
convolution and the frozen PQ2 replacement refuse this vector-only gate.

The API requires Torch, reusing the cached dense environment for optional
synthetic controls. It copies operand arrays, computes the declared F32 vector
transforms under inference mode, and compares exact bytes, not a float tolerance.
Its immutable receipt marks `source_file_verification` and
`converted_file_verification` as `not_checked`: the source hash authenticates
only against the caller's expectation, and the output array is not yet bound
to a GGUF file. Torch loaded pages are not attested. No model runtime approval
or whole-model conversion parity follows; actual file binding and matrix parity
are separate prerequisites for that operand-only API.

`verify_profile_gguf_vector(..., reader=..., source_name=..., source_values=...,
expected_source_sha256=..., expected_file_bytes=..., expected_file_sha256=...)`
adds a separate file-binding check. It requires the imported `GGUFReader` type,
read-only little-endian mmap and a one-dimensional F32 view with exact declared
name/shape/count/offset. It independently rereads the v3 header and selected
tensor-info/payload range, checking alignment/table end and shared mmap offsets.
The whole file is hashed before/after operand comparison in bounded unbuffered
chunks. `maximum_file_bytes` defaults to 64 KiB and is capped at 16 MiB;
`maximum_seconds` is a cooperative 30 s default/600 s cap shared by both hashes.

The caller constructs the reader first using the pinned environment and a
separately bounded file: parser construction is outside this API's budget.
Reader code is not attested by type checking; its receipt explicitly records
`gguf_reader_verification: not_checked`. It binds only the selected vector range,
not other tensor bytes, complete model metadata/tags or tokenizer behavior.
The returned immutable receipt has file bytes/hash/offset and
`converted_file_verification: caller_sha256_and_reader_range_match`; source-file
authentication and runtime approval remain absent. Keep files unchanged; the
checks do not guarantee immutable storage or safety under concurrent writers.

`verify_profile_converted_matrix(..., source_name=..., source_values=...,
converted_values=..., expected_source_sha256=...)` checks raw little-endian
uint16 BF16 matrices against declared source/output geometry and identity or
grouped-to-tiled V-head row/column transforms. Source/output arrays must be
contiguous and finite, and source bytes must match the caller pin. Comparison
is exact bits, including signed zero; no quantizer or float tolerance is used.
Frozen PQ2 replacements and F32 vectors/convolution refuse this API.
`maximum_bytes` defaults to 1 MiB and is capped at 16 MiB, bounding
`source.nbytes + 2 * converted.nbytes`, not peak RAM or projection validation.
The immutable receipt leaves both source/converted file verification
`not_checked`; GGUF matrix views, whole-model equality and source authentication
remain separate gates. Its transforms use NumPy, not an implicit Torch/model
dependency. Reviewed upstream Torch methods are exercised only by optional
synthetic controls in the already cached environment.

`iter_profile_bf16_matrix_chunks(..., source_name=..., source_values=...,
maximum_chunk_bytes=...)` uses the same profile plan to convert borrowed
little-endian uint16 BF16 matrices without whole-matrix F32 expansion.
It yields `(first_row, chunk)` with owned, contiguous, read-only rows in
destination order, including declared V row/head/column permutations.
The output cap defaults to 64 KiB and is at most 1 MiB; at least one row must
fit. Frozen PQ2 replacements and F32 vector/convolution paths are excluded.
No Torch/GGUF import occurs. Exhaustive finite BF16 storage equivalence and
the actual pinned converter transforms are checked by optional synthetic tests.

The source array or read-only mmap is caller-owned and not authenticated or
globally snapshotted. Only yielded rows are copied; later source changes may
affect subsequent chunks. Nonfinite values are refused per chunk, so consumers
must discard partial output if iteration fails. Retaining chunks grows memory;
the cap excludes source storage, index/mask temporaries and profile/projection
validation. This is not a peak-RAM guarantee, file verifier, GGUF writer or
runtime approval. See the [BF16 chunk checkpoint](handover.md#bf16-chunk-primitive-checkpoint-2026-10-16)
for the mmap ownership repair and scoped evidence.

`temporary_profile_bf16_matrix_payload(..., source_name=..., source_values=...,
expected_source_sha256=..., directory=...)` owns a disposable raw matrix file
inside private scratch and yields `(path, immutable_receipt)`. It streams the
iterator without whole-matrix F32 expansion, checks the caller's source-array
hash before creation/after writing/at exit, and checks payload identity/hash
before yield/at exit. `maximum_bytes` defaults to and is capped at 16 MiB;
chunk limits are unchanged. `maximum_seconds` is one cooperative 30 s
default/600 s cap including caller lifetime, not hard I/O/native cancellation.

Finish payload mappings/readers before exit; publish derived results only
after successful closure. Partial files/private directories are removed on
every exit, including source/payload changes, deletion, replacement, deadline
or consumer failure. Source files are not authenticated, and mutable-array
hash checks are not a continuous or atomic snapshot guarantee. The cap excludes
caller storage, profile validation, writer spooling and process peak RAM.
The optional pinned test uses `GGUFWriter(use_temp_file=True)`, requires it
retain no tensor array, closes mappings before owner exit, and closes its
temporary stream on failure. This is not a full converter or real-run approval.
See the [payload owner checkpoint](handover.md#bf16-payload-owner-checkpoint-2026-10-16).

`mapped_profile_safetensors_bf16_matrix(..., source_path=..., source_name=...,
expected_source_sha256=..., expected_file_bytes=..., expected_file_sha256=...)`
owns a read-only BF16 matrix mapping made from a retained no-follow/nonblocking
regular-file descriptor. Profile shard/header pins and selected geometry/range
are checked, with at-most-64-KiB positional hashing reads and file identities
checked before yield and after normal completion. It yields `(mapping,
immutable_receipt)`; derived results are publishable only after successful
closure. Unlike the older verifier, it does not copy the full operand or make
comparison bytes. Source and operand caps each default to and are at most
16 MiB. The cooperative 30 s default/600 s maximum includes caller lifetime.

Finish all borrowed mapping views/iterators before exit, and never use aliases
after closure. A read-only map does not prevent external writes or truncation;
the checks are not an atomic/continuous snapshot or protection from invalidated
mapped pages. Raw mapping verification does not imply finite values; the BF16
chunk consumer checks those. The retained-allocation fixture is not a peak-RAM,
mapped-RSS, OS page-cache or full-model memory claim. See the
[mapped source checkpoint](handover.md#bf16-mapped-source-checkpoint-2026-10-16).

`bounded_source_descriptor_record_v2(descriptor, maximum_bytes=..., deadline=...)`
is the explicit large-source positional hasher with a 6 GiB ceiling. It retains
the existing at-most-64-KiB reads, borrowed-descriptor position, regular-file and
identity/deadline guards. The legacy `bounded_native_descriptor_record` keeps
its 64 MiB ceiling and signature. Neither selects the other from file size or
environment; existing 16 MiB mapping/payload APIs are unchanged. The sparse
64 MiB + 17 byte control is synthetic only, not model execution or process-RAM
evidence. See the [v2 descriptor checkpoint](handover.md#v2-descriptor-checkpoint-2026-10-19).

`temporary_profile_bf16_matrix_payload_v2(...)` explicitly selects an at-most-
2 GiB raw payload policy, but retains the 16 MiB default caller budget. It
shares the legacy chunked writes, hashes, identity checks, deadlines and
private-file cleanup. V2 receipts add `payload_policy_version: 2` and
`payload_cap_bytes`; legacy 16 MiB APIs and receipt fields are unchanged.
It accepts a borrowed BF16 array or mapping, not source-file provenance:
compose with verified v2 shard ownership and publish only after both owners
exit successfully. A 32 MiB synthetic composition passes exact streamed
hashes, partial-write cleanup and the existing below-3-MiB traced Python
allocation bound. That is not a host/device RAM guarantee or a multi-GiB
artifact measurement. See the [v2 payload checkpoint](handover.md#v2-payload-checkpoint-2026-10-19).

`mapped_profile_safetensors_bf16_shard_v2(...)` explicitly selects the separate
6 GiB file/2 GiB selected-matrix policy and v2 descriptor hasher. Its file and
child budget defaults remain 16 MiB; larger extents require explicit caller
budgets. V2 immutable receipts add `source_policy_version: 2`,
`source_file_cap_bytes` and `source_operand_cap_bytes`. Serial ownership,
provisional publication, pinned header/range/file identity, deadlines, cleanup
and borrowed-alias limitations match the legacy owner. Legacy 16 MiB entry
points and their receipts remain unchanged. Synthetic evidence uses a sparse
source below 128 MiB and a 32 MiB matrix, not real weights or multi-GiB artifacts.
See the [v2 shard checkpoint](handover.md#v2-shard-checkpoint-2026-10-19).

`mapped_profile_safetensors_bf16_shard(..., source_path=..., source_shard=...,
expected_file_bytes=..., expected_file_sha256=...)` retains one verified
descriptor/header and yields `(matrix_factory, immutable_shard_receipt)`.
Use `with matrix_factory(source_name, expected_source_sha256,
maximum_bytes=...) as (mapping, receipt)` for serial child matrices. Only one
child may be live; caller routing is snapshotted, and cross-shard requests
refuse. Whole-shard hashing occurs once at entry and once at successful exit,
with bounded selected-range hashing and identity checks per child. Per-file
and per-child caps remain 16 MiB under one cooperative 30 s default/600 s max
deadline. The per-matrix mapping API now wraps this same owner.

Publish only after parent closure. A swallowed child failure or unfinished
mapping still refuses parent success; private mappings/descriptors close on
failure and closed-owner calls refuse. Finish all child contexts and aliases
first. Concurrency/truncation/issuer limitations are unchanged. See the
[shared shard checkpoint](handover.md#bf16-shared-shard-checkpoint-2026-10-16).

The optional pinned synthetic writer uses one shared shard around all 16 BF16
matrix consumers and checks exactly two whole-file hashes on the same descriptor.
It writes the GGUF header/metadata/tensors only after successful parent exit.
Changed/restored/replaced source shards after child closure must leave no
published output and close the writer's temporary stream. Existing selected-byte
live-mapping refusals remain separate. See the
[shared writer checkpoint](handover.md#bf16-shared-writer-checkpoint-2026-10-16).

`temporary_profile_mapped_bf16_payload(..., source_path=..., source_name=...,
expected_source_sha256=..., expected_file_bytes=..., expected_file_sha256=...,
directory=...)` composes this source mapping with the chunked raw payload owner,
requiring no caller source array. Its single cooperative 30 s default/600 s
maximum spans both owners and caller lifetime; 16 MiB source/operand/payload
caps and existing chunk limits remain unchanged. It yields a private payload
path and provisional immutable receipt with nested source evidence. Finish
payload users before exit, then publish only after successful closure. Both
mapping and temporary payload cleanup apply on failure. Existing source-mapping
concurrency/alias limitations remain. The pinned synthetic GGUF writer control
uses this composition for 16 matrices; selected-byte post-spool mutations fail
on its live-source array hash before the outer file check. See the
[mapped payload checkpoint](handover.md#bf16-mapped-payload-checkpoint-2026-10-16).

The bounded allocation regression compares warmed 64 KiB/2 MiB operands under
an isolated `tracemalloc` session. The 2 MiB copy-based control must expose two
source-sized buffers; mapped source and mapped payload checks require tracked
peaks below 3 MiB and growth at most 256 KiB across the two sizes. Setup/caller
inputs are excluded; mappings, native allocations and OS caches are not a
process-RAM measurement. The shared-shard API also runs this control. Only the
three peak-scaling cases skip when an external tracer already owns the process.
See the [allocation scaling checkpoint](handover.md#bf16-allocation-scaling-checkpoint-2026-10-16).

`temporary_profile_safetensors_bf16_payload(..., source_path=..., source_name=...,
source_values=..., expected_source_sha256=..., expected_file_bytes=...,
expected_file_sha256=..., directory=...)` composes that owner with
`verify_profile_safetensors_operand` before payload creation and after successful
caller completion/payload cleanup. The entire shard, profile-pinned header and
selected range must match; the final source receipt must equal the entry receipt,
including device/inode/size/mtime/ctime. Source access failures become `ValueError`
at this new boundary; existing no-follow/nonblocking regular-file checks remain.

One cooperative 30 s default/600 s maximum deadline covers the whole lifetime.
Source/payload caps remain 16 MiB; `maximum_bytes` retains the source verifier's
three-times-operand charge, copies and comparison-byte materialization, not a
process-RAM guarantee. The yielded immutable receipt nests the entry source
receipt but requires successful exit before derived results are published.
Finish payload readers/mappings first. Body errors skip success verification
but still remove private payloads; callers never publish failed-body results.
These are caller/profile pin checks, not issuer authentication, an atomic or
continuous snapshot, a full-shard low-memory reader, complete conversion or
runtime approval. See the [source-bound payload checkpoint](handover.md#bf16-source-bound-payload-checkpoint-2026-10-16).

The optional pinned `test_pinned_prism_qwen35_value_head_reorder_on_toy_tensors`
BF16 branch builds one at-most-16-MiB physical synthetic safetensors shard with
explicit fixture-only file/header/profile pins. All 16 selected matrices use
read-only source mappings, source-bound raw payloads and the temporary GGUF
writer, followed by the existing exact GGUF file-range controls. First/last
post-spool source mutations must refuse before output publication and close
the writer's temporary stream. The paired F32 branch remains unchanged. This
fixture assembles source bytes in memory and does not measure converter RAM;
see the [source-bound writer checkpoint](handover.md#bf16-source-bound-writer-checkpoint-2026-10-16).

`verify_profile_converted_convolution(..., source_name=..., source_values=...,
converted_values=..., expected_source_sha256=...)` checks declared convolution
weights from BF16/F32 3D source arrays to F32 2D outputs. It decodes BF16,
reorders only the V channel section when required and squeezes the singleton
axis, comparing exact F32 bits. Singleton kernels, wrong tensor kinds, layouts,
shapes, nonfinite operands and wrong source hashes refuse. `maximum_bytes`
defaults to 512 KiB and is capped at 16 MiB, bounding source plus two output
extents, not peak RAM. The receipt verifies operand arrays only; source and
converted files remain `not_checked`, and no runtime approval follows.

`verify_profile_gguf_matrix(..., reader=..., source_name=..., source_values=...,
expected_source_sha256=..., expected_file_bytes=..., expected_file_sha256=...)`
reuses the typed file-range checker behind the unchanged vector API. It requires
BF16 storage with raw uint8 reader rows, reversed GGUF dimensions and an exact
shared read-only mmap range. It independently rereads the v3 tensor-info record
and payload, then compares a raw little-endian uint16 copy with the bounded
matrix reference. Source layout and operand budgets refuse before optional
reader import, file hashing or mapped copies. `maximum_file_bytes` defaults to
16 MiB and is capped there; `maximum_bytes` defaults to 1 MiB and is capped at
16 MiB independently. The existing 30 s/600 s cooperative deadline covers the
two hashes and intervening comparison, not a hard I/O timeout.

Its immutable receipt marks the selected converted-file range as caller-pin
matched; source-file and reader-code verification remain `not_checked`.
Parser construction is outside the budget, and other tensor bytes/model tags/
tokenizer behavior are not proven. It is not permission for real-model reads,
whole-model conversion parity, or executing another profile.

`verify_profile_gguf_convolution(...)` accepts the same file/source pin arguments
for a declared 3D BF16/F32 source and 2D F32 convolution reader view. It reuses
the table-record, exact mmap/range, two-hash and cooperative-deadline checks,
then calls the bounded convolution operand gate. `maximum_file_bytes` defaults
to/caps at 16 MiB; `maximum_bytes` defaults to 512 KiB and caps at 16 MiB.
Singleton kernels and all early source/layout/budget refusals remain. Receipts
mark only this selected range as file-bound, with source-file/reader-code
verification `not_checked`; there is no complete-table or runtime approval.

`verify_profile_gguf_tensor_table(..., reader=..., source_values=...,
expected_source_sha256=..., expected_file_bytes=..., expected_file_sha256=...)`
checks every declared tensor range. Source arrays and digest maps require exact
unfrozen HF-name keys. It preflights layout, per-kind limits, minimum payload
extent and an aggregate `maximum_bytes` (default/cap 64 MiB) before copying arrays
or importing the reader. The charge is source plus twice output extent for
each dense tensor and three canonical frozen payload extents; not peak RAM.
`maximum_file_bytes` defaults to/caps at 16 MiB. Singleton convolution kernels
refuse. It snapshots maps/pins and retains read-only operand copies while
using the same typed range checks for dense and canonical PQ2 data.

`maximum_seconds` defaults to 30/caps at 600, shared across plan construction,
snapshots and all checks; caller reader construction and hard I/O/native
timeouts remain outside that guarantee. The immutable receipt holds immutable
per-tensor records and explicitly leaves `source_file_verification`,
`model_metadata_verification` and `gguf_reader_verification` as `not_checked`.
Whole-table tensor bytes against caller arrays do not authenticate their issuer,
tokenizer/configuration semantics, real source conversion or model quality.
The 16 MiB file bound intentionally excludes real complete checkpoints; relaxing
it or performing real reads requires a separately approved resource/provenance
gate. `PRISM_PARITY_PYTHON` enables this API in the synthetic prefill controls.

An optional `model_owner=...` binds this table API to a live
`VerifiedModelPolicy`, matching its profile ID/hash, model byte count/hash,
ABI and frozen payload pin. Hashing and independent table/payload rereads use
its borrowed descriptor, not the NumPy-resolved reader filename. Positional
reads and temporary duplicates preserve the owner's file position/lifetime;
ordinary path access still refuses symlinks. The receipt's `model_file_access`
is `verified_owner_descriptor` or `reader_filename`. Reader mapping/type/range
checks and all budgets remain unchanged. This is not loaded-page or reader-code
attestation. A table receipt remains provisional until owner exit succeeds:
pathname replacement can leave the original descriptor readable while changing
inode metadata, which the owner still rejects at exit. Never publish that
receipt or scores after a failed owner exit; destroy native users/mappings first.

`verified_profile_model_policy(..., tensor_reader=..., source_values=...,
expected_source_sha256=...)` optionally requires that same descriptor-bound
table gate before yielding the policy. Supply all three inputs together; partial
inputs, files over 16 MiB and invalid `maximum_tensor_bytes` (default/cap 64 MiB)
refuse. Maps/pins are snapshotted before opening the model; the table gate checks
and copies operands under its existing limits. Its immutable receipt is nested
at `owner.verification["tensor_table"]`, with the top-level
`tensor_table_verification` status. Parity, mutation, deadline and budget failures
free the native policy before descriptor closure and before exposing it to the
caller. With this option, the cooperative pre-yield deadline also covers elapsed
file/runtime checks and policy construction, but cannot interrupt blocking I/O
or native calls; parser/archive construction and caller prefill remain outside
it. Existing calls without these inputs retain file/runtime-only behavior and
do not gain table evidence. Destroy caller-owned readers before owner exit and
publish receipts/scores only after successful exit. Synthetic verified full,
parallel and recovery controls use this integrated gate when
`PRISM_PARITY_PYTHON` is set; vector-only controls remain compatible. Source
archive identities are producer-supplied synthetic evidence, not authenticated
HF weights, and model metadata/reader-code/loaded-page claims remain unchanged.

`verify_profile_gguf_policy_metadata(..., reader=..., expected_file_bytes=...,
expected_file_sha256=..., model_owner=None)` verifies 25 selected policy records:
architecture/provenance/execution strings, three root geometry fields and the
14 typed numerical declarations from `profile_model_specification()`. Exact
little-endian string, UINT32, FLOAT32 and INT32-array records must be read-only
shared reader ranges and match independent file reads. An independent bounded
walk checks all metadata keys, supported scalar/non-nested array types, record
extents, uniqueness and the tensor-table boundary; it anchors selected offsets
without interpreting tokenizer or unselected values. Keys are at most 64 KiB,
selected record bytes at most 8 KiB, file at most 16 MiB and keys at most 256.
File hashes bracket checks; a cooperative 30 s deadline (cap 600 s) covers the
gate, not caller parser construction or hard I/O timeouts. Owner access uses
positional borrowed-descriptor reads; ordinary path access remains no-follow.
The immutable receipt marks `model_metadata_verification` as
`selected_policy_records_only`, not complete configuration/template/tokenizer
semantics, source authenticity, reader code or loaded pages. A fresh caller hash
cannot make a changed selected value pass. This standalone API grants no runtime
approval and does not change table-only receipts or existing policy calls.

`verified_profile_model_policy(..., verify_policy_metadata=True)` explicitly
requires this gate after complete tensor parity and before yield, using the
same owner descriptor and remaining cooperative pre-yield deadline. The option
is a strict bool, defaults to false and requires the existing full tensor-reader/
source-array/pin inputs. An immutable selected-policy receipt is retained at
`owner.verification["policy_metadata"]` with top-level
`policy_metadata_verification`; the nested table receipt correctly still leaves
metadata unchecked. Metadata mismatches/mutation/deadlines free the policy
without yielding it. Opted-out/file-only/vector-only callers keep their prior
behavior. Synthetic parity-enabled verified full/parallel/recovery controls
enable both gates. Both receipts remain provisional until owner exit succeeds;
neither selected metadata nor runtime file pins attest loaded code, tokenizer
semantics, independent source-weight provenance or real-model quality.

`verified_profile_model_policy(..., source_paths=..., frozen_source_values=...,
maximum_source_file_bytes=...)` explicitly opts into complete source/converted
table evidence before yield. Supply both new inputs plus the existing reader,
unfrozen arrays and digest maps; do not include a frozen entry in those maps.
The frozen digest comes from the manifest, not a new caller override. Exact
complete source maps/layouts/digests and budgets are snapshotted before model
access; aggregate source file default/cap is 16 MiB. The source-gated deadline
also counts this early preflight. Descriptor-anchored composition uses remaining
pre-yield time and retains immutable `source_conversion_table`, top-level
`source_table_verification`/`source_conversion_table_verification`, and the
unchanged converted receipt at `tensor_table`. Selected metadata may be enabled
independently. Failures free before yield; pointers clear before descriptor close.
Omitted source inputs preserve all prior file-only/table-only behavior. Evidence
is pre-yield source identity, not immutable source storage through scoring, source-
to-frozen quantization equivalence, issuer or quality/runtime approval. Source
files must be quiescent. Destroy native users before context exit and publish
receipts/scores only after successful exit. Actual source-gated synthetic policy
creation/free and bounded native prefill are tested through unchanged hooks.
The two paired source-gated prefill controls use zero dense sources: uniform
logits/scores are an arithmetic/lifecycle check, not nonzero transform or quality
evidence. Separate nonzero seven-role controls are described below.

`verified_profile_model_policy(..., verify_source_files_on_exit=True)` separately
opts a source-gated owner into final selected-shard identity/digest checks. The
flag must be a strict boolean and requires the existing complete source inputs;
default `False` preserves pre-yield-only source verification. Source-owner
receipts explicitly distinguish `source_file_lifetime_verification` values
`pre_yield_only` and `entry_and_exit_checks_required`. The latter is a required
exit check, not a pre-yield assertion that exit succeeded: all evidence/scores
remain provisional until successful context closure.

The opt-in snapshots absolute source paths without resolving symlinks, plus
the verified device/inode/size/mtime/ctime identities before yield. Exit reopens
each selected shard no-follow/nonblocking, hashes against its profile pin in
<=64 KiB positional chunks, and compares descriptor/path identities to the
pre-yield records. Persistent changes, restored bytes/mtime, identical-byte
replacements, symlinks and missing files refuse. Caller path-map/path-like/cwd
or inventory changes cannot redirect the snapshotted exit checks. Source-file
aggregate default/cap remains 16 MiB; this is one additional cooperative
`maximum_seconds` phase (default 30, cap 600), not a whole-session or hard I/O/
native deadline. Even exit failure or caller exception frees the policy and
clears pointers before closing the model descriptor. Entry/exit checks do not
prove immutable storage between observations or an atomic multi-file snapshot,
issuer provenance, frozen quantization equivalence, quality or runtime approval.
Both paired BF16/F32 source-gated actual-loader controls exercise this opt-in;
the nonzero seven-role caller also opts in and tests deferred output/refusal as
described below.

Single/two-shard lifecycle controls exercise first/last-file changes, restored
bytes/mtime, replacement, symlink/missing inputs, caller failure, mutable maps/
path-like objects/cwd and default opt-out. A deterministic clock charges 20
seconds per real bounded hash against the same 30-second exit deadline: one
shard succeeds and two refuse after both hashes, with free-before-close in
either case. This tests shared phase accounting, not actual hash throughput.
The paired BF16/F32 controls also include two physical shards with separate
headers/local offsets/index entries/profile pins, complete 28-tensor parity,
selected metadata, native prefill and exit checks. Their receipt must count
both shards and exact aggregate bytes <=16 MiB; restored source writes and
fresh-pin converted corruption still refuse. These paired fixtures have zero
dense sources, separate from nonzero arithmetic and representative quality.

Each two-shard paired mode also changes either physical shard after successful
native prefill, restores a byte plus mtime, or replaces it with identical bytes
using <=64 KiB copy chunks. All 12 BF16/F32 shard/effect combinations require
nonempty withheld native bytes, then exact source-exit refusal, cleared handles,
closed public descriptor and baseline registry. A score-parse counter must not
advance on failure. This proves post-prefill cleanup/withholding for both files,
not atomic storage, nonzero model behavior or external pin authenticity.

`verify_profile_safetensors_operand(..., source_path=..., source_name=...,
source_values=..., expected_source_sha256=..., expected_file_bytes=...,
expected_file_sha256=...)` binds one unfrozen BF16/F32 operand to an actual
profile-pinned safetensors shard/header/range. The checked conversion plan
determines dtype/shape; unknown/frozen replacement operands refuse. Caller file
size/hash must equal the profile's full-file pin. Header size/hash must also
match, and the existing inventory parser validates complete offset coverage
before selected-range comparison. The source array is copied and digest-checked;
the file is opened no-follow/nonblocking, hashed before/after and read positionally
in <=64 KiB chunks. Device/inode/size/mtime/ctime and the pathname identity must
remain stable, including writes restored to original bytes. File default/cap is
16 MiB; operand default/cap is 16 MiB charged as three source extents, not peak
RAM or header/plan memory; header cap is 1 MiB plus its eight-byte length.
Cooperative default 30 s/cap 600 s checks do not interrupt blocking I/O. Immutable
receipt retains the device/inode/size/mtime/ctime tuple as `source_file_identity`
and reports only this source range; pin issuer, converted file and runtime
approval remain unchecked/not granted. Caller must separately bind the same
digest to converted tensor parity. The 16 MiB cap excludes real 9B shards;
no real reads or multi-file/whole-source claim is implied. Source synthetic
controls use genuinely pinned zero-filled disposable shards, not the frozen
candidate's source or representative numerical/quality evidence.

`verify_profile_frozen_source_tensor(..., source_path=..., source_values=...,
expected_file_bytes=..., expected_file_sha256=...)` uses the same file/header/
range gate for the frozen manifest's original BF16 tensor and declared source
digest, with the same bounds. Its immutable receipt records the separate frozen
payload digest, but explicitly sets `source_to_frozen_quantization_verification`
to `not_checked`. This verifies source identity only: it neither quantizes,
refits, rewrites codes/scales nor proves dense-to-frozen equivalence or quality.
The public unfrozen API still refuses frozen operands. The synthetic frozen
control uses a disposable manifest with its zero-source digest; the sole real
candidate and its manifest are unchanged.

`verify_profile_safetensors_tensor_table(..., source_paths=..., source_values=...,
expected_source_sha256=...)` verifies every selected conversion-plan source
tensor, including the frozen original source. Provide exact maps keyed by selected
shard filename and HF tensor name; no missing/extra entries. Frozen source digest
must equal its manifest declaration. Before file dispatch it validates every
dtype/shape/contiguous layout/digest and snapshots arrays/maps/encoded payload.
The payload must be a contiguous uint8 array. Aggregate operand default/cap is
64 MiB, charged as three source extents plus three canonical payload extents;
each operand retains the 16 MiB cap. Aggregate selected shard bytes, not just
each file, default/cap at 16 MiB. Plan/header memory is not peak-RAM bounded.
One cooperative 30 s/cap 600 s deadline covers all range checks and final shard
rechecks. Immutable inode/timestamp identity must match between operands and
again at completion. Shards must be quiescent; there is no atomic cross-file or
safe concurrent-writer guarantee. The immutable receipt contains all source
receipts, with frozen identity separate. Unselected/MTP operands, converted bytes,
quantization equivalence, pin issuer, metadata semantics and runtime/quality
approval remain unchecked/not granted. This standalone source gate is used by
the explicit composed owner opt-in, not automatic scoring. Default controls read
only disposable pinned zero-source shards; no real source reads are authorized.

`verify_profile_gguf_source_tensor_table(...)` composes complete selected-source
and converted table checks through shared copied read-only arrays/digests and
frozen payload. Supply the same complete source maps plus converted reader and
file pins. Both existing source/converted layout and aggregate preflights run
before source dispatch; the reader's exact tensor set and optional live owner's
identity must match. One cooperative deadline covers source check, converted
table and source recheck. Operand charge is the larger of the source table charge
and converted table charge plus three original frozen-source extents, capped at
64 MiB. Aggregate source shard files and the separate converted file each cap at
16 MiB, not 16 MiB combined. This is not a peak-RAM guarantee.
Immutable receipt nests unchanged source and converted receipts, preserving the
converted-only source claim. Restored inter-stage source writes refuse. Original
frozen source digest and canonical PQ2 payload remain distinct: quantization
equivalence is unchecked. Optional `model_owner` anchors converted reads, but
this gate runs in the owner only through the explicit source opt-in; receipts remain
provisional until owner exit succeeds. Quiescent files are required; no atomic
cross-file, issuer, model metadata/tokenizer, loaded-code, quality or runtime
approval claim is added.

`verify_profile_gguf_source_operand(...)` composes one unfrozen source-file
range and its typed converted GGUF range through the same copied read-only
operand and digest. Supply `expected_source_file_bytes`/
`expected_source_file_sha256` plus the usual converted file pins. One shared
cooperative deadline covers plan/snapshot, source check, converted check and a
source recheck after conversion; per-file cap remains 16 MiB, operand-stage
limits remain unchanged and snapshot/plan/header memory is not peak-RAM bounded.
The recheck compares retained file identity as well as bytes, refusing writes
restored between the two stages. Final pathname identity uses no-follow `lstat`.
Optional `model_owner` anchors the converted file; mismatched/closed owners and
wrong source dtype/shape refuse before source-file access. The immutable receipt
contains unchanged immutable `source` and `converted` receipts and explicitly
sets `cross_file_atomic_snapshot` to `not_guaranteed`. It does not retroactively
change the converted-only receipt's source claim. This is one selected range,
not a whole source/converted table, model metadata, issuer, loaded-code or
quality gate. Source files must be quiescent; rechecking does not guarantee
immutable storage or safe concurrent writers. A frozen replacement remains
unsupported, and no runtime approval is granted.

The optional paired-file controls use `PRISM_SOURCE_DIR` and
`PRISM_PARITY_PYTHON` from the synthetic complete-stage command above:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
   tests/test_prism_native_control.py -k source_to_gguf_selected_range
```

The original cases create pinned zero-filled safetensors and either four-tensor selected or
complete 28-tensor synthetic GGUF fixtures in disposable scratch. Selected cases
test BF16/F32 vectors/convolution, a BF16 matrix, and separate frozen BF16 source/
canonical PQ2 identities. Complete cases compose all ranges for BF16 and F32
vector/convolution variants. They refuse changed converted bits even with a fresh
file hash and restored inter-stage source writes. The two `complete-owner-*`
cases additionally require `MIMO_PRISM_RUNTIME_BUILD` and use the existing native
policy factory/free, selected metadata and actual source-gated native prefill.
They capture <=64 KiB provisional native output in disposable scratch and validate
it only after successful owner exit: genuine BitNet dispatch count, one repack,
typed inspect/edit/ask IDs 11/17/23 with labels A/B/C, zero generated tokens,
zero logits/uniform conditional scores and exact registry restoration. Native
users are destroyed before policy exit. Hooks and numerical tolerances are
unchanged; there is no real complete-model conversion, independent HF source,
quantization-equivalence or representative quality evidence.

Four additional `norm-nonzero-*`/`exp-nonzero-*` cases bind actual nonzero
BF16/F32 source vectors to selected GGUF F32 ranges. They recompute only disposable
source profile pins and update the corresponding synthetic inventory/manifest
identity, preserving header geometry and frozen payload. Expected norm-add values
use F32 arithmetic; negative exp uses the independently fixed four-head index
order 0/2/1/3. Freshly pinned norm no-op and omitted-head-permutation results
refuse, as do restored source writes and changed converted bits. This is nonzero
selected-vector arithmetic/range evidence, not complete-model/native nonzero
evidence, upstream pin-issuer authentication or representative decision quality.

Four further nonzero cases cover raw BF16 matrix QKV rows and output columns,
plus BF16/F32 convolution row permutation and 3D-to-2D squeeze. Their independent
toy index order preserves the first 128 Q/K rows and tiles four 32-wide value
heads in order 0/2/1/3; output columns use the corresponding 128 indices. Distinct
finite source bits expose omitted reordering. BF16 outputs are written as exact
encoded bytes, not round-tripped through a float quantizer. Fresh-pin unpermuted
matrix/convolution outputs, source restored writes and changed converted bits
refuse. These eight nonzero selected-range cases do not execute native prefill;
their transform discrimination is separate from the native controls below.

With `PRISM_PARITY_PYTHON` present, the existing seven-role profile/manifest
native prefill test also creates a separate nonzero source-gated variant. It
writes the same 27 dense operands plus a distinct zero-valued original frozen
source across two disposable BF16 safetensors shards, with separate headers,
local offsets and profile pins under one aggregate <=16 MiB bound. It updates
only that variant's identity declarations; the baseline stays single-shard.
Frozen PQ2 codes and independent FP16 row/group scales stay unchanged. Its
complete source/converted table and 25 selected policy metadata records pass
inside the six-runtime-file-pinned owner before native full prefill. Nonuniform
logits and typed zero-generation conditional scores match the existing
independent FFN/full-attention oracle at unchanged `2e-5` tolerances; the second
writer's oracle must match the original exactly. Descriptor access, source
recheck, registry cleanup and one genuine BitNet dispatch/repack are checked.
Both source shards and the second GGUF/archive use one disposable directory, removed
on success/failure. This is nonzero synthetic integration, not upstream issuer
provenance, nonzero recurrent-state coverage, source-to-frozen quantization
equivalence, tokenizer semantics, quality or any real-profile execution approval.

The separate `recurrent-*` cases add bounded nonzero recurrent integration
for all seven frozen FFN/full-attention roles, without changing the native hook. Sparse
BF16 QKV/gate/output matrices and a four-tap F32 convolution drive the existing
fixed-gate delta-state reference (two key heads, four value heads, 32 dimensions).
Dense inputs use the same explicit BF16 rounding as the other references. Toy
source operands invert the literal four-head order `[0, 2, 1, 3]` on V rows,
gate rows and output columns; the convolution coefficients are exactly BF16
representable. Complete two-shard source/converted parity remains required.
State and residual norms must exceed 0.01; skipping recurrence, resetting state
per token or resetting convolution history must each change selected logits
beyond the unchanged `2e-5` native tolerance. Both writers must reproduce the
full/peer/max128 references and all three fault deltas exactly. This extra
seven-role set runs 49 source-gated workflows, 42 exit refusals and 28 pre-yield
refusals, separately from the original seven zero-recurrent role fixtures.
Attention roles combine recurrent state with nonzero causal attention, including
the output role's different hidden/head geometry; dispatch/repack, typed order,
parallel contexts, chunk/reset, 128-token maximum and kernel recovery retain
the same checks and tolerances.
It is fixed-gate synthetic state coverage, not arbitrary learned gates,
initialized-state coverage, real-model numerical agreement or quality evidence.

The additional `dynamic-recurrent-*` probes use synthetic input-dependent
alpha/beta projections and nonuniform per-head dt biases, with A fixed at -1.
Its independent NumPy reference evaluates sigmoid beta and
`exp(-softplus(alpha + dt_bias))` decay, matching the pinned native expression.
Both gates must vary across heads by >0.01, and across tokens when inputs differ;
identical repeated tokens are not required to change gates. Replacing them with
fixed 0.5 gates joins the three recurrent fault variants and must change logits
outside `2e-5`. Source alpha/beta rows and bias heads invert the same literal
toy head order, and complete physical/converted parity still applies. This extra
seven-role set runs 49 workflows, 42 exit and 28 entry refusals, including
combined recurrent/full-attention paths and the different output geometry.
Together the fixed- and dynamic-gate recurrent sets add 98 successful workflows,
84 exit and 56 pre-yield refusals, separate from the zero-recurrent baseline.
These checks cover synthetic gate arithmetic, not learned-model quality or
initialized-state coverage.

The same source-gated variant also runs parallel independent contexts and kernel
recovery for all seven roles: three dispatch/prefill calls and one repack in
parallel, or one rejected call/kernel error followed by one successful dispatch
in recovery. Primary/peer logits match their independent oracles; IDs
11/17/23, labels A/B/C, inspect/edit/ask, three input/prefilled tokens and zero
generated tokens are explicit requirements. Both source-writer reference
sequences must exactly equal the original references. The unchanged registry
baseline and owner closure checks apply to all three source-gated modes.

The source-gated variant also covers chunked/reset prefill, typed option
reordering and the existing 128-token maximum. Chunk/reset require two successful
dispatches and one repack; chunk's last dispatch has two tokens. Maximum has
128 input/prefilled/last-dispatch tokens and one dispatch/repack. Reordered
IDs/labels/token IDs and logits/scores follow C/A/B order. All compare to the
existing independent oracle at unchanged `2e-5` tolerances; the second writer's
128-token oracle must also exactly equal the original. The complete surface is
49 positive source-gated role/workflow combinations, plus the 28 source refusals
below. Unsupported full-only/large-state controls remain refusals. This does
not widen the native hook, source/model byte or operand caps, subprocess deadline,
token bound, model support tier or real-model authorization.

All 49 nonzero source-gated workflows enable the exit recheck. Their caller
captures native stdout in a disposable temporary file, requires successful
prefill and <=64 KiB output, restores stdout before owner cleanup and releases
the buffer only after successful owner/source/runtime closure and registry/
pointer cleanup. Six further exit-refusal modes per role change either shard's
source byte after prefill, restore that byte plus mtime after writing, or replace
that shard with identical bytes via <=64 KiB copy chunks.
These 42 cases require successful prefill with a nonempty withheld buffer, then
source identity refusal, cleared pointers/restored registry and no score output.
This is distinct from pre-yield input refusal; it does not prove continuous
immutability, external issuer provenance, frozen equivalence or quality.

Two physical-source refusal modes run on either shard for every eligible role
(28 combinations). One changes a source byte without changing its pin and
requires `source operand file identity mismatch`; that byte is restored before
checking the other shard. The other writes a selected embedding byte (or another
unfrozen operand on the other shard) inconsistent with the archive, then computes
fresh shard/header/profile pins and matching GGUF identity metadata; it requires
`source operand file range byte mismatch`. Each fresh-pin writer starts from the
same unchanged two-shard fixture/operands. Thus selected-range equality is tested
without relying on stale hashes or metadata refusal. All must refuse before owner
yield/prefill, restore the registry baseline and emit no scores. Only disposable
synthetic bytes/pins change; frozen payload and original frozen-source identity
stay untouched.

Bounds: 427 tensors, 32 layers, <=19 GiB declared weight bytes plus 32 MiB
metadata/alignment reserve; the single projection retains P3's geometry bounds.
Python also reports `maximum_metadata_bytes` and `maximum_metadata_keys` (256).
The native factory checks magic/version, exact tensor count and <=256 metadata
keys before parsing, then uses pinned GGUF callbacks capped at 32 MiB and 64 KiB
per read. Padded data offset must fit 32 MiB; the whole file must fit tensor
bytes plus that allowance. Valid near-limit metadata, exact-key/file ceilings
and over-limit/header refusals pass. This corrects the historical 1 MiB reserve
blocker without changing v1 or the prefill test hook. The cached tokenizer-only
[planning evidence](research-audit.md#real-v2-planning-limits) is not a final
GGUF-size, real tokenizer/runtime or source-weight authentication claim.
The policy retains a canonical reference array in addition to compact resident
tensor blocks. Do not extend P3's zero auxiliary tensor-array count to total
policy/process memory. Synthetic table/refusal, isolated 8,192-row loader
arithmetic and fixed tiny complete-model loading/prefill through v2 pass.
No generic execution CLI, new source authentication or per-model runtime approval
is introduced. All MiMo v1 factories and refusals remain unchanged.

The compiled test control's `--prefill-profile-bitnet-control MODEL REFERENCE [MODE]`
and `--prefill-profile-bitnet-trace-control MODEL REFERENCE TRACE_DIRECTORY` routes
accept only the fixed four-layer F32 synthetic table, `synthetic-qwen35-prefill-v2`
declarations and its layer-3 FFN-down projection. Both files are capped at 1 MiB.
The reference must be a GGUF v3 single PQ2 tensor with the identity execution
contract and exact 32 x 256 geometry. It is supplied separately, never extracted
from the model under validation. These are test routes, not other-model execution
tools. Policy ownership lasts through all model/context/buffer lifetimes. The
shared prefill oracle exercises 1-128 tokens, recurrence/attention, context reset/
isolation, tracing, error/recovery and typed conditional scores with zero generation.
The Python profile/manifest adapter's BF16 plan is also tested end to end for
all seven eligible synthetic FFN/full-attention projections. `native_model_specification()` retains
backing arrays through the record's lifetime. `libprism_bitnet_prefill_test.so`
is a test library, not a production execution ABI. Its C entry point borrows the
checked record, policy and overrides, returning after all model buffers/contexts
are destroyed. It requires both recurrent and full-attention layers and caps
files at 16 MiB, layers at four, hidden/FFN/target dimensions at 1,024 and vocabulary
at 64. Loaded geometry/attention/SSM bounds apply before context allocation.
Real-profile execution still requires its own production caller, source/artifact
provenance gate and separate approval; this hook must not be widened for it.

Pinned BF16 CPU traits use BF16 dot operands and convert floating input rows to
BF16. Dense synthetic references therefore round both weights and inputs using
the pinned GGUF codec. F16 dot traits likewise round query/probability operands
for FP16 K/V-cache matmuls, not only cache values. Group-A8 BitNet references
retain their distinct contract.
This does not resolve historical real-model numerical differences or validate
quality. The all-attention-only Qwen3.5 fixture hit an empty recurrent-state
assertion and is refused by this test hook, not silently treated as supported.

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

The buffer owns each tensor's one repacked weight block array and native trait;
FP16 scale bits are retained in those blocks without a duplicate FP32 array.
Only the bounded named layer-3 PQ2 weight is
accepted. Sequential uploads are validated and packed once when complete;
rewriting packed weights is refused. Ordinary `MUL_MAT` performs native A8 for
1 through 128 tokens, allowing the token count to change between graphs.
`prism_bitnet_cpu_tensor_status_v1` reports cumulative kernel calls/repacks and
status 8 for incomplete or invalidated uploads. Graph compute status alone is
insufficient: failed native execution fills output with NaNs, and the tensor
status must be checked. Nonfinite input rejection can recover on a later batch.
`prism_bitnet_cpu_tensor_storage_bytes_v1` reports logical resident weight bytes
and auxiliary weight-array bytes (zero), returning the current tensor status.
Canonical PQ2 full/chunked/one-byte readback preserves the storage API even
though the private resident code layout is BitNet-specific. This is not a new
GGUF tensor type or permission for another tensor name or larger v1 geometry.

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

Source planning was checked on 2026-10-01, before the vocabulary-only preflight;
host RAM/swap/disk rows were refreshed on 2026-10-05:

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
| Host total RAM | 31,541,739,520 | Host observation, not a target-device budget |
| Host available RAM | 23,374,372,864 | About 21.8 GiB, time-sensitive; recheck before conversion |
| Host swap / currently used | 8,589,934,592 / 0 | Swap is not working-memory acceptance evidence |
| Cache filesystem available | 326,808,473,600 | About 304 GiB, time-sensitive; recheck before conversion |

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
Eight cases cross dense/BitNet FFNs and zero/nonzero recurrent and attention
outputs. Layer 0 optionally exercises causal convolution, Q/K L2 normalization,
gated-delta state with beta/decay fixed at one half, and gated RMS normalization.
Layer 1 optionally exercises grouped-query causal attention with one nonzero
rotary plane and FP16 KV rounding. The other attention/recurrent outputs remain
zero; layer 3 has nonzero gate/up/down arithmetic. The
dense case stores the down projection in F32; the BitNet case stores its exact
ternary codes and two independent FP16 scales per output row as PQ2 bytes.
This is synthetic test data, not another fitted/retained candidate or source
MiMo conversion. No source model weights are opened for this fixture.

`--prefill-control` and test-only `--prefill-bitnet-control` load the model through
the real public API, use one CPU thread and at most 128 batch/ubatch tokens, and
prefill numeric tokens `[3,5,7]`, or 128 copies of token 2 in `maximum` mode,
without sampling or an answer decode. The pinned
context rounds its 128 requested context slots to 256. The BitNet control uses an
explicit exact layer-3 override, a 1 MiB file cap, and four-layer/32 x 256 geometry
checks. It does not use or relax the public one-tensor metadata-gated factory.
Its internal model-tensor accessor is a pinned test dependency, not a public API.

Independent token-by-token NumPy convolution/state and causal-attention math,
plus RMS/SiLU/FFN/head math, checks final logits; the BitNet reference
uses nearest-even group-128 A8 and integer group partials with exact stored FP16
scales. Typed IDs `inspect/edit/ask` map to synthetic numeric slots `11/17/23`.
These are not genuine MiMo label tokens or task-quality evidence. Selected-label
softmax is conditional on these options, not calibrated confidence.

Dense `--prefill-text-control <gguf> <prompt>` accepts an already-rendered prompt
of 1..4096 bytes and 1..128 tokens. It tokenizes A/B/C into distinct single-token
labels, then requires each appended label to preserve the prompt token prefix
and add exactly that label token. The toy prompt `aba` yields `[4,2]`; 128 copies
of `a` yield 128 copies of token 2. Both match independent reference scores.
Empty/oversized prompts and a fixture where appending A changes BPE tokenization
are refused without scores. No sampler, chat-template rendering, or thinking
policy is selected; real MiMo weights are not tested by this synthetic path.

Each fixture checks full, split 1+2-token, reset-after-another-prompt, and reordered
option execution. Nonzero state/output changes are explicitly required, so these
are not zero-output state controls. BitNet has one counted kernel dispatch per
successful decode and one weight repack per model load, including context reuse.
Hybrid position
bounds report the intersection of KV/recurrent retained ranges: `[2,2]` for
three-token prefill, `[127,127]` for the maximum batch, and an empty range
immediately after reset. The additive ABI-v1
`prism_bitnet_cpu_tensor_last_input_tokens_v1` diagnostic measures three input
rows for full/reset/reordered execution, two for the final chunk, and 128 for
`maximum`. These are actual BitNet tensor dimensions, not prompt-length inference;
final-layer output masking does not reduce these controls to one projected row.
It returns current tensor status and the last attempted batch, including failed
input preparation; zero means no compute attempt yet. Reads use the existing
mutex, require a live owned tensor, and do not certify lifecycle-race safety.
Malformed +2 codes, negative/NaN
FP16 scales, NaN head weights, and unknown modes must fail without stdout scores.
Unknown command flags and missing/extra arguments are also rejected before CPU
discovery, rather than falling through to an unrelated successful default report.
The `parallel` mode creates both contexts before concurrent execution. One is
primed with `[4,9,19]`; it then consumes `[3,5,7]` on a worker thread while the
fresh context consumes `[3,5,7]`. Independent three-/six-token references check
both selected logits; nonzero-state cases must yield different selected logits
between the histories. There are three successful BitNet calls and one shared repack.
The per-weight mutex serializes its scratch/compute use; this is initialized
steady-state context isolation, not simultaneous registry/loading/free/unload
safety or a parallel-speedup claim.

The synthetic control also accepts `singletons` (the original three tokens,
three calls), `singletons-80` (80 one-token calls), and `prefill64-singletons`
(64-token prefill plus 16 one-token calls). The latter two use the same fixed
80-token stream, repeating token IDs 3, 5, 7 and ending 3, 5. Each call reuses
one actual `llama_context` and its native memory; no host cache handoff is used.
The independent NumPy reference checks final selected logits and conditional
scores with unchanged tolerances. Hybrid retained-position intersection is
`[79,79]` for both long schedules. BitNet requires 80/17 dispatches respectively,
one repack and one final input row; answer generation stays zero.
The deliberately faulty `singletons-reset` clears native memory before the
last token. The existing position guard must refuse with status 31 and no
stdout scores. This is a synthetic state-loss negative control, not a normal
reset API or authorization for learned-model execution. See the
[native loader continuation checkpoint](handover.md#native-loader-continuation-checkpoint-2026-10-16).

`reset-singletons` warms the same context with 80 singleton tokens (repeating
4, 9, 19 and ending 4, 9), checks position 79, clears native memory and checks
position -1, then replays 3, 5, 7 as singletons. The fresh-reference logits
and scores must match, final position is 2, and BitNet requires 83 dispatches,
one repack and one final input row. A separate NumPy uncleared-history oracle
must differ for nonzero recurrent/attention fixtures. See the
[native loader reset checkpoint](handover.md#native-loader-reset-checkpoint-2026-10-16).

`interleaved-singletons` alternates 80 singleton calls per context while sharing
one loaded model. The peer starts with a three-token prefix; both streams end
with 7, 3, 5 but have different histories. Independent references check both
selected-logit vectors, native final positions are 79/82, and BitNet requires
161 dispatches, one repack and one final input row. This is deterministic
context isolation, not simultaneous model loading/freeing or a speedup claim;
the existing concurrent-batch `parallel` control remains separate. See the
[native loader isolation checkpoint](handover.md#native-loader-isolation-checkpoint-2026-10-16).

BitNet-only `kernel-error` temporarily selects unsupported upward rounding after
loading. The test requires `llama_decode` success but tensor status 1, NaN logits,
zero counted kernel calls, one repack, and no stdout score report. In
`kernel-recovery`, scoped rounding is restored and all context memory is cleared;
the same model/context then reproduces reference scores with one successful
dispatch and no extra repack. Rejected decode attempts are reported separately.
GGML graph/decode success must never replace tensor-status and finite-logit checks.
Model/context/backend cleanup is scoped; files are removed even on failure.
The existing converter-name gate also uses bounded local headers when
`MIMO_LOCAL_DIR` is set, avoiding unnecessary HTTPS after restart.

After the versioned build above, reproduce just this bounded control with:

```bash
PRISM_SOURCE_DIR="$cache/native/prism-source" \
PRISM_CONVERTER_PYTHON="$cache/dense-venv/bin/python" \
MIMO_PRISM_RUNTIME_BUILD="$cache/native/jev-prism-runtime-v1-build" \
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
   tests/test_prism_native_control.py -k tiny_qwen35_native_prefill
```

The original eight-case checkpoint recorded 60 successful native invocations
and 76 refusal checks; current evidence is in the linked handover. These are
controlled fixed-gate/head/rotary fixtures, not
general recurrent/attention correctness, native MiMo weight loading, a full-model
file policy, quality, or a target benchmark.

### Approved Dense Native Reference

On 2026-10-05 the user approved one temporary text-only BF16 GGUF from the
existing pinned snapshot, bounded CPU prefill at at most 128 tokens, zero answer
generation, memory measurement, and cleanup. This does not authorize additional
bulk conversions, a retained quantized candidate, held-out scoring, or a
full-model BitNet file policy. Offline conversion reused the existing dense
environment and pinned converter with `--outtype bf16 --no-nextn --use-temp-file`;
lazy evaluation remained enabled. The writer's spill file lived in the same
temporary cache directory as the output. Sources and cached CPU/base libraries
were not changed.

| Observed Quantity | Result |
| --- | ---: |
| GGUF file bytes | 17,920,693,472 |
| Tensor payload bytes | 17,909,729,280 |
| BF16 tensors / payload bytes | 250 / 17,905,483,776 |
| F32 tensors / payload bytes | 177 / 4,245,504 |
| Text layers / tensors | 32 / 427 |
| Embedding and output shapes | Both 248,320 x 4,096 |
| Conversion elapsed / peak RSS | 5m40.42s / 10,266,840 KiB |
| Direct native load plus prefill elapsed / peak RSS | 4m37.03s / 17,667,176 KiB |
| Streamed BF16 reference elapsed / peak RSS | 50.98s / 1,018,188 KiB |

The temporary GGUF SHA-256 is
`8e34f1b5156e0844d8c3dbf7023ff0d67b7b54e4f6b9f4460160f86bb8fa076e`.
The pinned parser verifies no vision/MTP tensors and BF16/F32-only storage.
Selected prompt embedding rows and A/B/C output rows preserve their source BF16
bytes. These selected-row checks are not an all-tensor transformation audit.
Process measurements recorded zero swaps, but host swap use increased later;
this is not a swap-free host guarantee. Times are observations, not a target
benchmark or a fair cross-runtime speed comparison.

The existing `--prefill-text-control` loaded the real MiMo text architecture and
prefilled the `inspect-before-answer` engineering fixture: 80 tokens, A/B/C IDs
32/33/34, typed options `inspect/edit/ask`, finite full-vocabulary logits, and
zero generated answer tokens. Hybrid retained positions were `[79,79]`.
Requested context 128 was rounded to 256; batch/ubatch remained 128, CPU threads
one, KV F16 (8 MiB), and recurrent state F32 (50.25 MiB). Compute storage was
128.38 MiB. The prompt SHA-256 was
`e4692b6224d74f3e5d5b729573580401b6b9a36d7541624596dc1e9a50502572`.

| Label | Native Logit | Streamed BF16 Logit | Native Conditional Score |
| --- | ---: | ---: | ---: |
| A | 21.367546081542969 | 21.431333541870117 | 0.9705953960369732 |
| B | 15.470935821533203 | 15.568086624145508 | 0.002667920776189328 |
| C | 17.775672912597656 | 17.858489990234375 | 0.02673668318683748 |

Identical prompt hashes and token IDs were checked. Maximum selected-logit and
conditional-score absolute gaps were 0.09715080261230469 and
0.0005860534409651841. Both chose the same option; exact parity is not established,
and these observations are not a newly accepted tolerance or calibrated quality.
The pinned `ggml_get_rows`/`ggml_mul_mat` return F32, while streamed hidden values
are BF16; native KV/recurrent precisions also differ. Those are concrete precision
differences, not a complete attribution of the gap.

At `462dc12`, the real-reference regression is opt-in through
`MIMO_NATIVE_REFERENCE_GGUF`; it never converts or deletes weights. Its first
600-second run timed out despite the successful direct command. The
environment-matched rerun passed in 586.50 s, including metadata and exact
selected-row checks. Native subprocess execution strips the Python-reference
`OMP_NUM_THREADS` setting, retaining the explicit GGML one-thread context and
600-second timeout diagnostics. Resource variability and observed host memory
pressure prevent attributing the earlier timeout solely to OMP or treating the
passing duration as a speed guarantee. Run this large-memory test serially.
No full-model BitNet dispatch follows from this dense reference.

The temporary GGUF and writer/compiler spill directory were deleted after the
checks. Only 156 KiB of generated logs/JSON/timing reports remain outside Git at
`$cache/native/mimo-bf16-reference-report-20261005-vjwvUZ`. The sole retained
ternary candidate's two hashes are unchanged. Reproduction requires separate
approval for another temporary conversion; no test performs one automatically.
While an approved reference is present, run the focused check with:

```bash
PRISM_SOURCE_DIR="$cache/native/prism-source" \
MIMO_DENSE_PYTHON="$cache/dense-venv/bin/python" \
MIMO_LOCAL_DIR="$cache/models/mimo-2367e865d009c13ac81713a2878291d33ab28177" \
MIMO_PRISM_RUNTIME_BUILD="$cache/native/jev-prism-runtime-v1-build" \
MIMO_NATIVE_REFERENCE_GGUF="/path/to/approved-temporary-reference.gguf" \
OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
   tests/test_prism_native_control.py -k mimo_bf16_native_prompt_prefill
```

### Staged Approval Gates

1. **Completed, no weight conversion:** reconciled header budgets, isolated full
   runtime build, dependency provenance, guarded vocabulary-only conversion,
   real native/HF prompt and label parity, and existing one-tensor BitNet proofs.
   Separate synthetic native hybrid prefill, nonzero BitNet FFN arithmetic,
   nonzero recurrent/attention state, chunk/reset, typed order, initialized
   shared-context isolation, runtime error/refusal/recovery, measured 128-token
   BitNet batches, and dense contextual prompt/label scoring controls also pass.
2. **Approved one-shot conversion completed:** one text-only BF16 native
   reference from the existing snapshot retained the full embedding/head and
   excluded vision/MTP. Lazy conversion plus disk spill completed with actual
   peak/dtypes/bytes recorded above; the sole ternary candidate is unchanged.
   Further bulk conversion still needs separate approval. The temporary GGUF
   and spill directory have been deleted; only small measurement reports remain.
3. **Initial dense hosting completed, parity acceptance open:** real-architecture
   load/context/80-token prefill, finite logits, recurrent/cache state, label
   mapping, and zero generation pass directly. Identical-prompt streamed BF16
   comparison has the nonzero gaps above. Diagnose precision/graph differences
   before claiming exact parity or choosing broader acceptance tolerances. The
   environment-matched opt-in regression passes metadata, exact selected-source
   rows, native token/state/score contracts, and zero generation, not a new
   cross-runtime tolerance or BitNet full-model policy.
4. **Approved one-projection hosting completed:** the additive complete-model
   factory binds exact layer-3 identity bytes against the tagged reference while
   leaving the original one-tensor factory unchanged. Real 32-layer native
   prefill, measured 80-row BitNet dispatch, no-score kernel refusal, and cleared
   same-context recovery pass. The temporary mixed model and projection encoding
   are deleted. This is one frozen projection, not full-model ternary export;
   further conversion needs approval, and numerical acceptance remains open.
5. **Still not authorized:** full-model ternary conversion, compensation
   promotion, another fitted/retained quantized candidate, held-out inference,
   calibrated confidence, vision, or deployment benchmarks. Broader quantization
   requires separate resource, arithmetic, runtime, and representative quality
   gates. A successful dense native reference is not ternary quality evidence.

### Approved Single-Projection Native Model

On 2026-10-06 the user approved one temporary complete text model containing the
existing frozen layer-3 FFN-down projection, bounded CPU prefill, measurement,
and deletion. Offline lazy conversion reused the same source snapshot, pinned
converter, and dense environment with `--outtype bf16 --no-nextn --use-temp-file`.
An in-memory `GGUFWriter.add_tensor` hook substituted only
`blk.3.ffn_down.weight` using `pack_ternary_pq2_0`; every ternary code and FP16
row/group scale was preserved. No fitting or cached source edits occurred.

The additive `prism_bitnet_cpu_model_override_from_gguf_v1` requires GGUF v3,
427 tensors, `qwen35` with 32 layers/4096 hidden/12288 FFN dimensions, both full
BF16 vocabulary matrices, BF16/F32 for every other tensor, bounded nonoverlapping
payload ranges, and at most 19 GiB. Required string metadata is:

```text
jev.model.source_revision = 2367e865d009c13ac81713a2878291d33ab28177
jev.bitnet.model = mimo-qwen35-layer3-ffn-down-v1
jev.bitnet.execution = group128-a8-fp32-nearest-even-identity-v1
```

The sole PQ2 target must be 4096 x 12288 and match every byte of the separately
validated, at most 14 MiB, tagged one-tensor reference. `prism.hadamard.*` is
refused. The returned rule is the same anchored layer-3 override as before;
the legacy one-tensor factory still refuses complete models. Sparse fixtures
cover bad tags/revisions/geometry/types, changed bytes, transforms, extra tensors,
truncation, and no-score CLI refusal. Their scalar placeholders test declared
policy, not a loadable MiMo architecture. This factory does not authenticate all
source tensors or prove architecture completeness: the native loader checks
structure, and the caller must load these same unchanged files and retain the
library. It is not a hostile-file parser or a file-race/lifecycle guarantee.

Before this run, host RAM was 31,541,751,808 bytes with 23,175,872,512 available,
swap was 8,589,934,592 bytes unused, cache disk free space was 317,043,560,448
bytes, and `memory.max` was `max`. These are time-sensitive observations,
not peak-fit or target-device guarantees.

| Observed Quantity | Result |
| --- | ---: |
| Mixed GGUF file / payload bytes | 17,833,399,744 / 17,822,435,328 |
| BF16 / F32 / PQ2 tensor counts | 249 / 177 / 1 |
| Exact projection payload bytes | 13,369,344 |
| Conversion elapsed / peak RSS | 4m36.31s / 10,291,588 KiB |
| Direct native load plus prefill elapsed / peak RSS | 1m45.14s / 17,605,340 KiB |
| Streamed frozen-projection reference elapsed / peak RSS | 34.77s / 1,161,756 KiB |

The deleted model SHA-256 was
`e88e873825fc30167ab1c9ae3f54fbc7cb0bf6550132c12ca7b574b3c4c4db81`.
Both full 248320 x 4096 vocabulary matrices remain; vision/MTP are excluded.
The same `inspect-before-answer` prompt hash recorded above yielded 80 tokens,
A/B/C IDs 32/33/34, typed `inspect/edit/ask`, retained positions `[79,79]`, and
zero generated answer tokens. The actual BitNet tensor reported **one successful
dispatch, one repack, and 80 input rows**, not a count inferred from prompt length.
The one-thread/batch128/KV F16 controls and finite entire-vocabulary check remain.

| Label | Native Logit | Streamed Frozen-Projection Logit | Native Conditional Score |
| --- | ---: | ---: | ---: |
| A | 21.478885650634766 | 21.474821090698242 | 0.97170605857764214 |
| B | 15.454629898071289 | 15.387020111083984 | 0.0023508985041923954 |
| C | 17.855735778808594 | 17.81911277770996 | 0.025943042918165567 |

Maximum selected-logit and conditional-score absolute gaps were
0.06760978698730469 and 0.0009491202828953993. Both chose `inspect`; no exact
parity, new tolerance, representative quality, or calibrated confidence follows.
Different precision and thread configurations also prevent a speed comparison.

A saved-report-only check rounds streamed logits to BF16: `[21.5, 15.375, 17.875]`.
Its maximum rounding gap is 0.05588722229003906, matching the recorded head
diagnostic, but native residuals remain approximately `[-0.02111435, 0.07962990,
-0.01926422]`. Head-output rounding alone does not make these logits equal.
Native intermediate activations were not retained; isolating earlier divergence
requires another separately approved temporary-model diagnostic, not refitting.

`--prefill-model-bitnet-control <model> <projection> <rendered-prompt> [mode]`
accepts only `full`, `kernel-error`, and `kernel-recovery` for real prompts.
Unsupported upward rounding on the actual 80-token prompt gives decode status 0
but tensor status 1, nonfinite logits, zero successful calls, and one repack;
`kernel-error` returns 29 with no stdout. Restored rounding and cleared context
then recover finite typed scores with one successful call, one repack, and 80
actual input rows. Real recovery asserts this contract, not exact recovered/full
logit equality; synthetic recovery still checks independent reference logits.
The two real error/recovery regressions passed in 733.45 s combined, peak RSS
17,606,288 KiB, with a 600-second native limit per case. Large jobs run serially;
observed process swaps were zero, not a host or target-device guarantee.

While a separately approved model exists, the existing opt-in regression never
converts or deletes it and verifies exact artifact/reference/model payloads,
selected source embedding/head rows, tokenizer boundaries, dispatch, and scores:

```bash
PRISM_SOURCE_DIR="$cache/native/prism-source" \
MIMO_DENSE_PYTHON="$cache/dense-venv/bin/python" \
MIMO_LOCAL_DIR="$cache/models/mimo-2367e865d009c13ac81713a2878291d33ab28177" \
MIMO_PRISM_RUNTIME_BUILD="$cache/native/jev-prism-runtime-v1-build" \
MIMO_PROJECTION_ARTIFACT="$cache/quantized/layer3-ffn-down-rtn-searched-fp16" \
MIMO_NATIVE_BITNET_GGUF="/path/to/approved-temporary-model.gguf" \
MIMO_NATIVE_BITNET_PROJECTION_GGUF="/path/to/tagged-frozen-projection.gguf" \
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
   tests/test_prism_native_control.py -k 'mimo_bf16_native_prompt_prefill and one-bitnet'
```

Cleanup is verified: both GGUFs and all spill files under
`$cache/native/mimo-one-bitnet.WHaMyP` were deleted. Only 160 KiB of generated
reports remain at `$cache/native/mimo-one-bitnet-report-20261006-WHaMyP`.
The original sole candidate hashes, source pins, and cached CPU/base libraries
are unchanged. Further model conversion still requires separate approval.

## Bounded Prefill Precision Diagnostic

At `63e8b4c`/`2bc3ce0`/`6e8386c`, test-only native callbacks and the streamed
reference capture input embeddings, each decoder output, layer-3 FFN input/output,
and the selected final-normalized row. Each complete trace has 36 tensors and
48,513,024 bytes for the same 80-token `inspect-before-answer` fixture. Raw files
are finite little-endian F32, with exclusive creation and no destination reuse;
the capture caps are 128 tokens, width 12,288, 36 records, and 96 MiB.
`compare_precision_traces` checks a bounded manifest, controlled stage names,
contiguous layers, geometry, exact raw sizes and finiteness before reporting
F64 error metrics one stage at a time. Manifests alone do not attest prompt or
weight identity; bind those separately. This is not a calibration-data format.

The user approved one temporary conversion using the existing snapshot and sole
frozen artifact. It produced exactly the previous mixed-model SHA-256
`e88e873825fc30167ab1c9ae3f54fbc7cb0bf6550132c12ca7b574b3c4c4db81`,
17,833,399,744 bytes, in 10m10.81s with peak RSS 10,226,068 KiB.
The native traced run took 11m57.75s, peak RSS 17,608,600 KiB, and reproduced
the entire earlier native score report exactly, including logits
21.478885650634766, 15.454629898071289,
17.855735778808594. One successful dispatch, one repack, 80 actual input rows,
typed options, and zero generated tokens were preserved. These timings are
resource observations, not a speed comparison or target-device guarantee.

Default streamed BF16 remains unchanged. `fp32` promotes decoder weights/state
while preserving exact source BF16 values and the frozen projection. The pinned
CPU `ggml-cpu.c` BF16 trait instead selects BF16 dot operands and converts F32
right-hand activations. `ggml_bf16_rhs` therefore keeps F32 state/outputs but
rounds dense `torch.nn.Linear` inputs and the selected head input through BF16;
the custom BitNet projection still receives F32 for its own group-128 A8 rule.
It does not reproduce all native recurrent, attention, KV or reduction behavior.
Both nondefault modes require the frozen native artifact. All diagnostic modes
require a synthetic fixture and at least four layers; dataset, calibration and
activation-observer use is refused before model import.

All three references match embeddings exactly and first diverge at `l_out-0`,
before the frozen layer-3 projection. Representative measured RMSE values are:

| Stage | Streamed BF16 | Streamed FP32 | BF16-RHS Experiment |
| --- | ---: | ---: | ---: |
| Layer 0 output | 0.000319831 | 0.0000776813 | 0.0000103885 |
| Layer 3 FFN input | 0.000491040 | 0.000180698 | 0.000140436 |
| Layer 3 FFN output | 0.000845521 | 0.000483064 | 0.000428691 |
| Selected final norm | 0.0448547 | 0.0176708 | 0.0202983 |

BF16-RHS improves early agreement but not every later metric. Its selected logits
are 21.483444213867188, 15.45882797241211, 17.85195541381836; final logit equality
is not achieved. Its run took 1m11.44s, peak RSS 1,695,068 KiB. The original BF16
and FP32 traces took 40.43s/41.69s with peaks 1,163,700/1,696,908 KiB.
Replaying the saved projection on the captured native F32 input is bit-for-bit
equal to the captured native graph output. The existing independent integer/
group-scale reference reports maximum error 3.618173636255051e-7. Thus this
measured upstream divergence is separate from frozen-projection arithmetic;
it does not establish full-runtime equivalence or a new acceptance tolerance.

The selected source-head replay from the same captured native final norm also
isolates operand rounding: unrounded F32 inputs differ from native by at most
0.003826141357421875, while BF16-rounded inputs differ by only
1.9073486328125e-6 (one F32 ULP for A; B/C exact). This verifies the measured
head boundary, not equality of different decoder states or a general tolerance.

With a separately approved model already present, native trace mode is:

```bash
"$cache/native/jev-prism-runtime-v1-build/bin/prism_bitnet_loader_control" \
   --prefill-model-bitnet-trace-control "$approved_model" "$tagged_projection" \
   "$rendered_prompt" "$new_native_trace_directory"
```

The model, projection and rendered prompt contracts remain those of the ordinary
one-BitNet route. Its score-report schema is unchanged; a failed trace produces no
scores. The real opt-in trace regression has a 1,200-second native limit, based
on the observed traced duration; other real prefill cases remain at 600 seconds.
The test never converts weights. Run large-memory jobs serially.
For the source-streamed engineering reference, use a fresh trace directory:

```bash
OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
"$cache/dense-venv/bin/python" -m embedded_jev.streamed_text \
   --local-dir "$cache/models/mimo-2367e865d009c13ac81713a2878291d33ab28177" \
   --layers 32 --fixture tests/fixtures/agent_tool_smoke.json \
   --case-id inspect-before-answer \
   --native-ffn-library "$cache/native/jev-prism-runtime-v1-build/bin/libprism_group_scale.so" \
   --projection-artifact "$cache/quantized/layer3-ffn-down-rtn-searched-fp16" \
   --compute-dtype ggml_bf16_rhs --precision-trace-directory "$new_streamed_trace_directory"
```

Use `bf16` or `fp32` for the other diagnostic comparisons. Raw traces, both
temporary GGUFs and converter spill are disposable: remove them after the approved
checks and retain only small reports. No refitting, new candidate, compensation
promotion, held-out scoring or full-model ternary conversion follows from this
diagnostic. Further model conversion still needs approval; conditional option
scores are not calibrated confidence or representative quality evidence.

At `5b7e559`, the real opt-in trace regression passed in 746.54 s (12m27.58s
including test startup), peak process RSS 17,611,040 KiB. It verifies exact
frozen artifact/reference/model bytes, selected source embedding/head rows,
prompt token and label IDs, finite stage geometry, dispatch/state contracts,
and zero generation. This gate does not accept native/streamed numerical parity.
Cleanup is verified: the entire approved `native/mimo-precision.GiDmcd` scratch,
including both GGUFs, converter/compiler spill, failed/successful raw traces and
regression captures, was deleted. Only 27 small reports (240 KiB allocated) remain
outside Git at `$cache/native/mimo-precision-report-20261006-GiDmcd`.
Both original candidate hashes and both pinned tracked source worktrees are
unchanged. No installation, refitting, held-out inference, push or branch change
was performed. Another model conversion requires separate approval.

## Recurrent Operation Diagnostics

At `a072049`, the existing native graph smoke executable adds
`--qk-norm-control`. Five finite 128-wide vectors, including zero and small
magnitudes, verify native L2's clamped rule `x / max(norm(x), 1e-6)` independently.
The pinned Torch helper instead uses `x / sqrt(sum(x*x) + 1e-6)`; those formulas
are materially different for small inputs. The Hadamard default is unchanged,
and unknown/extra CLI arguments produce no report.

Fixture-only `--compute-dtype ggml_bf16_rhs_qk` adds clamped Q/K normalization
to the earlier BF16-RHS/F32-state experiment, disabling the Torch helper's own
normalization to avoid doing it twice. It requires the pinned norm epsilon and
existing frozen native artifact; prior modes/defaults are unchanged. Dataset,
calibration, activation-observer and shallow-prefix use remain refused.
Eight pinned Torch 2.10.0/Transformers 5.12.1 cases compare chunked output and
returned state with a separately normalized token-by-token recurrence across
1/7/64/80 tokens and zero/nonzero initial state. These are operation checks,
not proof of all native rounding, convolution, attention, cache or gate behavior.

Four-layer probes of both BF16-RHS variants pass on the existing 80-token
`inspect-before-answer` fixture, with one saved projection call and zero
generation. They differ first at `l_out-0` (RMSE 7.758425460008644e-6); this is
a comparison between streamed variants, not with disposed native activations.
The full Q/K-mode run passes all 32 layers, selected source head rows, typed
options and zero generation in 21.31 s, peak RSS 1,678,868 KiB. Its native
projection/reference maximum error is 1.1307724534503905e-7. Measured logits are
21.466835021972656, 15.425870895385742, 17.849796295166016. Against the saved
same-prompt native report, the maximum selected-logit gap is **0.028759002685546875**,
larger than BF16-RHS alone (**0.004558563232421875**). This negative result does
not promote the new mode or establish decision quality; matching one formula
is not full-runtime parity. Timings are observations, not a speed comparison.
All raw traces/compiler scratch under `native/mimo-qk-contract.HNEVga` are
deleted. Only 44 KiB of small reports remain outside Git at
`$cache/native/mimo-qk-contract-report-20261008-HNEVga`. Candidate hashes and
both pinned tracked source worktrees are unchanged; no complete-model conversion,
new fit, held-out scoring, installation, push or branch change followed.

At `e03b48e`/`9408f3f`, the same executable adds
`--gdn-control {1|7|64|80} {zero|nonzero} [raw]`. Actual GGML L2 and gated-delta
graph execution uses width 128, two Q/K heads, four value heads, scalar gates,
and one final-state slot. Sixteen cases compare every output/state value with an
independent NumPy FP64 recurrence at `rtol=2e-5, atol=2e-6`, verify unchanged
input state, and distinguish wrong interleaved head mapping. Raw-gate cases also
exercise sigmoid and softplus, including inputs around the threshold 20.
The actual kernel uses tiled Q/K broadcast and stores state as value-by-key.
The pinned `conversion/qwen.py` reorder base permutes grouped HF V rows, gates,
convolution V channels and output columns into that physical order; a kernel
head-order difference alone is therefore not evidence of a model mismatch.
This is a bounded numerical fixture, not all-source authentication, complete
model/recurrent-cache equivalence, or an accepted model-parity tolerance.

The additive `--gdn-file-control INPUT OUTPUT` accepts externally captured
recurrent operands on the supported little-endian Linux/x86-64 host. A 20-byte
little-endian uint32 header contains magic `0x3144474a` (`JGD1`), token count,
width, Q/K-head count and value-head count. Width is exactly 128, tokens 1-128,
heads 1-32, and value heads must be divisible by Q/K heads. It is followed by
contiguous little-endian F32 arrays in this order:

1. Query, then key: each `[tokens, key_heads, 128]`, unnormalized.
2. Value: `[tokens, value_heads, 128]` in native tiled head order.
3. Raw alpha, then beta logits: each `[tokens, value_heads]`, same head order.
4. Initial state: `[value_heads, 128, 128]`, value-by-key.
5. Delta-time bias, then negative A multiplier: each `[value_heads]`.

Every operand must be finite, each multiplier negative, and EOF must follow the
last array. The maximum input is 8,421,652 bytes. The actual pinned native L2
and raw-gate GDN graph runs on one CPU thread and leaves initial state unchanged.
Output is exclusively created little-endian F32: all token/value-head output
vectors, followed by final state in the same orientation (maximum 4 MiB).
Stdout contains only counts/geometry; malformed inputs and existing output
refuse with status 6 and no report. Computation/nonfinite failures have their
existing statuses. Use trusted regular offline files and an external deadline;
this diagnostic is not a source authenticator, model loader or hard I/O timer.
The compiled control verifies exact legacy raw-gate replay, maximum geometries
and refusals without real weights. Its `31+49`, `1+63+16`, `40+40`, 80 singleton
and 64-token-prefill/16-singleton initialized-state partitions carry each returned
F32 value-by-key state into the next packet, requiring bit-exact output and
final-state equality with the 80-token baseline.
This tests only GDN handoff on supplied Q/K/V and raw gates, not convolution or
full-model cache behavior; it does not change the original numerical tolerances.

The additive `--cached-gdn-control PARTITION [carry|reset]` uses exactly `80`,
`31+49`, `40+40`, `1+63+16`, `singletons` or `64+singletons`, with fixed 80
tokens, width 128, two Q/K/four tiled value heads, nonzero F32 initial state,
raw gates and one CPU thread. A separate native buffer/context owns the
262,144-byte value-by-key state across per-chunk graph/context/allocator
destruction. The GDN output tail begins after `tokens * 4 * 128` F32 outputs;
its `[128,128,4,1]` view feeds dependency-linked `ggml_cpy` into that owner.
Require exact post-teardown data/buffer ownership and state bytes, unchanged
source operands and bit-exact carry results against the old raw-gate control.
Readbacks verify the copy; they must not be used to restore runtime state.
Carry uploads initial state once and never restores it from the host. Reset
is an explicit lost-state fault with an independent reset-aware recurrence.
The [native recurrent ownership checkpoint](handover.md#native-recurrent-ownership-checkpoint-2026-10-16)
records all six schedules, CLI refusals and unchanged tolerances. This is a
synthetic primitive, not convolution/learned-driver/native-loader integration.

The [typed recurrent owner checkpoint](handover.md#typed-recurrent-owner-checkpoint-2026-10-16)
validates a scratch native-state helper for Q/K:value head geometries 2:4,
16:32 and 32:32. Keep the pinned `ggml_gated_delta_net` final `K` argument
**int64_t**, and verify output byte bounds before graph allocation. Require
finite F32 operands, negative decay multipliers, valid head divisibility,
append-only total tokens <=80, exact post-teardown state/address checks and
no host restores. Invalid input must leave the owner usable; an error after
native update must close it. Compare noncontiguous inputs and all six schedules
with compiled controls before learned integration. Record failed preflight
RAM/disk readings and no-child status without lowering initial 12 GiB RAM /
50 GiB disk prerequisites; missing historical readings remain unknown.

The [learned native state checkpoint](handover.md#learned-native-state-checkpoint-2026-10-16)
composes three independently parameterized native recurrent owners with the
existing four-layer resident driver. Only GDN state carry changes; host
convolution history and F32 residual/down sums remain explicit. Compare
all stage/cache records against the **same schedule** reference, enforce
readback-only snapshots and forbidden host restores, and close all owners on
partial advancement. Historical reports may have different schemas: compare
shared values and explicit output/history/state/cache hashes, never infer
unrecorded stage hashes or rerun a successful model job to replace evidence.

The [native state singleton checkpoint](handover.md#native-state-singleton-checkpoint-2026-10-16)
adds learned 64-prefill/16-singleton and all-80-singleton native recurrent
continuation. Preserve same-schedule reference stage/cache checks, the
original batch/singleton differences and all exit pins. The older 64+16 report
has no per-stage hashes; use the fresh in-memory reference for that comparison
and compare only actually recorded historical fields. The retained independent
all-singleton reference has full hashes and must not be recomputed merely to
obtain a newer receipt schema.

The same executable's `--cached-conv-control PARTITION [carry|reset]` is a
separate synthetic history-ownership primitive: eight channels, four nonuniform
taps/channel, 80 tokens, nonzero F32 history and one CPU thread. Reuses the six
strict cached GDN/attention partitions. Native concat prepends `[3,8]` history
to `[tokens,8]` inputs; SSM convolution/SiLU produce token-major outputs and a
strided last-three-input view feeds native CPY into the persistent history
buffer. Expand raw/SiLU roots before the copy; preserve source/filter bytes and
check all history bytes, data address and buffer identity **after** each graph
context/allocator dies. Carry uploads history once with zero host restores;
reset deliberately restores original history at boundaries. JSON exposes copy/
restore counts, raw/SiLU/history values and preservation flags. Independent
raw/history bits, reset-aware arithmetic, all six native schedule bits and
strict refusals are covered by the existing focused compiled test, with
unchanged SiLU tolerance. See the
[native convolution ownership checkpoint](handover.md#native-convolution-ownership-checkpoint-2026-10-16).
That control alone does not establish learned history integration.

The [learned convolution owner checkpoint](handover.md#learned-convolution-owner-checkpoint-2026-10-16)
separately validates a typed history owner (F32, channels 1-8192, four taps,
three retained inputs, total tokens <=80) and synthetic/learned coarse
four-layer resident composition with three native GDN/history pairs plus native attention
cache. Host snapshots are oracle-only; residual/down sums stay host F32. The
scratch module-hook adapter is synchronous and **not thread-safe**, not a
production loader. Keep all original stage/cache checks and source-only upload
guards. A post-update error must close all owners. The combined 64+16 learned
continuation stopped on the unchanged host-swap guard; combined all-singleton
execution remains untested. Do not retry an unchanged failed resource gate,
substitute another schedule seeking green, infer a completed numerical verdict,
or use wrapper RSS/swaps when the timing file is empty. Retain explicit null
experiment measurements, check termination/cleanup and recheck pins read-only.

The [reference lifetime checkpoint](handover.md#reference-lifetime-checkpoint-2026-10-16)
tests a separately pinned scratch variant that removes the no-longer-needed
F32 reference decoder during native replay. Require full bitwise BF16 round
trips and exact supplied down-weight matching before mutation; release all
original parameter objects, retain native operands independently, and restore
only down weight after successful replay for unchanged reference controls.
Do not run another reference forward on the meta-weight module. Keep the
original helper files/pins, numerical/control comparisons and resource guards;
record native-entry RSS and parameter counts without treating object bytes as
guaranteed OS/device savings. This new coarse-tested lifetime variant does not
retroactively pass the old resource-stopped continuation.

The [lifetime pin audit checkpoint](handover.md#lifetime-pin-audit-checkpoint-2026-10-16)
records a later final-publication identity refusal in that variant. Capture
initial full device/inode/size/mtime/ctime identities as well as hashes for
every watched source/proof before replay, and retain expected/actual values
on refusal without publishing a numerical report. A byte-identical touch
must still fail. Matching hashes after failure do not establish unchanged
content throughout execution or identify a writer. Preserve failed logs;
do not reconstruct missing entry identities, reset metadata or weaken guards
to accept a completed calculation. Test stable/touched/changed-byte cases and
publication-hook restoration synthetically before any learned use. Keep
supervisor, phase-marker, failure and numerical report filenames distinct.

The [file event checkpoint](handover.md#file-event-checkpoint-2026-10-16)
adds unprivileged parent-directory/file inotify observation. Treat modify,
close-write, attribute, replacement/deletion and overflow events as integrity
evidence; duplicate notifications are not distinct write counts, and inotify
does not attribute a writer PID. Buffer numerical publication until both
source/proof identity comparisons and event checks pass. Preserve observations
even when computation fails. A short stable synthetic window is not proof
that a longer learned run is safe; same-byte endpoint rewrites still refuse.
Save only relevant owned scratch buffers before execution, not unrelated user
documents, and never reset timestamps, alter user autosave settings or weaken
the audit to obtain acceptance.

The [detached execution checkpoint](handover.md#detached-execution-checkpoint-2026-10-16)
tests an alternative when editor-owned scratch sources cannot be flushed.
Use a disposable diagnostic-code snapshot with explicit type/count/byte bounds,
source identity checks through handoff, byte-identical copies and a retained
original/copy identity manifest. Do not copy weights, quantized arrays, shared
libraries or upstream source checkouts. Declare that copies, not the original
editor paths, become the execution inputs; this is not immutable storage or
continuous verification of the originals. Preserve every execution-file,
model/profile/runtime, event, numerical and resource guard. Exclude only the
current mode's prior output names, keep failed evidence untouched, verify
copied reports before removing the disposable tree, and clean raw packets in
the parent even on failure. Validate the handoff synthetically and natively
before learned use; do not reinterpret a previously refused run as accepted.

The [detached singleton checkpoint](handover.md#detached-singleton-checkpoint-2026-10-16)
records separately verified cached-MiMo 64+16 and all-80-singleton results
through that handoff. Both preserve exact same-schedule stage/cache evidence,
native history/GDN ownership, reference reuse and buffered publication. These
bounded dense four-layer controls do not certify a production loader, another
profile, complete-model BitNet execution, quality or device performance.

The same executable's `--attention-control TOKENS [FIRST_TOKEN]` is a separate
synthetic non-flash attention check: tokens 1/7/80, strict absolute position with
window <=80, width 256, four query/two grouped KV heads and 128 F16 cache slots.
It reports F32 scores/probabilities/token-major values and verifies unchanged
Q/K/V/mask bytes. CPU F16 matmuls round query and probability operands; compare
each operation against the matching rounded-operand reference, not an assumed
batch-invariant answer. The [nonflash attention checkpoint](handover.md#nonflash-attention-checkpoint-2026-10-15)
records the measured rounding discontinuity and allocator-lifetime repair.
This is prepopulated synthetic KV reading, not rotary, cache-write or learned
full-attention compatibility evidence. The focused command below runs it too.

For learned same-input comparisons, retain the synthetic oracle but do not
assume its fixed per-element tolerance holds for every cancelling dot. Diagnose
the error against the actual accumulator/operand precision without replacing a
failed acceptance check. Likewise, a rounded-probability delta times V explains
only part of an output difference unless both final contraction residuals are
accounted for. The separate [learned attention checkpoint](handover.md#learned-attention-checkpoint-2026-10-16)
records failed fixed checks, numerical attribution and fixed-KV read partitions;
it does not verify native rotary, learned KV writes or full-loader compatibility.

`--rope-control TOKENS [FIRST_TOKEN] [text|axes]` independently exercises pinned
partial IMRoPE: counts 1/7/80, absolute window <=80, four F32 heads of width 256,
64 rotary dimensions, sections `[11,11,10,0]`, base 10,000,000 and no YaRN
extension. Inputs are token/head/column; four position axes are shared or
distinct synthetic values in 0-79. It verifies original input/position bytes.
The compiled reference models the kernel's repeated F32 phase multiplication,
not ideal FP64 frequency powers, with independent rotation and bit-pattern-exact
tail/offset checks. See the [partial rotary checkpoint](handover.md#partial-rotary-checkpoint-2026-10-16)
for the failed ideal-phase check and evidence limits. The same focused test below
covers it without loading weights or validating learned/native-loader rotary.

The [learned rotary checkpoint](handover.md#learned-rotary-checkpoint-2026-10-16)
separately replays captured pre-rotary Q/K through that fixed primitive. Matching
the primitive's own phase-aware oracle does not imply bit-exact agreement with
HF phase generation; small differences can cross F16 query/key thresholds and
propagate through the unchanged attention core. Do not substitute one phase
policy or interpret this isolated comparison as native KV-write/loader parity.

The [connected attention checkpoint](handover.md#connected-attention-checkpoint-2026-10-16)
composes dense Q/K/V projections, Q/K norm, partial rotary, the non-flash core,
sigmoid gate and output projection on a trusted normalized input. Keep query/
gate splitting per head, RMS `weight+1` semantics and the F16-core/BF16-projection
boundaries distinct. For output-projection controls, capture gated inputs before
the BF16-RHS hook. This remains a diagnostic composition without native prefix,
residual/MLP, KV writes, full-loader or whole-model BitNet acceptance.

`--kv-write-control TOKENS [FIRST_TOKEN] [contiguous|permuted]` separately tests
actual pinned `ggml_set_rows`: 1/7/80 updates, strict logical window <=80,
width 256, two KV heads and 128 F16 slots. K writes contiguous GQA rows; the
non-flash transposed V path writes scalar indices `channel * 128 + slot`.
Synthetic slots are unique. Independent complete-buffer F16 bit-pattern checks
cover conversion, selected writes, every untouched sentinel and wrong V layout;
source F32 values and I64 indices remain byte-exact. See the [KV write checkpoint](handover.md#kv-write-checkpoint-2026-10-16).
This is not a persistent cache, attention-graph integration, eviction, concurrency,
mean-centered/padded/quantized/multistream or learned-loader acceptance. Any
future file/index API must refuse overlaps and out-of-range rows before GGML,
whose overlapping destination-row behavior is undefined.

The separate [KV composition checkpoint](handover.md#kv-composition-checkpoint-2026-10-16)
connects actual native writes to causal attention using copied F16 host cache
arrays between separate graphs. Require complete-buffer bit equality after
every write, matching-batch causal score/probability/output equality, zero
future probabilities and reset/layout faults. Compare monolithic batching
separately: existing singleton reduction/F16 probability-rounding differences
are not cache-write defects. This does not establish persistent native cache
ownership or integrate learned chunkwise projections, rotary or the loader.

`--cached-attention-control PARTITION [carry|reset]` is the separate synthetic
same-allocation lifetime control. Fixed partitions are `80`, `31+49`, `40+40`,
`1+63+16`, `singletons` and `64+singletons`; carry is the default. One explicit
native backend buffer owns 128 F16 K/transposed-V slots across each chunk's
temporary graph/context/allocator destruction. Actual write-output views feed
causal attention in the same graph. Check stable owner addresses, complete
cache bytes after teardown, unchanged sources/indices, actual final bit
readback and operation-level attention references. Carry restores no cache
from host arrays; reset deliberately restores initial state between chunks as
a lost-state fault. See the [persistent cache checkpoint](handover.md#persistent-cache-checkpoint-2026-10-16).
This is synthetic state ownership, not a learned persistent cache or loader
integration, eviction, concurrency, multistream or model acceptance.

The separate [learned persistent cache checkpoint](handover.md#learned-persistent-cache-checkpoint-2026-10-16)
extends that ownership pattern through a bounded append-only helper on fixed
captured post-rotary learned QKV. Host snapshots are readback oracles only;
forbid cache uploads after construction, compare independent complete-cache
bytes after every graph teardown and keep matching-batch causal read equality
separate from monolithic batching differences. Invalid-input refusal preserves
usable state; compute/readback failure must close the owner. Test idempotent
close, closed/exhausted refusal and normal/exceptional context cleanup. This
still excludes chunkwise learned projections/rotary and loader integration.

The [cached attention sublayer checkpoint](handover.md#cached-attention-sublayer-checkpoint-2026-10-16)
then composes chunkwise native QKV/norm/absolute rotary, persistent F16 cache
writes/core, sigmoid gate and BF16 output projection from trusted normalized
input. Keep absolute rotary offsets, per-head query/gate splitting and input
rounding boundaries explicit. Require per-step complete-cache equality and a
matching-core reference on the **chunk-produced** cache, not a monolithic
projection cache that may differ. Report stage differences and F16/BF16
crossings separately; do not assert projection-batch invariance or infer a
native prefix, complete decoder layer, loader or BitNet acceptance.

The [attention continuation checkpoint](handover.md#attention-continuation-checkpoint-2026-10-16)
adds fixed-prompt 80-singleton and 64+16-singleton schedules through the same
connected sublayer. Require the identical monolithic baseline, exact per-step
cache readback and matching-core bits; report projection, Q/K/V F16 and
output-input BF16 crossings plus final max/RMS differences. Singleton replay
does not generate tokens or establish final-output batching invariance, and
its measured differences do not authorize padding/compensation promotion.

The [attention decoder layer checkpoint](handover.md#attention-decoder-layer-checkpoint-2026-10-16)
adds input/post-attention RMS, both residuals and the full MLP to this attention
composition on trusted prefix output. Preserve `weight+1` norms and the
existing bounded `LayerKernel` gate/up row blocks and three-column-block down
sum. Residuals and block-result summation remain host F32, not a single native
loader graph. Match captured input/prefix identities, compare the complete
same-input HF reference, isolate same-product split/full down differences and
repeat source/runtime pin checks after all additional tensor/reference work.

The [attention layer continuation checkpoint](handover.md#attention-layer-continuation-checkpoint-2026-10-16)
extends that complete layer through fixed-prompt singleton and prefill/singleton
schedules. Require the identical monolithic full-layer hash/reference metrics,
exact cache state after MLP graph teardown and matching-operand core bits.
Separate native-input-norm layer results from earlier fixed-HF-normalized
sublayer results; their operands differ. Observed post-norm/MLP propagation
does not establish batching invariance or authorize numerical-policy changes.

The [native prefix composition checkpoint](handover.md#native-prefix-composition-checkpoint-2026-10-16)
feeds three separately parameterized recurrent native prefills into that complete
attention layer. Reuse the original recurrent/blocked-MLP helpers and verify
per-layer geometry, head order, direct gated scale and `weight+1` residual norms
synthetically. Require exact prior layer-0 packet hashes/metric reproduction,
separate same-input versus propagated HF comparisons, and exact prefix-output/
attention-input hash handoff. Stream weights serially and repeat all final pins.
Each recurrent layer starts with its own zero caches; replaying attention on a
fixed final native prefix is not end-to-end four-layer cached continuation.
Keep copied recurrent state, persistent native attention ownership and host F32
residual/down sums explicit; no loader, BitNet or quality claim follows.

The [four layer continuation checkpoint](handover.md#four-layer-continuation-checkpoint-2026-10-16)
carries separate recurrent histories/states through each coarse schedule before
matching and crossing attention schedules. Require a nonzero synthetic stack
with exact all-stage/cache carry controls and discriminating reset faults;
repeat the prior learned monolithic packet, metric and full-stack hashes.
Bind collected attention outputs to the replay hashes and compare the actual
four-layer result, not a fixed-prefix attention-only drift. This remains
layer-major streaming, not resident token-major routing or loader acceptance.

The [resident stack checkpoint](handover.md#resident-stack-checkpoint-2026-10-16)
adds token-major chunk routing through all four dense layers with their BF16
parameters retained in RAM. Compare against independent layer-major stage/cache
controls and prior matching-schedule learned output hashes; preserve absolute
rotary positions and prohibit attention cache restores. Validate input refusal
before mutation, per-layer state isolation and fail-closed cleanup after a
partially advanced step. Separate host swap growth from process swaps and keep
the unchanged deadlines/reserve. Resident routing does not repair batch drift
or establish full-loader, whole-model BitNet, device or quality acceptance.

The [resident singleton continuation checkpoint](handover.md#resident-singleton-continuation-checkpoint-2026-10-16)
adds 64-token prefill plus 16 singleton steps through all four dense layers.
Require exact resident versus independent same-schedule layer-major stages,
histories/states and final attention cache, not singleton/monolithic equality.
Keep original monolithic evidence and report propagated full-stack max/RMS and
cache crossings separately from fixed-prefix attention replay. This schedule
still consumes the fixed prompt and is not token generation or an all-singleton
result. Keep long reference/comparison stages serial and individually bounded.

The [resident all singleton checkpoint](handover.md#resident-all-singleton-checkpoint-2026-10-16)
completes all 80 fixed-prompt singleton steps through the four dense layers.
Run the long independent reference and resident comparison serially under
separate unchanged deadlines; retain only per-stage/cache hashes and scalar
metrics. Require identical runtime/source/helper provenance and exact hashes
and metric dictionaries, preserving monolithic evidence and measured drift.
Supervisor stage names must differ from child numerical-report names: both
write exclusive JSON files. If final supervisor metadata fails after a child
succeeds, preserve its report/logs/time evidence, identify missing host-sampling
aggregates explicitly and do not rerun the successful child for reassurance.
This remains a four-layer dense control, not full-loader/model parity.

Run the focused checks without any complete model or source-weight load:

```bash
PRISM_SOURCE_DIR="$cache/native/prism-source" \
PRISM_GGML_CPU_LIBRARY="$cache/native/prism-build/bin/libggml-cpu.so" \
MIMO_DENSE_PYTHON="$cache/dense-venv/bin/python" \
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
   tests/test_prism_native_control.py::test_pinned_prism_cpu_fwht_matches_dense_signed_reference \
   tests/test_prism_native_control.py::test_streamed_native_qk_diagnostic_matches_pinned_torch_recurrence
```

No new complete native model was retained or converted for these controls.
Matching full-loader native/reference captures remain a next diagnostic, not
a result established by saved scores or exploratory whole-layer composition.
The [coupled cache checkpoint](handover.md#coupled-cache-checkpoint-2026-10-15)
records bit-exact convolution-history/GDN-state handoff on identical projected
operands. Native projection batching is separately exact for `31+49`/`40+40`,
but not `1+63+16`; fixed-operand cache checks do not establish whole-layer parity.
The [batch dispatch control checkpoint](handover.md#batch-dispatch-control-checkpoint-2026-10-15)
localizes those differences to the singleton token and measures block propagation.
Diagnostic-only zero-row padding restores exact projection/block agreement on
these operands; it is not a production change or combined cached-decoder result.
The [combined recurrent cache checkpoint](handover.md#combined-recurrent-cache-checkpoint-2026-10-15)
then combines projection batching, convolution history and recurrent state:
`31+49`/`40+40` are exact unmodified, while singleton padding remains necessary
for exact `1+63+16` agreement on these operands. This is one recurrent block,
not full-layer/full-loader caching or BitNet full-model execution.
The [cached layer continuation checkpoint](handover.md#cached-layer-continuation-checkpoint-2026-10-15)
extends that diagnostic composition through the original split MLP, including
80 one-token steps and 64-token prefill plus 16 one-token continuations. Exact
controls still use diagnostic singleton padding; the unchanged unmodified
execution retains measured drift. No full-loader/BitNet/quality claim follows.
The [whole layer checkpoint](handover.md#whole-layer-checkpoint-2026-10-15)
records bounded MLP arithmetic, causal precision controls and real GDN state
handoff; its split matmul/routing path is not the full-loader graph. The
[real operand checkpoint](handover.md#real-operand-replay-checkpoint-2026-10-15)
records expanded approval, the swap-guarded failed conversion and two smaller
MiMo layer-0 replays. These observations do not resolve full-model parity or
authorize a guard-relaxed retry, refitting or held-out scoring.

The additive native command
`--prefill-text-trace-control MODEL PROMPT TRACE_DIRECTORY` combines the existing
text scorer with its finite F32 intermediate capture. The destination must not
exist; prompt <=4096 bytes, <=128 tokens, three distinct contextual option tokens,
<=36 stages and <=96 MiB trace bytes retain their original guards. Reports must
exactly equal untraced text scoring in synthetic controls. This is dense tracing,
not BitNet dispatch or profile-v2 execution. Real captures require an explicitly
approved temporary model; keep raw tensors outside Git and remove them after
verified small numerical summaries. The 2026-10-15 approved MiMo experiment's
resource/cleanup plan is in the [handover](handover.md#prompt-trace-checkpoint-2026-10-15).

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
the complete OS image is **not bit-reproducible**, and it embeds the building
host's exported root certificates. Before publishing benchmarks,
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
credentials, or generated checkpoints to Git. Repository ignore rules add only
the generated host certificate files to the user's existing rules.

## ML and GPU Work

Install the quantization ML stack only after host inventory. Use a dedicated
environment and lock, not ad hoc packages in the research base. The inspected
SemIf revision pins Torch 2.10.0 and Transformers 5.17.0; MiMo's configuration
records Transformers 5.12.1 and the base Qwen3.5-9B configuration 4.57.0.dev0.
These are useful compatibility anchors, not proof that all three projects share
one working environment or that one reference implementation fits every profile.

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