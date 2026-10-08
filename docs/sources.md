# Source Register

Initial audit: 2026-09-25. Updated: 2026-10-08. These primary sources and the
later scoped experiments below support the
[current handover](handover.md#current-checkpoint-2026-10-08).
The initial audit inspected source code without executing model weights or
benchmark suites; subsequent model/native controls are distinguished below.
Paper titles/abstracts and linked project documentation establish their scope,
not a full reproduction of every paper result.

## Model

Target: `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B`.

Observed HF revision: `2367e865d009c13ac81713a2878291d33ab28177`.

| Source | What it establishes | Limit |
| --- | --- | --- |
| [Pinned configuration](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B/blob/2367e865d009c13ac81713a2878291d33ab28177/config.json) | Qwen3.5 conditional-generation architecture; 32 text layers; 4,096 hidden width; 248,320 vocabulary; untied embeddings; vision and FP32 SSM configuration | Configuration is not a tensor-header inventory or runtime allocation trace. |
| [Model card](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B/blob/2367e865d009c13ac81713a2878291d33ab28177/README.md) | Qwen3.5-9B SFT, agentic/visual training mixture, MIT model-card license, reported benchmarks | No evidence of ternary robustness or single-pass decision calibration. |
| [Chat template](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B/blob/2367e865d009c13ac81713a2878291d33ab28177/chat_template.jinja) | `reasoning_content`, structured function/parameter tool calls, explicit non-thinking prefix | Rendering alone does not validate the tokenizer or vision processor. |
| [HF metadata API](https://huggingface.co/api/models/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B/revision/2367e865d009c13ac81713a2878291d33ab28177) | Revision and reported 9,409,813,744 BF16 safetensors parameters | Server metadata; separately reconciled against the pinned headers below. |
| [Pinned weight index](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B/blob/2367e865d009c13ac81713a2878291d33ab28177/model.safetensors.index.json) and bounded shard headers | 760 tensor entries, 4 shards; independently counted 9,409,813,744 parameters / 18,819,627,488 tensor bytes, matching the index total | Header-only: no tensor values, payload hashes, quality, or execution checked. Per-file hashes and byte counts appear in `python -m embedded_jev.inventory` output. |

Initial model/template reads used `main`; the full revision was then resolved
and the configuration re-read at that revision. Future executions must use the
immutable revision and file hashes, not this audit date or a moving branch.
The 2026-09-28 tokenizer-only check fetched just the pinned `config.json`,
`tokenizer_config.json`, `chat_template.jinja`, and 19,989,325-byte
`tokenizer.json`. Their SHA-256 hashes and the exact Transformers/tokenizers
versions appear in `python -m embedded_jev.label_probe` output. No model weights
or processor payloads were fetched for that check.
The later processor metadata check fetched only `processor_config.json`,
`preprocessor_config.json`, and `video_preprocessor_config.json` (1,191, 443,
and 385 bytes). Nested image settings match the standalone image config;
standalone video settings are a matching subset of the nested video settings.
The lightweight tokenizer-only environment initially could not construct the
processor. After explicitly approved CPU-only Torch 2.10.0, Torchvision 0.25.0,
and Pillow 12.1.1 were installed outside the repository, the pinned
`Qwen3VLProcessor` produced the same text-only prompt IDs as the tokenizer and
all-zero multimodal token types for five synthetic agent/tool cases. No image
processing, native runtime, model weights, or model judgments were tested.

The later opt-in [weight-slice reader](../embedded_jev/weight_slice.py) used the
same immutable revision and header offsets to fetch 2,048 BF16 bytes from
`model.language_model.layers.3.mlp.down_proj.weight` in shard 2. It recorded
per-row SHA-256 values; at the time it did not verify a complete shard hash. A
synthetic-activation local MSE screen on this slice is documented in
[docs/development.md](development.md) and is not a quality or accuracy result.
On 2026-09-29 the complete pinned BF16 snapshot was downloaded to one external
cache directory. The [pinned Hub blobs API](https://huggingface.co/api/models/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B/revision/2367e865d009c13ac81713a2878291d33ab28177?blobs=true)
provided four LFS content SHA-256 digests, each verified against the full local
shard; Git blob IDs or LFS SHA-256 verified the 13 nonweight files. The offline
header/index check agreed on 760 tensors and 18,819,627,488 BF16 weight bytes.
The snapshot does not demonstrate MiMo inference, ternary quality, or BitNet
model dispatch.

Subsequent CPU controls execute all 32 text decoder layers through the
[streamed BF16 reference](../embedded_jev/streamed_text.py), validate the
four-layer prefix against the normal Transformers forward, and score verified
one-token option labels without generation. One retained layer-3 FFN-down
projection can replace the dense matmul via the BitNet-derived grouped kernel.
These Python-hosted controls are not full-model ternary GGUF loading or
representative decision quality; commands and recorded results are in the
development guide and handover.

## Public Intent Proxy

On 2026-09-30, CLINC150 source was pinned at
`828f8093932c8fe6ca7936c3d2e52903b1c523de` from
[clinc/oos-eval](https://github.com/clinc/oos-eval/tree/828f8093932c8fe6ca7936c3d2e52903b1c523de).
Attribution: Stefan Larson et al., *An Evaluation Dataset for Intent
Classification and Out-of-Scope Prediction*, EMNLP-IJCNLP 2019,
[paper D19-1131](https://aclanthology.org/D19-1131/).
The pinned [CC BY 3.0 license](https://github.com/clinc/oos-eval/blob/828f8093932c8fe6ca7936c3d2e52903b1c523de/LICENSE)
and [README](https://github.com/clinc/oos-eval/blob/828f8093932c8fe6ca7936c3d2e52903b1c523de/README.md)
were retained with the external source bundle. The
[data file](https://github.com/clinc/oos-eval/blob/828f8093932c8fe6ca7936c3d2e52903b1c523de/data/data_full.json)
is 2,495,390 bytes, Git blob `7a7b26c5f2dfbbf213f3e67d2dd0727e1af545aa`,
SHA-256 `36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0`.
Original in-scope splits contain 15,000/3,000/4,500 train/val/test rows.

Changes: four deterministic cases per original split (seed 902), each with
its gold intent and three shuffled distractors; OOS excluded; normalized
duplicate utterances excluded across the selected subsets. These are public
intent diagnostics, **not** official CLINC150 evaluation or representative
agent/tool decisions. Model-training contamination is unknown. A frozen
validation comparison and four training activation captures were executed.
Later compensation diagnostics fit only training activations and report
validation metrics without fitting on them. Wider validation worsened versus
RTN; no new candidate was saved or promoted. The proxy held-out subset remains
unscored.

## BitNet

Inspected parent commit: `0b341e582afbf9e1011f24744b554c96a3477eb5`.
Its tree records llama.cpp submodule commit
`390c307752ab78fd8189f359d6954c9ba1be74af`, from `isHuangXin/llama.cpp`.
The parent and gitlink were checked out and a model-free `ggml-cpu` library
built with Clang 18 on x86-64. Its exported I2_S dot passed direct toy-vector
tests after explicit group-sum compensation. Subsequently the pinned native
BitNet control GGUF was SHA-256 verified and prefilled through a built `llama`
library; a debugger confirmed `llamafile_sgemm_i2s` dispatch. No MiMo model was
loaded into that BitNet fork or its Qwen3.5 integration validated.

Control model: [`microsoft/BitNet-b1.58-2B-4T-gguf`](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T-gguf/tree/a1f2f1c765812aa8af3f6eda4a313707064bba15),
revision `a1f2f1c765812aa8af3f6eda4a313707064bba15`, MIT. The sole
`ggml-model-i2_s.gguf` is 1,187,801,280 bytes and has SHA-256
`4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`.
The pinned fork's structured GGUF parser found 210 I2_S (type ID 36) among
332 tensors. This inventory, one finite-logit prefill, and a debugger breakpoint
are dispatch evidence for a **different model**, not a MiMo performance forecast.

| Source | What it establishes | Limit |
| --- | --- | --- |
| [BitNet README](https://github.com/microsoft/BitNet/blob/0b341e582afbf9e1011f24744b554c96a3477eb5/README.md) | Official inference project, supported model/kernel matrix, Python/CMake/Clang requirements | Does not advertise plug-and-play MiMo/Qwen3.5 or an RVV ternary path. |
| [I2_S implementation](https://github.com/microsoft/BitNet/blob/0b341e582afbf9e1011f24744b554c96a3477eb5/src/ggml-bitnet-mad.cpp) | `quantize_i2_s`, tensor-level FP32 scale, x86/ARM-specific packing and integer dot products | Our group-128 scales need adaptation; architecture/build variants require golden tests. |
| [Kernel header](https://github.com/microsoft/BitNet/blob/0b341e582afbf9e1011f24744b554c96a3477eb5/include/ggml-bitnet.h) | Public `ggml_bitnet_*` kernel functions and quantization restrictions | Not the PDF's proposed model-level `bitnet_init`/`bitnet_eval_prompt` interface. |
| [Pinned llama C API](https://github.com/isHuangXin/llama.cpp/blob/390c307752ab78fd8189f359d6954c9ba1be74af/include/llama.h) | Real model/context lifecycle, prompt batch evaluation, flagged logits, and sequence-state APIs | C struct layouts remain specific to this revision; no binary compatibility with Prism is assumed. |
| [Top-level CMake](https://github.com/microsoft/BitNet/blob/0b341e582afbf9e1011f24744b554c96a3477eb5/CMakeLists.txt) | `BITNET_ARM_TL1`, `BITNET_X86_TL2`, llama.cpp integration | Flags do not establish generated-kernel or model-format compatibility. |
| [CPU optimization notes](https://github.com/microsoft/BitNet/blob/0b341e582afbf9e1011f24744b554c96a3477eb5/src/README.md) | I2_S GEMM/GEMV and activation-parallel work, measured examples, x86/ARM support | Different models/hardware; not a forecast for MiMo on an N100 or SBC. |
| [Submodule declaration](https://github.com/microsoft/BitNet/blob/0b341e582afbf9e1011f24744b554c96a3477eb5/.gitmodules) | Native dependency origin | Record both parent and gitlink commit; tracking branch names are not reproducible pins. |

The isolated [AVX2 fixture](../native/bitnet_group_scale.cpp) adapts the pinned
I2_S `1x1` packed-code integer dot, with a [separate upstream MIT notice](../native/BitNet-LICENSE.txt).
Codes 0/1/2 require subtracting each 128-value activation sum to recover
ternary -1/0/+1 before applying row/group scales. This is not stock I2_S,
PTQ1_0 storage, or a BitNet/Qwen3.5 runtime integration.

## Bonsai and Runtime Formats

The demo documentation was inspected on `main`. Format and Qwen3.5 source
inspection used release tag `prism-b10735-842b188`, now resolved to full commit
`842b1880415d6f508f03b789e5ce70194def7bfd`. Its CPU `llama` library
was built with Clang 18 outside the repo. The earlier build's `libggml-cpu.so`
SHA-256 was
`52fe58a3333b2cf69ac82132c5db1518dd35c506546a280d42fd00187d01a4dd`;
`libllama.so.0` SHA-256 is
`3e585b7919a91662195ffccb85c2eb6eefdaee0deb383b5a182d309ecafcce06`.
After a container reopen, a persistent-cache CPU-only rebuild of `ggml-cpu`
had SHA-256 `adaaacaf406df3700fb5f05cb5749990e9b17d410d91e13d0c58263ae971a150`.
Hashes are build-specific. Native tests pass signed FWHT parity and a toy
`MAP_CUSTOM2` BitNet-derived grouped-dot node on two repeated evaluations.
The same toy node passes an opt-in two-key-head/two-repetition grouped-V
activation permutation before signs/FWHT, with independent scalar parity.
A later shared `MAP_CUSTOM2` bridge schedules the real frozen layer-3 FFN-down
projection during the Python-hosted 32-layer MiMo text forward. Direct and
graph paths have identical selected scores and final hidden hashes. Its
actual loaded GGML base/CPU library paths and hashes are recorded. This is
not registered model-loadable tensor/codec evidence or a native whole-model
inference result.
A native-A8 callback path also matches the direct and Python-prepared graph
paths' full-text hidden hashes/scores, with Python production-batch A8 forbidden
by the optional test. A separate in-memory signed-Hadamard experiment (seed 773)
checks matching dense weight/input transforms and native sign/FWHT/A8 parity.
Identity artifacts are refused under that rotated policy; no rotated candidate
was retained or promoted.
A pinned PQ2_0 control additionally uses the actual native decoder and
native tensor/MUL_MAT implementation: the full frozen projection decodes
exactly after separate byte conversion. Low-bit-first adjacent packing and
Q8_0/Q8_K activation dispatch were checked in the pinned source. This is
Prism codec/operator evidence only; it does not establish BitNet tensor
dispatch or a complete MiMo GGUF load path.
A separate registered CPU buffer/tensor-trait control now executes `MUL_MAT`
through the BitNet-derived grouped kernel after explicit PQ2 ternary repacking
and native group-128 A8. Counted dispatch matches the direct kernel exactly
for one-/two-token toy inputs and the sole full-size frozen projection. Its
registration/dispatch API was verified against pinned
[CPU traits](https://github.com/PrismML-Eng/llama.cpp/blob/842b1880415d6f508f03b789e5ce70194def7bfd/ggml/src/ggml-cpu/traits.h),
[trait dispatch](https://github.com/PrismML-Eng/llama.cpp/blob/842b1880415d6f508f03b789e5ce70194def7bfd/ggml/src/ggml-cpu/traits.cpp),
[CPU backend](https://github.com/PrismML-Eng/llama.cpp/blob/842b1880415d6f508f03b789e5ce70194def7bfd/ggml/src/ggml-cpu/ggml-cpu.cpp), and
[buffer ABI](https://github.com/PrismML-Eng/llama.cpp/blob/842b1880415d6f508f03b789e5ce70194def7bfd/ggml/src/ggml-backend-impl.h).
The built library exports the required C++ symbols. This is a scoped internal
ABI proof, not a stable public extension API, persistent/concurrent registration,
new GGUF type, or loader-selected BitNet execution. The original PQ2 operator
control above still exercises Prism's distinct Q8_0/Q8_K activation contract.
Later repeated-evaluation and opaque native handle controls reuse one weight
tensor/graph with one repack and counted dispatch, verifying copied byte
ownership, rejection/recovery, two live toy handles, and registry restoration.
The Python-hosted registered backend has identical 32-layer final hidden hashes
and selected scores to direct/callback execution. One reordered synthetic
typed-option case also has exact ID/label/conditional-score parity, with no
generation; it is not calibration or representative quality. Production batch
A8 is forbidden in Python by the native-A8 gate, and explicit native handle
release is checked. Full optional controls, including signed-Hadamard, were
rerun successfully at `4efa59b`; immutable upstream source/library pins did not
change. Neither persistent model-loader registration nor model hosting follows.
A subsequent standalone tagged toy weight import uses the pinned native
[GGUF API](https://github.com/PrismML-Eng/llama.cpp/blob/842b1880415d6f508f03b789e5ce70194def7bfd/ggml/include/gguf.h)
and [structured writer](https://github.com/PrismML-Eng/llama.cpp/blob/842b1880415d6f508f03b789e5ce70194def7bfd/gguf-py/gguf/gguf_writer.py).
Native shape getters verified the row-major byte-shape correction; explicit
identity/group-128/A8 metadata and named one-tensor geometry are required before
payload import. Counted BitNet parity holds after deleting the toy file, while
unsupported metadata/layout/payload/file contracts are rejected. This is not
automatic Prism model-loader dispatch, a complete MiMo GGUF, or a new retained
quantized model. All seven native controls with full-size reuse and module
coverage passed at `cb4fe74`; full-text/synthetic model gates remain `4efa59b`.
No full MiMo GGUF was loaded by Prism, and no full converter or native Qwen3.5
model hook has been validated.
A versioned early-registration control at `b1f306a` verifies actual CPU
discovery's one-time list snapshot and late-init refusal. Its stable buffer owns
per-tensor native traits and passes changing 1/2/128-token grouped-kernel parity
with one repack. The pinned
[model loader](https://github.com/PrismML-Eng/llama.cpp/blob/842b1880415d6f508f03b789e5ce70194def7bfd/src/llama-model-loader.cpp)
probes candidates with a zero-size buffer and a 512-column dummy `MUL_MAT`;
explicit custom-buffer overrides bypass that probe. The new buffer deliberately
refuses implicit dummy selection. At `9a1fbfa`, a bounded tagged-file factory
returns the exact public anchored override; the actual pinned loader, support
units, and backend registry are compiled into a temporary control linked to the
unchanged CPU/base libraries. Native `create_tensor` selection and `load_all_data`
upload lead to exact counted BitNet dispatch for toy and frozen full-size weights.
The full-size test uses an unchanged temporary GGUF encoding, deleted afterwards;
no new policy/candidate is retained. Untagged PQ2 remains ordinary CPU despite
the custom candidate. Initialized mixed-graph concurrency also passes, with no
registry mutation or lifetime race claim. The seven-control gate passes in
22.71 s; GCC bridge-only ASan/UBSan/leak checks pass. This is one-tensor loader
proof, not a complete MiMo/Qwen3.5 model construction or native hosting result.
At `3c4fa81`, a separate versioned CMake wrapper compiles the full pinned llama
library, registry, and dynamic helper against unchanged cached GGML CPU/base
libraries. The actual public `vocab_only` model-load path consumes a temporary
vocabulary-only GGUF created by the pinned converter in the existing dense
environment. Source-weight `get_tensor`/`get_slice` calls are forbidden by the
test, zero weight tensors are verified, and HF/native non-thinking prompt and
A/B/C token IDs agree without decoding/generation. The file is deleted after
the check. All seven native controls with this preflight pass in 41.59 s and
the packaged build passes two CTests; this proves neither full architecture
prefill nor dense/ternary task quality. Resource planning at `ecfcff5` uses only
reconciled source headers; observed RAM/disk and known FP32 converter staging
are distinguished from unmeasured final GGUF and native allocations.
At `50a943b`, the same pinned model builder and hybrid-memory implementation
load and prefill a temporary four-layer synthetic Qwen3.5 GGUF through public
native APIs. Nonzero layer-3 dense and BitNet/A8 FFN final logits match independent
NumPy arithmetic, preserving two FP16 groups per output row. The test-only exact
override has one repack and one counted dispatch per decode; full/chunked/reset
and reordered typed-slot controls pass without answer generation. Corrupt codes,
invalid scales, nonfinite head weights, and unknown modes emit no score report.
At `27cfd63`, invalid control arguments are rejected before discovery instead of
returning the default control's unrelated success report. All nine native controls
pass in 35.30 s, plus two packaged CTests. The pinned
hybrid retained-position range is the intersection of KV/recurrent ranges;
GGUF omits empty BPE merge lists, so the toy vocabulary supplies a valid merge.
The converter name-map check now reuses bounded cached headers when available.
This is synthetic compatibility/arithmetic evidence, not native MiMo loading,
nonzero recurrent/attention correctness, a production full-model override policy,
representative quality, or a benchmark. No MiMo weights were converted here.
At `cf16046`, eight dense/BitNet x recurrent x attention fixture cases extend
that native gate. Independent causal-convolution/gated-delta references use fixed
half decay/beta; grouped-query attention uses one rotary plane and explicit FP16
KV rounding. Material nonzero output changes, full/chunked/reset parity, and two
initialized shared-model contexts with independent three-/six-token histories
pass. Unsupported rounding proves decode success can coexist with BitNet status
1 and NaN logits; score refusal and same-context reset/recovery pass with no new
repack. The per-weight scratch/compute mutex is inspected and retained. All
15 native controls pass in 45.19 s, plus two packaged CTests; the default suite
is 159 passed/32 skipped. This is bounded arithmetic/state/error evidence, not
general head/gate/multimodal correctness, lifecycle-race safety, native MiMo
hosting, quality, or a benchmark. Cached upstream sources/libraries and the sole
retained candidate remain unchanged.
At `76cc0b2`, dense native text prefill tokenizes an already-rendered prompt and
requires distinct single-token A/B/C labels with prefix-preserving contextual
tokenization. Two-/128-token toy prompts match independent scores; invalid
lengths and contextual BPE merges are refused without generation. At `ecc6cbd`,
the stable BitNet tensor exposes a mutex-protected last-attempted-input diagnostic,
preserving existing ABI-v1 signatures and status semantics. Actual three-/two-/128-row
projection batches pass, including 128-token model prefill with nonzero recurrent
and attention paths; final-layer masking does not reduce these controls to one
row. All 15 native controls pass in 43.26 s, plus two CTests and 159 default tests
with 32 optional skips. These are synthetic arithmetic/runtime/tokenizer checks,
not native MiMo weight loading, calibrated confidence, or quality evidence.
At `462dc12`, one explicitly approved offline lazy/disk-spill text-only BF16
conversion of the pinned MiMo snapshot produced 427 tensors (250 BF16, 177 F32),
17,920,693,472 file bytes, both full vocabulary matrices, and no vision/MTP.
Selected embedding/head rows preserve source BF16 bytes. The actual native public
load/context/80-token prefill returns finite logits and typed A/B/C IDs 32/33/34,
with zero answer generation. Conversion/native peak RSS observations were
10,266,840/17,667,176 KiB. An identical-prompt 32-layer streamed BF16 reference
has maximum selected-logit/conditional-score gaps 0.09715080261230469 /
0.0005860534409651841; exact parity is not claimed. Pinned GGML embedding gathers
and matrix products return F32, unlike streamed BF16 hidden values; this is
precision evidence, not full attribution of the discrepancy. The opt-in real
regression passed separately in 586.50 s after an initial timeout; existing
15 native controls passed in 61.35 s, 159 default tests passed/33 skipped, and
two CTests passed. The temporary GGUF/spill directory were deleted, leaving
small reports only. No full-model BitNet policy, new candidate, held-out score,
calibrated confidence, or benchmark follows.
At `dda784a`/`669ed7b`, the separately approved complete-model control binds the
exact frozen layer-3 PQ2 bytes to a tagged reference and preserves BF16/F32
elsewhere. Declared tags, bounded ranges, full vocabulary, and one target are
validated; these declarations do not authenticate every source tensor or prove
architecture completeness. The native loader checks structure. Real MiMo
80-token prefill measures one BitNet dispatch, one repack, and 80 input rows,
with typed A/B/C scores and zero generation. Exact artifact/reference/model
bytes and selected original embedding/head rows pass. Actual-prompt unsupported
rounding proves decode success with tensor status 1/nonfinite logits; no-score
refusal and cleared-context recovery pass. Sixteen existing native controls,
three separate real BitNet cases, 159 default tests, and two CTests pass. The
native/streamed frozen-projection maximum logit/conditional gaps are
0.06760978698730469 / 0.0009491202828953993. Both temporary GGUFs/spill are
deleted; only 160 KiB reports remain and candidate hashes are unchanged. This is
one-projection arithmetic/runtime evidence, not exact parity, whole-model ternary
quality, calibrated confidence, a benchmark, or lifecycle/file-race safety.
The later `63e8b4c`/`2bc3ce0`/`6e8386c` precision controls use bounded native and
streamed raw F32 traces of the same engineering prompt. Embeddings match; first
divergence precedes the frozen FFN at layer 0, while native-input FFN replay is
bit-for-bit exact. In the pinned `ggml/src/ggml-cpu/ggml-cpu.c`, the BF16 CPU trait
sets `vec_dot_type = GGML_TYPE_BF16` and converts F32 RHS inputs through
`ggml_cpu_fp32_to_bf16`. This is operand-precision evidence, not proof that F32
graph outputs imply all-F32 matmul or that the limited BF16-RHS experiment matches
every native operation. See the
[trace contract and measurements](development.md#bounded-prefill-precision-diagnostic).
At `a072049`/`e03b48e`/`9408f3f`, native clamped L2 and gated-delta graph controls
verify normalization, tiled Q/K heads, transposed state and activated/raw gates;
pinned Torch chunked/recurrence checks pass separately. The source owners are
`ggml/src/ggml-cpu/ops.cpp`, `src/models/delta-net-base.cpp`, and
`src/models/qwen35.cpp`. The converter imports split modules: Qwen3.5 inherits
`_LinearAttentionVReorderBase` in `conversion/qwen.py`, which permutes grouped
V/gate/convolution rows and output columns to tiled order. This inspected source
contract is not all-source authentication. The streamed Q/K experiment's larger
selected-logit gap remains a negative diagnostic, not policy or quality acceptance.
See the [recurrent results](development.md#recurrent-operation-diagnostics).
In a model-free check of the pinned `gguf-py` Qwen3.5 tensor-name map, exact
MiMo `model.language_model.layers.3` FFN-down and attention-Q paths returned
no match. The converter's shared tensor filter removes `language_model.`
before mapping; executing that method on the sampled paths mapped them to
`blk.3.ffn_down.weight` and `blk.3.attn_q.weight`. The later metadata-only
check passed all 760 pinned index names through the source-derived text filter,
`.dt_bias` rename and name map: 333 vision-side names filtered, 427 text names
mapped, including 24 SSM biases. This is name coverage, not proof of tensor
value transformations, visual export, or loader compatibility. A separate
source-derived method test using isolated `torch==2.10.0+cpu` and NumPy 2.2.6
checks small QKV/Z/alpha/conv1d value-head permutations, A-log `-exp`, dt-bias
reorder and folded-versus-unrotated `out_proj` behavior. The folded path sets
the grouped-V runtime permutation flag instead of permuting stored columns.
The pinned metadata method writes `prism.hadamard.gdn_v_grouped=true` through
its bool writer API for a toy folded `ssm_out` manifest, and omits it when the
flag is off. A no-tensor GGUF round-trip preserves the bool type and weight
name. The local toy exporter still rejects `ssm_out` without verified
head geometry. These checks do not exercise full converter initialization,
all tensor-value paths, visual export, or the GGUF loader.

| Source | What it establishes | Limit |
| --- | --- | --- |
| [Bonsai demo](https://github.com/PrismML-Eng/Bonsai-demo) | Deployment harness, released files, required fork for Bonsai 2, reported retention | Not a demonstrated open end-to-end quantizer for arbitrary checkpoints. |
| [Format documentation](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/MODEL-FORMATS.md) | PQ2_0 versus PTQ1_0 versus group-64 Q2_0 and migration hazards | Must be matched to a release; old names/type IDs are ambiguous. |
| [Block layouts](https://github.com/PrismML-Eng/llama.cpp/blob/prism-b10735-842b188/ggml/src/ggml-common.h) | Exact block field sizes, scale placement, and effective bpw | A struct definition alone does not prove encoding order or kernel support. |
| [Qwen3.5 graph](https://github.com/PrismML-Eng/llama.cpp/blob/prism-b10735-842b188/src/models/qwen35.cpp) | Hybrid graph, gated projections, optional MTP path, final normalization and output projection | A capable graph does not prove this checkpoint's export or BitNet execution. |
| [Converter registration](https://github.com/PrismML-Eng/llama.cpp/blob/prism-b10735-842b188/conversion/qwen3vl.py), [shared tensor filter](https://github.com/PrismML-Eng/llama.cpp/blob/prism-b10735-842b188/conversion/base.py), [SSM transforms](https://github.com/PrismML-Eng/llama.cpp/blob/prism-b10735-842b188/conversion/qwen.py), and [GGUF tensor mapping](https://github.com/PrismML-Eng/llama.cpp/blob/prism-b10735-842b188/gguf-py/gguf/tensor_mapping.py) | Qwen3.5 registered; all 427 pinned text names map, and tiny synthetic QKV/Z/alpha/conv/out-projection paths passed. | Source-derived toy checks; full tensor transforms, converter and tokenizer/processor export remain unvalidated. |
| [Model loader](https://github.com/PrismML-Eng/llama.cpp/blob/prism-b10735-842b188/src/llama-model.cpp) | `prism.hadamard.*` metadata validation, width-keyed signs, optional GDN permutation, tied/inverse embedding rules, FP32 recurrent caches | Runtime schema is more constrained than arbitrary per-tensor transforms. |
| [Graph helpers](https://github.com/PrismML-Eng/llama.cpp/blob/prism-b10735-842b188/src/llama-graph.cpp) | Matmul transform ordering and inverse lookup ordering; source-disclosed multi-sequence recurrent-cache hazard | Backend FWHT dispatch and cache-risk reproduction remain native integration work. |

The exact proprietary Bonsai optimization procedure, calibration corpus, and
quality on our checkpoint remain unknown. The PDF's secondary reverse-engineering
claims and YouTube timestamps were not independently reproduced here.

## Jev and SemIf

Inspected SemIf commit: `23cf1f39fc9534fe81437200959b6dfc7106e45a`, branch `master`.
The repository retains the historical `SemIf-OpenJev` URL.

| Source | What it establishes | Limit |
| --- | --- | --- |
| [TypeSafe](https://typesafe.ai/) | Jev vendor describes typed decisions and its own architecture, sampler, and RLCD training | Product claims, not an open implementation or independent performance validation. |
| [Jev introduction](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | Public parallel-output claims and benchmark caveats; zero-type-error interpretation of hallucination | Does not supply a reproducible model architecture/training implementation. |
| [Choice](https://docs.typesafe.ai/primitives/choice), [Score](https://docs.typesafe.ai/primitives/score), [Noul](https://docs.typesafe.ai/primitives/noul) | Published categorical, ordinal-expectation, and yes-probability contracts | Our initial SemIf interface is a smaller subset, not a drop-in replacement. |
| [Confidence](https://docs.typesafe.ai/confidence) | Confidence is a statistic derived from the distribution, separate from raw option probabilities | The interactive demo explicitly uses an approximation, not a normative service formula. |
| [SemIf README](https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/23cf1f39fc9534fe81437200959b6dfc7106e45a/README.md) | Independent interface-pattern reproduction; multiple backends; published failure/quality evidence | Does not reproduce TypeSafe's undisclosed model/training. |
| [Method](https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/23cf1f39fc9534fe81437200959b6dfc7106e45a/docs/METHOD.md) | Fixed label logits, frozen workloads, metrics, conditional probability interpretation | A decision scorer can still be semantically wrong. |
| [Core](https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/23cf1f39fc9534fe81437200959b6dfc7106e45a/src/semif_phase1/core.py) | A-P labels, 2-16 option validation, structured evidence and prompt | MiMo integration still needs its exact tokenizer/template checks. |
| [llama.cpp backend](https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/23cf1f39fc9534fe81437200959b6dfc7106e45a/src/semif_phase1/llamacpp_backend.py) | Native token parity, flagged final logits, complete sequence serialization, serial restored branches | ABI-specific; its shared CPU path is not parallel suffix execution. |
| [Dependency metadata](https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/23cf1f39fc9534fe81437200959b6dfc7106e45a/pyproject.toml) | Exact upstream Python dependencies and llama-cpp-python extra | Installation not performed in the lightweight base environment. |

## Research Foundations

| Work | Relevant contribution | What not to infer |
| --- | --- | --- |
| [GPTQ](https://arxiv.org/abs/2210.17323) | Approximate second-order one-shot weight quantization; [reference implementation](https://github.com/IST-DASLab/gptq/tree/2d65066eeb06a5c9ff5184d8cebdf33662c67faf) under Apache 2.0 | A short modified loop automatically preserves arbitrary models at ternary precision. |
| [QuIP](https://arxiv.org/abs/2307.13304) | Incoherence processing and quadratic-proxy adaptive rounding for low-bit quantization | Orthogonal rotations improve the spectral condition number. |
| [QuIP#](https://arxiv.org/abs/2402.04396) | Randomized Hadamard processing, lattice codebooks, and fidelity fine-tuning | Its vector-codebook representation is interchangeable with scalar ternary kernels. |
| [QuaRot](https://arxiv.org/abs/2404.00456) | Function-preserving rotations for four-bit weight/activation/cache inference; [code](https://github.com/spcl/QuaRot) | Published four-bit retention guarantees equivalent three-level PTQ on MiMo. |
| [BitNet b1.58](https://arxiv.org/abs/2402.17764) | A ternary model/training recipe with low-bit activations | Merely packing an existing BF16 model reproduces its training and quality. |
| [T-MAC](https://arxiv.org/abs/2407.00088) | LUT-based mixed-precision CPU matrix multiplication; [implementation](https://github.com/microsoft/T-MAC) | Every format, model, ISA, or prefill workload receives the same speedup. |

## Evidence Rules

- Mathematical statements have explicit assumptions and small executable checks.
- Upstream benchmarks remain attributed results on their reported workloads.
- Hypotheses are labeled until an owned experiment measures them.
- Each experiment records immutable source/model/data revisions and artifact hashes.
- Do not extrapolate data-type support to graph support, or graph support to speed.
- Preserve upstream licenses/notices when vendoring or adapting code. Dataset and
  model licenses are separate from this repository's license.
- Hardware recommendations need actual availability, native measurements, and a
  dated delivered-to-Finland quote. No prices in the PDF were verified in this phase.