# Source Register

Accessed 2026-09-25. These are primary sources used for the initial audit.
Source code was inspected; model weights and benchmark suites were not executed.
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
per-row SHA-256 values; it did not verify a complete weight shard hash. A
synthetic-activation local MSE screen on this slice is documented in
[docs/development.md](development.md) and is not a quality or accuracy result.

## BitNet

Inspected parent commit: `0b341e582afbf9e1011f24744b554c96a3477eb5`.
Its tree records llama.cpp submodule commit
`390c307752ab78fd8189f359d6954c9ba1be74af`, from `isHuangXin/llama.cpp`.
The parent and gitlink were checked out and a model-free `ggml-cpu` library
built with Clang 18 on x86-64. Its exported I2_S dot passed direct toy-vector
tests after explicit group-sum compensation. Subsequently the pinned native
BitNet control GGUF was SHA-256 verified and prefilled through a built `llama`
library; a debugger confirmed `llamafile_sgemm_i2s` dispatch. No MiMo model was
loaded or Qwen3.5 integration validated.

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
inspection used release tag `prism-b10735-842b188`; resolve its full commit and
record the binary hash before adopting it as an execution dependency.

| Source | What it establishes | Limit |
| --- | --- | --- |
| [Bonsai demo](https://github.com/PrismML-Eng/Bonsai-demo) | Deployment harness, released files, required fork for Bonsai 2, reported retention | Not a demonstrated open end-to-end quantizer for arbitrary checkpoints. |
| [Format documentation](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/MODEL-FORMATS.md) | PQ2_0 versus PTQ1_0 versus group-64 Q2_0 and migration hazards | Must be matched to a release; old names/type IDs are ambiguous. |
| [Block layouts](https://github.com/PrismML-Eng/llama.cpp/blob/prism-b10735-842b188/ggml/src/ggml-common.h) | Exact block field sizes, scale placement, and effective bpw | A struct definition alone does not prove encoding order or kernel support. |
| [Qwen3.5 graph](https://github.com/PrismML-Eng/llama.cpp/blob/prism-b10735-842b188/src/models/qwen35.cpp) | Hybrid graph, gated projections, optional MTP path, final normalization and output projection | A capable graph does not prove this checkpoint's export or BitNet execution. |
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