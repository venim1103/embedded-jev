# Research Audit

Reviewed on 2026-09-25. The supplied 72-page conversation is a statement of
intent, not a working implementation or reliable benchmark. This audit separates
source-backed facts, mathematical deductions, and hypotheses requiring experiments.
No MiMo weights were downloaded or quantized during that initial audit.
Subsequent source downloads, streamed text inference, and bounded native
integration controls are recorded in the
[current handover](handover.md#current-checkpoint-2026-10-08). This document's
original mathematical/source conclusions are not model-quality measurements.

The later [bounded precision diagnostic](development.md#bounded-prefill-precision-diagnostic)
matches embeddings exactly, first diverges at layer 0, and reproduces the frozen
projection bit-for-bit when replayed on identical native inputs. BF16-RHS operand
rounding explains part of the measured difference, not the complete runtime gap;
no cross-runtime tolerance, calibration or quality acceptance follows.
Later [recurrent controls](development.md#recurrent-operation-diagnostics)
verify the different Q/K norm formulas and actual native output/state/raw-gate
arithmetic. Q/K-only alignment worsens the engineering selected-logit gap;
it is negative evidence against promoting that change, not a quality result.

## Recommendation

Pursue **MiMo ternary quantization + genuine BitNet-derived CPU execution +
SemIf-style typed decisions** as an experimental system. Preserve all three goals,
but do not describe them as an already compatible stack. Quantization quality and
runtime representation must be designed together before a full-model conversion.

The most important missing work is a **group-scaled ternary kernel integration**
for Qwen3.5, not another Python wrapper around a stock BitNet installation.
Retain a native BitNet checkpoint as a runtime control, not as an undisclosed
replacement for the requested MiMo model.

## What Jev Means Here

[SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev/tree/23cf1f39fc9534fe81437200959b6dfc7106e45a)
explicitly states that it reproduces an interface pattern, not TypeSafe Jev's
undisclosed architecture or training. A causal model reads state, a criterion, and
described options, then exposes final-position logits for fixed answer labels.
It does not sample an answer token. This is not a conversion into a JEPA/world
model and does not provide the model's usual multi-step generated reasoning for free.

The current upstream already includes a llama.cpp CPU backend, Torch CPU/GPU,
MPS, MLX, and an EXL3 bridge. The PDF's claim that SemIf is PyTorch-only is outdated.
Upstream reports nonzero error rates and cache-path disagreements, and correctly
calls its option probabilities conditional, not automatically calibrated.

[TypeSafe's own description](https://typesafe.ai/) claims a new architecture,
sampler, and Reinforcement Learning for Calibrated Decisions (RLCD). That is a
vendor statement, not evidence that a frozen MiMo logit readout reproduces it.
Our project targets the open decision interface and measures its own reliability.

### Public Jev Contract

The [official documentation](https://docs.typesafe.ai/) exposes three primitives:

| Primitive | Public semantics | Implication for our prototype |
| --- | --- | --- |
| Choice | Argmax option, categorical probabilities, and a separate confidence statistic; up to 255 options | SemIf's initial A-P implementation supports only 2-16 options. Do not claim full API parity. |
| Score | Probability distribution over 2-10 ordered descriptions; score is the probability-weighted mean of level indices | An ordinal score is not simply an arbitrary generated number. Preserve its distribution. |
| Noul | Probability that a yes/no proposition is true; no separate confidence field | A value near 0.5 is uncertainty, not necessarily a medium degree of the property. |

Jev's confidence documentation describes a statistic derived from the distribution,
not simply its maximum probability. Its interactive Choice demo labels its formula
an approximation; that is not a sufficient specification of the service's exact
confidence computation. Our result schema should name `max_option_probability`
and calibration status explicitly rather than imply an identical Jev confidence.

The vendor describes independent parallel questions in one request. A serial
SemIf CPU loop with prefix restore does not reproduce constant-cost parallel
execution. Likewise, guaranteed output types exclude malformed values, not wrong
semantic decisions. TypeSafe's published "zero hallucinations" discussion uses
schema validity; it must not be interpreted as zero factual or judgment errors.

## Verified Model Facts

The [MiMo model card](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B/blob/2367e865d009c13ac81713a2878291d33ab28177/README.md)
identifies this checkpoint as an SFT of Qwen3.5-9B, including agentic and visual
training. These facts do not establish resistance to ternary quantization.
The [published configuration](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B/blob/2367e865d009c13ac81713a2878291d33ab28177/config.json)
observed during this audit contains:

| Property | Value | Consequence |
| --- | --- | --- |
| Architecture | `Qwen3_5ForConditionalGeneration` | Use a Qwen3.5-aware multimodal loader, not an invented Qwen3.8 adapter. |
| Text layers | 32: 24 linear-attention, 8 full-attention | Discover layer types from configuration; do not assume a 64-layer 27B layout. |
| Hidden / intermediate width | 4,096 / 12,288 | Both divide by 1,024 and 128; other projection shapes still need inspection. |
| Vocabulary | 248,320 | Embeddings and vocabulary output are substantial memory costs. |
| Tied embeddings | `false` | Input embedding and output projection are separate matrices. |
| Full-attention KV heads / head width | 4 / 256 | Calculate cache size from these values, not another model's estimate. |
| Linear-attention value heads / key and value width | 32 / 128 / 128 | Recurrent state is a separate runtime allocation. |
| `mamba_ssm_dtype` | `float32` | Do not blindly force recurrent accumulators to BF16. |
| Vision depth / width | 27 / 1,152 | Vision requires its own processor, weight accounting, and performance tests. |
| Maximum positions | 262,144 | An architectural limit, not an affordable edge default. |
| Recorded Transformers version | 5.12.1 | Select and validate an implementation version explicitly. |

This table is a configuration observation, not an inventory of actual tensor
headers. The observed HF revision is `2367e865d009c13ac81713a2878291d33ab28177`;
HF metadata reports 9,409,813,744 BF16 parameters. The later bounded
[inventory](../embedded_jev/inventory.py) independently counted the same
9,409,813,744 parameters in 760 tensor entries across four shard headers,
totaling 18,819,627,488 tensor bytes.
The config declares one optional MTP layer, but the pinned index/headers contain
no MTP tensors. This is a storage inventory, not a model-quality or runtime
measurement. The [docs/sources.md](sources.md) register records the evidence boundary.

## Corrections to the Conversation

| PDF claim or implementation | Finding | Required action |
| --- | --- | --- |
| Weight fingerprints recover the exact Bonsai quantizer | Unsupported. Sign flips and scales do not uniquely identify an algorithm, calibration data, or training history. | Treat an open rotation/GPTQ recipe as our own hypothesis. |
| Rotation ensures near-lossless ternary quality | Unsupported for this checkpoint. QuaRot, QuIP, GPTQ, and BitNet address different representations and training regimes. | Measure held-out quality; budget for QAT/distillation recovery if PTQ fails. |
| Rotate weights after collecting the original Hessian, then run ordinary linears | Incorrect basis and forward function. The shown runners never rotate the corresponding activations. | Validate an equivalent rotated dense model before quantizing anything. |
| Hadamard rotation reduces the Hessian's spectral condition number by 10-100x | Mathematically false for an orthogonal similarity transform. | Distinguish coordinate incoherence from eigenvalue conditioning. |
| The GPTQ example has one scale per row/group | Incorrect. It computes a scalar across output rows for each column, then saves only the last column's scale for the group. | Fit and retain a scale for each output-row/input-group; test dequantization identity. |
| `pinv(H)` can replace the upper Cholesky factor in the same update loop | Incorrect algorithmic substitution. | Retry justified damping/factorization or fail explicitly; never silently switch meanings. |
| Generic decoder-layer calls are sufficient | They omit masks, positions, rotary inputs, and hybrid cache details; the capture path also sends CUDA tokens to CPU embeddings. | Capture and replay the actual architecture's arguments and device placement. |
| The GGUF examples produce runnable models | False. Packed `I8` arrays, arbitrary tensor names, missing tokenizer metadata, and `arch="qwen2"` do not implement a Qwen3.5 quantized model. | Use the chosen runtime's converter, registered types, logical shapes, and graph. |
| PQ2_0 is about 1.75 bpw | False. The checked layout is 2.125 bpw; PTQ1_0 is 1.75 bpw. | Separate ternary value count, serialized bit rate, and whole-model size. |
| I2_S is the same FP16-scale/group-128 layout as PQ2_0 | False in the inspected BitNet source. | Preserve group scales through a kernel/format extension or change the quantization scheme and re-evaluate quality. |
| `bitnet_init` and `bitnet_eval_prompt` are a ready-made runtime interface | No established API is demonstrated by the PDF. Its wrapper supplies neither those implementations nor ABI validation. | Bind to verified llama/ggml APIs in a pinned build, or write and test an explicit C shim. |
| `pip install llama-cpp-python` links it to a separately built fork | False by default. | Build a matching binding or use a small native executable with a versioned interface. |
| Arbitrary option strings can be scored using their last token | Incorrect for multi-token strings and ambiguous token boundaries. | Map semantic option IDs to distinct, context-validated single-token labels. |
| Softmax probabilities are calibrated confidence | False; sharpening with an arbitrary temperature is not calibration. | Fit temperature on separate labeled data, then test NLL/Brier/reliability on untouched examples. |
| Repeating two traces supplies 500k useful calibration tokens | False as a diversity or rank guarantee. | Track unique records, token distributions, and overlap; do not inflate coverage by repetition. |
| SWE-bench Lite `test` is calibration data | Evaluation contamination if reporting that benchmark. Gold patches also are not real execution traces. | Keep evaluation partitions isolated and label issue/patch pairs honestly. |
| INT16 is always sufficient for a 1,024-point integer FWHT | False without intermediate rescaling or a proven tighter bound. | Start with FP32 rotation; use INT32 or verified staged scaling for integer experiments. |
| BitNet means zero multiplication everywhere | False. I2_S uses integer multiply-accumulates; scaling, attention, norms, and recurrence remain. | Benchmark a real selected kernel, including preprocessing and non-linear work. |
| Zero weights automatically skip half the SIMD work | False for dense packed SIMD kernels. | Sparse execution needs an actual sparse layout/algorithm and an end-to-end benefit. |
| Generic RVV build flags enable optimized BitNet on RISC-V | Not established. The inspected documented CPU paths are x86 and ARM. | Treat RVV as a port with numerical and performance acceptance tests. |
| A 9B visual decision takes 120-600 ms on these boards | Not measured by the conversation and inconsistent with its simplistic compute model. | Publish no latency guarantee until target-hardware measurements exist. |
| All NPUs are fundamentally incapable of the approach | Too broad. Operator lowering, external transforms, and vendor programmability vary; T-MAC even has an NPU-related project. | CPU first for portability; assess specific accelerator SDKs separately. |

## Rotation and Curvature

Use row-vector activations $X\in\mathbb{R}^{N\times d}$, weights
$W\in\mathbb{R}^{m\times d}$, and an orthogonal rotation $R$.

$$
Y=XW^T=(XR)(WR)^T,\qquad RR^T=I.
$$

Weights and activations must use the **same complete transform**, including sign
vectors, permutation, block partition, and normalization. A signed Hadamard need
not be symmetric: writing transposes interchangeably is unsafe. Preserve bias.
Never commute rotations through SiLU, gates, softmax, RMSNorm scales, or recurrent
state updates without a separate equivalence argument.

For a token-averaged quadratic reconstruction proxy:

$$
G=\frac{2}{N}X^T X,\qquad G_R=R^T G R,
\qquad \kappa_2(G_R)=\kappa_2(G).
$$

Orthogonal rotation preserves eigenvalues and rank. It can redistribute coordinate
outliers and the diagonal, which helps quantization, but is not spectral
preconditioning. Adding $\lambda I$ changes eigenvalues; rotating does not.
Rank is at most $\min(N,d)$; $N\geq d$ is necessary, not sufficient, for full
rank. Rank deficiency can be handled with damping and does not alone prove failure.
At fixed sequence length, adding calibration sequences increases forward work
approximately linearly, not exponentially.

For a positive-definite damped proxy, define the upper factor $U$ by
$G_\lambda^{-1}=U^T U$. The GPTQ-style update uses

$$
e_j=\frac{w_j-q_j}{U_{jj}},\qquad
W_{:,j:}\leftarrow W_{:,j:}-e_j U_{j,j:}.
$$

Do not confuse this $U$ with the unfactored inverse. Group-scale search and
column traversal remain additional algorithm choices. For fixed ternary codes
$t_i$, a simple unweighted scale is
$s=\sum_i w_i t_i / \sum_i t_i^2$ when the denominator is nonzero.
This is not a claim to globally solve the Hessian-weighted ternary problem.

Use real-token masks in covariance accumulation. Padding can be valid if excluded
properly; a pad token is not inherently a singularity. Concatenated conversations
are not independent merely because an EOS token appears between them.

## Storage and Kernel Contracts

The [Prism release header](https://github.com/PrismML-Eng/llama.cpp/blob/prism-b10735-842b188/ggml/src/ggml-common.h)
defines these layouts:

| Type | Weights/block | Bytes/block | Effective bpw |
| --- | ---: | ---: | ---: |
| `PQ2_0` | 128 | 2 scale + 32 packed | 2.125 |
| `PTQ1_0` | 128 | 24 packed + 2 remainder + 2 scale | 1.75 |
| `Q2_0` | 64 | 2 scale + 16 packed | 2.25 |
| `TQ1_0` | 256 | 48 packed + 4 remainder + 2 scale | 1.6875 |

PTQ1_0's 24 bytes encode 120 trits, and its two remainder bytes encode eight
more. This is not four ordinary two-bit indicators per byte. Packing order and
trit decoding still require golden tests against the actual codec. A group-256
format cannot losslessly merge two arbitrary group-128 scales.

In [BitNet's I2_S quantizer](https://github.com/microsoft/BitNet/blob/0b341e582afbf9e1011f24744b554c96a3477eb5/src/ggml-bitnet-mad.cpp),
`quantize_i2_s` takes a maximum over the supplied tensor, stores an FP32 scale
after its packed values, and reserves alignment space. Its inspected default
packing interleaves 32-element lanes on x86 and 16-element lanes on ARM; the
alternative weight-parallel path changes layout again. Near-zero, negative, and
positive values become codes 1, 0, and 2 respectively. Calling this converter on
arbitrary group-scaled ternary weights loses their scale distinctions.

I2_S, TL1, TL2, T-MAC layouts, PQ2_0, and PTQ1_0 are not synonyms. Nor is
an IQ2 codebook a scalar ternary representation. Bonsai 2 also needs the exact
activation transform at runtime even when a loader recognizes its tensor type.

For group $g$ and row $r$, a CPU kernel must preserve

$$
y_r=\sum_g s_{r,g}\,a_g\sum_{i\in g}t_{r,i}z_i,
\quad t_{r,i}\in\{-1,0,1\},\quad z_i\in\mathbb{Z}_{8}.
$$

Here $a_g$ is the activation scale (or the appropriate per-token scale).
The row-dependent $s_{r,g}$ generally cannot be absorbed into one input scaling
shared by all output rows. LUT construction and rescaling are real costs.

An unnormalized 1,024-point FWHT on INT8 inputs bounded by 127 can produce
$1024\times127=130048$, beyond signed INT16. Rotating quantized inputs is also
not numerically equivalent to quantizing rotated floating-point inputs.

### Prism Transform Metadata

The pinned [loader](https://github.com/PrismML-Eng/llama.cpp/blob/prism-b10735-842b188/src/llama-model.cpp)
does not infer a rotation from a filename or ternary dtype. It reads:

| Metadata | Inspected constraint |
| --- | --- |
| `prism.hadamard.version` | 1 or 2; version 2 requires the tied-output mode, version 1 forbids it. |
| `prism.hadamard.block_size` | One positive power of two for the declared tensors; must divide every declared input width. |
| `prism.hadamard.transform` | Exactly `normalized-sylvester-walsh-hadamard`. |
| `prism.hadamard.axis` | Exactly `input-last-dimension`. |
| `prism.hadamard.sign_mode` | `identity` or `explicit`. |
| `prism.hadamard.weight_names` | Nonempty, unique, runtime-mapped tensor names on verified matmul paths. |
| `prism.hadamard.sign_widths` / `sign_values` | Explicit signs are +/-1, flattened and keyed by input width, not independently by tensor. |
| `prism.hadamard.gdn_v_grouped` | Optional Gated DeltaNet value-head feature permutation for `ssm_out`; must match offline weight ordering. |
| `prism.hadamard.inverse_weight_names` | Only supported token-embedding lookup tables; inverse transform follows lookup. |
| `prism.hadamard.tied_output` | Requires version 2, a latent input embedding, and no separate output weight. |

The [matmul helper](https://github.com/PrismML-Eng/llama.cpp/blob/prism-b10735-842b188/src/llama-graph.cpp)
applies optional feature permutation, then signs, then Hadamard, then packed
weight multiplication and any output scale. The embedding lookup path applies
Hadamard and then signs to recover the original basis. Shared activation transforms
are memoized. These observations reinforce why sign order and inverse direction
cannot be guessed from the PDF's symmetric unsigned-Hadamard example.

For our untied MiMo route, start with version-1-compatible semantics and unrotated
embeddings/head. An internal per-tensor manifest may be more expressive than this
runtime, but the exporter must reject configurations that cannot be represented
by its single block size and width-keyed signs. Do not silently discard differences.
The exact optimized `llama_mul_mat_hadamard` backend dispatch still requires
inspection and native testing during the integration milestone.

### Native API and Cache Risk

The pinned BitNet submodule's [C header](https://github.com/isHuangXin/llama.cpp/blob/390c307752ab78fd8189f359d6954c9ba1be74af/include/llama.h)
provides `llama_model_load_from_file`, `llama_init_from_model`, `llama_decode`,
`llama_get_logits_ith`, sequence-state serialization, and explicit free functions.
It is not the `bitnet_eval_prompt` API invented in the conversation. Calling
`llama_decode` on prompt batches is still prefill-only if no token is sampled.
Its name does not require an autoregressive generation loop.

The inspected Prism `build_rs_cache_view` source explicitly documents a
multi-sequence risk: relocating extra cache rows can overwrite a main row before
the fused recurrent operator reads it. No reproducer was run in this project.
Treat this as a source-disclosed integration risk, restrict the first prototype
to one sequence, and gate concurrency on a tested fix or conservative gather
path. The Qwen3.5 graph exposes `GGML_GDN_STATE_GATHER=1` as a legacy-path switch;
verify both correctness and performance before relying on it.

## Memory Accounting

Use decimal GB for disk estimates and binary GiB/MiB for allocations.
From the observed vocabulary and hidden width:

$$
248320\times4096=1017118720\text{ parameters per vocabulary matrix}.
$$

Each BF16 matrix is **2,034,237,440 bytes**, approximately **2.034 GB / 1.895 GiB**.
Both untied matrices total **4.068 GB**, before a single transformer block or
vision weight. At standard Q8_0's 34 bytes per 32 weights, each would occupy
approximately **1.081 GB**, subject to the converter and runtime supporting it.
Thus the PDF's under-3-GB complete runtime with both matrices in BF16 is impossible.

For full-attention FP16 KV at batch one, without replication:

$$
8\text{ layers}\times2\text{ (K,V)}\times4\text{ heads}\times256\times2
=32768\text{ bytes/token}.
$$

That is 128 MiB at 4,096 tokens. A conventional FP32 recurrent-state layout adds
$24\times32\times128\times128\times4=50331648$ bytes, or 48 MiB per sequence,
plus convolution state, scratch, alignment, and allocator overhead. Actual runtime
allocation and snapshot duplication must be measured.

A single FP32 12,288-square curvature matrix costs **576 MiB**. Factorization
workspaces and multiple copies can multiply this. One BF16 activation bank for
524,288 tokens of width 4,096 costs **4 GiB**; two banks cost **8 GiB**.
"One layer on GPU" alone is not a memory plan.

### Decision-Only Head Opportunity

If the runtime exposes the exact final normalized hidden state, only the output
rows for the fixed answer labels are needed to reproduce their raw linear logits.
Sixteen BF16 rows of width 4,096 cost **128 KiB**, not 2 GB. Softmax over those
rows equals full-vocabulary softmax conditioned on those same labels.

This is an exact algebraic opportunity, **not an existing converter feature**.
It needs a dedicated head/loader path and loses general text generation and
full-vocabulary perplexity unless the full head is retained separately. The input
embedding still needs its vocabulary. Verify hidden-state normalization and any
logit transforms against full-head inference first.

## Latency and Energy

No generated tokens means no autoregressive answer loop; prompt processing,
vision encoding, memory traffic, and recurrent updates remain. Long prompts can
be compute-bound. Short suffixes can be bandwidth- or overhead-bound.

$$
t\ \geq\ \max\left(\frac{B_{\rm moved}}{\mathrm{BW}_{\rm sustained}},
\frac{C_{\rm work}}{\mathrm{rate}_{\rm matching\ kernel}}\right).
$$

Use actual streamed bytes, not necessarily the whole file: embedding lookup does
not read every vocabulary row. For illustration, seven billion dense projection
weights applied to 512 positions imply about 7.168 trillion multiply/add-equivalent
operations. Completing that in 200 ms requires 35.84 trillion such equivalents
per second. Ternary/LUT execution changes the operation mix, so this is a sanity
check, not a literal FLOP requirement or an impossibility proof.

[T-MAC's published prefill example](https://github.com/microsoft/T-MAC#prefill-speedup)
reports 50.1 tokens/s for a 7B W2 model at four threads on Surface Laptop 7, not
thousands of prompt tokens/s on a smaller board. It is a different model/system,
not our forecast, but demonstrates why decode bandwidth cannot predict prefill.
Report cold load, warm prefill, cache restore, suffix scoring, vision time, p50/p95,
peak RSS, and joules/decision separately. Hardware pricing in the PDF is not
verified here and must not drive a purchase without a current delivered quote.

## Calibration Is Three Different Jobs

1. **Weight calibration:** representative activations for PTQ reconstruction.
2. **Activation calibration:** clipping/scale choices for A8 or other execution.
3. **Decision calibration:** a fitted probability map using labeled held-out tasks.

Use separate manifests and disjoint evaluation data. No universal mixture, damping
constant, or temperature is established by the conversation. Include the eventual
short, non-thinking decision prompts in activation coverage, not only long agent
traces. Include real image examples when validating a multimodal route.

The [MiMo template](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B/blob/2367e865d009c13ac81713a2878291d33ab28177/chat_template.jinja)
renders `reasoning_content` into `<think>...</think>` and structured `tool_calls`
into its own function/parameter syntax. `enable_thinking=False` appends an empty
thinking block to the assistant prefix. The PDF's manually inserted JSON tool
strings and fabricated reasoning are not a verified reproduction of that template.
Use actual reasoning only when supplied by a licensed source; never invent it and
call it an authentic trajectory. Handle tool-result roles and fail on malformed
schemas rather than silently dropping tools.

## Evidence Boundary

Source inspection establishes interfaces and formats, not model quality or speed.
The accompanying [docs/test_research_math.py](test_research_math.py) checks only
small algebraic and accounting claims. It does not validate a quantizer, a packed
runtime, multimodal processing, or safety of device-control decisions.
The separate [offline inventory tests](../tests/test_inventory.py) verify metadata
and header reconciliation; they do not validate model execution either.
The pinned [label probe](../embedded_jev/label_probe.py) verifies single-token
A-P continuations and matching rendered/tokenized template IDs for one text-only
sample prompt using Transformers 5.12.1. With CPU-only processor dependencies,
five synthetic agent/tool prompts also passed text-only Qwen3VLProcessor parity;
their recorded labels are fixtures, not model predictions. Neither probe tests
native tokenization, visual inputs, generated reasoning, or decision quality.