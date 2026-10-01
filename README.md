# Embedded Jev

An experimental on-device decision engine: present evidence, a question, and
described options; score the options directly instead of generating an answer.
The target is a ternary-quantized
[`MiMo-V2.6-Distill-Qwen-9B`](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B)
running through **genuine BitNet-derived CPU kernels**, with a SemIf-style typed
decision interface. Text comes first; vision and embedded x86, ARM, and RISC-V
targets follow only after native validation.

**Status:** The CPU research environment, bounded inventory, and text-only
tokenizer/processor checks work on the pinned model metadata. A small synthetic
agent/tool fixture checks prompt and option-label mapping, not decision accuracy.
Full-model quantization and model-loadable BitNet dispatch remain open. Pinned
MiMo text layers now stream in BF16 and produce no-generation A/B conditional
scores from selected output-head rows. A bounded synthetic agent/tool fixture
also returns typed option IDs with A-C conditional scores; these are not
calibrated confidence or validated decisions. The complete pinned BF16 source
snapshot (four SHA-256-verified shards, 760 indexed tensors) is cached outside
this repository; an offline inventory reconciles its headers. One real FFN-down
matmul can be replaced in the streamed BF16 text path by an in-memory
BitNet-derived ternary/A8 AVX2 adapter. One hash-checked, packed projection
fixture is retained outside Git; it is not a loadable quantized model or a
quality result. Separately, a supported 1.19 GB native
BitNet control checkpoint was loaded and prefilled without answer generation.
Its A-C labels can be scored directly from final-position logits, but it is not
a substitute for MiMo or evidence of decision quality or edge speed.

The last tested implementation checkpoint is `6d74ae8` (2026-09-30): native
A8 and matching signed-Hadamard controls pass, as do exact PQ2_0 conversion
and controlled native tensor execution. PQ2 dispatch is not BitNet dispatch.
Read the [current handover](docs/handover.md#current-checkpoint-2026-10-01)
for recorded gates, reusable caches, constraints, and the next native task.

## Get Started

Open the repository in the [CPU devcontainer](.devcontainer/devcontainer.json),
then from its workspace root run:

```bash
python .devcontainer/smoke.py
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider docs/test_research_math.py tests
PYTHONDONTWRITEBYTECODE=1 python -m embedded_jev.inventory
```

The inventory reads only small metadata and safetensors headers at the pinned
MiMo revision; it does **not** fetch weight shards. Its JSON output reports
tensor shapes, storage costs, initial ternary eligibility, and separate memory
estimates. To inspect the locally cached full snapshot without network access:

```bash
snapshot="$HOME/.cache/huggingface/embedded-jev/models/mimo-2367e865d009c13ac81713a2878291d33ab28177"
PYTHONDONTWRITEBYTECODE=1 python -m embedded_jev.inventory --local-dir "$snapshot"
```

The audit and exact byte counts live in
[the research notes](docs/research-audit.md) and
[development guide](docs/development.md). Only the default remote inventory
command needs a network connection; `--local-dir` and the tests do not.
See the development guide for the
optional, tokenizer-only [label-boundary probe](embedded_jev/label_probe.py).

## Approach

1. Establish a dense reference and validate the model's actual prompt and
   tokenizer boundary for fixed answer labels.
2. Preserve the function under matching activation/weight transforms; test
   group-scaled ternary reconstruction against that reference.
3. Run eligible projections through verified BitNet-derived CPU kernels without
   losing scales or activation transforms, then return conditional option scores
   without sampling output tokens.

Ternary storage alone is not BitNet execution, and conditional option scores
are not calibrated confidence or a reproduction of TypeSafe Jev's private model.
This project is independent of TypeSafe, SemIf, and PrismML. Consult
[the design](docs/design.md) before treating a format estimate as a runtime.

## Documentation

- [Roadmap](docs/roadmap.md): milestones, quality gates, and benchmarks.
- [Design](docs/design.md): artifact, runtime, and typed-decision contracts.
- [Research audit](docs/research-audit.md): verified facts, estimates, and open hypotheses.
- [Development guide](docs/development.md): setup, tests, and inventory usage.
- [Source register](docs/sources.md): upstream revisions and evidence limits.
- [Engineering handover](docs/handover.md): detailed status for the next session.

This repository's [license](LICENSE) does not supersede model, dataset, or
third-party runtime licenses.