# Embedded Jev

An experimental on-device decision engine: present evidence, a question, and
described options; score the options directly instead of generating an answer.
The target is a ternary-quantized
[`MiMo-V2.6-Distill-Qwen-9B`](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B)
running through **genuine BitNet-derived CPU kernels**, with a SemIf-style typed
decision interface. Text comes first; vision and embedded x86, ARM, and RISC-V
targets follow only after native validation.

**Status:** The CPU research environment, bounded inventory, and a text-only
check of the pinned chat template and A-P answer-token boundary are working.
Quantization, BitNet integration, model inference, and decision scoring are not
yet implemented. No model weights have been downloaded, and there are no
measured model-quality or device-performance results.

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
estimates. The audit and exact byte counts live in
[the research notes](docs/research-audit.md) and
[development guide](docs/development.md). A network connection is needed for
the inventory command, not for the tests. See the development guide for the
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