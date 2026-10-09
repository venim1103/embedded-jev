# Embedded Jev

An experimental on-device decision engine intended to support multiple
Qwen3.5 9B-class models through a SemIf-style typed decision interface. Given
evidence, a question and described options, it returns option IDs and conditional
scores instead of generating an answer.

The goal is ternary-quantized models running through **genuine BitNet-derived
CPU kernels**. [`MiMo-V2.6-Distill-Qwen-9B`](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B)
is the first test subject, not a permanent model requirement. Text comes first;
vision and embedded-device support follow only after native validation.

## Status

The CPU text prototype runs MiMo's text path and scores options without
generating answer tokens. A single quantized projection has also run inside
native MiMo through BitNet-derived kernels. This is a research prototype, not
a ready-to-deploy, fully ternary model. Other model variants are not yet validated.

Full-model ternary conversion, exact agreement across runtimes, representative
decision quality and calibrated confidence remain open. Embedded-device
performance, vision, and ARM/RISC-V support are not yet validated.

Conditional scores are not calibrated confidence, and synthetic checks do not
establish real-world accuracy. Ternary storage alone is not BitNet execution.
This project is independent of TypeSafe, SemIf and PrismML; it does not reproduce
TypeSafe Jev's private model.

## Documentation

- [Development guide](docs/development.md): setup, operation, tests and technical reference.
- [Design](docs/design.md): architecture, numerical contracts and rationale.
- [Roadmap](docs/roadmap.md): milestones and acceptance gates.
- [Research audit](docs/research-audit.md): verified facts, estimates and hypotheses.
- [Source register](docs/sources.md): upstream revisions and evidence limits.
- [Engineering handover](docs/handover.md): implementation status and developer checkpoints.

## License

This repository's [license](LICENSE) does not supersede model, dataset, or
third-party runtime licenses.