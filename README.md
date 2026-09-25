# Embedded Jev

Research toward an on-device semantic decision engine:
**MiMo ternary quantization -> BitNet-derived CPU execution -> SemIf-style typed decisions**.

Target model: `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B`, revision
`2367e865d009c13ac81713a2878291d33ab28177`.
Target platforms: Linux x86-64 first, ARM64 next, and an explicitly validated
RISC-V port later. Vision follows a correct text-only path.

## Current Status

The CPU development container builds and starts; its Python/C++ smoke checks and
nine model-free research tests pass on Linux x86-64. The quantizer, BitNet/MiMo
integration, and decision service are **not implemented yet**. No model weights
have been downloaded and no edge latency or quantization quality is claimed.

This is an independent project. It does not reproduce TypeSafe Jev's undisclosed
model/training or PrismML's proprietary quantization pipeline.

## Start Here

| Document | Contents |
| --- | --- |
| [docs/handover.md](docs/handover.md) | Self-contained restart guide for the next session inside the devcontainer |
| [docs/research-audit.md](docs/research-audit.md) | Corrections to the supplied AI conversation, mathematics, format and memory accounting |
| [docs/design.md](docs/design.md) | Quantization, genuine BitNet integration, artifacts, decision API, caching, and vision |
| [docs/roadmap.md](docs/roadmap.md) | Implementation milestones, proposed modules, acceptance gates, and benchmark protocol |
| [docs/development.md](docs/development.md) | Devcontainer setup, validation commands, dependencies, and future GPU workflow |
| [docs/sources.md](docs/sources.md) | Primary sources, inspected revisions, and limits of the evidence |

## Development

In VS Code, run **Dev Containers: Reopen in Container** using
[.devcontainer/devcontainer.json](.devcontainer/devcontainer.json).
The CPU image includes Python 3.12, Clang 18, CMake/Ninja, NumPy/SciPy, pytest,
and Ruff. Model runtimes, GPU access, and large downloads are deliberately opt-in.

Inside the container:

```bash
python .devcontainer/smoke.py
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider docs/test_research_math.py
ruff check --no-cache .devcontainer/smoke.py docs/test_research_math.py
```

For Docker/Podman CLI instructions, see [docs/development.md](docs/development.md).

## Key Findings

- Bonsai `PTQ1_0` is 1.75 bpw for its ternary blocks; `PQ2_0` is 2.125 bpw.
	Whole-model size includes higher-precision tensors and runtime memory.
- The inspected BitNet `I2_S` representation does not preserve arbitrary
	group-128 scales. A real kernel/format adaptation is required for this MiMo route.
- MiMo's untied input embedding and output head together require about **4.07 GB
	in BF16**, before transformer or vision weights. The supplied under-3-GB
	runtime estimate with both preserved in BF16 is not feasible.
- SemIf already has a llama.cpp CPU backend. Direct logits remove answer
	generation, not prompt computation, and are not automatically calibrated.

Next: inventory the pinned model's tensor headers, establish the host memory
budget and dense decision reference, then validate a small ternary quantization
slice together with its BitNet execution contract.