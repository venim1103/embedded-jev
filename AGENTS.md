# Project Agent Workflow

Build an experimental on-device decision engine around MiMo-V2.6-Distill-Qwen-9B, genuine BitNet-derived CPU execution, and typed conditional option scores without generated answer tokens. Preserve independent FP16 output-row/input-group scales; ternary storage alone is not proof of BitNet execution.

## Start With Context

- Read the newest relevant checkpoint in [docs/handover.md](docs/handover.md) before continuing implementation. Use [docs/development.md](docs/development.md) for setup, tools and technical reference, and [docs/design.md](docs/design.md) for architecture and numerical contracts.
- Check the actual worktree and nearby implementation/tests. Preserve user changes, including editor settings. Verify cached files, pinned source revisions, selected environments and resource budgets rather than assuming they survived a restart.
- Reuse existing caches and environments; do not duplicate model downloads or source checkouts. After an interrupted tool call with unknown outcome, verify whether it applied before repeating it. Distinguish implemented, tested, staged, integrated and measured work.

## Documentation Audience

- [README.md](README.md) is the GitHub front page for visitors. Keep it a concise public introduction with plain-language capabilities, maturity caveats, documentation links and licensing. Do not add discussions, progress updates, handovers, test counts, commit history, setup commands or technical implementation details there.
- Put reusable setup, operating instructions, tool contracts and implementation reference in [docs/development.md](docs/development.md). Keep architecture, numerical contracts and design rationale in [docs/design.md](docs/design.md).
- Put changing implementation status, decisions/discussions, test evidence, failed-check history and next-session handover in [docs/handover.md](docs/handover.md). Clearly distinguish historical notes from current capabilities.
- Keep source-backed facts, estimates and hypotheses in [docs/research-audit.md](docs/research-audit.md), upstream revisions/licenses and evidence limits in [docs/sources.md](docs/sources.md), and milestones/acceptance gates in [docs/roadmap.md](docs/roadmap.md).
- Keep stable agent workflow rules in this file. Update the public README only when the visitor-facing description or safety guidance genuinely changes, not at each development checkpoint.

## Work Independently

- The user prefers autonomous implementation within the agreed scope. Make routine implementation/testing decisions without repeatedly asking permission; continue through verification and useful checkpoints until done or genuinely blocked.
- Keep changes small and local. Form a concrete hypothesis, make the smallest useful edit and immediately run a focused check before expanding scope. Reuse existing tests/helpers and avoid unrelated cleanup.
- Give concise progress updates explaining what changed, what was verified and what remains. Be explicit about uncertain evidence and missing authorization.
- Do not spawn subagents unless the user explicitly requests delegation.
- Respect requests to pause, stop or finish for the day. Finish only the agreed checkpoint; a terminal notification or documentation request is not permission to resume paused implementation.

## Verification And Commits

- Prefer synthetic, bounded fixtures for autonomous arithmetic and integration checks. Optional tests must reuse the pinned environments and sources documented in [docs/development.md](docs/development.md); never start a model conversion or download implicitly from a test.
- Before every source checkpoint commit, run the default gate from the repository root, plus focused checks for the changed behavior:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider && \
ruff check --no-cache embedded_jev tests && \
git diff --check
```

- For native/runtime changes, run the applicable compiled controls and opt-in gates from the development guide. Report skipped optional tests and prerequisites accurately; default test success is not proof that native or full-model checks ran.
- Do not relax numerical tolerances, assertions, deadlines, token/tensor bounds or safety checks just to obtain green results. Do not automatically rerun a failed gate seeking green; diagnose it first. A substantive repair permits rerunning the focused check and a new gate.
- Do not rerun an unchanged successful gate merely for reassurance. Documentation/instruction-only edits need scoped content, link and whitespace validation; report accurately which checks ran.
- Scoped local checkpoint commits are authorized as part of the workflow. Commit meaningful tested increments periodically, not every probe and not only at the end of a large task. Review the diff and stage only files belonging to the checkpoint.
- Do not push or create branches unless explicitly requested. Never revert another person's changes.

## When To Ask For Help

Ask for a small, precise action or separate approval when work requires:

- Another complete-model conversion, additional weight/source downloads, refitting, a new retained candidate, compensation promotion, full-model ternary conversion, held-out scoring or broader quality experiments. An earlier one-shot approval does not authorize another artifact or experiment.
- Missing model/cache inputs, target-device access, a genuine blocker, ambiguous requirements or an expansion of agreed resource, security or environment constraints.
- Privileged or host-level changes. Do not change clocks/NTP, host/network settings, TLS/security or privilege configuration as a testing workaround.

Prepare the autonomous prerequisites first. Explain the bounded action, expected evidence, resource budget and cleanup plan. Never ask for passwords, tokens or other secrets. Reuse existing tools where possible; when installing necessary software, also update the devcontainer provisioning so it is reproducible.

## Safety Boundaries

- Keep storage-format correctness, actual BitNet dispatch, arithmetic agreement, runtime compatibility and model quality separate. Conditional option scores are not calibrated confidence; synthetic fixtures are not representative decision accuracy.
- Preserve the frozen candidate's codes, independent row/group scales and declared transforms. Do not silently refit, replace or promote it to hide an unresolved runtime difference.
- Keep weights, quantized artifacts, raw activations, compiler spill and large reports outside Git. Use disposable scratch for approved diagnostics, retain only small necessary reports, and verify copied evidence before cleanup.
- Run large-memory jobs serially. Measure RAM, swap and disk use; per-process counters are not host or target-device guarantees. Preserve pinned cached sources/libraries unless changes are separately scoped.
- Keep credentials and private data out of repository docs, logs, screenshots and memory. Do not claim deployment readiness, edge performance or whole-model ternary support from a bounded primitive or synthetic control.

## Leave A Clear Handoff

- Update [docs/handover.md](docs/handover.md) at meaningful checkpoints with the actual change, commands/results, failed-check history, limits, artifact cleanup and next concrete step. Preserve earlier results rather than rewriting failures as successes.
- Keep stable workflow rules here and changing implementation status in the handover. Document synthetic arithmetic, native integration, real-model diagnostics and representative quality evidence separately; do not describe an unintegrated primitive as a working full-model path.
- When stopping, give a concise outcome with commit IDs, verification, remaining boundaries and worktree state. Do not continue into the next implementation slice after the user's stop request.