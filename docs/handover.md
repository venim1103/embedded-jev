# Engineering Handover

Prepared: 2026-09-25. Updated: 2026-10-19. Phase: bounded text/native integration; full-model deployment remains open.

This document is intended to let a new developer or coding-agent session continue
after reopening the repository inside the devcontainer, without access to the
original conversation. Read this first, then follow the links for detail.

**Resumed by user, 2026-10-19:** Continue the previously approved synthetic-only
larger-file work with periodic local checkpoints. Starting `a9f82a1` matches
`origin/main`; untouched user editor settings remain untracked. An interrupted
resume had already edited the explicit v2 descriptor hasher and tests, and its
focused terminal command completed successfully. Those edits were inspected
and preserved rather than repeated. The required source gate now passes; read
the [template preparation checkpoint](#template-preparation-checkpoint-2026-10-19), followed by the
[standalone token checkpoint](#standalone-token-checkpoint-2026-10-19),
[vocabulary preparation checkpoint](#vocabulary-preparation-checkpoint-2026-10-19),
[context emission checkpoint](#context-emission-checkpoint-2026-10-19),
[metadata preparation checkpoint](#metadata-preparation-checkpoint-2026-10-19),
[architecture metadata checkpoint](#architecture-metadata-checkpoint-2026-10-19),
[frozen writer checkpoint](#frozen-writer-checkpoint-2026-10-19),
[frozen source owner checkpoint](#frozen-source-owner-checkpoint-2026-10-19),
[mixed writer checkpoint](#mixed-writer-checkpoint-2026-10-19),
[v2 mixed source checkpoint](#v2-mixed-source-checkpoint-2026-10-19),
[v2 resource checkpoint](#v2-resource-checkpoint-2026-10-19),
[v2 writer checkpoint](#v2-writer-checkpoint-2026-10-19),
[v2 payload checkpoint](#v2-payload-checkpoint-2026-10-19),
[v2 shard checkpoint](#v2-shard-checkpoint-2026-10-19) and
[v2 descriptor checkpoint](#v2-descriptor-checkpoint-2026-10-19).
The earlier pause below is historical. Existing real-model/resource boundaries
remain unchanged; no cached learned weights or full conversion is authorized.

**Paused by user, 2026-10-16:** Stop at the completed shared-source writer and
larger-policy proposal checkpoints, user-pushed through `30e3f15`. The last
source checkpoint is `622e307`: 677 default passes/89 optional skips and both
pinned BF16/F32 converter controls pass. The user subsequently approved the
synthetic-only larger-source policy/budgets below, but its implementation has
not started. The interrupted call contained only read-only lookup and a memory
update; actual worktree/memory checks confirm no source edits and no applied
memory update. No jobs or owned scratch remain. User editor settings are
untouched. Resume implementation only after an explicit user instruction;
neither the recorded approval nor tool notifications override this pause.

**Resumed by user, 2026-10-16:** Continue autonomously with periodic local
checkpoint commits. Starting `6848279` matched `origin/main`; only untouched
user settings were untracked. The bounded attention helper and learned-operand
precision/read-partition diagnostics below completed under the existing
cached-MiMo approval. Original failed numerical checks remain failed, not
replaced by looser acceptance. A subsequent bounded synthetic partial-rotary
source control also passes. Latest continuation starts at user-pushed `d88871b`
with only untouched editor settings untracked. A separately bounded native
recurrent-state ownership control, typed helper and learned four-layer coarse
resident integration now pass. Three native recurrent buffers carry state;
host readbacks are oracles. Those GDN-only controls keep host convolution history.
Learned 64-prefill/16-singleton and all-80-singleton continuation also pass
against unchanged same-schedule references. A subsequent native convolution
control/typed helper and combined learned coarse adapter also pass. Combined
64+16 continuation stopped on the unchanged swap guard; its numerical verdict
is unavailable and combined all-singleton execution was not started. No retry
or full conversion was attempted after that guard stop.
Latest resume starts at user-pushed `ab484a8`. A separate reference-lifetime
variant now passes synthetic/coarse learned checks with lower native-phase
retention and unchanged arithmetic/guards. Its longer continuation reaches
final publication but fails the source/proof identity guard; no accepted
numerical report exists. The new synthetic pin audit passes. Learned jobs
remain stopped pending source/proof stability. A later event-observed attempt
captures writes to the newly edited observer scripts and correctly withholds
publication. A separate disposable execution snapshot now passes bounded
fixture/native controls without editor-attached execution files; learned
64+16 validation of this materially different path also passes, with all
execution identities/events and same-schedule numerical evidence verified.
All-80-singleton validation now also passes. The selected bounded combined
native-history/GDN/attention continuation slice is complete; full-model/runtime
acceptance remains separate. The historical writer remains unknown.
After user-pushed `6294380`, bounded synthetic complete-loader controls also
cover singleton and 64+16 continuation through dense, BitNet v1 and profile
policy backends, plus deliberate state-loss refusal. No learned model ran in
this new source checkpoint.
After user-pushed `fb2852c`, long singleton warmup/reset/replay also passes
through the same three backends, without rebuilding runtime libraries.
After `443e1f1`, deterministic singleton alternation between two contexts
sharing one loaded model also passes independent history-sensitive references.
After user-pushed `1b9c2f6`, a profile-gated BF16 matrix chunk primitive passes
exhaustive finite-storage and pinned-transform controls. It is not a complete
converter or approval to retry the refused real-model conversion.
After `1ca8ac2`, a disposable raw payload owner also passes bounded lifetime
refusals and actual pinned GGUF temporary-writer integration for 16 matrices.
After `fc54e54`, a separate source-bound payload owner composes the existing
profile-pinned safetensors range verifier with matching entry/exit receipts.
After `75a9205`, all 16 pinned-converter matrix writes also pass through physical
synthetic safetensors mappings, with first/last post-spool source refusals.
After user-pushed `c03ef1e`, a descriptor-owned verified source mapping removes
whole-operand copies from the new bounded matrix reader, not the older verifier.
After `967e2cc`, that source owner composes with chunked raw payload production
without caller source arrays and passes the pinned 16-matrix GGUF control.
After `9f597bb`, a bounded two-size allocation control also verifies that the
new paths avoid the older verifier's source-sized Python allocation growth.
After `0211d3c`, a shared shard owner retains one descriptor/header across
serial matrix contexts, with two whole-shard hashes and fail-closed lifecycle checks.
After `e5c94ba`, the pinned synthetic writer uses that shared owner across all
16 matrices and withholds publication through final shard-identity checks.
After `622e307`, the larger-source API/resource proposal below was prepared
and explicitly approved for synthetic-only implementation; no implementation
started before the user's pause. Current API limits and real-model approvals stand.

**Historical pause, 2026-10-15:** Stop after the completed `f723065` checkpoint,
which the user pushed before this pause. Only a final read-only API lookup was
interrupted; no bounded ctypes attention helper or learned layer-3 capture was
created or run. No experiment process, raw replay packet, GGUF or owned spill
remains; user settings are untouched. Resume implementation only after an
explicit user instruction. Next: validate a bounded same-input attention helper
synthetically before capturing post-Q/K-norm/post-rotary learned operands. Native
rotary, KV writes, learned projections and full-loader parity remain separate.

**Historical resume, 2026-10-15:** Continue autonomously with periodic local
checkpoint commits. Starting `218d5c6` matched `origin/main`, with only untouched
user settings untracked. The combined recurrent-cache check below completed
within the existing cached-MiMo diagnostic approval and unchanged resource guards;
this does not authorize another full conversion or other-profile execution.
Whole-layer and token-by-token controls subsequently completed; read the
[nonflash attention checkpoint](#nonflash-attention-checkpoint-2026-10-15) and
[cached layer continuation checkpoint](#cached-layer-continuation-checkpoint-2026-10-15).

**Historical pause, 2026-10-15:** Finish only the completed singleton-dispatch
evidence checkpoint and stop. The combined projection/convolution-history/GDN
cached-block experiment has not started. Resume implementation only after an
explicit user instruction; a terminal notification is not a resume request.
See the [batch dispatch control checkpoint](#batch-dispatch-control-checkpoint-2026-10-15).

**Expanded approval, 2026-10-15:** After user-pushed `53074a8`, the user approved
necessary further work, including the previously blocked real-input diagnostic.
Selected next action: one disposable cached-MiMo dense BF16 conversion and
serial native/reference intermediate comparison on the existing 80-token
synthetic fixture. See the budget below; this is not a retained candidate,
refit, held-out quality run or another-model compatibility claim.
The one conversion attempt stopped on its swap guard before creating a GGUF;
two smaller authenticated layer-0 recurrent-operand replays completed, followed
by stateless-stage, connected recurrent-block and whole-layer composition probes.
After user-pushed `df1da32`, autonomous continuation added whole-layer precision
controls, bit-exact coupled convolution/core cache handoff and a separate
projection-batch precision diagnostic; full conversion was not retried.
Read the [nonflash attention checkpoint](#nonflash-attention-checkpoint-2026-10-15),
[cached layer continuation checkpoint](#cached-layer-continuation-checkpoint-2026-10-15),
[combined recurrent cache checkpoint](#combined-recurrent-cache-checkpoint-2026-10-15),
[batch dispatch control checkpoint](#batch-dispatch-control-checkpoint-2026-10-15),
[coupled cache checkpoint](#coupled-cache-checkpoint-2026-10-15),
[whole layer checkpoint](#whole-layer-checkpoint-2026-10-15),
[layer zero stage checkpoint](#layer-zero-stage-checkpoint-2026-10-15) and
[real operand replay checkpoint](#real-operand-replay-checkpoint-2026-10-15)
for the measured results and remaining full-model blocker.

**Resumed by user, 2026-10-15:** Continue autonomous bounded synthetic source/
provenance work after user-pushed `15d9d3c`. Starting `main` matched `origin/main`
with only untouched user settings untracked. Real-model approvals are unchanged.

**Resumed by user, 2026-10-14:** Continue autonomous bounded synthetic source/
provenance work after user-pushed `81d583a`. Starting `main` matched `origin/main`
with only untouched user settings untracked. Real-model approvals are unchanged.

**Historical pause, 2026-10-13:** Finish only the tested source-exit scoring
checkpoint below and stop for today. Multi-shard exit coverage is queued but
not started; resume implementation only after an explicit user instruction.
No background jobs, pushes, branch changes or user-settings changes. Real-model
approval boundaries remain unchanged.

**Resumed by user, 2026-10-13:** Continue autonomous code-only caller/provenance
preparation with bounded synthetic fixtures. Starting `5bbbd4e` matched
`origin/main`; user settings remain untouched. Real-model approvals are unchanged.

**Historical pause, 2026-10-12:** Stopped at tested source checkpoint `b6a9f0d`,
following `c18e90f` and `4973863`. The user pushed all three; `main` matched
`origin/main` before this documentation-only pause note. The default gate has
322 passes/54 optional skips, and both pinned BF16/F32 vector controls pass.
The planned synthetic GGUF vector-view binding has not started: the interrupted
read produced no result and made no edits. Resume only after an explicit user
instruction. No tracked source changes or background jobs remain; the untracked
user `.vscode/` directory is untouched. Real-model authorization is unchanged.

**Quick orientation (2026-10-16):**

1. The user explicitly resumed on 2026-10-16 after pushing `6848279`;
  autonomous bounded cached-MiMo diagnostics are active. The bounded
  synthetic profile/manifest-to-prefill and seven-role checkpoints below are verified.
  The user separately approved the code-only v2 metadata-budget correction below.
  After pushing `af62d7e`, the user approved code-only caller/provenance preparation;
  the verified-file owner below is the first tested increment in that scope.
  The latest approval now permits the bounded cached-MiMo diagnostic below;
  synthetic execution itself still grants no other-profile runtime approval.
2. Read the [larger source proposal](#larger-source-proposal-2026-10-16),
   then the [BF16 shared writer checkpoint](#bf16-shared-writer-checkpoint-2026-10-16),
   then the [BF16 shared shard checkpoint](#bf16-shared-shard-checkpoint-2026-10-16),
   then the [BF16 allocation scaling checkpoint](#bf16-allocation-scaling-checkpoint-2026-10-16),
   then the [BF16 mapped payload checkpoint](#bf16-mapped-payload-checkpoint-2026-10-16),
   then the [BF16 mapped source checkpoint](#bf16-mapped-source-checkpoint-2026-10-16),
   then the [BF16 source-bound writer checkpoint](#bf16-source-bound-writer-checkpoint-2026-10-16),
   then the [BF16 source-bound payload checkpoint](#bf16-source-bound-payload-checkpoint-2026-10-16),
   then the [BF16 payload owner checkpoint](#bf16-payload-owner-checkpoint-2026-10-16),
   then the [BF16 chunk primitive checkpoint](#bf16-chunk-primitive-checkpoint-2026-10-16),
   then the [native loader isolation checkpoint](#native-loader-isolation-checkpoint-2026-10-16),
   then the [native loader reset checkpoint](#native-loader-reset-checkpoint-2026-10-16),
   then the [native loader continuation checkpoint](#native-loader-continuation-checkpoint-2026-10-16),
   then the [detached singleton checkpoint](#detached-singleton-checkpoint-2026-10-16),
   then the [detached decode checkpoint](#detached-decode-checkpoint-2026-10-16),
   then the [detached execution checkpoint](#detached-execution-checkpoint-2026-10-16),
   then the [file event checkpoint](#file-event-checkpoint-2026-10-16),
   then the [lifetime pin audit checkpoint](#lifetime-pin-audit-checkpoint-2026-10-16),
   then the [reference lifetime checkpoint](#reference-lifetime-checkpoint-2026-10-16),
   then the [learned convolution owner checkpoint](#learned-convolution-owner-checkpoint-2026-10-16),
   then the [native convolution ownership checkpoint](#native-convolution-ownership-checkpoint-2026-10-16),
   then the [native state singleton checkpoint](#native-state-singleton-checkpoint-2026-10-16),
   then the [learned native state checkpoint](#learned-native-state-checkpoint-2026-10-16),
   then the [typed recurrent owner checkpoint](#typed-recurrent-owner-checkpoint-2026-10-16),
   then the [native recurrent ownership checkpoint](#native-recurrent-ownership-checkpoint-2026-10-16),
   then the [resident all singleton checkpoint](#resident-all-singleton-checkpoint-2026-10-16),
   then the [resident singleton continuation checkpoint](#resident-singleton-continuation-checkpoint-2026-10-16),
   then the [resident stack checkpoint](#resident-stack-checkpoint-2026-10-16),
   then the [four layer continuation checkpoint](#four-layer-continuation-checkpoint-2026-10-16),
   then the [native prefix composition checkpoint](#native-prefix-composition-checkpoint-2026-10-16),
   then the [attention layer continuation checkpoint](#attention-layer-continuation-checkpoint-2026-10-16),
   then the [attention decoder layer checkpoint](#attention-decoder-layer-checkpoint-2026-10-16),
   then the [attention continuation checkpoint](#attention-continuation-checkpoint-2026-10-16),
   then the [cached attention sublayer checkpoint](#cached-attention-sublayer-checkpoint-2026-10-16),
   then the [learned persistent cache checkpoint](#learned-persistent-cache-checkpoint-2026-10-16),
   then the [persistent cache checkpoint](#persistent-cache-checkpoint-2026-10-16),
   then the [KV composition checkpoint](#kv-composition-checkpoint-2026-10-16),
   then the [KV write checkpoint](#kv-write-checkpoint-2026-10-16),
   then the [connected attention checkpoint](#connected-attention-checkpoint-2026-10-16),
   then the [learned rotary checkpoint](#learned-rotary-checkpoint-2026-10-16),
   then the [partial rotary checkpoint](#partial-rotary-checkpoint-2026-10-16),
   then the [learned attention checkpoint](#learned-attention-checkpoint-2026-10-16),
   then the [nonflash attention checkpoint](#nonflash-attention-checkpoint-2026-10-15),
   then the [cached layer continuation checkpoint](#cached-layer-continuation-checkpoint-2026-10-15),
   then the [combined recurrent cache checkpoint](#combined-recurrent-cache-checkpoint-2026-10-15),
   then the [batch dispatch control checkpoint](#batch-dispatch-control-checkpoint-2026-10-15),
   then the [coupled cache checkpoint](#coupled-cache-checkpoint-2026-10-15),
   then the [whole layer checkpoint](#whole-layer-checkpoint-2026-10-15),
   then the [layer zero stage checkpoint](#layer-zero-stage-checkpoint-2026-10-15),
   then the [real operand replay checkpoint](#real-operand-replay-checkpoint-2026-10-15),
   then the [prompt trace checkpoint](#prompt-trace-checkpoint-2026-10-15),
   then the [dynamic gate roles checkpoint](#dynamic-gate-roles-checkpoint-2026-10-15),
   then the [dynamic gate probe checkpoint](#dynamic-gate-probe-checkpoint-2026-10-15),
   then the [recurrent source roles checkpoint](#recurrent-source-roles-checkpoint-2026-10-15),
   then the [recurrent source probe checkpoint](#recurrent-source-probe-checkpoint-2026-10-15),
   then the [nonzero multi shard refusal checkpoint](#nonzero-multi-shard-refusal-checkpoint-2026-10-14),
   then the [nonzero multi shard checkpoint](#nonzero-multi-shard-checkpoint-2026-10-14),
   then the [multi shard refusal checkpoint](#multi-shard-refusal-checkpoint-2026-10-14),
   then the [multi shard exit checkpoint](#multi-shard-exit-checkpoint-2026-10-14),
   then the [source exit score checkpoint](#source-exit-score-checkpoint-2026-10-13),
   then the [source exit owner checkpoint](#source-exit-owner-checkpoint-2026-10-13),
   then the [source gated workflow checkpoint](#source-gated-workflow-checkpoint-2026-10-13),
   then the [source gated refusal checkpoint](#source-gated-refusal-checkpoint-2026-10-13),
   then the [source gated recovery checkpoint](#source-gated-recovery-checkpoint-2026-10-13),
   then the [nonzero source prefill checkpoint](#nonzero-source-prefill-checkpoint-2026-10-13),
   then the [nonzero source permutation checkpoint](#nonzero-source-permutation-checkpoint-2026-10-13),
   then the [nonzero source vector checkpoint](#nonzero-source-vector-checkpoint-2026-10-13),
   then the [source gated prefill checkpoint](#source-gated-prefill-checkpoint-2026-10-13),
   then the [source gated owner checkpoint](#source-gated-owner-checkpoint-2026-10-13),
   then the [complete source conversion checkpoint](#complete-source-conversion-checkpoint-2026-10-13),
   then the [complete source table checkpoint](#complete-source-table-checkpoint-2026-10-13),
   then the [frozen source identity checkpoint](#frozen-source-identity-checkpoint-2026-10-13),
   then the [combined source range checkpoint](#combined-source-range-checkpoint-2026-10-13),
   then the [source operand file checkpoint](#source-operand-file-checkpoint-2026-10-13),
   then the [integrated policy metadata checkpoint](#integrated-policy-metadata-checkpoint-2026-10-13),
   then the [policy metadata API checkpoint](#policy-metadata-api-checkpoint-2026-10-13),
   then the [integrated table caller checkpoint](#integrated-table-caller-checkpoint-2026-10-13),
   then the [descriptor table checkpoint](#descriptor-table-checkpoint-2026-10-13),
   then the [complete table API checkpoint](#complete-table-api-checkpoint-2026-10-13),
   then the [complete synthetic tensor checkpoint](#complete-synthetic-tensor-checkpoint-2026-10-13),
   then the [convolution file checkpoint](#convolution-file-checkpoint-2026-10-13),
   then the [convolution operand checkpoint](#convolution-operand-checkpoint-2026-10-13),
   then the [BF16 matrix file checkpoint](#bf16-matrix-file-checkpoint-2026-10-13),
   then the [BF16 matrix operand checkpoint](#bf16-matrix-operand-checkpoint-2026-10-13),
   then the [complete synthetic vector checkpoint](#complete-synthetic-vector-checkpoint-2026-10-13),
   then the [GGUF vector binding checkpoint](#gguf-vector-binding-checkpoint-2026-10-13),
   then the [conversion plan and vector parity checkpoint](#conversion-plan-and-vector-parity-checkpoint-2026-10-12),
   then the [bounded runtime hashing checkpoint](#bounded-runtime-hashing-checkpoint-2026-10-12),
   then the [symbol origin checkpoint](#symbol-origin-checkpoint-2026-10-12),
   then the [runtime file pin checkpoint](#runtime-file-pin-checkpoint-2026-10-12),
   then the [verified file owner checkpoint](#verified-file-owner-checkpoint-2026-10-12),
   then the [metadata budget checkpoint](#metadata-budget-checkpoint-2026-10-12),
   then the [real comparison preparation](#real-comparison-preparation-2026-10-12),
   then the [projection role checkpoint](#projection-role-checkpoint-2026-10-12),
   then the [profile caller checkpoint](#profile-caller-checkpoint-2026-10-12),
   then the [synthetic v2 prefill checkpoint](#synthetic-v2-prefill-checkpoint-2026-10-12),
  then the [profile model policy checkpoint](#profile-model-policy-checkpoint-2026-10-09)
  at `6abb44c` and the [projection geometry checkpoint](#projection-geometry-checkpoint-2026-10-09)
   at `db002c1`, with resident compaction at `cedaeec`; then read the historical
   [real-model checkpoint](#current-checkpoint-2026-10-08) at `9408f3f`.
3. The goal is profile-based support for multiple Qwen3.5 9B-class models, with
   MiMo as the first test subject and most derivatives as the eventual aim (see
   the [support tiers](design.md#support-tiers-for-derivatives)). Existing real-model
   scoring tools verify MiMo's profile before loading; explicit offline inventory
   accepts the selected Defiant Fable metadata profile. Other-model execution
   remains unsupported (see the
   [coupling inventory](development.md#mimo-coupling-inventory)).
4. The user selected DavidAU's Defiant Fable safetensors source at `7af0a9c4` as
   the first non-MiMo profile. Its approved
   [preflight](development.md#preflight-results) ran on 2026-10-09 after the
   container rebuild with host certificates. Verified files and headers are
   cached; sampled `plusIQ` GGUF tensors match the source, but its template adds
   in-band controls. Never disable certificate verification.
5. P0.1-P0.4, bounded P3 and P4's separately versioned complete-model policy
  preparation plus synthetic actual-loader/prefill integration are implemented.
  The Python profile/inventory/manifest adapter now reaches actual BF16/PQ2
  model prefill for all seven eligible FFN/full-attention roles with independent
  synthetic references. Bounded P4 code-only preparation is complete; the next
  cached-MiMo numerical diagnostic has user approval within the budget below.
  Its one conversion attempt hit the swap guard; smaller real-operand kernel
  replays completed, not full-model tracing. The metadata-cap blocker is
  corrected within explicit bounds; this
  diagnostic is not new-profile P2, v2 real-model P4 or quality acceptance.
6. Keep the sole frozen candidate and untouched held-out data; no push, branch
   change or subagents.

**Documentation handoff review, 2026-10-09:** The user requested a documentation
check and a next-chat prompt, not more implementation. The starting source
checkpoint was user-pushed `db002c1`, with `main` matching `origin/main` and only
the untracked user `.vscode/` directory. Current checkpoint references, the
restart brief, coupling inventory and roadmap now consistently identify P0/P3
as complete and synthetic P4 preparation as next. The development guide includes
the exact bounded 11-control P3 command, separating it from historical real-model
inference commands. The obsolete constant-identity sentence in AGENTS.md was
corrected without changing approvals or safety boundaries. README and historical
model/numerical evidence are unchanged. Scoped content, local-link/anchor, fence,
table, Bash-syntax, editor-diagnostic and whitespace validation passed; no source,
native or model gates were rerun. Implementation stays paused in this chat;
the next chat must receive the user's explicit resume instruction.

## Template Preparation Checkpoint (2026-10-19)

After `104559a`, added `prepare_profile_model_template_metadata`. It requires
explicit `source` or `gguf_embedded` selection and snapshots immutable source
metadata before reusing context/source preparation. Extra GGUF template bytes
share the 32 MiB aggregate input budget. Source selection rejects extra bytes;
GGUF selection requires a separately declared pin and actual caller bytes,
with no fallback to source text. Exact length/SHA, nonempty strict UTF-8 and
NUL refusal precede the immutable selected-text receipt.

The receipt names `tokenizer.chat_template` but emits nothing. It retains
source/tokenizer/weight evidence limits, explicitly marks rendering
and prompt-boundary validity unchecked, and distinguishes the profile-declared
GGUF origin from independent source-GGUF verification. It does not load or
execute Jinja or any model repository code.

Evidence:

- Initial template/vocabulary/contract/context family: **59 passed,
  662 deselected in 0.25 s**.
- Refined the budget oracle so the GGUF case fits the source-only charge but
  fails when the extra selected template is correctly charged. Exact aggregate
  limits succeed; one-byte-under limits refuse without large allocations.
  Added exact UTF-8 round trips and the explicit origin-verification marker.
  Immediate focused family: **61 passed, 662 deselected in 0.26 s**.
- The 19 template cases separately cover source/GGUF selection, fresh-pin
  empty/NUL/invalid UTF-8, stale pins, absent/unbound external templates,
  mutable inputs, unsupported selection and unexpected external bytes.
- Required source gate: **875 passed, 96 skipped in 29.26 s**, Ruff and
  whitespace clean. Editor error checks for both changed Python files are
  clean. Optional controls were not rerun: this pure preparation increment
  changes no writer, parser-engine or native execution path.

No failed checks, new learned inputs, downloads, native builds, model jobs,
refits, candidates or quality experiments occurred. Owned scratch/model caches
and user settings remain untouched. Next: emit the selected pinned template
string using the trusted writer protocol, with duplicate/protocol preflight
and exact pinned GGUF string-byte verification. Complete tokenizer conversion,
rendered-prompt boundaries, full-model artifacts and runtime/quality acceptance
remain separate. See the [API contract](development.md#profile-gated-complete-model-policy).

## Standalone Token Checkpoint (2026-10-19)

After `abbd3b7`, added `prepare_profile_model_tokenizer_contract`. A private
builder lets vocabulary-only and contract preparation share the same pinned
source checks and one trusted parser instance without exposing that mutable
engine publicly. Vocabulary-only receipts are unchanged. Contract preparation
checks declared control-token IDs and standalone encodings, and all A-P IDs
and single-token encodings with `add_special_tokens=False`. Labels may not
have the parser's special-token flag; profile-declared controls need not have
that flag. Returned ID maps are immutable. Explicit markers leave rendered
prompt label boundaries, full tokenizer semantics, GGUF mapping, templates,
weights and runtime approval unchecked.

Evidence:

- Focused vocabulary/contract/context/architecture family: **62 passed,
  642 deselected in 0.52 s**. The six new default cases use a parser double
  to check one parse, ID and encoding mismatches, special-label refusal and
  nested receipt immutability.
- Pinned metadata-only control now uses explicit `identity`, `vocabulary`
  and `contract` modes. Actual cached-parser contract mode: **1 passed,
  2 deselected in 0.36 s**. Declared controls and all A-P labels pass, while
  fresh-profile-pin changed special IDs, a lowercasing normalizer and a label
  marked special each refuse. Existing exact context-byte checks remain.
- The two changed non-contract modes separately passed: **2 passed,
  1 deselected in 0.58 s**. Each child keeps the 15 s deadline and owned
  tensor-free output smaller than 64 KiB. No rendered chat prompt or model
  execution occurs.
- Required source gate: **856 passed, 96 skipped in 24.47 s**, Ruff and
  whitespace clean. Editor error checks are clean; optional-import warnings
  recorded in the preceding checkpoint remain environment prerequisites.
  Pylance finds both direct vocabulary call sites compatible; no direct
  callers exist for the parameterized pytest function, whose three dynamic
  modes were actually exercised.

No checks failed in this increment. No learned inputs, downloads, native
builds, model conversions, refits, candidates or quality jobs ran. Standalone
labels do not replace the existing `probe_label_boundary` requirement that
each label extend the actual rendered prefix by exactly one token. Next:
prepare explicit source-versus-separately-pinned-GGUF template selection
without rendering template code or guessing which template belongs in a
converted artifact. Full-model/runtime/quality work remains separately
approved. See the [API contract](development.md#profile-gated-complete-model-policy).

## Vocabulary Preparation Checkpoint (2026-10-19)

After `078f339`, added `prepare_profile_model_tokenizer_vocabulary`. It reuses
the exact bounded source/context preparation, then parses declared BPE JSON
with the trusted installed `tokenizers.Tokenizer` implementation. No code from
a model repository is imported. Vocabulary IDs must be unique, contiguous
from zero and fit profile capacity. Immutable ID-ordered token strings and
unassigned-profile-row counts do not imply a GGUF padding-token policy.
Special/label token behavior, full tokenizer semantics, GGUF mapping and
template/weight/runtime compatibility remain explicitly unchecked. No writer
is mutated or tokenizer/merge metadata emitted by this API.

Evidence and limits:

- The initial focused run had **10 failed, 46 passed in 1.15 s** because the
  new default test double omitted a local `sys` import. No new behavior was
  reached in those failing cases. After that fixture repair, the same family
  passed: **56 passed, 642 deselected in 0.54 s**. The default tests deliberately
  use a parser double to isolate pin/model/parser/layout refusal and ordering.
- A small ordering refinement avoids intermediate item tuples and an unused
  binding. Its immediate vocabulary-only check passed: **10 passed,
  688 deselected in 0.16 s**.
- The existing pinned metadata-only control now has inert identity-only and
  actual-BPE-vocabulary parameters. The new real cached parser case first
  passed **1 passed, 1 deselected in 0.39 s**; after the ordering refinement
  it passed **1 passed, 1 deselected in 0.34 s**. It verifies ordered synthetic
  tokens, a fresh-profile-pin invalid merge refusal, and the unchanged exact
  GGUF context bytes. The changed inert branch separately passed **1 passed,
  1 deselected in 0.31 s**. Each child retains the **15 s** deadline and the
  output remains a disposable tensor-free GGUF smaller than 64 KiB.
- Required source gate: **850 passed, 95 skipped in 22.91 s**, Ruff and
  whitespace clean. The normal run skips both opt-in metadata cases above;
  their actual execution is separate evidence. Editor error checks are clean.
  Pylance's explicit diagnostic query reports optional `tokenizers` unavailable
  in the editor environment, alongside existing optional `torch`/`gguf` import
  warnings. This is not a cached-runtime failure or a claim that the editor
  interpreter changed. Signature inspection finds no direct callers of the
  parameterized pytest function; both dynamic pytest branches actually ran.

No weights, learned tokenizer bytes or model code were read, and no downloads,
native builds, conversions, refits, candidates or quality jobs ran. All scratch
remains owned/disposable under the unchanged synthetic budgets. Next: check
declared profile special/label token behavior with this same trusted parser
on synthetic bytes before proposing GGUF vocabulary/merge emission. Full-model
artifacts and runtime/quality work remain separately approved. See the
[API contract](development.md#profile-gated-complete-model-policy).

## Context Emission Checkpoint (2026-10-19)

After `4c3bf2b`, added `write_profile_model_context_metadata`. It repeats bounded
source-byte preparation instead of trusting an externally supplied receipt,
then emits just `qwen35.context_length`. Architecture and context emitters now
share the existing single-output/protocol/duplicate preflight and typed-record
writer. Architecture records and evidence markers are unchanged. The context
receipt freezes emitted records while retaining pinned-source evidence and
unchecked tokenizer semantics, separate GGUF template and source weights.
No runtime approval follows. Preflight failures leave metadata untouched;
arbitrary later writer-method failures do not promise rollback.

Evidence:

- Focused default metadata/preparation/emission family: **52 passed,
  636 deselected in 0.44 s**, including exact values, source-pin refusal,
  duplicate refusal, wrong architecture and split-writer refusal.
- New pinned metadata-only control initially failed before child execution
  because its fixture builder omitted a local `hashlib` import. In that run
  the existing frozen/architecture writer control passed: **1 failed,
  1 passed, 5 deselected in 4.28 s**. The missing import was repaired; the
  failed context test alone then passed: **1 passed in 0.31 s**. The successful
  architecture control was not repeated without a behavior change.
- The pinned context control independently reads exact UINT32 bytes from a
  disposable GGUF smaller than 64 KiB, with no tensors and an inert `{}`
  tokenizer identity fixture. It checks duplicate refusal and closes the
  writer, optional spool and reader mapping before temporary cleanup. The
  child timeout remains **15 seconds**; no model is converted or executed.
- Required source gate: **840 passed, 94 skipped in 25.80 s**, Ruff and
  whitespace clean. Editor diagnostics for all three changed Python files
  are clean. The default run skips the optional controls above; their actual
  execution is reported separately.

Scoped documentation validation caught a newly mistyped development heading;
the link was corrected to the existing complete-model policy section and
rechecked without rerunning unchanged source gates.

No new learned inputs, downloads, native builds, model jobs, refits, retained
candidates or quality experiments ran. Context emission is not tokenizer/BPE
or template emission, full metadata acceptance or loader compatibility.
Next: prepare bounded tokenizer metadata through a trusted existing parser
using synthetic pinned bytes before attempting a complete streaming converter.
Keep full-model artifacts and real-model/runtime/quality work separately
approved. See the [API contract](development.md#profile-gated-complete-model-policy).

## Metadata Preparation Checkpoint (2026-10-19)

After `0111f89`, extracted `verify_profile_metadata` from the existing header
verifier without changing its metadata/shard error precedence. It verifies
the exact metadata set/lengths/digests without requiring weight headers.
`prepare_profile_model_context_metadata` now verifies immutable caller
metadata and separate tokenizer bytes against the profile, with a **32 MiB
aggregate input** ceiling before hashing/parsing. It reads the profile-selected
text configuration, requires declared integer `max_position_embeddings`
within **1,048,576**, and bounds an explicit positive integer requested context
to that source maximum. No silent default for missing/unsupported context.

The immutable receipt prepares `qwen35.context_length`, identifies the pinned
source template and explicitly leaves tokenizer semantics, separate GGUF
template, weights and runtime approval unchecked/not granted. Source-template
bytes are pinned, not rendered. JSON parsing/hash checks execute no tokenizer,
model-repository or template code. Tests deliberately accept an inert `{}`
tokenizer as a **byte-identity fixture only**, not a working vocabulary/merges
or prompt-format compatibility claim. No context field is written yet.

Controls cover both existing text-only/multimodal config paths, metadata sets,
changed sizes/bytes, tokenizer pin mismatch, missing/boolean/oversized source
context, invalid/too-large requests, mutable inputs, an injected aggregate
budget and receipt immutability. The original header verifier retains its
exact precedence when metadata and shard declarations are both invalid.
Final focused slice: **46 passed in 0.46 s**. Initial gate: **834 passed,
93 optional skips in 24.05 s**. After clarifying the receipt's source-versus-
GGUF-template boundary, focused checks pass again and the required source
gate is **834 passed, 93 optional skips in 21.93 s**, Ruff/whitespace clean;
source/test editor diagnostics were clean before that two-field clarification.
No failures or guard relaxation. Optional converter/native controls were not
rerun: no writer/runtime execution path changed.

No files, learned weights, downloads, native builds/model execution, complete
conversion, refit/candidate/quality work, background jobs or owned scratch.
Next: bounded source-bound context emission and separate tokenizer/template
semantic preparation using pinned implementations. Complete tokenizer/GGUF
metadata and public whole-table/loader acceptance remain open; real-model
jobs still require separate bounded approval.

## Architecture Metadata Checkpoint (2026-10-19)

After `67d10d5`, promoted the existing pinned fixture's checked architecture
emission pattern into `write_profile_model_architecture_metadata`. It reuses
the profile-validated model specification, emitting three base geometry keys
plus 14 typed fields: uint32, float32 and the int32 rotary-section array.
The writer must target `qwen35`, expose the expected methods and have exactly
one metadata/output part, matching the resource plan. Wrong architecture,
split writer, missing protocol and duplicate target keys refuse before
metadata mutation. Call before headers; this is not rollback protection
against arbitrary later writer-method failure.

The immutable receipt freezes names/kinds/values, but correctly labels them
declared/emitted rather than file-verified; source/tokenizer identity and
runtime approval remain unchecked/not granted. Default tests cover hybrid/
full attention and two existing synthetic geometries, exact records, repeated
emission refusal, immutability and preflight nonmutation. Final focused slice:
**21 passed in 0.47 s**. The pinned frozen-writer case now emits this subset
and independently verifies all **17 metadata fields' exact types and bytes**
with the pinned GGUF reader, alongside all 28 tensor checks: **1 passed,
5 deselected in 4.04 s**. Unchanged earlier optional cases were not rerun.

Required gate: **808 passed, 93 optional skips in 23.66 s**, Ruff/whitespace/
editor clean; no failures in this increment. Source/output fixture and
15-second child bounds are unchanged. No learned inputs, downloads, native
builds/model execution, complete conversion, candidate/refit/quality action
or retained scratch. User editor settings untouched.

This remains the checked architecture subset, not context/tokenizer metadata,
complete model metadata, public whole-table/loader acceptance or a complete
converter. Next: bounded source-pinned tokenizer/context metadata preparation
using existing profile metadata contracts, without executing model-repository
code or authorizing real-model artifacts.

## Frozen Writer Checkpoint (2026-10-19)

After `dfe097f`, added one separately selected frozen-provenance writer case
beside all five existing pinned synthetic cases. It explicitly enables
frozen source ownership, verifies that raw range against the manifest, and
spools the existing packed payload unchanged as `PQ2_0`; no dense reconversion
or refit. The same parent owns **28 selected tensors: 16 BF16 matrices,
11 F32 auxiliaries and the frozen projection**, with exactly two whole-file
hashes on one descriptor. Parent source/payload identity success still
precedes GGUF publication. All original five source-mutation publication
refusals run with the explicit frozen opt-in too.

The output has exactly the declared 28 tensor names; every entry retains
exact typed file/range/byte and corruption/refusal checks. The existing
internal bounded projection verifier checks the frozen packed bytes, while
the independent source context checks dense-source provenance. These are
distinct checks, not quantization-quality proof. Fixtures/source/output remain
below 16 MiB, declared storage below 256 MiB, and child timeout **15 seconds**.

Focused new case: **1 passed, 5 deselected in 10.58 s**. After shared test
changes, the prior five cases separately pass: **5 passed, 1 deselected in
18.22 s**. All six actually executed; no unchanged successful case was rerun
for reassurance. Required gate: **788 passed, 93 optional skips in 28.74 s**,
Ruff/whitespace/editor clean. No failures in this increment. Before continuing,
a read-only host snapshot showed **19,041,157,120 bytes available RAM, zero
swap used**; time-sensitive, not model-job headroom or device evidence.

This covers the tiny selected tensor set, not complete architecture/tokenizer
metadata, public whole-table/loader acceptance, a complete streaming converter
or any learned model job. No download, native build/runtime, conversion/refit,
candidate/quality action or retained scratch. Next: bounded profile-declared
architecture metadata emission and pinned validation; tokenizer/template
identity and any real-artifact approval remain separate.

## Frozen Source Owner Checkpoint (2026-10-19)

After `83f2870`, the mixed v2 source owner accepts a strict boolean
`include_frozen_source` opt-in, default **false**. Default mixed and both
matrix-only APIs still refuse frozen ranges. Explicit opt-in maps only the
declared frozen BF16 source, binding caller digest to a snapshot of the
manifest's source digest and actual bytes to the profile-pinned raw range.
The frozen source name, digest, payload digest and layout are retained at
entry; later caller manifest mutation cannot replace them. Receipts label
the child `frozen_source` and explicitly leave source-to-frozen quantization
verification unchecked. The existing BF16 converter still refuses this row.
No code/scale regeneration, refit or candidate promotion.

Successful parent exit checks frozen payload bytes, dtype, shape and contiguous
layout against entry. Synthetic byte/shape/dtype mutations each refuse;
fixtures are fully restored in `finally`. Tests also preserve exact payload
hashes through successful mapping, exercise caller/manifest digest mismatch,
wrong declared manifest digest versus actual range, immutable snapshot and
default refusal. Nonboolean flags refuse before planning/opening. BF16/F32
auxiliary/matrix lifetimes and legacy frozen verifier checks remain intact.
Final focused slice: **131 passed in 10.00 s**; no failures in this increment.

Required gate: **788 passed, 92 optional skips in 79.44 s**, Ruff/whitespace/
editor clean. Five cached pinned regressions pass in **47.31 s**, each retaining
its unchanged 15-second subprocess deadline. These are aggregate test-suite
times, not an individual synthetic artifact-stage or measured host-memory
guarantee; no deadline/resource cap was raised and no successful gate was
rerun. Pylance confirms all **4 public** and **3 private** resolved call sites
compatible with the optional signatures. No learned input, download, runtime
build/model execution, complete conversion, new candidate or quality action.
No owned scratch/background jobs; user settings remain untouched.

Next: include frozen source proof and the existing frozen payload in bounded
pinned writer publication controls, without dense conversion. Metadata/
tokenizer and complete-converter acceptance remain separate; another real
artifact still needs its own bounded approval and cannot reuse the failed
conversion unchanged.

## Mixed Writer Checkpoint (2026-10-19)

After `3ee4ac1`, added two pinned synthetic mixed-writer cases, using BF16/F32
auxiliary sources alongside the three existing BF16/v1, F32 and BF16/v2 cases.
Each mixed case owns **27 unfrozen operands: 16 matrices and 11 auxiliaries**
under one source parent and exactly two whole-file hashes on one descriptor.
Vectors/convolution are decoded from verified mappings and passed through
the pinned converter transforms, existing exact arithmetic gates and temporary
writer spooling without retained writer tensor arrays. Every written operand
then passes its typed GGUF range/byte gate and file/budget/view/shape/payload
corruption refusals. The existing five source-mutation publication controls
also run for mixed owners. The pure resource planner gates declared storage
at **256 MiB** before opening the writer. Actual source/output files remain
**below 16 MiB**; no larger artifact or learned source is used.

Initial expanded check: **2 failed, 3 passed in 17.34 s**, with child stderr
lost in the oversized subprocess-command traceback. The local test now
explicitly asserts a zero return code with captured stderr, preserving the
failure check and deadline. Diagnostic mixed-only check: **2 failed in
7.39 s**; both report `verify_profile_gguf_vector() got an unexpected keyword
argument 'maximum_bytes'`. The mixed test now uses the vector wrapper's
supported explicit file budget and fixed 64 KiB operand contract; matrices
and convolution retain their typed operand budgets. Vector file-budget
refusal replaces the inapplicable caller-operand-budget argument; no public
API or production cap changed. All five cases then pass in **19.11 s**.

Required gate: **782 passed, 92 optional skips in 37.84 s**, Ruff/whitespace/
editor clean. The two new default skips were actually run in the separate
five-case pinned check. **Timeout correction:** earlier checkpoint prose
described a 30-second child bound; this test's source uses **15 seconds**.
That stricter value was discovered on read and remains unchanged, not raised
to the approved ceiling. No native build/model execution, downloads, learned
inputs, refit/candidate/quality work or retained scratch. Next: separately
versioned frozen-source provenance ownership without dense reconversion or
changes to frozen codes/scales; metadata/tokenizer and complete-converter
acceptance remain separate, and real-model jobs still need explicit approval.

## V2 Mixed Source Checkpoint (2026-10-19)

After `33cc013`, added explicit `mapped_profile_safetensors_conversion_shard_v2`
for serial BF16 matrices plus declared BF16/F32 one-dimensional vectors and
three-dimensional convolution kernels. Auxiliary raw ranges cap at **1 MiB**;
existing v2 file/matrix ceilings, 16 MiB defaults, deadlines, single-child
ownership and two full-file hashes are preserved. Both matrix-only APIs stay
matrix-only, with unchanged receipts. The mixed owner adds policy/auxiliary
cap and child dtype/shape/kind receipt fields. Read-only BF16/F32 mappings use
the retained descriptor; frozen replacements still refuse. Raw ownership
does not grant arithmetic, finiteness, runtime or model approval.

BF16/F32 positives map vectors, convolution and a matrix serially with exact
raw hashes, immutable layouts, overlap refusal, cleanup and exactly two
whole-file hashes. Injected oversized auxiliary geometry refuses before
mapping at the tighter 1 MiB boundary; no large operand is created. The
existing 23-case matrix lifetime/allocation family also runs through the mixed
owner, including source mutation, deadline, body failure and 32 MiB adjacent
composition controls. Initial focused result: **117 passed, 1 failed in 9.67 s**.
The failure was the old vector-rejection oracle: the mixed owner admits the
vector, then correctly rejects the unrelated matrix digest. Its new case now
requires exactly `mapped source tensor identity mismatch`; legacy matrix-only
expectations are untouched. Final focused slice: **118 passed in 9.43 s**.

Required gate: **782 passed, 90 optional skips in 25.09 s**, Ruff/whitespace/
editor clean. Cached pinned BF16/v1, F32 and BF16/v2 controls: **3 passed in
8.90 s**. Pylance confirms all **3 private-owner call sites** remain compatible
after its optional auxiliary flag. No new dependencies, learned inputs,
native builds/runtime jobs, artifacts or broader quality work. Scratch is
disposable and cleaned. Next: exercise mixed-source auxiliary arithmetic and
writer publication with the existing pinned synthetic converter fixtures;
complete metadata/tokenizer/frozen integration and real-model approval remain
separate.

## V2 Resource Checkpoint (2026-10-19)

Resumed after user-pushed `8a16571`, tracked clean with only untouched user
editor settings untracked. Added pure `profile_model_streaming_resource_plan_v2`
over the existing profile-validated conversion plan. Caller budgets are
explicit strict integers, capped at **6 GiB/input file**, **2 GiB/source or
output operand** and **48 GiB aggregate logical storage**. Invalid budgets
refuse before conversion-plan construction. No file opening or work launch.

The conservative sum counts every pinned source shard and metadata input,
the independent frozen payload input, the full 32-byte-aligned writer spool,
one provisional output with the existing 32 MiB metadata allowance, and the
largest nonfrozen temporary payload. Frozen codes/scales are never regenerated.
Cached input sizes are still charged, so this is not merely additional free
disk. Per-file/tensor counts are detached from caller inventory/profile.
Receipts explicitly mark source/tensor verification unchecked, memory
unmeasured and runtime approval not granted. Existing file acceptance and live
resource guards are not replaced by this planning bound.

Independent synthetic size oracles cover BF16/F32 auxiliary operands, one/two
shards and small/32 MiB declared embedding geometry. Exact storage budgets
pass; one-byte-under file/operand/aggregate budgets refuse. The larger full
staging plan correctly refuses **256 MiB**, unlike the earlier selected-only
composition. All planning checks prohibit file opening and create no large
fixtures. Strict invalid-budget checks cover zero, negative, boolean, float
and above-ceiling values. Final focused slice: **31 passed in 0.50 s**.
Required gate: **757 passed, 90 optional skips in 20.68 s**, Ruff/whitespace/
editor clean. No failures in this increment. Optional pinned converter/native
controls were not rerun: this is a pure static preflight, not changed execution.

No learned inputs, downloads, conversions, runtime builds/execution, candidate
or quality work; no owned scratch or background jobs. Next: source ownership
for the small vector/convolution operands needed alongside serial BF16 matrix
streaming, using existing pinned arithmetic controls and bounded fixtures.
Metadata/tokenizer integration, complete converter/resource validation and
separate approval for any real-model artifact remain open.

## V2 Writer Checkpoint (2026-10-19)

After `e96a830`, extended the existing optional pinned converter/GGUF control
with an explicit BF16/v2 case alongside unchanged BF16/v1 and F32 cases.
The v2 case selects only the v2 shard owner, descriptor hasher and raw payload
owner, and checks their version/cap receipt fields. All **16 matrices** retain
exact pinned Torch/transform/GGUF-byte checks, temporary writer spooling
without retained tensor arrays, and exactly **two whole-file hashes on one
descriptor**. GGUF publication remains after successful shared-parent exit.
All **five publication refusals** also run for v2: selected first/last matrix
bytes changed before payload exit, plus changed/restored/replaced source
after child contexts close. Exact original error assertions, cleanup and
no-published-bytes checks remain. Legacy composed-owner controls are preserved.

The initial focused attempt failed all three cases in **0.37 s**, before
launching child work: `_profile_parity_fixture` returns JSON text, not a dict.
The local test now parses and reserializes that JSON to add the explicit
policy field; no helper contract or guard changed. Same focused check after
repair: **3 passed in 8.22 s**. Required gate: **734 passed, 90 optional skips
in 20.29 s**, Ruff/whitespace/editor clean. The extra default skip is the new
opt-in v2 case; the separate three-case run actually executed it.

This pinned fixture remains **below 16 MiB** and each child retains its
original **30 s** timeout. The larger **32 MiB** mapping/payload and
below-3-MiB traced-Python allocation evidence is the separate default-test
fixture in the preceding checkpoints, not a larger pinned GGUF measurement.
No source/cache download, runtime/native build, learned-weight read,
conversion/refit/candidate/held-out scoring or retained artifact. All scratch
is disposable and cleaned by its owners. README and user editor settings
remain untouched.

The approved synthetic-only v2 policy increment now has explicit descriptor,
shared source, raw payload and pinned writer coverage. This does not yet make
a full streaming converter: metadata/tokenizer/vector/frozen-provenance
integration and a complete memory/resource plan remain distinct work. Any
real artifact or complete-model job still requires separate bounded approval;
the earlier guarded conversion must not be retried unchanged.

## V2 Payload Checkpoint (2026-10-19)

After `f8b2f85`, added `temporary_profile_bf16_matrix_payload_v2` with an
explicit **2 GiB** policy ceiling and unchanged **16 MiB** default. Legacy
raw payload ownership still caps at **16 MiB**. Both public wrappers share
the existing chunked conversion, source/payload entry/exit hashes, identity
checks, cooperative deadline and `finally` cleanup. Only v2 receipts add
`payload_policy_version: 2` and `payload_cap_bytes`. Payload ownership alone
still does not verify a source file or grant runtime approval.

All 18 existing refusal/cleanup cases now run through both versions; unchanged
head/index transform workflows also pass. The larger sparse fixture now
composes v2 source mapping with a **32 MiB raw payload**, checks the exact
output hash in 64 KiB reads and source/output scratch below **256 MiB**, and
proves successful private-file/directory cleanup. Legacy/v2 default payload
budgets and above-2-GiB budgets refuse before file opening. The exact 2 GiB
ceiling accepts this bounded input. An injected failure after the first chunk
also removes partial output. Traced Python allocations during the successful
32 MiB payload lifecycle stay below the existing **3 MiB** streaming bound;
this is not RSS, mapped-page, OS-cache or full-converter memory evidence.

Final focused checks: **47 passed in 1.85 s**. Required gate: **734 passed,
89 optional skips in 24.83 s**, Ruff/whitespace/editor clean. Cached pinned
BF16/F32 converter controls: **2 passed in 6.33 s**. No failures, downloads,
builds, learned-weight reads or retained artifacts in this increment.
Next: exercise explicit v2 owners through the existing pinned synthetic GGUF
publication workflow, without increasing real inputs or authorizing a
complete conversion/runtime/quality job.

## V2 Shard Checkpoint (2026-10-19)

After `51eb96b`, added `mapped_profile_safetensors_bf16_shard_v2` beside the
unchanged legacy entry point. Both share a private ownership implementation;
legacy ceilings remain **16 MiB**, while the explicitly named v2 API permits
at most **6 GiB per file** and **2 GiB per selected matrix**. Defaults stay
**16 MiB** for both file and child budgets: callers must opt into larger
values even after choosing v2. V2 uses only the new descriptor hasher and
adds source policy version/file-cap/operand-cap fields to immutable receipts.
Legacy receipt fields, numerical checks and native/v1 policies are unchanged.

The full existing entry/lifetime/allocation family also runs through v2:
**138 focused passes in 8.86 s** before the larger fixture. The synthetic
fixture helper now accepts an explicit vocabulary size with its prior default
unchanged. One sparse source is **above 64 MiB and below 128 MiB**, with a
**32 MiB** embedding at an offset beyond the old 16 MiB boundary. Source setup
uses a header and sparse extent; expected zero-stream hashes and conversion
use bounded chunks, not whole-shard arrays. Source plus selected operand fits
the approved **256 MiB** scratch budget. Exact chunk hashes, read-only mappings,
cleanup and exactly two whole-file hashes on one descriptor pass.

Legacy and v2-default file refusals occur before open. Above-6-GiB and boolean
file budgets also refuse before open; above-2-GiB and boolean operand budgets
refuse before mapping. Exact ceiling values are exercised with bounded actual
inputs, not multi-GiB artifacts. V2 child defaults still refuse the 32 MiB
operand. Final larger-range/lifetime subset: **93 passed in 6.41 s**.
Required gate: **716 passed, 89 optional skips in 19.71 s**, Ruff/whitespace/
editor clean. Cached pinned BF16/F32 converter controls both pass in **5.76 s**;
no dependencies were downloaded or rebuilt. No failures in this increment.

This is approved synthetic source policy preparation, not a larger raw-payload
owner, complete converter, process-RAM measurement, learned-weight read or
runtime/quality acceptance. All files are disposable scratch; no learned
inputs, full conversion, refit or new candidate. Next: an explicitly versioned
raw payload owner with at-most-2-GiB policy and bounded 32 MiB synthetic tests,
keeping the old 16 MiB payload API unchanged.

## V2 Descriptor Checkpoint (2026-10-19)

Added `bounded_source_descriptor_record_v2` with an explicit **6 GiB** ceiling,
sharing the existing private positional hashing loop. The old
`bounded_native_descriptor_record` retains its **64 MiB** ceiling and signature;
all source mapping/payload APIs retain their **16 MiB** limits. No environment
variable, profile or file-size inference selects the new entry point.

Both hashers run the existing descriptor-position, deadline, nonregular-file,
growth/truncation and malformed-budget controls. A disposable sparse
**64 MiB + 17 byte** fixture proves the old entry refuses before reading and
the explicit v2 entry hashes the exact extent in at-most **64 KiB** reads,
preserves descriptor position and returns the independent zero-stream SHA-256.
The isolated traced Python allocation peak is below **1 MiB**; this is not
process RSS, mapped-page, OS-cache or full-model memory evidence. Cleanup runs
in `finally`. No multi-GiB fixture or model file was created or read.

Recovered edits from the interrupted resume passed the required source gate:
**692 passed, 89 optional skips in 18.14 s**, Ruff/whitespace/editor clean.
No current failing check, dependency installation, native build, optional
converter run, learned input, conversion, refit, candidate or quality experiment.
This implements only the descriptor primitive of the approved larger-source
policy, not a larger-source adapter or runtime authorization. Next: add the
separately versioned shard owner with explicit 6 GiB file/2 GiB operand bounds,
while leaving existing APIs unchanged and actual synthetic fixtures within
the approved 128 MiB file/32 MiB operand/256 MiB scratch budget.

## Larger Source Proposal (2026-10-16)

After `622e307`, the bounded shared-shard BF16 source-to-writer prerequisites
are verified. Further work toward real source geometry crosses the existing
**16 MiB** source-file/operand/payload caps. The user explicitly approved the
synthetic-only API/test expansion below after `30e3f15`, then paused before
implementation started. This is **not implemented or approval for a real-model
conversion**. Resume requires a fresh user instruction, not repeated scope approval.

A structured read of the repository's pinned MiMo profile, with no cached
weight reads, gives largest shard **5,276,436,216 bytes** and BF16 embedding/head
extent **2,034,237,440 bytes**. Proposed separate versioned source policy:

- File cap **6 GiB (6,442,450,944 bytes)** and operand/raw-payload cap
  **2 GiB (2,147,483,648 bytes)**, each still requiring caller/profile identity.
- Keep all current 16 MiB APIs and the existing bounded descriptor hasher's
  64 MiB maximum unchanged. Use an explicit opt-in versioned policy, not
  inference from file size, environment variables or profile runtime approval.
- Preserve 64 KiB reads/default chunks, 1 MiB chunk/header caps, one live matrix,
  shared entry/exit hashes, zero full-matrix F32 expansion, provisional receipts,
  child/parent lifetime refusals and existing numerical/BitNet boundaries.
- First validation is **synthetic only**: no more than **128 MiB per source
  fixture**, **32 MiB selected operands**, **256 MiB aggregate logical scratch**,
  serial stages at most **60 s** with parent/finally cleanup. Existing opt-in
  converter subprocess tests retain their 30 s bound. Reuse caches/environments;
  no downloads, library rebuilds or changes to host RAM/swap/security guards.
- Exercise offsets/file extents above the old caps, streaming fixture creation,
  exact bytes, hash counts, allocation scaling, malformed budgets and partial
  failure cleanup. Boundary tests for proposed 6 GiB/2 GiB caps must reject
  before opening/allocating; no actual multi-GiB synthetic artifact is needed.

The user's approval for this bounded API/test expansion does **not** authorize reading
cached learned weights, another complete-model conversion/artifact, runtime
execution, refit, candidate, compensation promotion, held-out quality or another
model. A future real-model job still needs its own complete artifact/resource
plan and explicit approval; historical guard/numerical failures remain open.
Existing source gate is unchanged from `622e307` and is not rerun for this
documentation-only proposal. Next after explicit resume: add a separately
named v2 descriptor hasher backed by the existing bounded loop, preserving the
old 64 MiB hasher and all 16 MiB APIs. The first targeted descriptor-test lookup
was interrupted without usable output; no new hasher, test or larger file exists.

## BF16 Shared Writer Checkpoint (2026-10-16)

After `e5c94ba`, the existing optional pinned BF16 converter control uses one
shared source shard around all **16 matrix** contexts, rather than a new shard
owner per operand. Each read-only matrix mapping feeds the existing chunked raw
payload owner and actual `GGUFWriter(use_temp_file=True)` spool. Payload and
source mappings close before the next matrix. The control asserts **one
whole-shard hash** while all operands are spooled, then **exactly two** on the
same descriptor after successful parent exit. GGUF header/metadata/tensor output
is written only after that parent closure. Existing exact file/range/shape/
operand/corruption controls remain unchanged.

Three new publication refusals change the source only after both first/last
selected matrix consumers have finished: changed bytes, restored bytes/mtime,
and identical-file replacement. Each must fail the parent's exact
`mapped source file identity changed` guard, leave no published GGUF bytes,
close the temporary writer stream and restore only the disposable synthetic
source. The earlier two selected-byte mutations inside a mapped payload still
require the earlier live-source array-hash error. No source-identity or output
comparison assertion was loosened.

Both optional **BF16/F32 controls pass in 4.92 s**, with cached pinned sources
and the existing 30-second subprocess bound. Required source gate:
**677 passed, 89 optional skips in 18.31 s**, Ruff/whitespace/editor clean.
No failures, learned inputs, complete-model conversion, downloads, native
runtime build/execution, refits, candidate changes or quality run occurred.
All generated shards/payloads/GGUFs/spools are disposable fixture scratch.

The bounded descriptor-to-chunks-to-writer path and shared-shard publication
gate are now verified. This is not a complete converter, measured process-RAM
reduction, runtime approval or real-model parity. Existing caps remain
**16 MiB**. Next: prepare a separately versioned large-file policy proposal
with explicit file/operand/resource budgets; do not raise existing API limits
or retry the guarded complete-model conversion without separate approval.

## BF16 Shared Shard Checkpoint (2026-10-16)

After `0211d3c`, added `mapped_profile_safetensors_bf16_shard`, retaining one
verified source descriptor and parsed pinned header across serial matrix
contexts. It yields a matrix context factory and immutable provisional shard
receipt. Each child checks the snapshotted profile plan/routing, selected raw
range SHA-256, geometry and source identity, and owns one read-only mapping.
At most one child is live. Whole-shard hashing occurs **once at entry and once
at successful exit**, not for every selected matrix; all positional reads stay
at most **64 KiB**. File/operand caps remain **16 MiB**, and one cooperative
**30 s default/600 s max** deadline includes all consumers and exit checks.

The earlier per-matrix API now wraps this owner while preserving its preflight
guards and receipt. The mapped payload API and native/v1 paths remain unchanged.
Caller inventory routing changes cannot redirect the retained snapshot.
Cross-shard/undeclared-shard operands refuse. Child exceptions invalidate normal
shard closure even when swallowed by the caller; unfinished children refuse
publication and are forcibly unmapped. Closed-owner calls refuse. Finish all
child contexts/aliases and payload users before parent exit; publish only after
successful parent closure, not individual child closure. Existing concurrent
mutation/truncation and borrowed-alias limitations remain.

Focused controls prove two serial matrices share one descriptor with exactly
two whole-file hashes, forbid overlap, and check routing mutation isolation,
swallowed errors, unfinished mappings and between-child writes. The existing
entry/lifetime/retained-allocation/peak-scaling cases also run directly through
the shared API. A two-shard fixture proves that valid geometry/zero-byte hashes
cannot substitute an operand from another shard. **115 focused passes in
8.17 s**; required gate **677 passed, 89 optional skips in 19.42 s**, Ruff,
whitespace and editor diagnostics clean. Both existing optional pinned
**BF16/F32 converter controls pass in 5.41 s** after the wrapper refactor.
No failures, learned inputs, complete-model conversion, runtime build/execution,
downloads, refits, candidates or quality run occurred in this increment.

Next: use one shared shard across all 16 pinned synthetic GGUF matrix consumers,
buffer publication until shard closure and verify the two-hash count there.
Larger files, resource budgets and real conversion remain separately approved.

## BF16 Allocation Scaling Checkpoint (2026-10-16)

After `9f597bb`, a bounded `tracemalloc` probe compared warmed copying,
descriptor-mapped and mapped-payload paths on the same synthetic shard. Input
operands were **65,536 bytes** and **2,097,152 bytes**; fixture construction,
caller inputs and warmup were outside tracing. Observed peak tracked bytes:

| Path | 64 KiB Operand | 2 MiB Operand |
| --- | --- | --- |
| Copy-based source verifier | 2,405,621 | 4,377,855 |
| Descriptor-owned source mapping | 2,407,157 | 2,407,085 |
| Mapped source to raw payload | 2,529,134 | 2,528,902 |

The probe used the explicit existing `/opt/venv/bin/python`, not an inferred
editor interpreter, and automatic temporary cleanup. These are Python-tracked
allocation peaks, not process RSS, native allocations, mapped pages, OS cache,
whole-converter RAM or attribution of the original swap-guard failure.

The existing lifecycle fixture now runs a two-size peak-scaling regression
for both new consumers. The 2 MiB copying control must expose at least two
operand-sized buffers; the mapped path must stay below **3 MiB** tracked peak
and grow by at most **256 KiB** from the 64 KiB case. These newly introduced
budgets passed unchanged. All mappings/descriptors and private payloads must
close. If an external allocation tracer already owns the process, only these
two isolated-tracer cases skip. Focused controls: **46 passed in 3.54 s**.
Required gate: **653 passed, 89 optional skips in 23.96 s**, Ruff/whitespace/
editor clean. No new failures, dependency installs, learned inputs, conversion,
native execution/build, candidate change or quality run occurred.

The verified source/payload and source-file caps remain **16 MiB**. Current
per-matrix mapping hashes the shard repeatedly; this is intentionally bounded
evidence, not yet an efficient full-model source adapter. Next: retain one
verified shard owner across its selected matrix consumers before proposing a
separately capped larger-file policy. Never raise the current limits or retry
the guarded complete-model conversion implicitly.

## BF16 Mapped Payload Checkpoint (2026-10-16)

After `967e2cc`, added `temporary_profile_mapped_bf16_payload`, composing the
descriptor-owned source mapping with the existing chunked raw payload owner.
The caller supplies source-file/tensor pins, not a complete source array. One
cooperative **30 s default/600 s max** deadline spans source verification,
payload production, caller lifetime and both owners' exit checks. Source and
operand/payload caps remain **16 MiB**; chunk limits remain **64 KiB default/
1 MiB max**. The immutable provisional receipt nests the mapped-source receipt.
Finish payload users first and publish only after both contexts close
successfully. Older copy-based APIs, v1 factories and native policies are unchanged.

All 22 source-entry/lifetime/allocation cases now run for both direct mappings
and mapped payloads: **44 pass in 2.70 s**. Both retain less than **1 MiB** of
additional traced Python allocations while owning the **2 MiB** selected source
or payload. File reads stay at most 64 KiB; mapping/descriptor/private-directory
cleanup is checked on normal and failed exits. This remains retained-allocation
evidence, not peak RAM, mapped RSS, OS cache or whole-converter memory evidence.

The pinned synthetic BF16 GGUF writer now uses this API for all **16 matrices**
without caller-created source mappings/arrays passed to the producer. Existing
exact GGUF range/shape/byte checks remain intact. First/last selected-byte
post-spool mutations now fail on the earlier live-source array-hash guard,
`temporary BF16 payload source identity mismatch`, before outer shard recheck;
the assertion requires that exact error. Unrelated shard changes remain covered
by the unit file-identity guards. Both optional **BF16/F32 controls pass in
5.02 s** using unchanged cached dependencies and deadlines. Required gate:
**651 passed, 89 optional skips in 25.66 s**, Ruff/whitespace/editor clean.

No new failures occurred in this increment. Synthetic source setup still
assembles its small shard in memory. No complete-model conversion, native
build/execution, learned inputs, downloads, refits, candidate or quality run.
All new files are automatically cleaned scratch; user editor settings remain
untouched. Next: compare bounded allocation peaks against the older copying
verifier before designing any larger-file conversion plan. Real artifacts and
larger resource budgets remain separately approved; historical failures stand.

## BF16 Mapped Source Checkpoint (2026-10-16)

Resumed autonomously after user-pushed `c03ef1e`, with only untouched user
editor settings untracked. Added `mapped_profile_safetensors_bf16_matrix`
beside the unchanged copy-based source verifier and payload owners. It opens
one no-follow/nonblocking regular descriptor, checks the profile's whole-shard
and header pins, parses the selected BF16 matrix geometry/range, and hashes
the raw selected bytes through at-most **64 KiB** positional reads. Its
read-only NumPy mapping is constructed from that retained descriptor, not a
reopened pathname. Whole-file hash and device/inode/size/mtime/ctime checks
run before yield and after successful caller completion. The mapping and
descriptor close on success, body exception or failed verification.

Source file and selected operand caps remain **16 MiB** each; unlike the older
verifier, the selected extent is not multiplied by three because no full
operand copy/comparison string is made. One cooperative **30 s default/600 s
max** deadline covers hashing, caller lifetime and exit checks. The provisional
immutable receipt requires successful exit before publication. Finish every
borrowed view/chunk iterator before exit; closed mapping aliases must not be
used. This is not immutable storage or protection from concurrent truncation,
which can invalidate live mappings. Finiteness remains a chunk-consumer check,
not a claim made by the raw mapping owner.

Twenty-two focused mapped-source cases cover normal/chunked reads, pins/budgets,
unsupported operands, symlink/FIFO refusal, pre-yield and lifetime writes,
restored bytes/mtime, identical replacement, deletion, deadline/body errors,
cwd changes and descriptor/mapping cleanup. Every positional read is bounded
to 64 KiB. A warmed **2 MiB** selected matrix retains **less than 1 MiB** of
additional traced Python allocations while owned; no source-sized owned array
exists. This is not a peak-allocation, mapped RSS, OS cache or whole-converter
RAM measurement, and excludes caller-owned inputs/profile validation costs.

The first control failed before opening the mapping on an unbound `math`
reference (**44 passed, one failed in 2.17 s**); direct two-dimensional extent
calculation fixed the same **45 cases in 2.44 s**. Tightened failure oracles
later exposed an incorrect deletion expectation (**66 passed, one failed in
3.71 s**): unlink changes descriptor ctime, so the existing earlier identity
guard refuses before missing-path lookup. Correcting only that oracle gives
**67 passes in 3.75 s**. Required gate: **629 passed, 89 optional skips in
13.87 s**, Ruff/whitespace/editor clean. No native/converter/model run or
download was used; previous failed real-model guards remain unchanged.

Next: compose this source owner with the existing chunked raw payload and
exercise it in the bounded pinned GGUF writer before any larger file budget
or separately approved complete-model conversion.

## BF16 Source-Bound Writer Checkpoint (2026-10-16)

After `75a9205`, the existing pinned converter control writes all **16 BF16
matrices** through the source-bound payload owner into
`GGUFWriter(use_temp_file=True)`. A physical synthetic safetensors shard,
at most **16 MiB**, contains the selected nonzero matrix/vector/convolution
operands, explicit zero unselected norm vectors and the distinct zero original
frozen source. Only the fixture's shard/header/profile/manifest identity fields
are updated; frozen codes/scales and cached profiles remain unchanged. The
source-bound writer reads each matrix through a read-only selected-range mmap,
checks receipt range/file/profile pins, closes payload mappings before exit,
then closes source mappings. Existing exact converted GGUF range/shape/byte
and corruption checks are retained.

Two additional controls mutate the source after the first/last selected matrix
is actually spooled. Each refuses on the final source-file hash, removes the
private raw payload/directory, leaves no published GGUF bytes, closes the writer
and temporary stream, and restores only the disposable synthetic source.

The initial fixture run failed in **2.02 s**. Local inspection found the shard
builder assumed all inventory vectors were among the seven selected-vector
operands; three unselected norm vectors were absent. Supplying explicit zeros
only for missing BF16 vectors, while refusing missing matrices/convolutions,
made the same test function pass. The original terminal capture did not retain
the child stderr; no exact exception diagnosis is claimed. After adding the
post-spool refusals, both optional **BF16/F32 controls pass in 5.52 s** using
the existing pinned Prism source and cached dense interpreter. Required source
gate: **607 passed, 89 optional skips in 19.91 s**, Ruff/whitespace/editor clean.

This is bounded synthetic source-to-payload-to-writer evidence, not complete
model conversion, native execution, quality or measured RAM reduction. The
fixture still assembles its small source shard in memory; source verification
still copies selected operands. All new shard/GGUF files live in automatic
temporary directories; no downloads, learned inputs, retained candidate,
runtime builds or full-model experiments were used. Scoped docs checks pass.
Next: a genuinely low-memory verified source-range reader requires its own
bounded synthetic implementation and allocation/lifetime checks; larger source
budgets and a complete-model conversion remain separately scoped/approved.

## BF16 Source-Bound Payload Checkpoint (2026-10-16)

After `fc54e54`, added `temporary_profile_safetensors_bf16_payload`, composing
the existing selected safetensors operand verifier with the disposable raw
BF16 matrix payload owner. Before creating a payload and after normal caller
completion/cleanup, the whole source shard, pinned header and selected range
must match the profile and caller operand. Both source receipts must also
match exactly, including device/inode/size/mtime/ctime, so restored bytes/mtime
or an identical replacement are not accepted as unchanged storage.

The source path is made absolute without resolving symlinks; existing no-follow,
nonblocking regular-file checks remain intact. Source-access `OSError`s become
`ValueError` at the new owner boundary only. One cooperative **30-second
default/600-second max** deadline covers both source checks, payload production,
caller lifetime and cleanup checks. Source/payload byte caps remain **16 MiB**;
the reused source verifier still charges **three times the operand bytes**.
It copies the operand and materializes comparison bytes; this composition is
not the lower-memory full-shard reader needed by a complete-model converter.

The immutable receipt nests the entry source receipt and explicitly marks
entry/exit verification as required, not already complete. Publish derived
results only after successful owner closure, with payload readers/mappings
finished before exit. Consumer exceptions and all checked failures clean the
private payload/directory without deleting the caller's source file. No atomic
or continuous storage snapshot, pin-issuer identity, runtime approval, converter
RSS reduction or complete-model acceptance follows.

The existing bounded zero-shard fixture now checks matrix entry refusals through
the new owner, successful bytes/immutable receipts, changed/restored/replaced/
deleted source lifetime refusals, cooperative exit deadline and consumer-error
cleanup. The first run had **24 passes and one failure in 1.71 s**: deletion
correctly refused and cleaned but leaked `FileNotFoundError`. Normalizing source
access errors at the new boundary fixed the same **25 cases in 1.47 s**, with
unchanged refusal assertions. Expanded entry/lifetime coverage gives **45 passes
in 2.41 s**. Required source gate: **607 passed, 89 optional skips in 15.12 s**;
Ruff, whitespace and source/test editor diagnostics pass. No new native build,
optional pinned converter run, learned weights, download, conversion, frozen
candidate change or quality run was used in this increment. The prior pinned
16-matrix GGUF writer evidence belongs to the earlier array-pinned owner, not
this new source-bound composition.

Next: integrate the source-bound owner into the existing bounded pinned GGUF
fixture before considering a whole-model writer. Full-shard low-memory source
access and real conversion/artifact/resource approval remain separate work.

## BF16 Payload Owner Checkpoint (2026-10-16)

After `1ca8ac2`, added `temporary_profile_bf16_matrix_payload` beside the
unchanged model-policy owners. It consumes the chunk primitive into an
exclusive raw payload inside a private temporary directory. Total payload
size is at most **16 MiB**; the chunk cap remains **64 KiB default/1 MiB max**.
One cooperative **30-second default/600-second max** deadline covers source
hashing, chunk production/writes, payload verification and the caller lifetime.
It is not a hard interruption of I/O, profile validation or caller code.

The caller's source-array SHA-256 must match before creation, after writing
and at successful exit. The payload's device/inode/size/mtime/ctime and bounded
SHA-256 reads must match before yielding and at exit. An immutable receipt
explicitly requires entry/exit checks and leaves source-file verification
`not_checked`; publish derived results only after successful owner closure.
Private payload/directory cleanup runs on success, partial-write failure,
consumer exception or failed exit verification. Existing caller files remain
untouched. All mappings/readers using the payload must finish before exit.

Ten existing equal/unequal-head mapped-source workflows pass exact raw bytes,
receipt hashes and cleanup in **0.50 s**. Eighteen additional lifecycle cases
exercise malformed pins/budgets, late nonfinite values, injected mid-stream
failure, changed source, changed/restored/replaced/deleted payload, caller
failure and exit deadline. The first run had **one missing local `os` import**
in the restored-byte fixture (**27 passed, one failed in 0.87 s**); adding that
import, without changing checks, gives **28 passes in 0.83 s**.

The existing pinned upstream converter control now feeds all **16 BF16
matrices** to `GGUFWriter(use_temp_file=True)` through these disposable raw
payloads/read-only mappings. It asserts the writer retains no tensor array,
closes each mapping before payload-owner exit, and explicitly closes the
writer's temporary stream on failure. Existing exact GGUF range, shape,
operand and corruption checks pass in **3.15 s**. Required source gate:
**587 tests passed, 89 optional skips in 16.03 s**, Ruff/whitespace clean;
the optional compiled/native controls did not run implicitly. Editor and
scoped docs validation pass.

External `bf16-matrix-payload-owner-evidence.json` has reread SHA-256
`cc069a0a8c8ab0907d423f05b8b26ba0d22d05fac532e9ae607c85438a0f5d07`.
Pinned Prism source revision/tracked files are unchanged/clean. All new raw
payloads use automatic temporary cleanup; no learned weights, frozen-candidate
changes, download, complete-model conversion, runtime build or quality run.
This is a bounded matrix-payload consumer and synthetic GGUF integration,
not a complete converter, source-shard authentication, atomic/continuous
immutability guarantee, measured converter-RAM reduction or real-model parity.
Next: bind the borrowed source to verified safetensors ranges before considering
a whole-model writer; real artifacts/experiments remain separately approved.

## BF16 Chunk Primitive Checkpoint (2026-10-16)

Resumed after user-pushed `1b9c2f6`; only untouched editor settings were
untracked. The first bounded lower-memory conversion prerequisite now exists:
`iter_profile_bf16_matrix_chunks` uses the existing profile conversion plan
to emit owned, contiguous, read-only little-endian uint16 BF16 row chunks.
It implements identity and the declared grouped-to-tiled V row/head/column
permutations through bounded index maps, without whole-matrix F32 expansion
or optional Torch/GGUF imports. Frozen PQ2 replacements, vectors/convolutions,
wrong shapes/dtypes/endian/layout, invalid budgets and nonfinite chunks refuse.
The output cap defaults to **64 KiB**, at most **1 MiB**; a row must fit.

An optional pinned Torch-to-GGUF probe tests all **65,280 finite BF16 encodings**
(including signed zeros and subnormals) and passes exact bytes in **1.37 s**.
This establishes finite raw-BF16 roundtrip equivalence, not nonfinite support.
Existing literal-index identity/permutation controls validate chunk boundaries
and ownership; focused **21 cases pass in 0.55 s**, then **37 cases in 0.79 s**
with added refusal/borrowed-source checks. The existing pinned converter control
now checks all **16 BF16 matrices** through the iterator against upstream Torch
transforms, initially passing in **2.98 s**.

A subsequent read-only mmap boundary check exposed **four ownership failures**
in unequal-head QKV/Z/alpha/beta row gathers (**six cases passed, 0.64 s**);
all numerical bytes matched. Normalizing the borrowed source with zero-copy
`np.asarray` before gathering removes subclass propagation. The same unchanged
ten ownership/oracle cases then pass in **0.47 s**. No ownership assertion,
numerical criterion or bound was relaxed. The original failure log is retained.
Final-source pinned matrix control passes in **2.88 s**; the required gate
passes **569 tests, 89 optional skips in 16.83 s**, Ruff/whitespace clean.
The earlier pre-mmap gate (**569/89, 14.00 s**) is historical, not substituted
for this final-source gate. Editor diagnostics and scoped docs checks pass.

Source arrays/mappings remain borrowed: chunks snapshot only their own rows,
later writes can affect later chunks, and a late nonfinite chunk fails after
earlier chunks were yielded. Consumers must discard partial output on failure
and perform separate source/file/whole-artifact verification before publication.
The chunk cap does not bound caller source storage, retained output chunks,
index/mask temporaries, profile/projection validation or process peak RAM.
No whole-converter RSS improvement is measured or claimed.

External `bf16-matrix-chunk-final-evidence.json` has reread SHA-256
`be749a48c6001b728ed28b72d90b4ede17f9bf95a77c7268ad3755e822f2f7f3`;
it preserves the earlier report hash, final source identities and copied,
hash-verified mmap failure log outside Git. The pinned converter checkout is
unchanged/clean. Synthetic mmap/GGUF fixtures use normal bounded temporary
cleanup; no learned weights, frozen artifact, full conversion, download,
native runtime build, refit, candidate, quality or other-model execution ran.
Next: a bounded disposable payload consumer and verified source-range lifetime
before integration into a full writer. Another real artifact/experiment still
requires separate approval and a materially changed memory plan.

## Native Loader Isolation Checkpoint (2026-10-16)

After `443e1f1`, added `interleaved-singletons` to the existing bounded tiny
actual-loader control and NumPy-backed test. Two native contexts share one
loaded model. The peer first consumes 4, 9, 19 in one batch; both contexts
then consume 80 singleton calls each, alternating primary/peer deterministically.
The primary stream remains repeating 3, 5, 7 and ending 3, 5. The complete
83-token peer stream is 26 repetitions of 4, 9, 19 followed by 4, 9, 7, 3, 5.
Thus both end with **7, 3, 5**, but have different histories. Checked native
hybrid position intersections are **[79,79]** and **[82,82]**, respectively.

Independent NumPy references check both contexts' selected logits at unchanged
**rtol/atol 2e-5**; primary typed conditional scores and zero generated answer
tokens remain checked. Active recurrent and/or attention references must differ
at the existing **rtol/atol 1e-4** history discriminator despite identical final
three tokens. BitNet paths require **161 genuine counted dispatches**, one
shared weight repack and one final input row. No host cache restore or second
model load occurs. The existing concurrent `parallel` batch control remains
separate and passes; alternation is not a new production concurrency guarantee.

Immediate focused recurrent-plus-attention profile-policy check passes in
**1.33 s**. All **12** backend/cache combinations pass in **12.60 s**, adding
12 long interleaved workflows, two reference streams per case and nine active
history discriminators. The required source gate passes **553 tests, 88 optional
skips in 17.66 s**, Ruff and whitespace clean. Optional compiled checks ran
separately; source/test editor diagnostics and scoped documentation checks pass.
No failed arithmetic/native check, relaxed tolerance or expanded deadline.

Only the control executable was rebuilt. Six runtime library hashes and both
pinned upstream revisions/tracked-file cleanness are reverified unchanged.
External `tiny-native-singleton-isolation-evidence.json` retains source,
executable and runtime hashes, stream descriptions and actual results, reread
SHA-256 `a6dd44978d593f03681358bac5d3031a1ebe99d825001c0391e8464bad0414aa`.
Existing temporary fixture cleanup, geometry/token bounds and 30-second
per-control deadlines are unchanged. No learned weights, frozen-artifact
changes, conversion, download, candidate/refit, quality or other-model work.
This proves bounded synthetic context isolation with shared repacked weights,
not simultaneous loading/freeing, registry mutation safety, speedup, learned
full-loader parity, whole-model BitNet, representative quality or device support.
The selected synthetic continuation/reset/isolation slice is complete; next
learned full-runtime work requires its separate artifact/resource approval.

Read-only next-step preparation after this check: host readings were
**17,032,589,312 bytes available RAM**, **988,655,616 bytes swap used** and
**544,872,685,568 bytes free scratch disk**. These are time-sensitive host
observations, not target guarantees or evidence that the earlier swap refusal
would now pass. The pinned converter's entry point already defaults to lazy
evaluation, and the refused command already supplied `--use-temp-file`;
`--no-lazy` explicitly increases RAM. No simple untried memory-saving CLI flag
was identified. Its `conversion/base.py` preparation promotes BF16 to F32
before tensor transforms and GGUF quantization. A 248,320 by 4,096 embedding or
head has a **4,068,474,880-byte F32 representation** before packing overhead;
this is source-backed allocation-size reasoning, not measured failure attribution.

Next proposed prerequisite is a synthetic-first, chunk-bounded conversion path
that demonstrates byte-identical BF16 output and unchanged model-specific
transforms without mutating pinned upstream sources. It is not implemented or
approved for a real artifact here. Another full conversion must have separate
approval and a materially changed memory plan or verified host conditions;
do not repeat the old command merely because initial preflight passes. Any
future approved comparison should reuse authenticated caches/references, run
serially with the existing 1,200-second conversion/900-second native limits,
32 GiB address-space/48 GiB scratch caps, 2 GiB host reserve and 512 MiB
swap-growth stop, and retain only small verified reports before deleting
temporary model/spill/raw traces. Native prefix comparison and whole-model
parity/BitNet/quality acceptance remain distinct gates.

## Native Loader Reset Checkpoint (2026-10-16)

Resumed after user-pushed `fb2852c`, with only untouched user editor settings
untracked. Extended the existing bounded tiny-loader control and its NumPy
reference test with `reset-singletons`: 80 singleton warmup calls on repeating
4, 9, 19 (ending 4, 9), a checked native hybrid position intersection of
**[79,79]**, `llama_memory_clear(memory, true)` with both positions **-1**,
then three singleton calls on 3, 5, 7 through the **same context and model**.
Final native position is **[2,2]**, and selected logits/conditional scores
must reproduce the independent fresh three-token reference at unchanged
**rtol/atol 2e-5**. Typed options and zero generated answer tokens remain checked.

BitNet paths require **83 actual counted dispatches**, one weight repack and
one final input row. The independent oracle also evaluates the uncleared
83-token stream; every nonzero recurrent and/or attention case must differ
from the fresh reference at the same tolerance. Thus cache contamination is
numerically distinguishable, not merely assumed absent from position reset.
All-zero cache fixtures remain supported but do not claim this discriminator.
The old short-batch reset now also checks its pre-clear position **[2,2]**.

Immediate focused recurrent-plus-attention profile-policy check passes in
**1.56 s**. All **12** existing backend/cache combinations pass in **12.36 s**,
adding 12 positive long-reset workflows and nine active-cache fault-oracle
discriminators. Existing continuation, state-loss refusal, parallel, trace,
kernel-error/recovery and malformed-input checks remain included. The required
source gate passes **553 tests, 88 optional skips in 15.65 s**, Ruff and
whitespace clean; optional compiled checks ran separately. Editor diagnostics
and scoped documentation validation pass. No failed check or tolerance change.

Only the loader-control executable was rebuilt. Six cached runtime library
hashes and both pinned source revisions/tracked-file cleanness are reverified.
Small external `tiny-native-singleton-reset-evidence.json` records the source,
executable, runtime pins and results, reread SHA-256
`4e7c4dd5fb1d7fea4378a6cd811fbf3c7ad0437ff34249560a917cb061ac429f`.
Synthetic models use the existing bounded temporary fixture/finally cleanup;
the 30-second subprocess deadlines and native geometry bounds are unchanged.
No learned weights, frozen artifact, conversion, download, refit, candidate,
held-out scores or other-model execution were used. This is tiny native
context reset/reuse, not production concurrency, learned full-loader parity,
whole-model BitNet, P4 acceptance, representative quality or device evidence.
Next bounded source work may test singleton context isolation; a learned
full-runtime comparison still needs separate artifact/resource approval.

## Native Loader Continuation Checkpoint (2026-10-16)

Resumed after user-pushed `6294380`; only untouched user editor settings were
untracked. The learned detached continuation slice was complete, but its
synchronous diagnostic stack did not itself test repeated calls through the
actual complete-model loader. Extended the existing small C++ loader control
and its existing NumPy-backed test, without changing runtime libraries,
production v1/v2 policy factories or model approval boundaries.

The existing synthetic four-layer fixture has hidden width 32, FFN width 256
and vocabulary 64, with independently zero/nonzero layer-0 recurrent and
layer-1 attention paths. Its selected layer-3 FFN-down uses synthetic ternary
codes and independent per-output-row/input-group FP16 scales. Three new
control modes reuse **one actual native model/context/memory** per invocation:

- `singletons`: three one-token calls for the original 3, 5, 7 token fixture.
- `singletons-80`: 80 one-token calls on 26 repetitions of 3, 5, 7 plus 3, 5.
- `prefill64-singletons`: the same 80 tokens, split into one 64-token call
  followed by 16 one-token calls.

The unchanged independent NumPy recurrence/attention/FFN/output reference checks
selected logits and conditional option scores at **rtol/atol 2e-5**. Tokens,
typed option IDs/labels, final hybrid native-memory position and zero generated
answer tokens remain checked. BitNet paths require **3/80/17 genuine counted
dispatches**, one weight repack per model load, and one input token on the last
call. Native hybrid position intersection is **[79,79]** for both long schedules.
This does not assume bitwise batch invariance or calibrated confidence.

Deliberate `singletons-reset` fault clears native memory immediately before
the last token. The existing final-position guard returns **31** before JSON
publication; all backends must have empty stdout. It is not a supported
continuation/reset workflow. Existing full/chunked/reset/reordered/parallel/
maximum/kernel-recovery/trace, corruption and CLI-refusal checks still run.

Immediate focused compiled checks pass after each local addition: three-token
singleton **1.37 s**, long schedules **1.40 s**, reset fault **1.32 s**.
Final full existing matrix passes **12 cases in 11.90 s**: dense, BitNet v1
and profile-policy backends times independently zero/nonzero recurrent and
attention paths. This adds **36 positive schedule/backend/path checks and 12
state-loss refusals** to that matrix. No numerical tolerance, assertion or
30-second per-control deadline was relaxed. No failed arithmetic or native
check was retried for green.

Final diff review corrected only one over-indented C++ guard line. Reversing
that whitespace change reproduces the exact source hash used by the full
compiled matrix. The final control rebuild and focused recurrent-plus-attention
profile-policy case pass in **1.35 s**; the unchanged full matrix was not rerun.

The loader-control target alone was compiled/linked in the existing owned
runtime build. Six cached runtime library hashes are reverified unchanged;
both pinned upstream source revisions and tracked files are unchanged/clean.
The original frozen artifact was not read/refit/replaced. Fixtures are bounded
temporary synthetic GGUFs, removed by the existing finally block; trace fixtures
remain under pytest's normal temporary-directory lifecycle. No learned weights,
model conversion, source download, retained candidate, compensation, held-out
scoring, broader quality or other-model action was started.

The initial source gate passed **553 tests, 88 optional skips in 10.84 s**.
The required final-source checkpoint gate passes **553 tests, 88 optional skips
in 12.02 s**, Ruff and whitespace clean. The 12 compiled controls above ran
separately; default skips are not counted as compiled success. Source/test
editor diagnostics are clean. Original external evidence
`tiny-native-singleton-continuation-evidence.json` remains unchanged, SHA-256
`76c9ee0350428504e8f73f57e19504337f80907a502c1e0a573d58f71f7064b4`.
Separate `tiny-native-singleton-continuation-final-evidence.json` records the
final source/executable hashes, whitespace-only reconstruction, focused result
and final gate, reread SHA-256
`8094a0d1f3919027067a0ee30e44799f5db23978e01abf3dcafd43fc2cedb4d0`.
The first final-evidence command was rejected by shell history expansion before
Python ran; a quoted heredoc corrected the command without changing the check.
Docs/link/editor/whitespace validation accompanies this source checkpoint.

This adds bounded **actual tiny-loader** continuation and genuine synthetic
BitNet dispatch coverage. It is not learned full-loader parity, complete-model
BitNet execution on MiMo, representative quality, another-profile compatibility
or target-device certification. Historical direct-source/conversion failures,
fixed attention-score failure and full-model selected-logit gap remain open.
Next full-runtime comparison still requires its artifact/resource plan and
separate conversion/broader-experiment approval; do not repeat the old guarded
conversion or infer its result from these synthetic controls.

## Detached Singleton Checkpoint (2026-10-16)

After `151e4c5`, the remaining cached-MiMo **all-80-singleton** continuation
through the tested detached execution handoff passes in **171.24 s**,
**3,153,772 KiB** peak child RSS, minimum host-available RAM
**14,492,889,088 bytes**, zero host swap growth/process swaps and observed
filesystem growth **151,552 bytes**, under unchanged 300-second/resource guards.
The existing independent same-schedule reference is reused, not recomputed.

The snapshot has **527 files, 3,719,720 bytes**, no weight/library copies.
Every copied input identity remains exact after the child. All ten watched
source/proof identities remain exact, zero mutation events occur, and the
buffered numerical report publishes only after those checks. Retained report
hashes and the embedded observation hash verify; the temporary execution
directory is confirmed absent.

The **entire same-schema four-layer numerical dictionary** exactly matches
the retained native-state all-singleton result; the in-child independent
reference/stage/cache checks pass too. Six history and six GDN state owners
all close, with zero host history/state restores. Host snapshots remain
oracle-only; host F32 residual/down sums are unchanged. All 11 original F32
reference parameter objects are released during replay; only the verified
down weight is restored afterward. Observed release RSS
**2,978,881,536 -> 2,173,558,784 bytes** is not a device-memory guarantee.

Retained/reread report SHA-256:

| Report | SHA-256 |
| --- | --- |
| `execution-snapshot-singletons-event-audited-lifetime-history-resident-singletons-comparison.json` | `36999668c43eaa3482660ca6bbe6ffe9fc3338ba6835bf8546ab89eb10ae6539` |
| `execution-snapshot-singletons-verification.json` | `1dcabd9805730b16f863e1e49f55d8ef2f3a5dc412ad96e055adad9a1b050d10` |
| `detached-execution-final-evidence.json` | `7048f63d079120c95cef52031b4255a205dc451aca07f898c34f3e627202c01f` |

Final post-run audit verifies six runtime/three frozen hashes, nine original
helper hashes, profile/metadata, original authenticated shard identities
(not another whole-weight rehash), installed model-code hash and both pinned
upstream revisions/clean tracked sources. All retained reports from native,
64+16 and all-singleton runs are reread/hash-verified. All three owned
execution trees are absent; no owned child/raw replay packet/spill file remains.
Scratch totals **4,080,698 bytes**, only small diagnostic code/evidence/tools.
No cached library/source rebuild, model conversion/download/refit/new candidate,
compensation, held-out scoring, quality or other-model action was started.

**Completed scope:** selected bounded dense four-layer continuation with
native history, native GDN state and native attention cache across the prior
coarse schedules plus 64+16 and all-80-singleton schedules. This is synchronous
scratch integration, not a production loader, complete-model BitNet execution,
v2 P4 acceptance, representative quality or edge-device certification. Preserve
all old failed direct-run/conversion evidence; the original fixed-score failure
and full-model selected-logit gap remain unresolved. The old editor-attached
execution path still has an unidentified writer; the verified detached
execution handoff is the tested alternative, not proof that the writer stopped.

Next full-runtime comparison needs an explicit artifact/resource plan; another
complete-model conversion or broader experiment remains separately approval-
gated. Do not repeat the old guarded conversion or manufacture full-model
evidence from this slice. No further unchanged successful gate is rerun:
`b52e20b` default source gate remains **553 passed, 88 optional skips**; only
scoped docs/link/editor/whitespace checks apply to these source-tree updates.

## Detached Decode Checkpoint (2026-10-16)

After `17e3514`, one cached-MiMo **64-prefill / 16-singleton** diagnostic
through the tested detached execution handoff passes in **85.85 s**,
**3,152,896 KiB** peak child RSS, minimum host-available RAM
**13,928,288,256 bytes**, host swap growth **87,728,128 bytes**, zero process
swaps and observed filesystem growth **2,412,544 bytes**. The unchanged
300-second/resource guards permit completion; host-global swap growth is not
attributed to this process or treated as a device-performance measurement.

The snapshot contains **511 files, 3,358,711 bytes**, no model weights/libraries.
All copied input identities remain exact after execution. All ten watched
source/proof identities remain exact, zero mutation events occur, and the
buffered report publishes only after those guards pass. Copied-back reports
are reread/hash-verified; their observation hash matches the numerical report.
The temporary execution tree is confirmed absent.

The **entire same-schema four-layer numerical dictionary** exactly matches
the retained native-state 64+16 result. Fresh in-child same-schedule reference
and native-owner checks also pass. All **six history and six GDN state owners**
close, with zero host history/state restores. Three native history/state pairs
and native attention cache persist per stack; host readbacks remain oracles,
and residual/down sums remain host F32. The lifetime guard verifies/releases
all 11 original F32 parameter objects (**838,895,616 bytes**) and restores only
the **201,326,592-byte** down weight after replay. Observed reference-release
RSS **2,977,492,992 -> 2,172,170,240 bytes** is not a device-memory guarantee.

Retained/reread report SHA-256:

| Report | SHA-256 |
| --- | --- |
| `execution-snapshot-decode-event-audited-lifetime-history-resident-decode-comparison.json` | `7f2e1c3892b634b15584d54ec7e63e7edeabb2720109434704a73cd825d19a3d` |
| `execution-snapshot-decode-verification.json` | `49ecc7194a5477005230f2202ec28affee84dc3fd62439d74abe0415980086f2` |

This is accepted **bounded dense four-layer native-history/GDN/attention
64+16 ownership evidence**, not full-model/BitNet/v2 P4/quality/device acceptance.
Neither old direct-run refusal is reinterpreted; the original fixed-score
failure/full-model discrepancy remains. Next: detached all-80-singleton
continuation against the retained independent same-schedule reference, without
recomputing that reference or changing guards, tolerances or frozen artifacts.
No conversion/download/refit/new candidate/quality/other-model action. Default
source gate remains unchanged `b52e20b` (**553 passed, 88 optional skips**),
not rerun for docs-only updates; scoped docs validation applies.

## Detached Execution Checkpoint (2026-10-16)

Resumed after user-pushed `b599678`; all ten current source/proof identities
still match the last refusal endpoint. The user could not inspect the editor
buffers. A bounded read-only inspection of installed edit/save code did not
establish a supported document-specific flush operation; no extension internals
were invoked. One disposable text-fixture observation initially failed its
byte-exact precondition because the fixture lacked the required newline. After
repairing only that fixture, the same observation completes in **52.14 s**,
**19,000 KiB**, minimum host-available RAM **15,832,399,872 bytes**, zero swap
growth/process swaps, with no watched mutation of the fixture or the two prior
observer scripts. The single filtered wakeup contains no watched events or
observable holder. This neither reproduces nor identifies the historical
writer; the failed setup log remains.

Added separate scratch `execution_snapshot.py` as an alternative to depending
on editor-buffer access. It copies only the small existing diagnostic tree
into a disposable sibling directory, with **1 MiB per file, 8 MiB aggregate,
1,024-file** bounds and a strict diagnostic/ELF-smoke-executable allowlist.
Weight/raw-data suffixes, symlinks and nonregular files refuse. No model
weights, quantized arrays, shared libraries or upstream checkouts are copied.
Existing weights/libraries stay at their original pinned cache locations.

Capture original full identities/hashes, verify every copy's hash/bytes and
recheck origins through the handoff. Record both identity tables in a small
external manifest. After handoff, the **copies** are the executed source;
original editor buffers are not execution inputs. This deliberately establishes
a new provenance boundary, not a claim that original paths stayed unchanged
after handoff. Copies are ordinary files, not immutable storage or a security
boundary. The unchanged inner helper/proof and file-event guards still apply
to actual execution files, and every copied input identity is rechecked after
the child. Model/profile/runtime checks retain their original external inputs.

Only prior output filenames belonging to the selected diagnostic mode are
excluded to avoid exclusive-write collisions; original failed reports stay
untouched. After execution, copy only bounded new JSON/stdout/stderr/time
reports back under a distinct prefix and reread/hash-verify them. Parent-finally
raw-packet cleanup precedes removal of the entire owned execution tree. No
source code, numerical tolerance, guard, user setting, permission or clock is
changed inside this handoff.

Tiny snapshot self-test passes in **0.116 s**, **25,364 KiB**, zero swap
growth/process swaps: exact copied bytes/distinct inodes, later origin-touch
isolation, changed-origin-during-copy refusal, per-file/aggregate/type/symlink
refusals and success/failure cleanup. Scoped Ruff passes. Scratch Python is
outside the editor workspace; no Pylance diagnostic-clean claim is made.

The first **detached native synthetic** control passes in **36.58 s**,
**1,065,472 KiB** peak child RSS, minimum host-available RAM
**14,994,997,248 bytes**, zero host swap growth/process swaps and observed
filesystem growth **167,936 bytes**, under unchanged 60-second/resource guards.
The snapshot has **498 files, 3,065,002 bytes**. All copied input identities
remain exact; the eight-file event audit reports zero mutations. Its **entire**
numerical dictionary matches the retained native-history synthetic control.
All retained report hashes verify and the temporary execution directory is
confirmed absent. No learned model ran in this checkpoint.

Retained/reread report SHA-256:

| Report | SHA-256 |
| --- | --- |
| `execution-snapshot-selftest-result.json` | `e452f58c971ac710b9659e252eba6836901d7a2c46c925d0f2f0d6a499958775` |
| `execution-snapshot-native-manifest.json` | `11e6d058131850bb70b323a45eec30517ba957bb44d0bce0026d5c096d62fb29` |
| `execution-snapshot-native-verification.json` | `a69db2ef7139d2339e73627befe94d960863518f5430f609e0e96b0b510e3ae2` |

Next: one separately observed **64+16** cached-MiMo diagnostic through this
tested detached execution handoff, with unchanged 300-second/resource guards,
reference/pin/event checks and buffered publication. This is a material source
ownership change, not an unchanged retry or acceptance of either old failed
run. Preserve those failed verdicts and do not recompute the retained independent
reference. The editor-side prerequisite remains for the old direct-edit path;
the alternative path must establish its own learned evidence. Unchanged
successful `b52e20b` source gate (**553 passed, 88 optional skips**) is not
rerun for this docs-only source-tree checkpoint. Original conversion,
fixed-score/full-model/BitNet/v2 P4/quality/device limits are unchanged.

## File Event Checkpoint (2026-10-16)

Resumed after user-pushed `a455312` with only untouched editor settings
untracked. A before/after ordinary editor-read probe leaves all five watched
identities/hashes unchanged. No `inotifywait` executable is installed; no
software, privilege, host or editor-setting changes were made.

Added scratch `pin_events.py`, using typed libc inotify bindings with a
nonblocking/close-on-exec descriptor and both parent-directory/file watches.
Synthetic fixtures verify reads produce no mutation events, while touches,
writes, same-byte replacement and deletion are detected. Queue overflow,
truncated packets and closed reads refuse; descriptor cleanup is exact.
Completed source-matched proof passes in **0.066 s**, **19,040 KiB**, zero
swap growth/process swaps. The earlier proof and explicit stale-source-proof
refusal remain; source extensions use a separately named proof, never overwrite
old evidence. Scoped Ruff passes. Inotify does **not** identify a writer PID;
file/directory watches can produce duplicate notifications.

New **watched synthetic native** control passes in **37.38 s**, **1,064,524 KiB**
peak child RSS, minimum host-available RAM **16,115,978,240 bytes**, zero host
swap growth/process swaps under unchanged 60-second/resource guards. All eight
watched helper/proof identities remain exact, with zero mutation events.
Its entire synthetic native-history result matches the retained control
exactly. No learned weights are loaded. This new observation window does not
prove longer-run stability or identify the historical writer.

Added separate `observed_lifetime.py` around the unchanged pin-audited lifetime
adapter. It watches ten source/proof files, records all initial/final identities
and mutation events, and **buffers** the numerical report until both full
identity equality and event checks pass. Six synthetic cases across two event
observers cover success, metadata touch, content write, replacement, deletion
and injected compute failure. All required refusal/observation/publication
checks pass in **3.17 s**, **326,788 KiB**, zero swap growth/process swaps,
under unchanged synthetic guards. Original helper files/proofs, native
arithmetic, resource guards and numerical tolerances are unchanged.

One fresh event-observed cached-MiMo **64+16** diagnostic returns **1** after
**85.02 s**, **no resource refusal**, **3,154,012 KiB** peak child RSS, minimum
host-available RAM **14,358,519,808 bytes**, zero host swap growth/process swaps
and observed filesystem growth **1,777,664 bytes**, under unchanged 300-second
limits. The inner replay/reference/owner/original-pin checks complete and the
report reaches the publication buffer. The outer observation check refuses:

- Original eight pinned helper/proof identities remain unchanged.
- Newly edited `pin_events.py` and `observed_lifetime.py` retain their device,
  inode, size and endpoint SHA-256 values, but both mtime/ctime pairs change
  to the same observed timestamp during the run.
- Each script has four modify and two close-write **notifications**, including
  possible duplicate file/directory notifications, not four distinct writes.
- This proves writes occurred; endpoint hashes do not exclude transient
  content changes. No accepted numerical report is published or reconstructed.

This is materially stronger failure evidence than the old missing-entry-pin
case, not an accepted learned continuation. The earlier failed evidence remains
unchanged. No retry follows this event failure, and combined all-80-singleton
execution remains **not started**. A delayed editor save is a **hypothesis**,
not writer attribution: a post-event open-descriptor scan finds no current
holder, and no workspace autosave override was found. The available VS Code
command tool is restricted to workspace creation; no broad save-all command,
user-setting change, timestamp reset or privileged tracing was attempted.

Retained/reread report SHA-256:

| Report | SHA-256 |
| --- | --- |
| `pin-events-native-observation.json` | `fb36c9a44f4de2501a6b1e1280cf53f0e91cf54428bb52595929234ad3d04f19` |
| `observed-lifetime-selftest-result.json` | `8f5a7f661fd8657049f56987972695831abde145513b3b5ea5a7dcdc37104e6b` |
| `event-audited-lifetime-decode-learned-observation.json` | `ac2c9f7938c814b00a09c22fa89dcdf8177f3f66d28f131f4efb495d889a7cb9` |
| `event-audited-lifetime-rewrite-diagnosis.json` | `236bfd811db6651d86d8e943f58e010aebc443dd08c14d8e75bb4575add4baa0` |

Post-failure six runtime/three frozen hashes, nine original helper hashes,
profile/metadata, authenticated source shard identities, installed model code
and pinned Prism/BitNet revisions verify unchanged; Prism tracked files remain
clean. Scratch totals **3,053,422 bytes**, no owned children/raw packets/spill
files. All failed logs remain outside Git. No model conversion, download,
refit, candidate, compensation, quality or other-model action was started.

**Editor-side prerequisite:** save pending buffers for the two named scratch
scripts without changing their content, and confirm whether they were open or
autosaving when these writes occurred. Do not save unrelated user buffers or
change settings automatically. Then verify source/proof stability before a
separately justified learned attempt; do not retry the unchanged failed job or
remove the outer audit. Unchanged successful `b52e20b` source gate (**553 passed,
88 optional skips**) is not rerun for this docs-only source-tree checkpoint;
scoped docs/link/editor/whitespace checks apply. Original fixed-score failure,
full-model discrepancy and BitNet/v2 P4/quality/device limits remain open.

## Lifetime Pin Audit Checkpoint (2026-10-16)

After `9495903`, the materially changed reference-lifetime **64-prefill /
16-singleton** job ran under the unchanged 300-second/resource guards. It
returned **1** after **102.32 s**, with **no resource refusal**, peak child RSS
**3,153,836 KiB**, minimum host-available RAM **13,513,908,224 bytes**, host
swap growth **7,770,112 bytes**, observed filesystem growth **4,096 bytes**
and zero process swaps. This is not a retry of the old resource-stopped helper.

The traceback reaches final report publication after the existing numerical,
same-schedule reference and native-owner checks. It fails the lifetime
wrapper's **full source/proof identity equality assertion**, not an earlier
arithmetic assertion. The report is correctly withheld: no accepted decode
numerical verdict exists. The native-entry marker confirms all reference
parameters were meta, **838,895,616 parameter bytes** were removed and entry
RSS was **2,173,444,096 bytes**; none of those observations overrides the
failed provenance check. The original fixed attention-score check remains
false, as recorded in stdout.

Read-only diagnosis finds all four current source/proof SHA-256 values still
match the retained coarse result. The wrapper's current timestamps are
consistent with a touch during replay. **Entry identities were not retained**,
so the exact mismatched pin and its before/after identities cannot be recovered.
Matching current hashes do not prove there were no transient content changes,
or establish who changed metadata. No source touch was normalized, no identity
check weakened and no learned job rerun after this failure. Combined native
history all-80-singleton execution remains **not started**.

Added separate scratch `lifetime_pin_audit.py`; original helpers/proofs stay
unchanged. It records all four initial hashes and full device/inode/size/
mtime/ctime identities before replay, retains exact expected/actual values on
the first mismatch and still refuses publication. Future phase and numerical
reports have distinct names, avoiding existing evidence and supervisor files.
This is diagnostic preparation, not a repair or accepted learned result.

The initial tiny stable-file/metadata-touch/changed-byte test passes in
**3.52 s**, **326,604 KiB**, zero swap growth/process swaps. The extension
through the actual lifetime publication hooks initially failed because its
synthetic fixture used the old proof filename; that log remains. After the
single fixture-name repair, the same focused check passes in **3.37 s**,
**327,116 KiB**, minimum available RAM **16,876,089,344 bytes**, zero swap
growth/process swaps. All three synthetic cases verify publication/withholding,
retained identities and hook restoration, without loading weights or replaying
a learned model. Scoped Ruff passes. Scratch files are outside the editor
workspace; no Pylance diagnostic-clean claim is made.

Retained/reread report SHA-256:

| Report | SHA-256 |
| --- | --- |
| `lifetime-history-decode-pin-diagnosis.json` | `ad5798752cac83be61b144a79764126becc87aa08f4ec795246df83155aea3e3` |
| `lifetime-pin-audit-plumbing-selftest-result.json` | `facd83000163839525849ce1114ec96bb5ee129c801075b87289643db98be597` |

Post-failure verification confirms all six runtime and three frozen hashes,
original nine helper hashes, profile/metadata, original authenticated shard
identities (not another complete shard rehash), installed model-code hash and
pinned Prism/BitNet revisions. Prism tracked files remain clean. Scratch totals
**3,001,008 bytes**; owned children, raw replay packets and spill files are
absent, and failed logs remain. No cached library/source rebuild or model
conversion/download/refit/candidate/compensation/quality/other-model action.

Next: identify/prevent writes to watched scratch sources/proofs before another
learned attempt; retain initial identities with the tested audit and preserve
the original guards/reference. Do not rerun the failed job unchanged or treat
this as evidence that the guard was spurious. Docs/link/editor/whitespace
validation accompanies this docs-only source-tree checkpoint; successful
`b52e20b` source gate (**553 passed, 88 optional skips**) is unchanged and not
rerun. Full-model/BitNet/v2 P4/device acceptance remains open.

## Reference Lifetime Checkpoint (2026-10-16)

After user-pushed `ab484a8`, no unchanged guarded job was retried. Nearby
inspection found the complete attention-layer helper retains its entire F32
reference decoder throughout native replay, although afterward it only needs
the down-projection weight for existing same-input controls. Added separate
scratch `reference_lifetime.py` and `lifetime_history_resident.py`; all original
helpers, installed model code, native arithmetic, frozen artifact and their
existing pins remain unchanged.

Before native replay, verify **every F32 parameter bit** survives BF16/F32
round trip, in <=65,536-element chunks, and the supplied BF16 down weight
matches the original F32 down weight exactly. Then move the reference's 11
parameters to meta, require all original parameter weak references to die,
and call the unchanged replay. After success restore only the exact F32 down
weight from its verified BF16 copy. All remaining reference parameters stay
meta; no further reference forward is allowed/needed. A failed replay does not
recreate the source weights. Native cache owners retain their existing
fail-closed behavior. Hooks restore on exit; this remains synchronous scratch
diagnostic code, not a thread-safe production loader.

Synthetic lifetime test passes in **2.52 s**, **232,372 KiB** child RSS,
zero swap growth/process swaps: release/restoration and outputs match bits,
injected post-release failure leaves meta parameters, mismatched down and
lossy F32 source refuse before mutation. Separate 11-parameter hook-plumbing
test passes in **4.72 s**, **337,408 KiB**, zero swap growth/process swaps,
with exact source/replay/restored-down outputs and restored functions. Both
remain under unchanged 60-second/resource guards; scoped Ruff passes.

The new cached-MiMo learned **80, 31+49, 40+40, 1+63+16** coarse job passes in
**34.18 s**, **3,155,144 KiB** peak child RSS, minimum host-available RAM
**15,622,369,280 bytes**, zero host swap growth/process swaps, unchanged
300-second/resource guards. Every original stage/cache/control check passes.
Direct comparison verifies the **entire** same-schema four-layer numerical
dictionary, original native-owner records and runtime/model/helper pins match
the retained combined native-history coarse result exactly.

Measured native-phase lifetime change: **838,895,616 bytes** of F32 reference
parameters removed, all 11 original parameter objects released; observed RSS
**2,980,425,728 -> 2,175,102,976 bytes** at release. Only **201,326,592 bytes**
of down weight are restored after replay for the original reference control.
The removed parameter-byte count is not an OS/device RAM guarantee. The new
small native-entry report records the phase boundary so any future guard stop
can be located before/after release. Host availability also improved between
sessions; zero swap growth is **not** attributed solely to this change and
does not establish target-device performance or explain the old guard event.

Retained/reread report SHA-256:

| Report | SHA-256 |
| --- | --- |
| `reference-lifetime-selftest-result.json` | `517484af12c406d74e7df5bda6bde11d52e3946d92268b77145ff202f3b2ee7e` |
| `lifetime-history-selftest-result.json` | `1bfef57b42207644e24b6369fd0943c7ef6f7ada50fe4d677df185282afd99bc` |
| `lifetime-history-resident-comparison.json` | `a82b34b11f35e1ef37c2b5e662926f0864e49b9cc06897103831b81a9f705855` |

Scratch totals **2,976,018 bytes**, no raw arrays/replay packets/GGUF. Original
failed logs/diagnosis remain untouched; the old combined 64+16 run still has
no numerical verdict, and no old all-singleton job was started. This new
variant is a substantive lifetime change, not a looser guard or substitution
of a different schedule to seek green. Next: bounded new-variant 64+16 against
the unchanged same-schedule reference, after verifying the new synthetic
proof/source pins; no unchanged old-command retry. Docs/link/editor/whitespace
checks accompany this docs-only checkpoint; unchanged successful source gate
`b52e20b` (**553 passed, 88 optional skips**) was not rerun. No conversion,
download/refit/new candidate/compensation/quality/other-model action started;
original numerical/full-model/BitNet/v2 P4/device acceptance limits remain.

## Learned Convolution Owner Checkpoint (2026-10-16)

After `b52e20b`, scratch `PersistentConvolution` validates finite F32
`[channels,3]` initial history and `[channels,4]` filters, channels 1-8192,
append-only total tokens <=80. It uses the pinned native concat/SSM-conv/SiLU/
strided-tail CPY graph and explicit cache buffer/context/backend lifetime.
Only current input/filter tensors are uploaded after initialization; the
resident adapter hooks refuse history restores. Snapshots are independent host
oracles, never runtime history inputs. Output dimensions are checked before
graph allocation, all source bytes are preserved, history bytes/address/buffer
base are checked after graph teardown. Invalid input refuses before mutation;
post-update failure closes the owner. There is no reset API.

Typed helper self-test: **18 cases**, channels **8, 32, 8192** times six
schedules, raw/SiLU/full-history bits exact to the unchanged host-history
helper; all six eight-channel cases also match the compiled control exactly.
Eight invalid steps, noncontiguous equivalence, exhausted/closed refusal,
idempotent close and injected post-native-update failure cleanup pass. A
separate scratch control binary was compiled without replacing cached modules.

`native_history_resident.py` adds three such history owners to the already
verified three native GDN owners and persistent attention owner. Projections,
head order, norms, blocked MLP and host F32 residual/down sums are unchanged.
The adapter only substitutes convolution/GDN carry. It routes the synchronous
scratch calls with restored-on-exit module hooks; it is **not thread-safe** or
a production loader. Readback histories/states are oracle arrays. Native
history/GDN positions match the resident position after each successful step;
partial advancement closes every owner instead of pretending rollback.

| Check | Seconds | Peak Child RSS KiB | Min Host Available Bytes | Max Host Swap Growth Bytes | Process Swaps |
| --- | --- | --- | --- | --- | --- |
| Typed history helper, 18 synthetic cases | 1.47 | 270,084 | 14,520,766,464 | 0 | 0 |
| Combined seven-token monolithic/singleton synthetic stack | 43.19 | 1,062,516 | 12,320,645,120 | 0 | 0 |
| Combined cached-MiMo four coarse schedules | 39.53 | 3,580,176 | 10,441,805,824 | 284,766,208 | 0 |

Combined synthetic results match **every** recurrent/attention stage and final
history/state/KV cache of the unchanged layer-major reference. Four invalid
resident inputs, injected mid-layer failure, closed refusal/idempotent close
pass; nine history and nine GDN owners all close. The learned fixed 80-token
**80, 31+49, 40+40, 1+63+16** schedules also pass unchanged prior checks.
Direct comparison with the current native-state report verifies its **entire
same-schema** four-layer stage/hash/cache/metric dictionary exactly, plus
identical runtime/model/helper pins. Twelve history and twelve GDN owners all
close with zero host restores. This is the first bounded learned **coarse**
combined native-history/GDN ownership evidence, not a full-model result.

**Resource-blocked continuation:** the combined native-history 64-prefill /
16-singleton child was stopped at **27.02 s**, status **-15**, after host swap
growth **596,918,272 bytes** exceeded the unchanged **512 MiB** guard. Minimum
observed available RAM was **12,590,112,768 bytes**; observed filesystem growth
was **270,336 bytes**. This is a host-global guard observation, not proof of
which process caused the swap growth. The timing file is empty: the retained
**10,440 KiB** supervisor fallback and zero child-swap counter are **not** usable
experiment measurements. Separate guard diagnosis explicitly records experiment
RSS/swaps as **null**. Stderr contains only the existing Transformers fallback
warning, no completed numerical verdict. No decode numerical report exists;
the combined all-singleton job was **not started**. Neither the stopped job nor
another schedule was retried to seek green, and no bound/tolerance was relaxed.

Post-stop read-only diagnosis observed **14,719,406,080 bytes** available RAM,
**1,328,173,056 bytes** used swap and **544,906,784,768 bytes** free scratch.
That later RAM availability does not erase or explain the guard event. No
unrelated process was killed, host setting changed or missing fast-path library
installed as a workaround. Verified no owned experiment children survived and
no raw arrays/replay packets/GGUF/spill files or partial decode report remained.
Then rechecked all six runtime/frozen hashes, profile/metadata, initial
authenticated shard **identities** (not new full-weight hashes), installed
Transformers model code, nine helper hashes and compiled-control source pin.
All match. The old GDN-only learned singleton proofs remain valid, but they
do not prove combined native-history singleton execution.

Retained/reread report SHA-256:

| Report | SHA-256 |
| --- | --- |
| `persistent-conv-selftest-result.json` | `9b95734aeb6758115a63d493f8de66e508a589490784699b714bb287e0de1b05` |
| `native-history-resident-selftest-result.json` | `f1316e386af2c2128d439513b66d93717baf2d51e1312a20cc5c2a1f17c449d8` |
| `native-history-resident-comparison.json` | `4db102ba2fa7807e1505f62f50493bc782703f5392c11843d86eb5d81de782d9` |
| `native-history-resident-decode-guard-diagnosis.json` | `3941e90f7af7120b7bdc6d1eb17225ef720cd76cf14bda05e55062c9edaefb14` |

Separate compiled control SHA-256:
`d02c8bc78712e6b29fb8a61dee4f1c5369cfed115299977a9516f7d3d031a62a`.
Scratch totals **2,879,176 bytes** outside Git, only small scripts/reports/tools;
original reports and failed logs are preserved. Scoped Ruff, focused synthetic,
coarse learned and docs/link/editor/whitespace checks accompany this docs-only
checkpoint; successful source gate `b52e20b` (**553 passed, 88 optional skips,
12.93 s**) is unchanged and was not rerun. No full conversion/download/refit/
new candidate/compensation/quality/other-model action started. Original score
failure, full-model logit gap and conversion blocker remain open. Next learned
continuation is blocked on lower host memory pressure under the same guards;
do not retry unchanged or reinterpret the incomplete run as numerical evidence.
No full-loader/whole-model BitNet/v2 P4/quality/device acceptance is established.

## Native Convolution Ownership Checkpoint (2026-10-16)

After `d428f69`, the existing smoke executable adds separately bounded
`--cached-conv-control PARTITION [carry|reset]` using the same six strict
partitions as cached GDN/attention. This fixed synthetic control has eight
channels, four nonuniform taps/channel, 80 tokens, nonzero F32 `[3,8]` history
and one CPU thread. A cache-only context and explicit **96-byte** history
buffer survive each chunk graph/context/allocator destruction.

The graph uses native `ggml_concat(history,input,0)` to form `[tokens+3,8]`,
actual `ggml_ssm_conv` and SiLU. A strided `[3,8]` view of concatenated inputs,
offset `tokens*sizeof(float)` with row stride `(tokens+3)*sizeof(float)`, feeds
native `ggml_cpy` into the history owner. Raw/SiLU graph roots are expanded
before the copy root; the copy depends on the materialized concatenation, so
the old history is consumed before its owner is overwritten. Both F32
features/filters remain byte-exact. After all step resources die, the history
address, buffer identity and every byte must match the independently advanced
last-three-input oracle. That host oracle is never a runtime restore.

Carry uploads initial history once and performs one native copy/chunk with
zero host restores. Reset deliberately reuploads original nonzero history at
chunk boundaries; it is a fault control, not a runtime API. JSON distinguishes
initial uploads, fault restores and actual copy counts. Existing cached GDN,
attention, raw-file replay and other commands are unchanged.

Extended the existing pinned compiled test, not a new test file. Six schedules
times carry/reset match an independent FP64 four-tap oracle **bit-exactly** for
raw F32 outputs and full history. SiLU uses the existing `rtol=2e-5, atol=2e-6`.
All carry schedules match native monolithic raw/SiLU/history bits; reset keeps
the first chunk exact and changes later output by more than `2e-5`, with all
reset values still matching the independent reset-aware oracle. Six default
carry forms and eight malformed/extra-argument status-5, empty-stdout refusals
pass. This discriminates singleton tail stride, channel routing, reset state,
graph ordering and post-teardown ownership without learned weights.

Focused compiled check: **1 passed in 6.03 s** using the cached pinned Prism
headers/CPU/base libraries, no library rebuild. Required source checkpoint
gate: **553 passed, 88 optional skips in 12.93 s**, Ruff and whitespace clean;
native/test editor diagnostics and documentation/link/table validation clean.
No test failure, tolerance relaxation, model conversion/download/refit or
learned experiment accompanied this source slice. Earlier GDN learned results
remain valid but still use **host convolution history**; this primitive alone
is not a typed helper, learned convolution owner, full loader/BitNet/v2 P4/
quality/device acceptance. Next: a bounded typed history owner checked against
the compiled control and unchanged host-history helper, then synthetic
full-layer routing before any learned integration.

## Native State Singleton Checkpoint (2026-10-16)

After `e60e4a1`, the **unchanged** native-state resident adapter also passes
cached-MiMo 64-prefill/16-singleton and all-80-singleton learned continuation.
Both jobs include the monolithic 80-token control; three independent native
GDN state buffers and the persistent attention cache survive each step graph.
Convolution history still uses host arrays; recurrent state readbacks are
oracle-only and post-initialization host state uploads remain forbidden.
Six recurrent owners per job were created and closed. Four-layer arithmetic,
head routing, F32/BF16-RHS policy and all tolerances/guards are unchanged.

| Native-state learned job | Seconds | Peak Child RSS KiB | Min Host Available Bytes | Max Host Swap Growth Bytes | Process Swaps |
| --- | --- | --- | --- | --- | --- |
| 64-prefill / 16-singletons | 84.70 | 3,481,992 | 11,757,305,856 | 14,827,520 | 0 |
| 80-singletons | 165.09 | 3,465,440 | 12,071,075,840 | 0 | 0 |

The first job compares **every** recurrent stage/history/state and attention
stage/cache directly with a fresh unchanged layer-major reference. The second
matches every retained independent all-singleton stage/cache hash; that
successful old reference was **not** recomputed. Every resident step still
checks all attention cache bytes and same-operands core bits. Actual native
recurrent copies occur without host restores; all input bytes remain exact.
These are process/observed-host measurements, not target-device performance.

The extra whole-dictionary comparison against the historical 64+16 report
failed because it predates added stage-hash fields. A following over-strict
lookup also failed with `KeyError: stage_sha256`; those old hashes do not
exist. Read-only schema-aware comparison then verifies **435 shared leaves**,
all available output/history/state/cache hashes and pins exactly (eight
new-only fields). Missing historical per-stage hashes remain unavailable;
the separate fresh in-child all-stage comparison supplies that evidence.
No model rerun or acceptance relaxation followed either reporting failure.
For all singletons, both reports have the same schema: **640 shared leaves**,
all recurrent/attention stage hashes and final cache/output hashes match
exactly, with zero added/missing fields.

Observed schedule differences remain the prior values:

| Schedule | Max Output vs Monolithic | RMS vs Monolithic | Max Output vs HF Four Layers | Changed F16 K / V Elements |
| --- | --- | --- | --- | --- |
| 64-prefill / 16-singletons | 0.004378318786621094 | 0.00008131785828618599 | 0.012221336364746094 | 14,455 / 14,692 |
| 80-singletons | 0.012897491455078125 | 0.0002458041778928901 | 0.014368057250976562 | 73,233 / 74,340 |

These are measured batch/singleton numerical differences, not relaxed parity
tolerances. Monolithic output/HF metrics and the original failed score test
remain unchanged. Frozen codes/independent scales, six runtime libraries,
source/profile/metadata identities, installed Transformers and helper pins
recheck after both jobs. The all-singleton report's independent reference
digest also matches the existing reference pin.

Retained/reread report SHA-256:

| Report | SHA-256 |
| --- | --- |
| `native-state-resident-decode-comparison.json` | `97d07ac053304da170c02309a7d676bc2761d29936e0c2f8a23b5465862b223b` |
| `native-state-resident-singletons-comparison.json` | `154fb7481ed1b27bc9e61b2dfce6a9b9ea8feced4eff93ee71ea7143ff59a7aa` |

Output SHA-256 remains `f6a2224433cfd8eddf20a245b207e75c0e66321e2cbf16f2b9b1f5fd91533bdc`
for 64+16 and `2335d3e629a28ee6a67f5b7d656ec61c5eef309997234d67ba1b324caef6fcef`
for all singletons. Scratch totals **2,678,107 bytes** outside Git, no raw
arrays/replay packets/GGUF/owned spill. Parent-finally cleanup covers all own
packet paths. Docs/link/editor/whitespace validation accompanies this docs-only
checkpoint; unchanged source gate `6eb8959` was not rerun. Native recurrent
state ownership is now verified in these selected learned four-layer schedules,
not a full-model loader/BitNet/v2 P4/quality/device acceptance. The historical
full-model logit gap and swap-blocked conversion remain open. Next bounded
candidate: native convolution-history ownership, beginning with a synthetic
operation/layout control; no new conversion/download/refit/candidate/quality
run or other-profile execution is authorized by these results.

## Learned Native State Checkpoint (2026-10-16)

After `10f938e`, scratch `native_state_resident.py` replaces only the resident
driver's recurrent core file/host-state handoff. Each of layers 0-2 has a
separately parameterized `PersistentGDN` buffer, native GDN/state-tail CPY
updates, stable post-graph address/byte checks and a hook refusing all host
state uploads after initialization. Returned host snapshots are **oracles**,
not restoration inputs. Reuses unchanged BF16 projections, tiled head order,
convolution, norms, blocked MLP, host F32 residual/down sums and persistent
attention owner. Convolution histories remain independent **host** arrays.
The adapter opens no raw recurrent packet; unchanged reference controls may
still create disposable packets. Parent finally blocks cover all own packet
paths. All three recurrent owners close with the resident stack, including a
failure after layer 0 has advanced but layer 1 has not.

Scoped Ruff first caught an unused import before any child launch; removing
it permitted the same focused check. Full-geometry, nonzero sparse seven-token
synthetic monolithic/all-singleton stack passes in **37.63 s**, **1,058,528 KiB**
peak child RSS, minimum host-available RAM **14,176,804,864 bytes**, zero swap
growth/process swaps. Every recurrent/attention stage and final cache is
bit-exact to the unchanged layer-major reference. Four invalid inputs refuse
before mutation; injected mid-layer failure, closed refusal and idempotent
close pass. Nine recurrent owners were created and all closed.

The authorized cached-MiMo fixed 80-token learned run passes **80**, **31+49**,
**40+40**, **1+63+16** schedules in **39.53 s**, **3,609,732 KiB** peak child RSS,
minimum host-available RAM **11,848,736,768 bytes**, maximum host swap growth
**65,191,936 bytes**, zero process swaps. Unchanged 300-second/resource guards
remained active. All prior native prefix output/history/state and four-layer
output/metric checks pass; 12 recurrent owners close with zero host restores.
Every-step full attention-cache/matching-core and source-input byte checks
remain exact. Monolithic output remains
`3ecd5277dfed96c30180bcae87163b67bbea2d8def49f61a5f7ce1209c1504b2`;
monolithic/HF and chunk/monolithic gaps remain the historical measurements,
not relaxed acceptance limits or quality claims.

An extra **whole-dictionary** comparison with the historical coarse resident
report failed because that older schema lacks stage hashes and uses older
receipt names. A read-only structural diff found no changed shared value;
schema-aware comparison verifies all **561 shared leaves**, explicit four-layer
outputs, per-layer output/history/state hashes, attention caches and identical
runtime/model/helper pins. There are 76 added and 16 old-only nested fields.
Missing historical stage hashes remain unavailable; no model rerun or schema
rewrite was used to hide this comparison failure. Existing in-child stage/cache
metric and monolithic byte checks were unchanged.

Retained/reread report SHA-256:

| Report | SHA-256 |
| --- | --- |
| `native-state-resident-selftest-result.json` | `90722a3fefebab1264eae6a841434d3489d9288a0947121091e6dc6ea3b87a2c` |
| `native-state-resident-comparison.json` | `c7cb60534f39abfe7f697a5e57715a0be4a3aa60ac71f179e6dfefbcb605f83b` |

Adapter/helper source pins match the synthetic proof and remain unchanged
through learned execution; original six runtime libraries, frozen candidate,
source/metadata identities and installed Transformers model pins recheck at
exit. Scratch totals **2,584,163 bytes**, no raw arrays/packets/GGUF/owned spill.
Original reports/scripts are not overwritten. The retained inherited report
scope still says host recurrent state before the appended adapter-specific
replacement note; this checkpoint's native-owner/oracle distinction is the
current interpretation, not a claim about the old reference implementation.
Docs/link/editor/whitespace checks accompany this docs-only checkpoint; source
gate `6eb8959` is unchanged and not rerun. Next: matching learned 64-prefill /
16-singleton and all-80-singleton checks under the same limits. This does not
establish native convolution-history ownership, a complete-model loader,
BitNet whole-model dispatch, v2 P4 acceptance, decision quality or device
performance. Original score failure, full-model logit gap and conversion
blocker remain open; no conversion/download/refit/new candidate or broader
quality action started.

## Typed Recurrent Owner Checkpoint (2026-10-16)

After `6eb8959`, added a scratch typed `PersistentGDN` helper using the same
pinned GGML libraries and explicit native state-buffer/context ownership.
Its constructor requires finite F32 `[value_heads,128,128]` state and per-head
bias/negative multipliers, with heads 1-32 and value heads divisible by Q/K
heads. Each append-only F32 step consumes at most the remaining 80-token
window. There is no reset/restore API. Snapshots return independent host arrays
for checking, not runtime carry. Actual L2/raw-gate GDN and dependency-linked
state-tail `ggml_cpy` update the native owner; graph resources die before
address/buffer-base and copied state bytes are verified. Input bytes remain
exact. Invalid input refuses before mutation; computation/readback failure
closes the owner instead of reusing partially advanced state.

The initial supervisor preflight refused before launching the helper and
did not save its RAM/disk readings. A subsequent read-only diagnostic showed
**14,122,102,784 bytes** available RAM and **545,150,955,520 bytes** free disk,
both above unchanged prerequisites; the earlier exact readings are unknown.
Repaired only refusal observability: failed preflights now record actual
RAM/swap/disk readings, required 12/50 GiB reserves and `child_started=false`.
Two synthetic probes one byte below each threshold verify exact records and
forbid any child launch. No threshold was reduced or unrelated process killed.

The first actual helper run failed in **2.22 s** with nonsensical allocator
dimensions. The pinned header proved the ctypes GDN `K` argument must be
`int64_t`, not `int`; repaired that declaration and added an output-byte bound
before graph allocation. Failed logs remain. The repaired test passes in
**2.97 s**, **282,220 KiB** peak child RSS, minimum host-available RAM
**14,286,848,000 bytes**, zero swap growth/process swaps, unchanged 60-second
and resource guards. This is a binding repair, not a numerical change.

All **18 cases** (Q/K:value heads **2:4, 16:32, 32:32**, six schedules each)
match monolithic bits. Six 2:4 cases match the new compiled native-owned control;
the larger monolithic geometries match the legacy compiled raw-file GDN control
exactly. All schedules forbid host state uploads after initialization. Eight
invalid step cases, noncontiguous-input equivalence, exhausted/closed refusal,
idempotent close and injected failure **after** a native state update pass.
Parent and child finally blocks remove disposable raw packets. No learned
model, conversion, candidate/refit or quality run started.

Retained/reread `persistent-gdn-selftest-result.json` SHA-256:
`d96b93ba8414ff5214cea55aa65760e58690b54056b21d7e765302aae008d834`.
The separately compiled scratch control SHA-256 is
`7f91b5f91ff3aeff78dea4b88f5056fb5612be9b5276bab7fef759988d004b8f`;
cached libraries/binaries were not replaced. Scratch totals **2,496,789 bytes**
outside Git, with no raw arrays/replay packets, GGUF or owned compiler spill.
Scoped Ruff, focused synthetic and documentation/link/whitespace/editor
validation accompany this documentation-only checkpoint. The successful source
gate at `6eb8959` is unchanged and was not rerun. Native ownership is still not
integrated into the learned resident driver; convolution history also remains
host-carried. Next: a synthetic full-layer adapter comparison before learned
resident-driver integration, preserving earlier stage/cache hashes and all
original numerical/resource/frozen-candidate/full-loader approval boundaries.

## Native Recurrent Ownership Checkpoint (2026-10-16)

After user-pushed `d88871b`, added the separately bounded synthetic command
`--cached-gdn-control PARTITION [carry|reset]` to the existing native smoke
executable. Supported partitions are exactly `80`, `31+49`, `40+40`, `1+63+16`,
`singletons` and `64+singletons`. It uses width 128, two Q/K heads, four tiled
value heads, nonzero initial F32 value-by-key state, raw alpha/beta gates and
one CPU thread. No weights, model conversion or learned execution is involved.

A dedicated cache-only context and explicit backend buffer own the **262,144
state bytes**. Each chunk uses a new graph/context/allocator. Actual native L2
and raw-gate GDN output contains token vectors followed by final state; a
dependency-linked view of that tail feeds actual `ggml_cpy` into the persistent
state tensor. After chunk graph/context/allocator destruction, the state data
address and buffer owner must still match, and every state byte must equal
the GDN output tail captured before destruction. Input operands remain exact.
Readbacks are oracles, not runtime state restores. Carry mode uploads initial
state once, performs one native state copy per chunk and no host state restore.
Reset mode explicitly reuploads initial state between chunks as a lost-state
fault, not an accepted continuation policy.

The existing compiled regression now checks all six carry/reset schedules
against its independent NumPy FP64 recurrence with unchanged `rtol=2e-5,
atol=2e-6`. Every carry output and final-state bit exactly matches the old
80-token initialized raw-gate control. Reset preserves first-chunk bits but
changes later outputs; its entire output/state still matches the reset-aware
independent recurrence. Six default-carry invocations and eight malformed-CLI
refusals also pass. The shared partition parser preserves neighboring native
attention behavior; its existing controls run in the same focused test.

Focused pinned compile/execution: **1 passed in 11.75 s**. Required source
gate: **553 passed, 88 skipped in 24.80 s**, clean Ruff and whitespace. Scoped
editor diagnostics and Pylance test-file syntax checks pass. CPU/base library
SHA-256 pins reproduce exactly and the cached pinned source remains clean;
no cached library rebuild, CTest, sanitizer or model gate is claimed. No
failed check, numerical change or resource-limit relaxation occurred.

This is a synthetic native recurrent-state ownership primitive, not convolution
history ownership, a learned cached block/decoder, full-loader/model parity,
BitNet/v2 P4 acceptance, device performance or quality. The learned resident
driver below still carries recurrent state in host arrays; its previous results
are not retroactively native-owned. Frozen candidate, MiMo v1 factories and
profile/runtime approvals remain unchanged. The original failed score criterion,
full-model logit gap **0.0971508** and swap-guarded conversion remain unresolved.
Next: validate a bounded typed native-state helper against this compiled control
before replacing any learned driver's host-state handoff. Keep source/policy
integration and model-artifact approval separate from this synthetic result.

## Resident All Singleton Checkpoint (2026-10-16)

After `b9686cc`, completed bounded **80-singleton** dense four-layer continuation
on the fixed prompt. A seven-token all-singleton nonzero synthetic stack first
matches every resident recurrent/attention stage and final cache with its
independent same-schedule layer-major reference; refusal/failure cleanup also
passes in **39.49 s**, **1,047,732 KiB** peak child RSS, zero swap growth.

The learned reference and resident comparison run as separate serial jobs,
each retaining the original 300-second guard. Only small per-stage/final-cache
hashes and scalar metrics cross the job boundary; no raw activation is retained.
The reference child completes successfully in **198.78 s**, **3,323,236 KiB**
peak RSS, zero process swaps, with final source/runtime/helper checks. However,
its supervisor's final metadata write raises `FileExistsError`: the stage name
collided with the exclusive numerical report name. The numerical report,
stdout/stderr and `/usr/bin/time` evidence are intact. **Host minimum RAM,
swap-growth and disk-growth aggregates were not saved and remain unavailable.**
The monitor/guards were active; do not invent those aggregates or rerun a
successful reference for reassurance. A separate recovery report records this
failure and only independently retained child status/timing facts.

The resident comparison uses a distinct supervisor name and passes in
**208.44 s**, **3,337,568 KiB** peak child RSS, minimum host-available RAM
**10,500,919,296 bytes**, zero host swap growth/process swaps. Every recurrent
stage/history/state hash, attention-stage hash, final native K/V hash and scalar
metric dictionary exactly matches the reference. Both jobs use the same script,
helper/GDN/runtime/source pins; monolithic output/cache hashes and HF metrics
also reproduce. Every resident step passes full-cache/matching-core bit checks
after MLP work, with zero host attention cache restores; all owners close.
Reference-only records explicitly do not claim these resident per-step checks.

| Recurrent Layer | Output Max vs Monolithic | Final History Max | Final State Max |
| --- | --- | --- | --- |
| 0 | 0.0003347397 | 0.000007629395 | 0.000007629395 |
| 1 | 0.001392365 | 0.001676083 | 0.0001898408 |
| 2 | 0.006101608 | 0.004032135 | 0.0006805509 |

Four-layer all-singleton output has max **0.01289749**, RMS **0.0002458042**,
versus native monolithic, and max **0.01436806**, RMS **0.0002501368**, versus
trusted HF. Final F16 K/V differs from full-stack monolithic in **73,233 /
74,340** elements; changed prefix operands are included. These are observed
differences, not acceptance thresholds. Output SHA-256:
`2335d3e629a28ee6a67f5b7d656ec61c5eef309997234d67ba1b324caef6fcef`.
Original fixed score criterion remains false; full-model logit gap **0.0971508**
is unresolved. No conversion, download, candidate/refit, compensation, padding
promotion, output head/scoring or generated answer tokens.

Retained/reread small-report SHA-256 values:

- `resident-stack-singletons-selftest-result.json`:
  `78a198d898367cb0e64f6f58ae85b88ac8f8fef452e95db2a89c752db78fca60`.
- `resident-stack-singletons-reference.json`:
  `0a263efd63091405f571f6f20af5dbfece80c82b8e178edb589de0499435576c`.
- `resident-stack-singletons-reference-supervision-recovery.json`:
  `a7e8c37b446c4545c12b8343f7b98181f84807d1a540360f10f746916c6b5aa8`.
- `resident-stack-singletons-comparison.json`:
  `fbcbeb915a1a82c0a627927e85d031b7f4918339f5cad73eb3d101f3810bf32e`.

Scratch totals **2,398,709 bytes** outside Git; no raw arrays/replay packets,
GGUF or owned compiler spill remains. Scoped Ruff, synthetic/learned and
documentation/link/whitespace/editor checks accompany this documentation-only
checkpoint; unchanged source gates were not rerun. This completes the selected
dense four-layer coarse/prefill-singleton/all-singleton controls, not native
recurrent-state residency, full-loader/model parity, whole-model BitNet/v2 P4,
target-device performance or quality acceptance. Next substantive integration
gate remains matching native-loader/reference traces on an approved model
artifact. The stopped full conversion remains blocked under unchanged resources;
prepare a revised bounded artifact/resource proposal before any new conversion,
and preserve the original failed numerical criteria and frozen candidate.

## Resident Singleton Continuation Checkpoint (2026-10-16)

After `7fbb893`, extended the bounded resident driver to **64-token prefill
plus 16 singleton steps** on the fixed 80-token MiMo prompt. An independent
layer-major reference executes the same partition with each layer's own
history/state and absolute attention offsets. No generated tokens, output
head, padding or new runtime API is involved.

The full-geometry nonzero synthetic stack uses `4+1+1+1`. Every recurrent
stage, history/state, attention stage and final native cache matches its
same-schedule layer-major reference bit-for-bit, without assuming batching
invariance. Input-refusal, injected-failure closure and raw-packet cleanup
checks also pass. Synthetic runtime is **29.58 s**, **1,048,700 KiB** peak
child RSS, zero swap growth.

Learned resident versus independent layer-major matches **every recurrent
stage, final history/state, attention stage and final K/V cache bit-for-bit**.
The preceding monolithic recurrent hashes and full HF metric dictionary
reproduce exactly. Per-step full-cache/matching-core checks hold after all
MLP work, with zero host attention cache restores; all owners close.

| Recurrent Layer | Output Max vs Monolithic | Final History Max | Final State Max |
| --- | --- | --- | --- |
| 0 | 0.00006723404 | 0.000007629395 | 0.000003814697 |
| 1 | 0.0007896423 | 0.001676083 | 0.0001956820 |
| 2 | 0.003899574 | 0.005183101 | 0.0003766641 |

The complete four-layer result has max **0.004378319**, RMS **0.00008131786**,
versus native monolithic; versus trusted HF, max **0.01222134**, RMS
**0.0002478968**. Final F16 K/V differs from full-stack monolithic in
**14,455 / 14,692** elements. These include changed prefix operands and are not
fixed-prefix attention-only crossings or acceptance thresholds. Final output
SHA-256: `f6a2224433cfd8eddf20a245b207e75c0e66321e2cbf16f2b9b1f5fd91533bdc`.

The learned stage passes in **91.07 s**, **3,464,932 KiB** peak child RSS,
minimum host-available RAM **12,149,075,968 bytes**, zero host swap growth and
process swaps. All original 60/300-second/resource guards and final runtime/
frozen/source/profile/metadata/installed-code/GDN/helper pins hold. No failed
check or refusal occurred. Original fixed score criterion remains false;
full-model logit gap **0.0971508** is unresolved. No conversion, download,
refit, compensation, padding promotion, scoring or generation.

Retained/reread small-report SHA-256 values:

- `resident-stack-decode-selftest-result.json`:
  `07f6c75904b438c38d8bcae5a72a38e6809562c94776d2c265d8dd9f90fd652f`.
- `resident-stack-decode-comparison.json`:
  `227f43af882d51722c89d8d8cfd6547e3299805b6c45e0eccaec97d29dfd7d74`.

Scratch totals **2,299,409 bytes** outside Git; no raw arrays/replay packets,
GGUF or owned compiler spill remains. Scoped Ruff, synthetic/learned and
documentation/link/whitespace/editor checks accompany this documentation-only
checkpoint; unchanged source gates were not rerun. This is dense four-layer
prefill/singleton continuation, not all-80-singleton execution, full-loader/
whole-model BitNet/v2 P4, target-device performance or quality acceptance.
Next: separately budget an all-singleton layer-major reference and resident
comparison, retaining only small hashes/metrics and preserving each existing
300-second deadline rather than adding both long paths to one stage.

## Resident Stack Checkpoint (2026-10-16)

After `2dd5f65`, added a bounded scratch token-major resident owner: every
chunk traverses recurrent layers 0-2 and the complete attention layer before
the next chunk. Four layers' BF16 projection parameters remain in memory;
the three recurrent histories/states remain separate copied host arrays and
attention owns one persistent native F16 cache. Original native helpers,
unpadded arithmetic, host F32 residuals/down sums and absolute rotary offsets
are unchanged. No model loader or new public runtime API was integrated.

The nonzero full-geometry synthetic control compares each resident recurrent
stage/final cache with the independent layer-major driver for `7` and `3+4`;
all attention stages also match monolithic bits. Four malformed inputs refuse
before mutation while the owner remains usable. A later injected layer-1
failure closes the partially advanced owner and releases all histories/states;
closed input refuses and close is idempotent. Complete attention cache/core
checks run after each full chunk, with cache restores forbidden. Initial lint
caught an unused import before execution; removing it permitted the first
synthetic run, which passes in **31.69 s**, **1,034,964 KiB** peak child RSS,
zero swap growth. No arithmetic or resource failure occurred.

The learned resident run tests `80`, `31+49`, `40+40` and `1+63+16`. All four
output hashes exactly equal the preceding layer-major matching-schedule
hashes. Monolithic recurrent output/history/state hashes reproduce exactly;
other schedules reproduce all prior recurrent stage/history/state difference
dictionaries. `31+49`/`40+40` additionally retain bit-exact every-stage and
complete recurrent/attention-cache agreement with monolithic. The `1+63+16`
output remains max **0.05065632**, RMS **0.0002436861**, from monolithic:
token-major ordering does not repair that measured drift. Its final F16 K/V
cache differs from full-stack monolithic in **72,614 / 73,840** elements,
because its incoming native prefix also differs; do not compare these counts
with fixed-prefix attention-only crossings. Every per-step full-cache and
matching-core check passes with no cache restores, and all owners close.

Learned runtime is **103.27 s**, **3,568,904 KiB** peak child RSS, minimum
host-available RAM **9,595,006,976 bytes**, host swap growth **1,241,088 bytes**,
zero child process swaps. This remains below the unchanged 512 MiB swap-growth
guard and above the 2 GiB reserve; host observations are not device guarantees.
Original deadlines and final runtime/frozen/source/profile/metadata/installed-
code/GDN/helper pins hold. The monolithic complete HF metric dictionary also
reproduces exactly. Original score criterion remains false; full-model logit
gap **0.0971508** remains open. No conversion, download, refit, compensation,
padding promotion, output head/scoring or generated tokens.

Retained/reread small-report SHA-256 values:

- `resident-stack-selftest-result.json`:
  `8c7cc21b8a29ef3e36f89f5f2317d2022ec969d0beaf750738a0c5ea25042505`.
- `resident-stack-comparison.json`:
  `b0cf390ece1642949a34ebf0dcbf759dae0890d69a03727cb4aff5182788919d`.

Scratch totals **2,260,937 bytes** outside Git; no raw arrays/replay packets,
GGUF or owned compiler spill remains. Scoped Ruff, synthetic/learned and
documentation/link/whitespace/editor checks accompany this documentation-only
checkpoint; unchanged source gates were not rerun. This establishes bounded
dense resident four-layer coarse continuation, not full-model/native-loader
parity, all-singleton continuation, whole-model BitNet/v2 P4, target-device
performance or quality acceptance. Next: budget a bounded resident prefill/
singleton schedule with an independent same-schedule layer-major reference;
do not expand the 300-second learned guard to fit a longer experiment.

## Four Layer Continuation Checkpoint (2026-10-16)

After `5fb6222`, added an unpadded recurrent partition driver around unchanged
`CachedLayer`, carrying separate convolution history and GDN state for each
layer. A seven-token full-geometry synthetic three-layer stack uses nonzero
sparse BF16 projections: `3+4` matches every monolithic stage and final cache
bit-for-bit. Resetting history/state preserves the first three outputs but
changes final outputs by maxima **0.1520386 / 0.2920475 / 0.7619047** across
layers 0/1/2. The synthetic control passes in **18.18 s**, **961,604 KiB** peak
child RSS, zero swap growth. It is not a representative quality fixture.

The learned run carries independent per-layer caches through `31+49`, `40+40`
and `1+63+16`, without padding. It exactly reproduces the prior monolithic
recurrent packet/metric dictionaries and four-layer output/reference evidence.
`31+49` and `40+40` are bit-exact at every recurrent stage, final history/state
and complete attention-layer output. For each prefix schedule, all four
attention schedules are measured separately; direct output hashes bind the
collected full-stack outputs to each attention replay. Every full-cache and
matching-operand core check passes, with zero native attention cache restores.

| Recurrent Layer, `1+63+16` | Layer Output Max vs Monolithic | Final History Max | Final State Max |
| --- | --- | --- | --- |
| 0 | 0.0003347397 | 0 | 0.000001907349 |
| 1 | 0.001392365 | 0 | 0.00008445233 |
| 2 | 0.005715370 | 0.003884912 | 0.0007173680 |

Matching `1+63+16` through all four layers has final max **0.05065632**, RMS
**0.0002436861**, versus native monolithic, and max **0.05208683**, RMS
**0.0002698306**, versus trusted HF. Do not substitute the preceding fixed-prefix
attention-only gap **0.0002279878** for this propagated result. The two other
matching schedules retain monolithic HF max/RMS **0.01222134 / 0.0002470044**.
These are measured unmodified differences, not relaxed acceptance thresholds.
Monolithic four-layer SHA-256:
`3ecd5277dfed96c30180bcae87163b67bbea2d8def49f61a5f7ce1209c1504b2`.
Matching `1+63+16` SHA-256:
`333ee8f5f9e6e64111fdc983ab8de873fc1fb5bd6a0d2202a946f2c398464e80`.

The learned stage passes in **99.07 s**, **2,033,604 KiB** peak child RSS,
minimum observed host-available RAM **13,294,895,104 bytes**, zero swap growth.
Original 60/300-second/resource guards and final runtime/frozen/source/profile/
metadata/installed-code/GDN binary checks hold. No failed check or refusal
occurred. Original fixed score criterion remains false; full-model logit gap
**0.0971508** is unresolved. No conversion, download, refit, compensation,
padding promotion, output head/scoring or generation.

Retained/reread small-report SHA-256 values:

- `native-prefix-partition-selftest-result.json`:
  `074e0ef4c451beb82b67186661f7cf0127a78d35f95479897015a29591c24f56`.
- `native-prefix-partition-comparison.json`:
  `2313e08020ea9e0e42b95e3871543962d253f65fa6a7c1114e8a597c2f00ff36`.

Scratch totals **2,200,689 bytes** outside Git; no raw arrays/replay packets,
GGUF or owned compiler spill remains. Scoped Ruff, synthetic/learned and
documentation/link/whitespace/editor checks accompany this documentation-only
checkpoint; the unchanged source gate at `c79b06e` was not rerun. Execution is
layer-major with serially streamed weights and copied host recurrent caches,
not a resident token-major decoder or single loader graph. Native persistent
attention ownership is separately checked. No full-model BitNet/v2 P4,
device-performance or quality acceptance follows. Next: a bounded token-major
resident four-layer routing control against these exact schedule hashes, with
independent per-layer state and unchanged resource guards.

## Native Prefix Composition Checkpoint (2026-10-16)

After user-pushed `4754bca`, composed separately parameterized native recurrent
decoder layers 0-2 with the complete dense attention layer on the existing
80-token MiMo fixture. A scratch parameter adapter reuses unchanged
`CachedLayer`: BF16 projections, `weight+1` input/post norms, direct gated-norm
scale, per-layer convolution and reordered bias/negative-exp decay. Independent
marker/order/scale checks, invalid geometry refusal and full-geometry zero-weight
residual identity pass in **4.72 s**, **576,416 KiB** peak child RSS, zero swap.

Each recurrent layer processes one 80-token prefill with its own zero initial
history/state. Weights stream serially, not three retained copies. Layer 0 must
reproduce both prior raw GDN packet hashes and its full-layer metric dictionary
exactly. Layers 1/2 separately compare native execution on the propagated native
input against a same-input HF call and against the independent trusted HF
prefix. No diagnostic singleton padding is used.

| Recurrent Layer | Same-Input Native/HF Max | Propagated Native/HF Max | Propagated RMS |
| --- | --- | --- | --- |
| 0 | 0.0003347397 | 0.0003347397 | 0.000007723088 |
| 1 | 0.0002883747 | 0.001556396 | 0.00005988301 |
| 2 | 0.005167007 | 0.007488251 | 0.0001387589 |

The final native prefix hash must exactly match the attention layer's input
hash. Complete four-layer output versus the trusted HF prefix/attention
reference has max **0.01222134**, RMS **0.0002470044**. Replaying only attention
on that fixed native prefix retains exact `31+49`/`40+40` stage agreement;
`1+63+16` final max/RMS are **0.0002279878 / 0.00001261400**, with **2/3** F16
K/V crossings. Complete-cache and matching-operand core checks pass with zero
native attention cache restores. These are measured arithmetic differences,
not acceptance thresholds or end-to-end four-layer cached decoding.

The first learned run stopped after **13.98 s** on the diagnostic wrapper's
overly broad recurrent-only assertion when the existing helper called its
complete layer-3 HF attention reference. Raw packets were cleaned. Repaired
only that dispatch boundary to pass through the expected layer-3 full-attention
call; retained failed logs, unchanged numerical checks and resource limits.
The repaired stage passes in **27.34 s**, **1,923,248 KiB** peak child RSS,
minimum host-available RAM **14,582,423,552 bytes**, zero swap growth. Runtime,
frozen/source/profile/metadata/installed-code checks and final GDN binary hash
hold. Original fixed score check remains false; full-model logit gap **0.0971508**
is unresolved. No new conversion, download, candidate, head/scoring or generation.

Retained/reread small-report SHA-256 values:

- `native-prefix-adapter-selftest-result.json`:
  `0495a2569e5400f53f9d60202b2a2aad8ceeb9bcecb5c8121d4e7ce1129db2fe`.
- `native-prefix-attention-comparison.json`:
  `f997f3e1386628b60b4e2268b4f1e4f6686f71ee9848dfd20e2abfdb941abc40`.

Scratch totals **1,970,829 bytes** outside Git; no raw arrays/replay packet,
GGUF or owned compiler spill remains. Scoped Ruff, synthetic/learned and
documentation/link/whitespace/editor checks accompany this documentation-only
checkpoint; the unchanged source gate at `c79b06e` was not rerun. Recurrent state
is copied host state; attention cache has its separate persistent native owner.
Residuals/down-block sums remain host F32. This is bounded dense four-layer
prefill composition, not a single loader graph, full-model BitNet/v2 P4,
target-device performance or quality acceptance. Next: bounded coarse-partition
continuation across all four layers, preserving each layer's history/state and
absolute attention positions; do not infer it from fixed-prefix attention replay.

## Attention Layer Continuation Checkpoint (2026-10-16)

After `c5d7716`, extended the same complete dense attention-layer diagnostic
to 80 singleton calls and 64-token prefill plus 16 singleton calls. The
synthetic schedules retain independent full 12,288-wide blocked MLP, norm,
product and residual references and exact per-step cache/core checks. They
pass in **9.93 s**, **462,828 KiB** peak child RSS, zero swap growth.

The learned schedules use the same fixed trusted layer-2 output, not generated
tokens or an accumulating native prefix. They preserve native input/post RMS,
chunkwise native attention/cache, the original seven MLP projection blocks,
host F32 residuals and host F32 down-block sums. Every complete-cache readback
and matching-operand core check passes after the entire chunk's MLP work;
cache restores remain forbidden and all owners close. The monolithic complete
layer output hash and complete HF-reference metric dictionary reproduce the
preceding coarse-partition report exactly. Monolithic output SHA-256:
`65159b0ec994b9a20e12b72b0d133289bffbc62ae5bc633222528aa22dddf15f`.

| Schedule | Attention Output Max Gap | Final Layer Max Gap | Final Layer RMS | Changed F16 K / V |
| --- | --- | --- | --- | --- |
| 80 singletons | 0.0003318787 | 0.0005140305 | 0.00004026168 | 129 / 107 |
| 64 + 16 singletons | 0.0001254082 | 0.0002212971 | 0.00001401778 | 26 / 21 |

Input norm is exact across batching in both schedules. Post-norm/gate/up/
product/down maxima versus monolithic are **0.0003151298 / 0.001105130 /
0.0008133650 / 0.0004100800 / 0.0005104989** for 80 singletons and
**0.0001059324 / 0.0005363673 / 0.0004519150 / 0.0003412962 /
0.0002264977** for 64+16. These are observed unmodified downstream propagation,
not tolerance gates, runtime corrections or representative quality evidence.
This layer includes a native input norm, unlike the earlier sublayer replay
on fixed HF-normalized input; do not conflate their operands or gap values.

The learned stage took **105.48 s**, **1,900,632 KiB** peak child RSS, minimum
observed host-available RAM **14,586,802,176 bytes**, zero swap growth. All
original 60/300-second/resource guards hold, with explicit final runtime/
frozen/source/profile/metadata/installed-code pin checks after the extra
reference and MLP work. No failed check or resource refusal occurred. The
original learned fixed score check remains false, and the complete-model
max logit gap **0.0971508** remains unresolved. Nothing was padded, refit,
compensated, promoted, converted or downloaded; no output head or scoring ran.

Retained/reread small-report SHA-256 values:

- `attention-decoder-decode-selftest-result.json`:
  `874560886099b35ddedaaa549826d6045a2e5d0fb443a05164517f48be5fdca8`.
- `attention-decoder-decode-comparison.json`:
  `71ab9a474bf5c8c6a255ae4b195daaec79786e199aa71fefec4ff8056d4930fb`.

Scratch totals **1,917,848 bytes** outside Git; no raw arrays/replay packet,
GGUF or owned compiler spill remains after empty-directory cleanup. Scoped
syntax/Ruff, focused synthetic/learned and documentation/link/whitespace/editor
checks accompany this documentation checkpoint; the unchanged source gate at
`c79b06e` was not rerun. This completes bounded complete-layer attention cache
continuation on a trusted prefix, not a native-prefix stack, full-loader parity,
whole-model BitNet/v2 P4, target-device guarantees or quality acceptance.
Next: retain the same layer-wise arithmetic and ownership boundaries when
composing a bounded native recurrent prefix with this attention layer. The
stopped complete conversion remains blocked under unchanged resource guards.

## Attention Decoder Layer Checkpoint (2026-10-16)

After `bd2aa9a`, extended the bounded composition to the **complete dense
layer-3 attention decoder layer** on the same authenticated trusted layer-2
output: input RMS with `weight+1`, native attention/cache, first residual,
post-attention RMS with `weight+1`, gated MLP and second residual. Reused the
unchanged `LayerKernel`: gate/up each use two bounded row blocks; down uses
three 4,096-column blocks. Residual additions and the blocked down-result sum
remain **host F32**, as in the earlier layer-0 control; this is not one native
loader graph or a new reduction policy.

The synthetic control uses hidden width 256, four query/two KV heads and the
full 12,288-wide MLP. Independent norm, BF16 projection, SiLU-product, blocked
down-sum and Torch-vs-NumPy residual oracles pass unchanged tolerances. Complete
cache bytes remain exact after every full-layer chunk, including subsequent
MLP graph destruction; matching-cache native core scores/probabilities/outputs
are bit-exact. Monolithic, 31+49, 40+40 and 1+63+16 pass in **8.34 s**,
**469,236 KiB** peak child RSS, zero swap growth.

The learned capture verifies the original layer-2 output hash, normalized
layer-3 input hash and post-rotary QKV hashes before adding cached operands.
It reuses six materialized attention parameters and separately loads five
norm/MLP tensors, including the previously observed input norm, yielding 11
target-layer parameters. A complete installed-HF reference uses the same
F32/BF16-RHS convention and must reproduce the captured input norm bit-exactly.
No complete model, output head, model-repository code or artifact is loaded.

| Comparison | Max Absolute Gap | RMS |
| --- | --- | --- |
| Monolithic composed layer vs same-input HF | 0.002464294 | 0.00009452965 |
| 31+49 vs monolithic composed layer | 0 | 0 |
| 40+40 vs monolithic composed layer | 0 | 0 |
| 1+63+16 vs monolithic composed layer | 0.0002372116 | 0.00001105487 |

The native input norm differs from HF by max **1.525879e-5**. Propagated
monolithic post-norm/gate/up/product/down maxima versus HF are **0.001324274 /
0.003574371 / 0.003620356 / 0.001648366 / 0.001816869**. A separate same-HF-
product down control has native-split vs Torch-split max **9.536743e-7**;
Torch-split vs Torch-full max **1.192093e-7**. These distinguish source operand
propagation from that same-input reduction boundary without assigning the
unresolved full-model discrepancy to either. For 1+63+16, attention/post-norm/
MLP-down maxima versus monolithic are **3.385544e-5 / 3.635883e-5 /
0.0002369359**, with only **3 / 6** F16 K/V crossings. Every per-step cache
and matching-core check still passes, with no cache restores.

The learned stage took **20.63 s**, **1,925,632 KiB** peak child RSS, minimum
observed host-available RAM **14,244,102,144 bytes**, zero swap growth. Explicit
exit verification is repeated after the extra layer tensors/reference/native
work, preserving all six runtime hashes, frozen codes/scales/manifest, profile/
metadata/source and installed implementation identities. Resource limits and
60/300-second deadlines are unchanged. No failed check occurred in this slice.
The original fixed learned score failure and full-model max logit gap
**0.0971508** remain unresolved; no tolerance, padding or compensation changed.

Retained/reread small-report SHA-256 values:

- `attention-decoder-layer-selftest-result.json`:
  `01fe000869748ba04a2779c84535e795f8c007e8ab992773c9bd1a0782ceb986`.
- `attention-decoder-layer-comparison.json`:
  `ec0c13bf2c57c8f79f4bcd310e390adc825b2c529c8d6f04cc76b5958f1fae19`.

Scratch totals **1,871,632 bytes** outside Git, no raw arrays/replay packet,
GGUF or owned compiler spill after empty-directory cleanup. Scoped syntax/
Ruff, synthetic/learned and documentation/link/whitespace/editor checks
accompany the commit; the unchanged source gate at `c79b06e` was not rerun.
No conversion, download, refit, candidate change, generation, scoring or other-
profile execution occurred. This is a dense full-layer composition from a
trusted prefix, not native-prefix propagation, full-loader/whole-model parity,
real-model v2 P4, BitNet execution, target-device performance or quality.
Next: bounded singleton/prefill-singleton continuation of this complete layer.

## Attention Continuation Checkpoint (2026-10-16)

After `5d1be75`, extended only the existing diagnostic's schedule selection
with separate exclusive synthetic/learned reports: 80 singleton calls and a
64-token prefill plus 16 singleton calls, alongside a monolithic baseline.
These replay the original **80 fixed prompt tokens**, with no output head,
option scores or generated answer tokens. The same trusted normalized input,
native QKV/norm/absolute rotary, persistent F16 write/core, sigmoid gate and
BF16 output projection are used. No arithmetic, padding or scale change.

The synthetic singleton schedules pass unchanged independent projection,
norm/rotary/gate/output references and the wrong-offset fault in **6.57 s**,
**365,568 KiB** peak child RSS, zero swap growth. Every learned chunk then
passes exact full-cache readback after graph teardown and bit-exact standalone
core reads on its own produced operands. Host cache restores are forbidden;
owners close after each schedule. The monolithic core hash and full HF-gap
dictionary reproduce the earlier baseline; the monolithic final output also
matches the coarse-partition report's SHA-256:
`13228fa679ca8bfd68421cdb619a4fe971e51100854b52abcb34db68dfe2da2f`.

| Schedule | Core Max Gap vs Monolithic | Final Output Max Gap | Final Output RMS |
| --- | --- | --- | --- |
| 80 singletons | 0.0006724000 | 0.0003695488 | 0.000003163927 |
| 64 + 16 singletons | 0.0003272891 | 0.0003170967 | 0.000001094056 |

The first schedule changes **491 / 130 / 106** F16 Q/K/V values and **1,348**
BF16 output-projection inputs; the second changes **97 / 26 / 21** and **112**.
Q/K/V projection maxima are **1.144409e-5 / 4.291534e-6 / 2.503395e-6** for
80 singletons and **6.675720e-6 / 2.503395e-6 / 2.026558e-6** for 64+16.
Gated maxima are **0.0001022816 / 0.00002644211**. These measurements retain
unmodified batching/rounding propagation; fixed-operand persistence alone did
not establish chunkwise projection or final-output invariance. No new
tolerance, compensation or padding policy is promoted from this diagnostic.

The learned stage took **30.24 s**, **1,484,240 KiB** peak child RSS, minimum
observed host-available RAM **15,527,874,560 bytes**, zero swap growth. The
unchanged 60/300-second, address-space, RAM/swap/disk guards hold. Six runtime
hashes, frozen codes/scales/manifest, profile/metadata/source and installed
implementation identities pass entry/exit checks. The original learned fixed
score check remains false; the full-model max logit gap **0.0971508** remains
unresolved. No failed gate or resource refusal occurred in these two stages.

Retained/reread small-report SHA-256 values:

- `cached-attention-decode-selftest-result.json`:
  `88c60e7705ea8aef8e417c8bda7768ca599c834aaca928c44c9276831e72b389`.
- `cached-attention-decode-comparison.json`:
  `8a5659deeebbe2f07fcb10fc63a346550be18b2ff3d23c356dd37496804354a7`.

Scratch totals **1,802,924 bytes** outside Git, with no raw arrays/replay
packet, GGUF or owned compiler spill after empty-directory cleanup. Scoped
syntax/Ruff, synthetic/learned and documentation/link/whitespace/editor checks
accompany the commit. The unchanged successful source gate at `c79b06e`
was not rerun for documentation-only edits. No conversion, download, refit,
new candidate, compensation promotion or other-profile execution occurred.

This completes the bounded **attention-sublayer** coarse/singleton cache
continuation diagnostic, not a complete attention decoder layer, input norm,
residual/MLP, native prefix, loader, whole-model BitNet/v2 P4, target-device
performance or quality acceptance. Next: synthetic-first input-normalization,
residual and MLP composition for a complete attention decoder layer, keeping
fixed-input sublayer evidence separate from native-prefix propagation. The
stopped full conversion is not authorized to retry under unchanged resources.

## Cached Attention Sublayer Checkpoint (2026-10-16)

After `1a4e653`, composed the existing verified native attention primitives
with the persistent cache owner, without duplicating projection/norm/rotary
logic. Narrow adapters add each chunk's absolute rotary positions and route
only current K/V updates into the append-only owner. Each chunk now runs four
BF16 projections, two Q/K RMS operations with learned `weight+1`, two native
partial rotary operations, one actual-cache-write/causal-core graph and the
sigmoid gate: ten native graph calls. The normalized input and three-layer
prefix remain trusted Torch, not a native prefix or complete decoder layer.

The 80-token synthetic control independently checks per-chunk projections,
norm, phase-aware rotary, sigmoid/gated values and BF16 output projection with
unchanged oracles. Resetting rotary offsets is a discriminating fault. Every
cache buffer must match independently accumulated F16 bytes after each graph
teardown; every core score/probability/output bit matches a standalone native
read on those exact chunk-produced cache operands. Cache host restores are
forbidden, source features stay unchanged and all owners close. Monolithic,
31+49, 40+40 and 1+63+16 synthetic composition passed in **7.42 s**,
**366,800 KiB** peak child RSS, zero swap growth. The learned capture wrapper
preserves the original attention forward return value.

The learned run verifies the same normalized-input hash and reproduces the
earlier monolithic core SHA-256
`3abedc6de1f9014e43bd8c1f542459a98c30a9b812ffd598b3e03cb9a08164a2`.
The full monolithic output-versus-HF metric dictionary is also identical:
max **0.002011299**, RMS **2.913632e-5**. All cache-byte, matching-core-bit,
absolute-position, no-restore and cleanup checks pass in every partition.

| Partition | Final Output Max Gap vs Monolithic | Changed F16 Q / K / V | Changed BF16 Output-Projection Inputs |
| --- | --- | --- | --- |
| 31+49 | 0 | 0 / 0 / 0 | 0 |
| 40+40 | 0 | 0 / 0 / 0 | 0 |
| 1+63+16 | 0.00003385544 | 8 / 3 / 6 | 34 |

For 1+63+16, Q/K/V projection maxima are **3.337860e-6 / 3.814697e-6 /
9.536743e-7**, core max **0.0001220703**, gated max **8.866191e-6**.
These are unmodified chunkwise arithmetic differences and quantization
crossings, not cache-write errors, accepted model accuracy or replacement
tolerances. Singleton projection/core/gate/output continuation is not yet
measured by this connected diagnostic.

The learned stage took **10.42 s**, **1,484,208 KiB** peak child RSS, minimum
observed host-available RAM **15,275,798,528 bytes**, zero swap growth. All
runtime/frozen/profile/metadata/source and installed implementation entry-exit
pins pass under unchanged 60/300-second deadlines and resource guards. The
original fixed learned score check stays false; the prior complete-model
max logit gap **0.0971508** remains open. There was no new failed check.

Retained/reread small reports:

- `cached-attention-sublayer-selftest-result.json`, SHA-256
  `8b8a2153043b68cae44a191364498cc1fac6307218bba4835c571f0abdea3a97`.
- `cached-attention-sublayer-comparison.json`, SHA-256
  `a247e4bfb9e5acc8c8d57291cedf1c12cbe55298b0b9e7e2d30f405baad8dfc7`.

Scratch totals **1,771,750 bytes** outside Git; no raw arrays/replay packet,
GGUF or owned compiler spill remains after empty-directory cleanup. Scoped
syntax/Ruff and documentation/link/whitespace/editor checks accompany the
commit; the unchanged source gate at `c79b06e` was not rerun. No output head,
generation, scoring, download, conversion, refit or candidate change occurred.
This remains dense attention-only evidence, excluding input norm, residual/MLP,
native prefix, loader, cache eviction/concurrency/multistream, whole-model
parity, v2 P4/BitNet or quality acceptance. Next: bounded singleton and
prefill-plus-singleton continuation of this same connected attention sublayer.

## Learned Persistent Cache Checkpoint (2026-10-16)

After `c79b06e`, a separate bounded ctypes owner implements the same explicit
native-buffer lifetime. It accepts finite F16 initial K/transposed-V arrays,
1-4 KV heads and 1-16 grouped query heads, width 256 and 128 slots. Steps append
1-80 finite F32 Q/K/V tokens within a fixed total window of 80; no arbitrary
indices, reset, file or restore API is exposed. Each graph writes actual
`ggml_set_rows` and consumes the write-output views in causal attention. After
temporary graph/allocator teardown, cache addresses and complete cache bytes
must remain exact. Returned arrays are readback oracles, **not runtime carry**.

Invalid input is refused before native calls and leaves the owner usable.
Compute/readback failure instead closes the owner, preventing partially
updated-state reuse. Context exit, including exceptional exit, frees the cache
buffer, backend and owning context; close is idempotent. Twelve synthetic
partition/geometry cases cover 4Q/2KV and maximum 16Q/4KV, all bit-exact against
copied matching-batch native references; six also match compiled output/cache
bits. Eleven invalid-step cases, exhausted/closed-owner refusals, injected
compute failure and exceptional context cleanup pass. This stage took
**4.27 s**, **277,896 KiB** peak child RSS, zero swap growth.

An additional composition independently carries only **expected oracle bytes**
on the host. Actual runtime cache remains in one native allocation per
partition. A setter hook refuses any cache-buffer upload after construction;
only six source/query/mask/index payload uploads occur per step. All six
partitions pass exact per-step full-cache checks and matching-batch causal
score/probability/output bits, with zero future probability. Synthetic
composition took **6.62 s**, **341,892 KiB** RSS, zero swap growth.

The learned run reproduces the prior authenticated post-HF-rotary layer-3 Q/K
and V hashes and the same final F16 cache hashes. All six partitions pass
every persistent-cache and matching-batch check with **zero host cache restore
attempts**. The five earlier copied-cache metric dictionaries reproduce
exactly, including singleton max output gap **0.0003134012** versus monolithic
batching; persistence introduces no additional difference. The original fixed
score check remains false, with probability/output checks true. No tolerance,
code, scale, candidate or failed historical result was changed.

The learned stage took **9.01 s**, **1,486,160 KiB** peak child RSS, minimum
observed host-available RAM **16,422,150,144 bytes**, zero swap growth. Runtime/
frozen/profile/metadata/source entry-exit pins and installed trusted model-code
identity remain verified, under unchanged 60/300-second deadlines and resource
guards. The inherited original capture's broad native-rotary/cache-runtime
flag remains conservative; the narrower persistent-cache result is explicit
inside this diagnostic and is not full-runtime acceptance.

Retained/reread small-report SHA-256 values:

- `persistent-attention-selftest-result.json`:
  `616d10ce490ff9e297e9f4ee88fec76be1d61a2ea219b71f2a0aeaa6f7674adc`.
- `persistent-attention-composition-selftest-result.json`:
  `13224d19daa71b06411ef5544b24c975822434d14dbeda89f393bb7efcfb5365`.
- `persistent-attention-comparison.json`:
  `1c9bd9d4fc504810c0c20778477c0630c876eb6c794971a8660e7e88719baf08`.

Scripts, small reports and diagnostic executables remain outside Git; scratch
totals **1,727,201 bytes**, with no raw arrays/replay packet, GGUF or owned
compiler spill after empty-directory cleanup. No new failure occurred in these
three stages. Scoped syntax/Ruff, documentation/link/whitespace/editor checks
accompany the commit; the unchanged source gate at `c79b06e` was not rerun.
This is fixed learned-operand native-cache evidence, not chunkwise learned
projections/native rotary, gate/output projection, native prefix, loader cache,
eviction/concurrency/multistream, full-model parity, v2 P4/BitNet or quality.
Next: compose chunkwise native projections/norm/rotary with the verified
persistent core, retaining separate batching-propagation evidence and unchanged
full-conversion/resource/other-model boundaries.

## Persistent Cache Checkpoint (2026-10-16)

After `e0077c5`, added the bounded synthetic command
`--cached-attention-control PARTITION [carry|reset]` beside unchanged controls.
`PARTITION` is exactly `80`, `31+49`, `40+40`, `1+63+16`, `singletons` or
`64+singletons`; unknown partitions/modes, missing and extra arguments refuse
with code 5 and no stdout. Geometry is fixed: 80 total tokens, width 256,
four grouped query heads, two KV heads, 128 F16 slots, one CPU thread and
non-flash attention. No file, model, arbitrary index or native policy input.

A cache-only GGML context owns K and transposed V through one explicitly
allocated backend buffer. Each chunk allocates a **new** context/graph/graph
allocator, writes actual `ggml_set_rows` updates, then reads dependency-linked
views of those write outputs through QK, causal softmax and value contraction.
The K view preserves physical token/GQA-row strides; the V view preserves
channel-major transposition. After the chunk's allocator and graph context
are freed, the owning K/V addresses/buffers must be unchanged and both complete
cache buffers must exactly match expected F16 bytes. Source/query/mask/index
bytes remain unchanged. Normal carry uploads the two initial cache buffers
only once and performs **zero host cache restores** across graph lifetimes.
The JSON emits actual final readback bits, not just a status flag.

The existing compiled regression now checks all six partitions with independent
NumPy complete-cache bit oracles and per-chunk score references, including
nonzero unwritten future slots. Probability/output operation references retain
`rtol=2e-5, atol=2e-6`, with exactly zero future probabilities. The monolithic
causal scores/probabilities/outputs match the earlier prepopulated control
bit-for-bit. Other partitions retain the explicit F16-probability-delta
attribution rather than asserting false batching invariance. An intentional
`31+49 reset` performs two host restores and changes later outputs by more
than 0.01, while its first 31 outputs stay exact. Default/explicit carry and
eight malformed invocations are covered.

The first focused run failed in the **new Python oracle's KV-head broadcasting**
(`[1,2,1]` instead of `[2,1,1]`), after native readback succeeded. Corrected the
reference and explicitly modeled the cache's per-chunk future-slot state;
no tolerance was relaxed. The repaired check passed in **10.15 s**, then
**8.06 s** after reporting actual readback and covering explicit carry mode.
The mandatory default gate reports **553 passes / 88 optional skips in
22.76 s**, Ruff and whitespace clean. Scoped documentation/editor checks
accompany the commit. The opt-in compiled control uses pinned existing CPU/
base libraries; no cached source/library rebuild, CTest or sanitizer is claimed.
The final pin probe initially passed symlink names to the helper that refuses
symlinks; hashing their strictly resolved targets, as in preflight, verifies
both original library hashes and the clean pinned source revision.

This establishes bounded synthetic same-allocation cache ownership across
temporary graph destruction, not learned persistent-state replay, cache
eviction/concurrency/multistream, padded/centered/quantized caches, learned
projection/rotary batching, full-loader parity, whole-model BitNet, real-model
v2 P4 or quality. No weights, output head, generation, conversion, download,
candidate or refit is involved. Prior learned score failures and the complete-
model max logit gap **0.0971508** remain open. Next: synthetically validate a
similarly bounded persistent-cache helper against this compiled control before
any fixed learned-operand replay; do not reuse host-restored cache evidence as
persistent-state proof.

## KV Composition Checkpoint (2026-10-16)

After `bee4723`, a separate bounded ctypes helper reuses the actual native
F16 K/transposed-V `ggml_set_rows` layouts. It accepts 1-80 F32 updates,
1-4 KV heads, width 256, 128 F16 cache slots and unique I64 indices in 0-127.
Wrong shapes/types, duplicate/out-of-range slots, nonfinite data and updates
that overflow F16 refuse before any GGML call. Sources/indices, caller arrays
and every unwritten native destination byte are preserved. Twelve synthetic
cases pass independent exact-bit oracles, ten also match the compiled control;
four-head/last-slot-127 boundaries and twelve unsafe-input refusals pass.
This stage took **2.72 s**, peak child RSS **248,644 KiB**, zero swap growth.

A separate composition then writes actual cache updates per chunk, copies the
returned F16 cache arrays through the host, and reads them through the existing
native non-flash attention helper. It checks the complete expected cache after
every write and the final cache against one monolithic write. Causal scores,
all probabilities and outputs must be bit-exact against a **matching-batch**
read from the fully populated cache. Future probabilities must be zero even
when future cache bytes retain nonzero sentinels. The five partitions are
31+49, 40+40, 1+63+16, 80 singletons and 64+16 singletons. The synthetic
composition passes in **6.57 s**, **343,100 KiB** RSS, zero swap growth.

The learned diagnostic replays the identical authenticated layer-3 post-HF-
rotary Q/K and V hashes. All five partitions pass every cache-byte and
matching-batch causal-read check. Against the monolithic read, 31+49 and 40+40
are exact throughout; 1+63+16 has a causal score max gap **3.051758e-5** but
exact probabilities/outputs. Both singleton partitions retain the known output
max gap **0.0003134012**, not a write error or a new acceptance tolerance.
Resetting the first 31 cached tokens causes output max gap **3.054394**;
reading V in the wrong untransposed layout causes **9.992188**. Both faults
discriminate. The original fixed score-oracle check remains **false**;
probability/output checks remain true, without changing any tolerance.

This learned stage took **27.24 s**, peak child RSS **1,459,452 KiB**, minimum
observed host-available RAM **12,984,352,768 bytes**, zero swap growth. The
60/300-second deadlines, address-space and host RAM/swap/disk guards are
unchanged. The six runtime hashes, frozen codes/scales/manifest, profile/
metadata and source identities pass entry/exit checks. Only the installed
trusted implementation is used; there is no model-repository code execution,
new download, conversion, output head, generated token or quality experiment.

Retained/reread small-report SHA-256 values:

- `kv-write-kernel-selftest-result.json`:
  `594e2eeb1096b6eab0dcfc0f8aa2a3d95459c0d157dd591aa4727d2a831cff2a`.
- `cached-attention-core-selftest-result.json`:
  `c6f0e5f6dc66bb93f22701c408d0d559e6487750b1da6798831eda1edecb9ace`.
- `cached-attention-core-comparison.json`:
  `40f864ce4fbb52433752d478970885a41311b84f305bd87568a01809d8681053`.

All scripts, small reports and diagnostic executables remain outside Git.
Scratch totals **1,608,495 bytes**; arrays are memory-only, with no raw replay
packet, GGUF or owned compiler spill after empty-directory cleanup. Syntax/
Ruff, focused synthetic/learned checks and scoped documentation/link/whitespace
validation accompany this documentation checkpoint. The successful source gate
at `bee4723` is unchanged and was not rerun for these documentation-only edits.

This proves bounded native writes feeding matching causal reads with **copied
host cache carry between separate native graphs**, not persistent native cache
ownership, projection-batch invariance, connected learned projections/rotary,
eviction, concurrency, loader parity, real-model v2 P4, BitNet or quality.
The complete-model max logit gap **0.0971508** remains unresolved. Next: a
bounded synthetic same-allocation native cache lifetime/write/read control;
do not promote host-array carry into persistent-state evidence or retry the
stopped full conversion under unchanged resources.

## KV Write Checkpoint (2026-10-16)

After `9b224ac`, inspected the pinned `llama_kv_cache::cpy_k/cpy_v` ownership
path rather than treating prepopulated attention reads as cache updates. Added
`--kv-write-control TOKENS [FIRST_TOKEN] [contiguous|permuted]` beside unchanged
existing commands. It accepts 1/7/80 updates with strict integer logical windows
within 80, fixed width 256, two KV heads and 128 F16 slots, CPU1. All malformed,
negative, overflow/trailing, unsupported-mode and extra arguments refuse with
code 5 and no stdout. Destination slots are unique by construction: contiguous
absolute slots or multiplication by 37 modulo 128. This is a single-stream,
unpadded, noncentered, dense synthetic cache, not a model/profile policy.

The actual pinned `ggml_set_rows` operations write contiguous 512-element GQA
K rows and scalar-indexed **transposed V**. A V source element at token/channel
is written at physical index `channel * 128 + slot`, matching the non-flash
writer's flattened view. F32 update values are converted into initialized F16
destination buffers. Both graphs run and both complete buffers are read back.
Updates and I64 indices stay live through byte checks; the new report deliberately
says `source_and_indices_unchanged`, not that mutated destinations are unchanged.

The existing compiled regression exercises ten cases: 1/7/80 at zero, seven at
64 and one at 79, each with contiguous/permuted slots. Independent NumPy
oracles require **F16 bit-pattern-exact** readback for every destination element,
not just selected rows. Every unwritten slot must match its initialized sentinel
bytes; the permuted path reaches scattered cache addresses without overlap.
Treating V as untransposed fails an explicit fault control. Two default-argument
forms and ten malformed invocations are also checked. No numerical tolerance
is added or relaxed for the store: rounding and untouched-slot preservation are
exact. Earlier FWHT/QK/GDN/attention/IMRoPE cases still run in the same test.

The focused compiled test passed in **6.65 s**, then **6.94 s** after clarifying
the report's source-only immutability wording and including the directly used
standard utility header. There was no failed arithmetic check. The required
source gate reports **553 passes / 88 optional skips in 21.57 s**,
Ruff/whitespace clean. Scoped editor/documentation checks accompany the commit.
Only a disposable executable was compiled against existing pinned CPU/base
libraries; no cached source/library rebuild, CTest or sanitizer is claimed.

This control loads no weights, writes no model artifact/raw replay packet and
uses no output head, generated token, option score, conversion, download, refit,
candidate or quality data. It verifies bounded synthetic writes/readback,
**not** their connection to the attention graph, persistent native cache
continuation, mean-centered/padded/quantized/multistream caches, eviction,
concurrent access, full-loader parity or v2 real-model P4/BitNet acceptance.
The prior learned score failure and complete-model max logit gap remain open.

Next: reuse these verified layouts in a strictly bounded helper that refuses
overlapping/out-of-range indices before calling GGML; validate it synthetically
before any learned write/read composition. Preserve destination precision,
unwritten bytes and explicit source/head/slot layouts. Do not infer cache-write
behavior from the preceding prepopulated-read evidence or retry the stopped
full conversion under unchanged resources.

## Connected Attention Checkpoint (2026-10-16)

After user-pushed `cb3d461`, connected the previously verified primitives from
the **same trusted normalized layer-3 input**: three BF16 Q/K/V projections,
two F32 Q/K RMS operations with independent learned `weight+1` scales, two
native partial IMRoPE rotations and the non-flash F16 KV core. Query projection
is split into query/gate **per head**, not by cutting the whole vector in half.
The seven-token synthetic composition has exact independent rounded-operand
projection references; RMS, rotary and each attention operation pass their
unchanged oracles. The unused gate split is preserved. No primitive bounds or
earlier helpers were changed; the synthetic stage took **4.47 s**,
**346,456 KiB** peak child RSS, zero swap growth.

The learned composition reproduces the authenticated HF operand baseline and
uses eight native graph calls. Q/K/V projection maxima versus the trusted
Torch reference are **1.239777e-5 / 5.722046e-6 / 3.695488e-6**. Propagated
Q/K norm maxima are **9.536743e-6 / 6.675720e-6**, versus only **1.907349e-6**
each when normalizing the identical HF projection inputs. Core F16 query/key/
value crossings versus HF are **505 / 146 / 124**. The resulting core differs
from the same-HF-operands native core by score/probability/output maxima
**0.005935669 / 3.305078e-5 / 0.0006723404**; versus the original F32 HF core,
output max **0.003058434**, RMS **0.0001325029**. This run took **7.07 s**,
**1,461,148 KiB** RSS, zero swap growth. These are measured differences, not
accepted tolerances or proof of the complete-model discrepancy's cause.

A separately bounded native **sigmoid-times-feature** helper then passed small
and maximum 80-by-4096 synthetic geometry, extreme gates from -100 to 100,
seven invalid-input refusals, byte-preserved input lifetimes and a wrong-SiLU
fault. Its independent FP64 probability/product references passed unchanged
`rtol=2e-5, atol=2e-6`. That helper stage took **2.37 s**, **249,352 KiB** RSS.
The extended synthetic composition independently verifies gating and a BF16
output projection in **5.67 s**, **348,788 KiB** RSS. Both had zero swap growth.

The final learned diagnostic extends the same composition through that gate
and BF16 `o_proj`, ten native graph calls in all. It requires the preceding
connected-core SHA-256 to reproduce exactly before reporting anything. A
prepended hook captures the HF gated tensor **before** its BF16-RHS rounding,
so same-input controls compare the intended boundary rather than an already
rounded intermediate:

| Comparison | Max Absolute Gap | RMS |
| --- | --- | --- |
| Complete native attention output vs HF F32/BF16-RHS | 0.002011299 | 0.00002913632 |
| Propagated gated input vs HF gated input | 0.0005471706 | 0.000008805949 |
| Same-input native sigmoid vs Torch sigmoid | 0.00000005960464 | 0.000000001649476 |
| Same-input native sigmoid product vs HF gated input | 0.0000004768372 | 0.000000001189732 |
| Same-input native output projection vs Torch | 0.000001192093 | 0.00000001185630 |

There are **37,253** BF16 crossings at the output-projection input; final
outputs are not bit-exact. The final stage took **9.32 s**, peak child RSS
**1,460,692 KiB**, zero swap growth and minimum observed host-available memory
**16,983,191,552 bytes**. All synthetic/learned stages retain their 60/300-second
deadlines, 32-GiB address-space and unchanged host RAM/swap/disk guards.

Retained/reread small-report SHA-256 values:

- `connected-attention-selftest-result.json`:
  `bf0f42814d206dc84653849f01e9905e8f4b700d1ee843fb1eddb2d41d196a83`.
- `connected-attention-comparison.json`:
  `36e2b1d57b124f2b224851e2927cb3edafc370343de68f1e0a6a461e208252f8`.
- `attention-gate-selftest-result.json`:
  `586376bd9176b469d3d180e8f5396f2c26953811c440889b188ee07a279bed10`.
- `complete-attention-selftest-result.json`:
  `d241f89f92ef17a1342e9e8f3e9ac7c1ca92a8f1aa8b038dffb21643b2767fee`.
- `complete-attention-comparison.json`:
  `cc49ac4d5dc261a434ded37039b99fd84204a9fc6992e29bb61ca4cf59ecf9f8`.

The frozen codes/independent scales/manifest, six runtime hashes, profile/
metadata and source identities remain pinned on entry/exit. Helpers, failures,
small reports and diagnostic executables remain outside Git; arrays are
memory-only, and no replay packet, GGUF or owned compiler spill remains after
empty-directory cleanup. Only the expected installed Torch fallback warning
appeared. Syntax/Ruff, focused synthetic/learned checks and scoped documentation/
whitespace/editor checks ran. The unchanged successful `054190f` source gate
was not rerun for this documentation-only checkpoint; no cached library rebuild,
CTest or sanitizer is inferred.

This is a **dense attention sublayer diagnostic composition**, not a native
residual/MLP, native prefix, actual KV-write cache, full loader, model-quality
or whole-model BitNet path. It neither changes a production phase/precision
policy nor substitutes the frozen ternary candidate. There were zero generated
tokens, output-head/option scores, new conversions, downloads, refits, retained
candidates, held-out runs or other-profile execution. The original learned fixed
score failure and historical full-model max logit gap **0.0971508** remain open.

Next bounded boundary: inspect the pinned loader's KV-write operation and
physical layouts, then verify writes/readback on synthetic operands before any
learned cache composition. Do not describe the current prepopulated KV reads
as actual cache updates or retry the refused full conversion under unchanged
resource conditions.

## Learned Rotary Checkpoint (2026-10-16)

After user-pushed `054190f`, the bounded scratch `RotaryKernel` passed 12
synthetic cases, including 16 heads, ten bit-pattern-exact comparisons with a
new scratch copy of the committed C++ control and ten invalid-input refusals.
It binds the pinned public API with explicit F32/I32 types, CPU1 and deterministic
allocator/backend/context cleanup. Features, four position axes and unrotated
tails remain byte-exact; token/head/width bounds are 80/16/256, with positions
0-79 and the same fixed 64-dimensional IMRoPE parameters. The supervised helper
stage took **2.17 s**, peak RSS **245,868 KiB**, zero swap growth.

A separate learned diagnostic captured **pre-rotary** Q/K after the trusted HF
Q/K norms, reusing the same three-layer streamed prefix, 80 known prompt tokens
and 14/14/14/7 materialized parameters. It preserves the original HF outputs
and verifies that post-HF-rotary operand hashes reproduce the preceding
attention capture. The native wrapper rotates the same pre-rotary values;
original source/profile/frozen/six-module pins are checked again on entry/exit.
No projection, norm, prefix or loader was changed or claimed native by this run.

Both query/key native rotations pass the unchanged operation-oracle tolerance,
using the explicit F32 phase recurrence and independent rotation. Max gaps to
that oracle are **1.090135e-6 / 1.107157e-6**; with independently rounded F32
trigonometric coefficients they are **8.978659e-7 / 8.868561e-7**. The unrotated
tails are bit-pattern exact and rotated norms are preserved. Native versus
installed HF rotation is **not exact**: query/key maxima
**1.454353e-5 / 9.119511e-6**, with **36 / 12** changed F16 values. The distinct
HF/native phase-generation conventions are retained, not silently corrected.

Holding the native F16 attention core and V fixed isolates propagation:

| Rotary Operands | Score Max Gap | Probability Max Gap | Output Max Gap | Changed F16 Probabilities |
| --- | --- | --- | --- | --- |
| native Q, HF K | 0.002784729 | 0.00002735853 | 0.0002212524 | 24 |
| HF Q, native K | 0.002281189 | 0.00002360344 | 0.00005495548 | 11 |
| native Q and K | 0.002784729 | 0.00002741814 | 0.0002212524 | 35 |

All three outputs differ from the same-HF-rotary/native-core baseline.
Independent rounded-probability delta contractions explain those output changes
within the unchanged fixed tolerance, max residual **1.953449e-7**. This
isolates a same-input numerical propagation effect, not its contribution to
the unresolved complete-model selected-logit gap.

The learned stage took **38.66 s**, peak RSS **1,459,368 KiB**, zero swap growth;
minimum observed host-available RAM **14,563,262,464 bytes**. Existing 300-second,
32-GiB address-space and host-memory/swap/disk guards were unchanged. Only the
expected installed Torch fallback warning appeared. No output head, generated
token, option score, complete conversion, download, refit, new candidate,
held-out data or other-profile execution was used. No raw arrays/packets/GGUF
were written; only scripts, small scalar/hash reports and scratch executables
remain. Empty owned compiler-spill directories were removed after inspection.

Retained/reread report SHA-256 values:

- `rotary-kernel-selftest-result.json`:
  `4dc2d7d2f9ad19461e4318dc7eef2bfea7b367854b236c372dc0ccfa07e548ef`.
- `learned-rotary-comparison.json`:
  `17a3c867c75eb6184c86bb481724644352d3f7bfd6ddb9f7edd968b44bebbe90`.

The new report's `same_input_native_rotary_verified=true` means only the bounded
operation just checked; its inherited negative rotary/cache flag remains a
conservative full-runtime boundary, not a claim that KV writes were checked.
The original learned fixed score failure and probability-only Torch/native
delta failure remain recorded. No tolerance, payload, scale axis, phase policy
or cached library was modified. Syntax/Ruff, supervised synthetic/learned
checks and scoped documentation/whitespace/editor checks ran; the successful
`054190f` source gate was not rerun for this documentation-only increment.

Next bounded boundary: connect learned attention projections, Q/K norm, native
rotary and the verified core using the same trusted input, keeping gate/output
projection and full-loader/cache-write evidence explicit. This remains a
diagnostic composition, not v2 real-model P4, quality or whole-model BitNet
acceptance. Do not retry the refused complete conversion or promote padding.

## Partial Rotary Checkpoint (2026-10-16)

After `aca1646`, added `--rope-control TOKENS [FIRST_TOKEN] [text|axes]` to the
existing native smoke executable. Counts remain 1/7/80 and absolute query
windows stay within 80; malformed, negative, trailing, overflowing and extra
arguments refuse with code 5 and no stdout. Earlier commands and refusals are
unchanged. Geometry is fixed at four F32 heads, width 256 and 64 rotary
dimensions. The actual pinned `ggml_rope_multi` IMRoPE kernel uses sections
`[11,11,10,0]`, base 10,000,000, unit frequency/attention scales and no YaRN
extension. Four I32 position axes are bounded to 0-79; the first three are
either shared text positions or distinct deterministic axes. This tests a
synthetic configuration, not a profile-independent rotary policy.

The existing compiled regression now compares ten rotary cases to an independent
NumPy reference: 1/7/80 at zero, seven queries at 64 and one at 79, each in both
position modes. It also checks default arguments, ten malformed invocations,
rotated norms, wrong noninterleaved-axis discrimination, unchanged input/position
bytes and **bit-pattern-exact** unrotated 192-dimensional tails/offset slices.
The native input layout is token/head/column, not the attention control's
head/token/column layout. All graphs run CPU1 and retain input lifetimes.

Failed-check history: the first new 80-token oracle used ideal FP64 frequency
powers and failed at two elements, max violating error **3.094109e-6**. The
pinned CPU cache instead initializes each position phase in F32 and repeatedly
multiplies by an F32 `powf` frequency step. The corrected operation oracle
models that explicit phase schedule and independently computes rotation; it
also requires a measurable distinction from the ideal-phase reference.
The original `rtol=2e-5, atol=2e-6` stays unchanged. The same focused compiled
regression then passed in **6.94 s**, and in **6.72 s** after strengthening
tail/offset equality to compare F32 bit patterns, including signed zero.

Required source gate: **553 passes / 88 optional skips in 34.00 s**;
Ruff/whitespace clean. The focused opt-in test compiled only a disposable
executable against pinned cached CPU/base libraries; no cached library/source
rebuild, CTest, sanitizer, model weight load, conversion, download, generated
token, refit or quality evaluation was performed. Repository editor and scoped
documentation checks accompany the checkpoint. No model/profile execution
approval, original numerical tolerance, frozen payload or scale axis changed.

Next: validate a bounded ctypes rotary helper against this synthetic control
before using captured **pre-rotary** learned Q/K. Compare native and installed
HF phase conventions explicitly; synthetic kernel arithmetic does not prove
learned projection/norm/rotary agreement, KV writes, full-loader parity or
real-model v2 P4/BitNet acceptance. The failed learned score check and historical
full-model logit gap remain open.

## Learned Attention Checkpoint (2026-10-16)

After user-pushed `6848279`, verified a scratch ctypes attention helper against
the unchanged pinned public GGML CPU/base APIs. It preserves input storage
through byte checks and bounds heads to 16 query/4 KV, width 256, queries/window
to 80 and cache slots to 128. F32 Q, F16 K/V, grouped heads, causal scaled
softmax and token-major F32 output match the preceding non-flash control.
Five synthetic cases are bit-exact with a newly compiled scratch copy of the
committed C++ control; a sixth 16/4-head case matches independent FP64/F16
operation references. Nine invalid inputs refuse. The supervised helper stage
finished in **6.63 s**, peak child RSS **252,428 KiB**, zero swap growth.

Captured learned MiMo layer-3 Q/K/V **after installed HF Q/K norm and rotary**
on the same 80-token `inspect-before-answer` prompt. Selected embedding rows
and three serially materialized recurrent-prefix layers use the existing trusted
Torch F32/BF16-RHS/clamped recurrent-QK diagnostic convention. Exactly
14/14/14/7 parameters are materialized for layers 0/1/2/3; layer 3 loads only
input norm and attention. There is no output head, generation or option scoring.
The native core receives identical captured operands, with explicit F16 KV
storage and 48 causally masked zero cache slots. It does not compute learned
projections, normalization, rotary, cache writes or the prefix natively.

The original fixed `rtol=2e-5, atol=2e-6` **score check fails at five elements**;
its max FP64-dot gap is **5.430827e-5**. The worst tolerance ratio is a cancelling
dot near -0.00125: absolute-product sum **216.8815**, error **5.846610e-6**,
fixed bound **2.024951e-6**. Every score fits the separately derived F32
accumulation bound `gamma_256 * sum(abs(products))`, with unit roundoff `2^-24`
and `gamma_n=n*u/(1-n*u)`. This is a diagnostic bound, **not a replacement
tolerance or a passing original score check**. The pinned vector path performs
F32 accumulation; wrong tiled KV heads differ by **363.7505**. Independent
softmax/value operation checks pass, with maxima **9.899434e-8 / 4.768372e-7**.

Native output versus the original F32 HF core differs by **0.003058434**;
versus a Torch core with matching rounded Q/K/P/V operands by **0.0003657341**.
There are **113** changed F16-rounded probabilities in the latter comparison.
A separate diagnostic initially stopped after **8.52 s**: attributing the
whole output delta only to rounded-probability changes fails at one element,
residual **2.609566e-6**. Its script, stderr and failed stage record are retained.
The next diagnostic independently contracts both probability arrays in FP64
and accounts for **both** final value-contraction residuals. Native/Torch
value residual maxima are **4.768372e-7 / 2.390938e-6**; both meet their own
original elementwise checks and derived F32 bounds. The original probability-
only delta check remains false; no tolerance, guard or production path changed.

Five partitions read the same prepopulated learned F16 KV arrays at absolute
query positions; every future probability is zero and every input byte survives:

| Queries | Scores Exact | Probabilities/Outputs Exact | Output Max Gap | Changed F16 Probabilities |
| --- | --- | --- | --- | --- |
| `31+49` | yes | yes | 0 | 0 |
| `40+40` | yes | yes | 0 | 0 |
| `1+63+16` | no | yes | 0 | 0 |
| 80 singletons | no | no | 0.0003134012 | 33 |
| `64` then 16 singletons | no | no | 0.0003134012 | 9 |

Singleton-read score/probability maxima versus the full batch are
**6.103516e-5 / 9.536743e-7**. Rounded-probability contraction explains each
partition's output delta within the unchanged fixed tolerance; its max residual
is **4.768372e-7**. This is fixed-KV reading, not a learned KV-write/cache
continuation acceptance or an explanation of the historical full-model logit gap.

The initial learned capture took **22.91 s**, peak RSS **1,480,896 KiB**;
the complete precision/read-partition stage took **8.17 s**, **1,460,564 KiB**.
Both stayed within the unchanged 300-second, 32-GiB address-space, memory/swap
and disk guards, with zero swap growth. Source identities, metadata/profile,
installed model implementation, frozen manifest/codes/independent scales and
six runtime-module hashes match entry/exit pins. Torch reports the expected
installed fallback for unavailable optional fast kernels; nothing was installed.

Retained reports, reread and SHA-256 verified:

- `attention-kernel-selftest-result.json`:
  `df82e8dd8824d714b31cc9efcad92441250a639c7b978ae5e9b54c2fe0220d48`.
- `learned-attention-comparison.json`:
  `9e323d282011e6789d55174658fd22f657a656599cd4e6c14ab8f209e3ba446c`.
- `learned-attention-precision-comparison.json`:
  `385928a74935a32032f6ce13e6789cb5108c76785bfd364d9f00f4150b3c5866`.

Helpers, failures and small reports stay outside Git in the existing diagnostic
scratch root; raw arrays were memory-only, and no replay packet or GGUF was
created. Python syntax/Ruff and the synthetic compiled comparison ran; repository
source tests, CTest, sanitizers and cached-library rebuilds did not. The previous
553-pass/88-skip source gate is unchanged, not rerun for documentation. Scoped
documentation checks cover this checkpoint. No push, branch, subagent, refit,
new candidate, download, held-out quality run or other-profile execution occurred.

Next bounded boundary: independently verify native partial rotary on supplied
synthetic operands before combining it with learned projection/norm operands.
Keep fixed-KV reads, actual cache writes, native loader parity and real-model
v2 P4/BitNet/quality acceptance separate. The full-model selected-logit gap
**0.0971508** remains unresolved; no conversion retry or padding promotion.

## Nonflash Attention Checkpoint (2026-10-15)

After `897ce02`, added a bounded synthetic full-attention primitive beside the
existing FWHT/QK/GDN controls. The loader explicitly disables flash attention and
selects F16 K/V caches, so this control mirrors the pinned non-flash
`llama-graph.cpp` sequence: QK matmul with declared F32 precision, causal scaled
softmax, F16-value matmul, head permutation and contiguous output. On the pinned
CPU path, F16 matrix dot traits still round the right-hand operands: queries
for QK and softmax probabilities for V. Declared F32 accumulation is not proof
of unrounded F32 operands.

The additive `--attention-control TOKENS [FIRST_TOKEN]` command accepts only
1/7/80 queries, with a strict integer absolute position and total window within
80 tokens. Geometry is fixed at width 256, four query/two KV heads, 128 F16 cache
slots and one CPU thread; source/profile/candidate inputs are not used. Its head
broadcast is **grouped**, unlike the recurrent core's tiled V-head layout.
Outputs expose finite F32 scores, probabilities and token-major attention values;
all Q/K/V/mask bytes must remain unchanged. Invalid counts, windows, negative,
trailing and overflow positions refuse with code 5 and no report.

The existing compiled control now covers **83 attention cases**: 1/7/80 prefill,
a seven-query window at position 64, and every remaining singleton position
1-79. Independent FP64 dot/value references explicitly round the appropriate
operands to F16; softmax has its own independent stable causal reference.
Original `rtol=2e-5, atol=2e-6` arithmetic tolerances are unchanged. Future-slot
probabilities are exactly zero, and wrong-head/unmasked faults discriminate.
This tests prepopulated synthetic KV reads, not cache writes, rotary positions,
learned projection/norm/gate behavior or full-loader integration.

Failed-check history is retained, not rewritten as exact prefill/decode parity:

- A first absolute-position test incorrectly required direct output closeness
  across batch sizes. It failed at singleton position 41: score max gap
  **4.768372e-7**, probability gap **3.725290e-9**, one changed F16-rounded
  probability and output gap **1.049042e-5**. Independently contracting that
  rounded-probability delta with V explains the output delta **exactly**. The
  corrected oracle keeps each operation's arithmetic tolerances and checks this
  causal delta relation; no tolerance was raised, and exact batch parity is not
  claimed. The corrected focused control passed in **3.85 s**.
- Adding post-graph input-byte checks then refused at the first query. A
  disposable debug executable/GDB showed K storage reused after its last graph
  consumer; the pinned allocator frees/reuses owned non-output buffers. Marking
  these synthetic inputs as live graph outputs preserves their lifetime through
  verification. Byte checks remain, and the same focused control passed in
  **3.21 s** after this local lifetime repair. The debug executable was removed;
  no host/security setting, debugger configuration file or cached library changed.

Required final source gate: **553 passes / 88 optional skips in 16.95 s**,
Ruff/whitespace/editor clean. No model weights, generated tokens, output head,
option scores, conversion, download, refit or quality data were used by the new
synthetic control. No CTest, sanitizer, cached-library rebuild, BitNet full-model
dispatch or new-profile compatibility is inferred.

Reread/verified small `attention_control_evidence.json` SHA-256
`71a6892b5784f9b693231062c653b0212b6fa7758b4b80feae61ebd00376d466`.
It retains scalar precision effects, source/test/executable hashes and input
preservation, not raw attention arrays. Existing scratch totals 1,247,925 bytes;
no raw replay packets, GGUF or owned spill remains. Original source identities,
frozen candidate/manifest and all six runtime hashes still match preflight;
both frozen scale axes and earlier evidence remain unchanged. No push, branch
or subagents occurred.

Next bounded boundary: inspect the selected MiMo full-attention projection,
Q/K norm, rotary and KV-write contracts before a same-input learned-operand
comparison. Do not describe synthetic prepopulated KV reads as a working learned
cache or silently replace the unresolved full-model selected-logit comparison.

## Cached Layer Continuation Checkpoint (2026-10-15)

After `c435eb7`, extended the combined recurrent-cache composition through
post-attention RMS and the existing split MLP/residual, using exactly the same
14 layer-0 parameters and 80 known tokens. Gate/up retain two row blocks; down
retains three input blocks and the original F32 sum order. No cached library,
production policy, frozen codes/independent FP16 scales, intermediate casts,
bound, tolerance, source-profile approval or model-quality data was changed.

Before new partitions, every captured whole-layer native stage, output and final
state reproduces the preceding monolithic composition exactly. Original whole-
layer reference-comparison metrics, split-down controls and full GDN replay bytes
also match. Unmodified `31+49` and `40+40`, and diagnostic singleton-padded
`1+63+16`, then require **bit-exact** equality at every stage, residual and final
history/state; all pass. Unmodified `1+63+16` remains nonexact: layer max gap
**0.0003347397**, MLP max **0.0003361702**, final-state max **1.907349e-6**.
These maxima are not new accepted tolerances or proof of the historical
full-model discrepancy's cause.

The two independent `31+49` faults still discriminate through the complete
layer. Lost convolution history yields layer/MLP maxima **0.6954083 / 0.4530827**;
reset recurrent state yields **1.145927 / 1.461804**. Their final-state gaps
remain **1.678816 / 5.347192**. These are controlled faults on this fixture,
not representative quality judgments.

A second serial experiment exercises **80 one-token steps** and **64-token
prefill plus 16 one-token continuations**. With diagnostic singleton padding,
every stage of every step and final history/state are bit-exact against the
authenticated monolithic baseline. An unmodified 80-singleton run remains
different at all 80 layer outputs: max **0.0003347397**, final-state max
**7.629395e-6**. All inputs are the existing known tokens; **zero generated
tokens**, no output head or option scores. Padding is diagnostic only, not a
production cache fix or BitNet/full-loader/v2 real-model acceptance.

Extended the existing compiled synthetic GDN file regression, not a new test
file: added `40+40`, 80 singleton chunks and `64+16` singleton continuation to
the same nonzero-initialized raw-gate fixture. Original two partitions, malformed
packets, maximum geometries, exact per-token output/final-state assertions and
15-second child deadlines remain unchanged. Focused compiled control: **1 pass
in 3.47 s**. Required source gate: **553 passes / 88 optional skips in 25.59 s**;
Ruff/whitespace/editor clean. No failed syntax/lint/arithmetic run or tolerance
relaxation occurred; no cached-library rebuild, CTest or sanitizer is claimed.

Supervised real stages (seconds / peak process RSS KiB): cached full layer
**27.20 / 2,001,964**; token continuation **146.61 / 2,012,272**. Minimum host
available RAM was **20,247,728,128 / 20,209,770,496 bytes**, with zero swap
growth/process swaps under unchanged 300 s/32 GiB/2 GiB reserve/512 MiB swap
guards. Continuation makes 177 chunk GDN calls plus its baseline/capture calls;
this diagnostic subprocess overhead is not deployment performance evidence.

Verified/reread summaries: `cached_layer_evidence_summary.json` SHA-256
`98371669b643f7a5fa4f2c2bb9bde7094349e73f210faa818de0a495c07df73e`;
`cached_decode_evidence_summary.json` SHA-256
`c25216ecf0af969aeb23be6617e0386f50e85d483ea6ae0e1a7b24b9c1d7eb95`.
All scratch scripts/reports/two tools total 1,247,229 bytes, with no raw replay
packets, GGUF or owned spill. Source identities and frozen candidate/manifest/
six runtime hashes were rechecked before/after both stages and remain unchanged;
earlier evidence is preserved. No full conversion retry, download, refit,
retained candidate, held-out scoring, push, branch or subagents occurred.

The learned layer-0 recurrent composition now has exact diagnostic-padded cached
and token-continuation evidence on this fixture, not full-model runtime parity.
The next numerical boundary is full-attention/KV-cache composition with its own
head geometry, rotary positions and storage precision; inspect and bound that
path before any real probe. Do not reuse recurrent state-layout assumptions or
claim the historical selected-logit discrepancy, BitNet full-model dispatch,
quality, device performance or another profile's execution is resolved.

## Combined Recurrent Cache Checkpoint (2026-10-15)

After user-pushed `218d5c6`, combined actual native projection batching with
convolution-history and GDN-state carry on the same authenticated 80-token MiMo
layer-0 operands. A disposable helper composes input RMS, four input projections,
causal convolution/SiLU, raw-gate GDN, gated RMS, output projection and residual.
All original monolithic stages and final state are bit-exact, including original
full replay input/output hashes and reference-comparison metrics. This baseline
check precedes all new partitions; model-repository code is not executed.

Six cases preserve the distinction between unmodified and diagnostic-padded
execution. Every captured stage, residual, final three-column convolution history
and final recurrent state is **bit-exact** for unpadded `31+49` and `40+40`, and
diagnostic singleton-padded `1+63+16`. No intermediate casts or tolerances are
added. Unpadded `1+63+16` retains the preceding block gap **3.719330e-5** across
15 output tokens and final-state gap **1.907349e-6**; combined cache carry does
not erase the native singleton projection difference.

Separately losing only convolution history at token 31 changes block output by
**0.3304701** and final state by **1.678816**. Resetting only recurrent state
changes block output by **0.3158770** and final state by **5.347192**. Both affect
all 49 remaining output tokens. Thus the exact positive cache cases are
discriminating, not zero-state/no-history degeneracies. These are real source
operands on one synthetic prompt, not representative quality or full-loader/v2
real-model acceptance. Diagnostic singleton padding is still not promoted.

The single supervised stage completed in **13.58 s**, peak process RSS
**998,372 KiB**, minimum available host RAM **22,054,682,624 bytes**, zero host
swap growth/process swaps. Existing 300 s, 32 GiB address-space, 2 GiB reserve,
512 MiB swap-growth and disk guards remain unchanged; process measurements are
not device guarantees. Syntax/Ruff and strict baseline/cache assertions passed
without arithmetic retries, tolerance changes or cached-library rebuilds.

Verified/reread `cached_block_evidence_summary.json` SHA-256
`4974ca69c635e94816a0c54df6dc5ef26a8f0273d6ceff46f74ed4f004748339`.
Scratch scripts/reports/two tools total 1,012,077 bytes; raw replay files and owned
spill were removed, and no GGUF was created. Source identities, frozen candidate/
manifest and all six runtime hashes were rechecked before/after execution and
remain unchanged. Earlier evidence remains intact. No conversion retry, download,
refit, retained candidate, held-out scoring, push, branch or subagents occurred.
Repository changes are documentation only, validated with scoped link/content,
whitespace and editor checks; unchanged successful source/native gates are not
rerun. v1/v2 policies and the production execution path remain untouched.

Next bounded question: extend the same cached composition through post-attention
RMS and the existing split-MLP/residual path, keeping its matmul-order and padding
limits explicit. This is not full-loader cache, BitNet full-model dispatch,
resolution of the historical selected-logit discrepancy or model-quality evidence.

## Batch Dispatch Control Checkpoint (2026-10-15)

Continued the approved bounded cached-MiMo diagnostics after `d3e4f4a`, then
honored the user's pause. No repository source, native library, policy, frozen
candidate, independent FP16 scales, tolerance or resource guard was changed.
All new scripts/reports remain outside Git in the existing scratch root; this
checkpoint changes only documentation. No combined cached-block experiment
was started, and diagnostic padding is not promoted into production.

Localized the ten non-bit-exact `1+63+16` projection cases: every discrepancy
is confined to token 0, with tokens 1-79 bit-exact. Previously saved maximum
gaps and unequal-element counts reproduce exactly. BF16 output rounding removes
all QKV/Z/alpha/beta differences on these operands, but leaves two unequal
output-projection elements per native type: BF16-matrix max **3.051758e-5**,
F32-matrix max **6.103516e-5**. This is output rounding, not a change to the
native F32 execution path or justification for adding intermediate casts.

Applied `1+63+16` projection batching to the existing connected native recurrent
block, while keeping convolution/GDN monolithic. The old monolithic baseline
reproduces all comparison metrics and original replay input/output hashes.
QKV/Z/alpha/beta rounded inputs remain identical; their token-0 output changes
propagate to **all 80 core-output tokens**, with max output gap **3.576279e-7**
and final-state gap **1.907349e-6**. The output projection sees **16 unequal
BF16-rounded inputs**, giving block max gap **3.719330e-5** across 15 tokens.
These are measured precision effects, not a passed cache-parity tolerance.

A reversible diagnostic pads each singleton linear batch with one zero-input
row and discards that row afterward, leaving weights and all real operands
unchanged. The existing native SGEMM selector can then accept the two-row
batch. All **30 of 30** BF16/F32 projection/partition cases become bit-exact
against the unchanged 80-token baseline. The separately padded connected-block
control also has **bit-exact** projection outputs, rounded inputs, core outputs
and final state, including identical full GDN replay input/output hashes. This
supports the singleton reduction-path hypothesis on these exact inputs; it is
not instrumented dispatch proof, a production workaround, a general padding
contract or representative quality evidence. Original unpadded results remain
preserved. Convolution/core are still monolithic in the connected-block control,
so actual combined cache handoff remains a distinct untested question.

Supervised stages (seconds / peak process RSS KiB): token localization
**8.27 / 988,248**, unpadded block propagation **8.22 / 978,004**, padded
projections **13.29 / 991,412**, padded block **14.13 / 993,724**. All observed
zero host swap growth/process swaps and minimum available RAM above 16.2 GiB
under unchanged guards. Scratch syntax/Ruff checks passed before execution;
strict padding and baseline assertions passed without tolerance relaxation or
arithmetic retries. None of these host observations is a target-device guarantee.

Verified/reread small summaries: `batch_propagation_evidence_summary.json`
SHA-256 `6a22dc8046f0d5ea3ab3e53e839655857e7817f2a0bdc8f176e254ae6e6949a9`;
`padding_control_evidence_summary.json` SHA-256
`2d25fbf20a5dc5e322d728f98d73c77c674608eb0d8284d32ccab5df52c0c493`.
The latter save was interrupted with unknown outcome; checked that it was absent
and spill remained before completing only the save/cleanup after the pause.
All scratch files total 940,875 bytes; no GGUF, raw packets or spill remains.
Source identities, candidate/manifest and six runtime hashes still match the
original preflight; earlier evidence is preserved. Documentation link/content,
whitespace and editor checks apply; the unchanged successful source/native gates
are not rerun for reassurance. No download, full conversion retry, refit, new
candidate, held-out scoring, push, branch or subagents occurred.

After an explicit resume, the next bounded question is to combine native
projection batching with convolution-history and recurrent-state carry on the
same authenticated operands, keeping the diagnostic padding distinction explicit.
Full-loader/v2 real-model parity, the historical selected-logit discrepancy,
BitNet full-model execution, model quality and device performance remain open.

## Coupled Cache Checkpoint (2026-10-15)

Continued after `5f54ba7` without repository source changes or another complete
conversion. This increment closes the preceding fixed-projected-operand cache
question, but records a separate projection-batch precision limitation. v1/v2
policies, frozen codes/independent FP16 scales, numerical tolerances, token/tensor
caps, cached libraries and user settings remain unchanged.

A disposable convolution helper carries exactly three projected-input columns
for the four-tap causal convolution. Each unchanged native call includes that
history and therefore admits at most 125 new tokens under the existing 128-token
bound; 126 is refused. A 32-channel synthetic fixture with nonzero initial
history matches independent Torch raw convolution exactly. Raw convolution,
SiLU output and final history are bit-exact for `31+49`, `1+63+16` and `40+40`;
the lost-history fault is discriminating before real inputs are used.

Captured the existing real 80-token F32/BF16-RHS layer-0 convolution operands
and coupled native convolution/SiLU to the pinned raw-gate GDN file control.
The real geometry is 8,192 convolution channels, width 128, 16 key/32 value
heads, zero initial history and the captured value-by-key recurrent state.
Grouped source V heads are explicitly converted to native tiled order; raw
alpha/beta operands and A/bias remain unchanged. All three partitions require
bit-exact raw convolution, SiLU, core output, final recurrent state and final
three-column history against the same native-operation baseline. All pass.
Losing only convolution history changes core output by up to **0.4066229**;
resetting only recurrent state changes it by **1.3415415**. Native-versus-Torch
SiLU produces a separately measured **4.768372e-7** core-output gap; it is not
silently counted as bit-exact original-packet agreement. Original full-packet,
output and executable hashes still match the preceding retained report.

A separate diagnostic splits each of the five real recurrent projections in
both BF16 and F32 native matrix types. **20 of 30** projection/partition cases
are bit-exact: every `31+49` and `40+40` case. All ten `1+63+16` cases differ,
with maximum batch-shape gap **7.629395e-6** and 21 to 6,021 unequal elements
per case. Monolithic projection comparisons exactly match the earlier report.
The pinned `ggml-cpu.c` calls the SGEMM selector, and
`llamafile/sgemm.cpp` refuses `n < 2` on this platform. This is consistent with
a one-token reduction-path difference, not an instrumented dispatch attribution
or a correction. No cached source/build, tolerance or guard was changed to hide
it. Coupled cache equality on fixed projected operands therefore does **not**
establish bit-exact projection batching, whole-layer or full-loader caching.

Supervised stages (seconds / peak process RSS KiB): synthetic convolution cache
**1.92 / 239,100**, real coupled cache **7.17 / 731,276**, projection batching
**10.88 / 992,112**. All observed zero host swap growth/process swaps, available
RAM above 16.6 GiB, and unchanged deadline/address-space/reserve/swap guards.
These are host observations, not target-device guarantees. Scratch syntax/Ruff
checks passed before each new real stage; no failed arithmetic assertion,
guard relaxation or retry occurred. The projection non-equality is a retained
diagnostic result, not a passed exact-parity assertion.

Small scripts/reports/two tools total 798,741 bytes in the existing scratch root.
Both new summaries were reread and verified: `coupled_cache_evidence_summary.json`
SHA-256 `16dedded0ce9c81c31bf31e707e3791554b8b6d742122247903536c8b4c04201`;
`projection_batch_evidence_summary.json` SHA-256
`66eedac4f5f9ebfcebf1c2aa5cf9b62d1b19c801d131031b51575881d176fc22`.
No GGUF, raw packets, spill or experiment process remains. Source identities,
candidate/manifest and six runtime hashes, both pinned tracked source worktrees
and all earlier reports are unchanged. No push, branch, download, refit, new
candidate, held-out scoring, subagent, BitNet/full-model acceptance or other-model
execution is inferred. This repository checkpoint changes documentation only;
scoped content/link/whitespace checks apply, not a repeated unchanged source gate.

Next bounded question: locate projection mismatches within the token partitions
and measure their propagation through a coupled recurrent block before making
any broader cache claim. Exact full-loader intermediate parity and the historical
full-model selected-logit discrepancy remain unresolved; the swap-refused complete
conversion is not retried under unchanged conditions.

## Whole Layer Checkpoint (2026-10-15)

After user-pushed `df1da32`, continued the approved cached-MiMo investigation.
Starting `main` matched `origin/main`, with only untouched user settings. No
full conversion retry, download, refit, retained candidate, held-out scoring or
other-profile execution followed. The source change is only an extension of
the existing compiled replay test; real diagnostics remain disposable scripts.

Added explicit full-MLP composition bounds: dimensions <=12,288 and at most
50,331,648 weights, using unchanged native calls <=8,192 rows/4,096 inputs.
Up/gate use two row blocks; down uses three input blocks and F32 addition.
Native GGML SiLU/product is separately bounded. Full-size sparse fixtures
have bit-exact up/down answers and reject oversized shapes; activation checks
pass before real inputs. This is not the unblocked full-loader matmul order.
The actual same-input down control measures native/Torch split and Torch
split/full maximum gaps of **4.768372e-7** each, not enough to explain the
larger propagated error.

Loaded exactly 14 layer-0 parameters and the existing prompt's embedding rows.
The complete dense composition propagates input norm, recurrent block,
residual, post-attention norm and gated MLP/residual through 18 native primitive
calls. Tensor routing, source A conversion and residual/split addition use
Torch/NumPy; no output head, scores, frozen substitution, full model or v2
policy is involved. Captures compare BF16, F32/BF16-RHS and clamped-Q/K F32
references from pinned Torch/Transformers. Metadata is reauthenticated, source
file identities checked before/after, and all earlier scripts/reports preserved.

| Reference | Recurrent Output Max Gap | MLP Output Max Gap | Whole-Layer Max Gap | Whole-Layer RMS Gap |
| --- | --- | --- | --- | --- |
| BF16 | 0.02888966 | 0.01967311 | 0.04315233 | 0.0003198404 |
| F32/BF16-RHS | 0.0001754761 | 0.0003361702 | 0.0003347397 | 1.030825e-5 |
| F32/BF16-RHS, clamped Q/K | 9.346008e-5 | 0.0003361702 | 0.0003347397 | 7.723088e-6 |

Two separately labeled causal controls preserve the negative/remaining results:

- Replacing only the two ordinary native RMS outputs with same-input Torch F32
  expressions lowers clamped-reference whole-layer RMS gap to **5.226836e-6**,
  but leaves maximum gap **0.0003347397** unchanged. Post-attention maximum gap
  actually grows to **0.0004983544**. Thus ordinary RMS alone does not explain
  the maximum. Projection inputs match at alpha/beta/QKV/Z after this control,
  but 123 recurrent-output and 58,047 MLP-down BF16-rounded inputs still differ
  against the clamped reference; upstream rounding discontinuities remain.
- Pinning only the MLP input to the clamped reference gives exactly matching
  rounded gate/up operands. Its intrinsic MLP max/RMS gaps fall to
  **9.179115e-6 / 2.779796e-7**, with 412 remaining rounded down-input
  differences. Whole-layer maximum is then **9.346008e-5**, preserving the
  uncorrected recurrent-prefix difference. This isolates propagation, not a
  runtime fix or accepted model tolerance. Neither diagnostic is promoted.

Extended the pinned compiled synthetic GDN file test with `31+49` and
`1+63+16` chunks, carrying returned F32 value-by-key state. Every output and
final-state bit equals the original initialized 80-token raw-gate run. Real
captured 16-key/32-value-head operands additionally pass `40+40`; all three
partitions are **bit-exact**. Full captured input/output/executable hashes
match the preceding diagnostic. Resetting state at the `31+49` boundary changes
core output by up to **1.3415413**, so the carry test is discriminating. This is
GDN state handoff on preconvolved operands, not convolution history, projection
batching, full-layer/loader caches, BitNet dispatch or representative quality.

Supervised stages (seconds / peak process RSS KiB): sparse MLP control
**3.87 / 491,784**; whole layer **10.83 / 1,588,516**; RMS control
**9.73 / 1,640,956**; MLP control **11.33 / 1,605,940**; real core cache
**6.77 / 727,632**. All observed zero host swap growth/process swaps under the
unchanged 300 s/32 GiB/2 GiB reserve/512 MiB swap guards; available RAM stayed
above 15 GiB. No device or aggregate-process peak guarantee is inferred.

Compiled focused control: **1 pass** in 3.82 s. Mandatory source gate:
**553 passes, 88 optional skips** in 28.97 s; Ruff/whitespace/editor clean.
Editor test discovery returned no tests, so the exact opt-in compiled test ran
directly. No failed syntax/lint/arithmetic check, tolerance relaxation, cached
library rebuild, CTest or sanitizer occurred in this increment.

Reports/scripts/two tools total 676,605 bytes outside Git in the existing
scratch root. Combined `whole_layer_evidence_summary.json` was reread and
verified; SHA-256
`6ff7c8fb72ccc25bc6066e03aeded40ee542aa0e55b9d063ce8b3fa5c89074f0`.
No GGUF, raw replay input/output or compiler spill remains. Candidate/manifest
and all six runtime hashes, source identities and both pinned tracked source
worktrees are unchanged. No push, branch, subagents or user-settings changes.

Next bounded check: include convolution-history handoff with core state, before
any full-layer cached claim. Exact full-loader intermediate parity and the
historical full-model selected-logit discrepancy remain open; do not retry the
refused conversion under unchanged conditions or silently replace its candidate.

## Layer Zero Stage Checkpoint (2026-10-15)

After `75392c1`, continued the approved cached-MiMo numerical investigation
without another complete conversion or repository source changes. Disposable
ctypes probes bind the actual pinned public GGML API with explicit signatures,
bounded dimensions, one native CPU thread and deterministic graph/buffer/backend
cleanup. They reuse the authenticated source and exact 80-token prompt below;
no model-repository code, MLP, output head, score or quality data is executed.
Source and six runtime/candidate identities are rechecked, not assumed from
successful arithmetic.

Synthetic projection control passed in **1.62 s**, peak RSS **235,556 KiB**:
BF16 exact and F32 max gap 4.627509e-8 versus independent FP64 dot products;
a separate wrong-unrounded-RHS result must differ. Synthetic RMS/gated-RMS,
four-tap convolution and SiLU control passed in **2.02 s**, **239,080 KiB**:
convolution bit-exact, RMS max gap 2.339752e-7. These self-tests ran before
real operands. Source/library revision, public declarations, enum sizes and
returned byte counts were checked rather than guessed.

Five real layer-0 recurrent projections compare identical BF16-rounded inputs
through BF16 and F32 native matrices. Both native types give the listed same
maximum gap against Torch F32; rounding native output to BF16 still differs
from Torch BF16 for some elements, so near-equal accumulation is not bit parity.

| Projection | Native/Torch F32 Max Gap | RHS Rounding Max Effect | Rounded Native/Torch BF16 Unequal Elements |
| --- | --- | --- | --- |
| Alpha | 5.722046e-6 | 0.01130962 | 0 |
| Beta | 1.430511e-6 | 0.003746033 | 0 |
| QKV | 1.525879e-5 | 0.03787613 | 84 |
| Z | 1.335144e-5 | 0.02638054 | 36 |
| Output | 4.768372e-6 | 0.008557796 | 96 |

Stateless probes compare input RMS norm (source weight +1), recurrent gated
RMS norm (no +1) and zero-history four-tap convolution in both precision modes.
Native raw convolution is **bit-exact against Torch F32 on identical inputs**
in both modes; SiLU max gap is 9.536743e-7 in BF16-input mode and 4.768372e-7
in F32/BF16-input mode. BF16 raw convolution differs from native F32 by up to
0.1159668, but rounding native raw output to BF16 restores exact convolution
output equality. This does not establish equal downstream SiLU rounding.
Input norm differs from the F32 expression by up to 3.814697e-6; gated norm
by 7.152557e-7. Against original BF16 computation, gaps are much larger
(input norm 0.1218643, gated norm 0.01136088); final BF16 rounding alone does
not reproduce every intermediate gated-norm cast. These are measured precision
effects, not reasons to relax a complete-model acceptance threshold.

Finally connected native input norm, five BF16 matrix projections, convolution/
SiLU, raw-gate GDN and gated norm, each exactly once, using Torch only for
tensor routing and source-derived A conversion. Outputs from each native stage
feed the next; explicit grouped/tiled head transformations surround GDN. This
is an exploratory recurrent-block composition, **not** the full loader graph,
v2/P4, complete layer, cache reuse or BitNet path. Against BF16 Torch, block
output max/RMS gaps are **0.02888966 / 0.000212030**; against the F32/BF16-input
reference, **0.0001754761 / 7.744376e-7**. After residual addition, maximum gaps
are 0.03126431 and 0.0001754761 respectively. The remaining nonzero difference
is preserved, not hidden by promotion or a new tolerance.

Real projections: **6.01 s / 981,876 KiB** peak process RSS; both stateless
modes: **5.17 s / 747,072 KiB**; connected block: **6.22 s / 923,804 KiB**.
All five stages observed zero host swap growth/process swaps under unchanged
guards; available host RAM stayed above 16 GiB. No target-device performance
or host-wide peak guarantee is claimed. Script lint initially caught deletion
of a decoder name retained by callbacks; clearing its cell after execution
fixed that before the connected real-input run. All script syntax/Ruff checks
and five supervised stages passed; no failed arithmetic run was retried.

The original `evidence_summary.json` and scripts remain unchanged. Additional
`stage_evidence_summary.json` was copied/reread and verified with SHA-256
`88c91a9ff0f0b3b3b3b08369e8e8e6b4c413fa04d1903185479dd9a8cb1cd7a1`.
The scratch root now retains 540,123 bytes of small reports/scripts/two tools,
not raw activations, model artifacts or compiler spill. Block replay files
were removed in child and parent finally paths; no experiment process remains.
Source identities, frozen payload/manifest and all six runtime hashes match
the earlier baseline. No library rebuild, download, refit, held-out run, push,
branch, subagent or user-settings change followed. Repository source gate at
`75392c1` is unchanged, not rerun for reassurance; this checkpoint changes only
the scoped handover documentation. Content/link/anchor/fence/table and whitespace
checks passed: eight files, 316 local links, 41 tables.

Next bounded diagnostic would add explicit MLP geometry/budgets and its own
controls before attempting a propagated whole layer. Full-loader intermediate
agreement and the historical full-model logit discrepancy remain open; the
one refused conversion is not automatically retried.

## Real Operand Replay Checkpoint (2026-10-15)

After `fe17c9e`, authenticated all four cached MiMo source shards against the
profile, metadata/tokenizer, the sole frozen candidate, six runtime modules and
the exact historical 80-token fixture prompt. The supervisor's output-size
sanity range was corrected before execution to include the documented
17,909,729,280-byte dense payload (16.68 GiB); no executed acceptance gate or
numerical tolerance was relaxed. Offline rendering passed in 8.98 s.

The single approved dense BF16 conversion was stopped after **71.05 s** because
host swap increased **545,566,720 bytes**, exceeding the declared 512 MiB limit.
Minimum observed available RAM was 11,470,553,088 bytes. The converter reached
`output.weight` but created no GGUF or tensor spill. Its entire process group
was terminated and no converter remained. Interrupted `/usr/bin/time` output
was empty: the supervisor's 10,236 KiB wrapper RSS is **not** converter peak RSS.
There was no retry, guard relaxation, download or candidate change.

Added a separately bounded `--gdn-file-control INPUT OUTPUT` to the existing
native graph smoke executable, not the v1/v2 model factories or prefill hook.
It replays finite raw recurrent operands through actual pinned GGML L2/GDN
operations, width 128, <=128 tokens, <=32 Q/K and value heads, divisible head
counts and negative decay multipliers. Input is <9 MiB; output <=4 MiB, created
exclusively after successful finite computation. Existing synthetic commands
are unchanged. The old 80-token nonzero-state/raw-gate output and state replay
bit-for-bit; malformed headers, dimensions, short/nonfinite/trailing operands
and existing output refuse without reports. Zero-input 16/32 and maximum
32/32 heads at 128 tokens also pass.

Smaller serial diagnostics loaded only the prompt's embedding rows, layer-0
input norm and recurrent block from authenticated source. No full embedding,
MLP, output head, complete model or frozen substitution was loaded. BF16 and
F32-with-BF16-rounded-linear-input captures preserve the exact prompt. The
native replay receives identical captured operands; HF grouped V heads/gates
are explicitly tiled, and returned state is restored to grouped order and
value-by-key orientation for comparison. Raw gates use F32 sigmoid/softplus.
These are core-operation observations, not matching whole-layer captures.

| Reference Against Native Core | BF16 Output Max Gap | BF16 State Max Gap | F32/BF16-Input Output Max Gap | F32/BF16-Input State Max Gap |
| --- | --- | --- | --- | --- |
| Original HF chunk | 0.01220393 | 0.03936386 | 2.622604e-6 | 4.553795e-5 |
| F32 recurrent, original captured beta | 9.807348e-4 | 0.02059507 | 2.682209e-6 | 4.625320e-5 |
| F32 gates, additive Q/K norm | 2.548099e-6 | 4.601479e-5 | 2.682209e-6 | 4.625320e-5 |
| F32 gates, native clamped Q/K norm | 4.768372e-7 | 5.722046e-6 | 4.768372e-7 | 2.384186e-6 |

BF16 sigmoid rounding alone changes beta by up to 0.0019530654; that difference
is zero in the F32 diagnostic. Matching gate precision and normalization
substantially narrows this core's gap. This does **not** attribute or resolve
the historical full-model selected-logit discrepancy: projections, convolution,
gated norm, residual/MLP, attention/cache and accumulated rounding remain
outside the identical-operand replay. No model-parity tolerance, quality result,
BitNet model dispatch or other-profile runtime acceptance is inferred.

BF16 diagnostic: **6.37 s**, peak process RSS **528,352 KiB**; F32/BF16-input:
**5.02 s**, **729,284 KiB**. Both observed zero host swap growth/process swaps;
host available RAM stayed above 17 GiB. These are host/process observations,
not target-device guarantees. Each retained report was reread and raw-file
hashes independently verified before parent-finally cleanup. No GGUF, raw
operands/output or compiler spill remains. Small reports, scripts and two
diagnostic executables remain outside Git at
`$cache/native/mimo-layer0-20261015`. Combined `evidence_summary.json` SHA-256:
`a55ab5fd360567b64c14d8f461e41903e2e77c48a4d25c157213c475cb0da365`.
Candidate and all six runtime hashes, source file identities and both pinned
tracked source worktrees are unchanged.

Focused compiled control: **1 pass** in 3.28 s; Ruff, Pylance syntax and editor
diagnostics clean. Initial test collection failed from an over-indented new
block, repaired before native execution; its focused rerun passed in 3.05 s,
then maximum-geometry/header cases passed in 3.28 s. The disposable capture
script first failed Ruff for one unused import, removed before execution.
Required source gate: **553 passes, 88 optional skips** in 26.50 s; Ruff and
whitespace clean. Documentation: eight files, 314 local links, 40 tables clean.
No cached-library rebuild, CTest, sanitizer, refit, held-out scoring, push,
branch or subagents.
Untracked user `.vscode/` remains untouched.

Next useful bounded slice: compare real layer-0 projection/convolution stages
with matching native arithmetic and precision, without a complete conversion.
Do not retry the full conversion under unchanged conditions or weaken its swap
guard. Full native traces and real-profile v2/P4 acceptance remain unverified;
Defiant Fable remains offline-only.

## Prompt Trace Checkpoint (2026-10-15)

After user-pushed `53074a8` and expanded approval, added
`--prefill-text-trace-control MODEL PROMPT TRACE_DIRECTORY` to the native CLI.
It composes existing prompt/token/option validation and bounded finite trace
capture without changing v1 factories or the synthetic v2 prefill hook. All
four tiny dense recurrent/attention combinations require exact traced/untraced
score reports, complete eight-stage geometry, finite bytes and existing-trace
destination refusal for two-token and 128-token prompts.

Verification: nonzero recurrent+attention dense **1 pass** in 1.37 s; remaining
three dense cases **3 passes** in 3.06 s. Direct compiled early/late init controls
pass; gate **553 default passes, 88 optional skips** in 22.06 s; Ruff and
whitespace clean. The editor CMake integration had no targets and failed to
configure with no diagnostics. Fallback compiled only the changed CLI with
clang++-18 against cached libraries in disposable scratch; no cached runtime
library/source overwrite or fresh complete-runtime build/CTest is claimed.

Approved experiment plan (not yet executed at this checkpoint): reuse cached
MiMo `2367e865` and pinned sources/environments; authenticate inputs and preserve
the sole frozen candidate. One dense BF16 text-only conversion, no vision/MTP,
no model-repository code, no download/refit/quality data. Stages run serially:
conversion <=1200 s, native <=900 s, each streamed diagnostic <=300 s, one
80-token fixture and zero generated tokens. Scratch cap 48 GiB; virtual address
cap 32 GiB; stop if host available RAM falls below 2 GiB or swap grows by more
than 512 MiB from the stage baseline. Initial host observation: about 17 GiB
available RAM, 342 MiB swap used, 510 GiB free disk. Historical conversion/native
RSS was about 10/17 GiB, not a current resource guarantee. On any refusal stop
and diagnose, not automatically rerun. Preserve a small report before deleting
all temporary GGUF/spill/raw traces. No model-parity tolerance is being accepted.
Scratch root: `$cache/native/mimo-layer0-20261015`; cached libraries unchanged.

## Dynamic Gate Roles Checkpoint (2026-10-15)

After `ec9a06f`, added only the remaining six dynamic role parameters. All
seven frozen roles now pass with synthetic input-dependent alpha/beta/bias
gates and nonzero recurrent state; Q/K/V/O also include nonzero full causal
attention. Both hidden1024/head32 and output hidden128/head64 geometries retain
complete physical two-shard source/converted parity, selected metadata/runtime
pins, exact full/peer/max128 references and unchanged `2e-5` native tolerances.

The dynamic set verifies **49 successful workflows, 42 post-prefill exit and
28 pre-yield refusals**, separately from the identical-count fixed-gate set and
the original zero-recurrent set. Combined new recurrent coverage is **98
successful workflows, 84 exit and 56 pre-yield refusals**. Every dynamic role
rejects omitted recurrence, per-token state reset, lost convolution history and
fixed gates through independent oracle comparisons; head/token gate variation
is explicit. Typed zero-generation scores, genuine BitNet dispatch/repack,
parallel/chunk/reset/reorder/maximum/recovery and cleanup checks remain active.

Verification: dynamic attention-output **1 pass** in 101.86 s, peak RSS
255,676 KiB; dynamic FFN-up/gate **2 passes** in 237.45 s, peak RSS 311,056 KiB;
dynamic Q/K/V **3 passes** in 312.06 s, peak RSS 316,800 KiB; swaps zero.
Dynamic-down and fixed/zero representatives were verified in the immediately
preceding checkpoint and are unchanged, not rerun for reassurance. Gate
**553 default passes, 88 optional skips** in 36.04 s; Ruff, whitespace and
diagnostics clean. No failed checks in this parameter-only increment. Earlier
collection failure and repair remain recorded in the probe checkpoint below.

No production/native changes, rebuild/CTest/sanitizer, real input reads/hashes/
conversion/execution/quality, downloads/refits/new candidates, push/branch or
subagents. Scratch removed; frozen codes/independent FP16 scales, v1, hook/
byte/operand/tensor/token/deadline bounds, caches and user settings unchanged.
This completes the bounded zero-initial-state recurrent source integration,
not arbitrary learned-model gates, injected initialized-state integration,
issuer/frozen equivalence, atomic storage, representative quality or device
performance. The next real-profile comparison still requires separate approval
and the existing [comparison preparation](#real-comparison-preparation-2026-10-12).
The historical real MiMo discrepancy and other-model execution remain open.

## Dynamic Gate Probe Checkpoint (2026-10-15)

After `64c497d`, added one `dynamic-recurrent-down` synthetic case. Sparse
per-head alpha/beta BF16 projections and nonuniform F32 dt biases drive sigmoid
beta and `exp(-softplus(alpha + bias))` decay with A fixed at -1. Verified that
expression in the pinned native Qwen3.5 source before editing the independent
NumPy oracle. Source alpha/beta rows and bias heads invert literal toy order
`[0, 2, 1, 3]`; complete source/converted parity remains an independent gate.

Both gates must vary across heads >0.01 and across changing token inputs >0.01;
the repeated-token maximum does not incorrectly require time-varying gates.
Fixed 0.5 gates now join omitted recurrence/reset state/reset history as a fourth
faulty oracle that must differ outside unchanged `2e-5` tolerance. Both writers
must exactly reproduce full/peer/max128 and all fault deltas. The new case
passes seven two-shard source-gated workflows, six post-prefill and four
pre-yield refusals with existing typed zero-generation/dispatch/repack/cleanup
requirements. No native or production code changes.

Verification: new dynamic probe **1 pass** in 94.96 s, peak RSS 311,240 KiB;
fixed-gate recurrent-down and original zero-recurrent attention-output **2
passes** in 164.46 s, peak RSS 311,228 KiB; swaps zero. Gate **553 default
passes, 82 optional skips** in 54.02 s; Ruff, whitespace and diagnostics clean.
Initial collection-only probe failed because a new decorator parameter had no
function argument. Replaced it with exactly one intended target (not a Cartesian
product); same collection check then passed with 17 selected cases before any
native run. No arithmetic/native failure or relaxed assertion/budget/deadline.

No rebuild/CTest/sanitizer, real input reads/hashes/conversion/execution/quality,
downloads/refits/new candidates, push/branch or subagents. Scratch removed;
frozen codes/independent scales, v1, bounds, caches and settings unchanged.
Input-dependent synthetic gates are not learned-model/initialized-state quality,
issuer/frozen equivalence, atomic storage or target performance. Real MiMo
discrepancy and all real-profile approvals remain open. Next bounded slice:
the other six dynamic frozen roles, with the same source/oracle/exit gates.

## Recurrent Source Roles Checkpoint (2026-10-15)

After `2e7b990`, extended only the recurrent case parameters to all seven
eligible frozen FFN/full-attention roles. Attention-output's hidden128/head64
geometry passes alongside hidden1024/head32 cases. Nonzero recurrent residuals
feed full causal attention for Q/K/V/O; FFN cases retain nonzero delta-state
before their selected BitNet projection. No oracle, hook, bounds, tolerance,
frozen payload, source transform or baseline-control changes in this increment.

The extra recurrent set now covers **49 successful two-shard source-gated
workflows, 42 post-prefill exit refusals and 28 pre-yield source refusals**,
separate from the original zero-recurrent seven-role set. For every recurrent
role, omitted recurrence, per-token state reset and lost convolution history
must change logits beyond `2e-5`; both writers exactly reproduce full/peer/
max128 references and diagnostic deltas. All existing typed zero-generation,
BitNet dispatch/repack, parallel/chunk/reset/reorder/maximum/recovery, complete
source/converted/selected-metadata/runtime pins and cleanup checks remain active.

Verification: attention-output **1 pass** in 55.77 s, peak RSS 255,956 KiB;
FFN-up/gate **2 passes** in 273.37 s, peak RSS 311,284 KiB; Q/K/V **3 passes**
in 349.66 s, peak RSS 316,908 KiB; swaps zero. Recurrent-down and all nine
original controls were verified in the immediately preceding checkpoint and
are unchanged, not rerun for reassurance. Gate **553 default passes, 81 optional
skips** in 27.96 s; Ruff, whitespace and diagnostics clean. No failed checks.

No production/native changes, build/CTest/sanitizer, real input reads/hashes/
conversion/execution/quality, downloads/refits/new candidates, push/branch or
subagents. Scratch removed; caches/user settings unchanged. Host observation:
29 GiB RAM, about 18 GiB available, 8 GiB swap with 1.1 MiB used and 518 GiB
free disk; per-process swap counts above are not host/target-device guarantees.
Fixed-gate synthetic recurrence is not learned-gate or initialized-state
integration, issuer/frozen equivalence, representative quality or performance.
Historical real MiMo discrepancy and all real-profile approval boundaries remain.

## Recurrent Source Probe Checkpoint (2026-10-15)

After user-pushed `15d9d3c`, added one `recurrent-down` case to the existing
profile/manifest synthetic fixture. Sparse BF16 QKV/gate/output and four-tap
F32 convolution now drive nonzero delta-state/residuals before the frozen
FFN-down projection. Reused the existing tiny fixed-gate recurrence with
two key/four value heads and 32 dimensions, explicit BF16 dense-input rounding
and literal inverse toy V head layout on source rows/columns. No native hook,
geometry, source/model/operand/token bound, deadline or tolerance was widened.

The oracle requires state and residual norms >0.01. Omitting recurrence,
resetting state every token or dropping convolution history must each produce
logits outside the unchanged `2e-5` native tolerance. The second source writer
must reproduce full/peer/max128 references and fault deltas exactly. The new
probe passes seven two-shard source-gated workflows, six post-prefill exit
refusals and four pre-yield source refusals, with complete 28-tensor parity,
25 selected metadata records, six runtime-file pins, typed zero-generation
scores and existing dispatch/repack/cleanup assertions. Baseline fixtures keep
their zero recurrent state and previous independent references.

Verification: initial new probe **1 pass** in 176.69 s; discriminating-oracle
follow-up **1 pass** in 85.99 s, peak RSS 311,116 KiB. All **9 preserved native
role/refusal controls** pass in 664.39 s, peak RSS 317,096 KiB; swaps zero.
Gate **553 default passes, 75 optional skips** in 34.64 s; Ruff, whitespace and
diagnostics clean. No failed checks. All scratch removed; no new native
build/CTest/sanitizer run or production/native changes.

No real input reads/hashes/conversion/execution/quality, downloads/refits/new
candidates, push/branch or subagents. Frozen codes/independent FP16 scales,
v1, cached inputs/libraries and user settings unchanged. Fixed-gate synthetic
state is not arbitrary learned gates, initialized-state integration, issuer/
frozen equivalence, atomic storage, model quality or target-device evidence.
Historical real MiMo discrepancy and other-model runtime approvals remain open.
Next bounded slice: extend this proven recurrent fixture to the other six
eligible frozen roles, preserving all references and limits.

## Nonzero Multi Shard Refusal Checkpoint (2026-10-14)

After `acc5f12`, the separate nonzero source-gated variant independently tests
stale-file and fresh-pin selected-range pre-yield refusals on both physical
shards for all seven roles: **28 refusal combinations**. Stale mutations are
restored before testing the other shard. Fresh-pin writers independently start
from the unchanged two-shard fixture and archive, select an unfrozen operand on
the requested file (embedding where present), flip one disposable byte and
update exact file/header/profile pins plus GGUF identity metadata. Each requires
its specific identity/range error before owner yield/native prefill, baseline
registry and empty score stdout. No stale hash is used to pass a range test.

Verification: FFN-down **1 pass** in 57.18 s, peak RSS 310,984 KiB; remaining
six eligible roles plus unsupported fixtures **8 passes** in 384.80 s, peak
RSS 316,648 KiB; swaps zero. The **49 successful nonzero two-shard workflows**
and **42 post-prefill exit refusals** remain intact with exact full/peer/max128
references and unchanged `2e-5` native tolerances. Gate **553 default passes,
74 optional skips** in 17.76 s; Ruff, whitespace and diagnostics clean. No failed
checks or relaxed bounds/assertions; all scratch removed.

No production/native changes, build/CTest/sanitizer, real inputs/reads/hashes/
conversion/execution/quality, downloads/refits/new candidates, push/branch or
subagents. Frozen codes/scales, independent original-frozen-source identity,
v1, caches, user settings, file/operand/token/tensor limits and deadlines remain
unchanged. This completes the queued bounded multi-shard caller checks, not a
real-model/runtime/quality gate. The next real-profile comparison still needs
separate approval and the existing [comparison preparation](#real-comparison-preparation-2026-10-12).
Issuer/frozen equivalence, nonzero recurrent-state integration, atomic storage,
the historical real MiMo discrepancy and other-model execution remain open.

## Nonzero Multi Shard Checkpoint (2026-10-14)

After `3b373d2`, only the separate source-gated nonzero seven-role variant now
uses two physical BF16 shards. Each has its own header, local offsets and exact
profile pin; all files share the existing aggregate <=16 MiB source cap. A new
matching two-shard fixture supplies both initial and fresh-pin refusal writers.
The baseline fixture remains single-shard. Frozen payload bytes/codes/independent
FP16 scales are identical, and the second writer's full/peer/max128 references
must exactly equal the baseline before native comparisons at unchanged `2e-5`
tolerances. Complete receipts require both shards/exact aggregate bytes.

All **49 nonzero source-gated workflows** now pass with two physical shards,
complete 28-tensor source/converted parity, 25 selected metadata records and six
runtime-file pins. Changed bytes, restored bytes plus mtime and identical-byte
replacement after native prefill each refuse for either first or last shard:
**42 post-prefill refusals** require native success/nonempty withheld buffer,
exact exit identity refusal, cleared pointers/restored registry and no scores.
The **14 pre-yield source refusals** and unsupported fixtures remain intact.
Scratch cleanup, token/tensor/byte/deadline bounds and hooks are unchanged.

Verification: initial two-shard FFN-down **1 pass** in 46.56 s, then FFN-down
with both-shard exit refusals **1 pass** in 56.41 s, peak RSS 310,904 KiB;
remaining six eligible roles plus unsupported fixtures **8 passes** in
312.53 s, peak RSS 316,428 KiB; swaps zero. Gate **553 default passes,
74 optional skips** in 11.75 s; Ruff, whitespace and diagnostics clean. No
failed checks, assertion relaxation or new native build/CTest/sanitizer run.

No production/native changes, real inputs/reads/hashes/conversion/execution/
quality, downloads/refits/new candidates, push/branch or subagents. Caches/user
settings untouched. Nonzero FFN/full-attention arithmetic is not nonzero
recurrent state, issuer provenance, frozen quantization equivalence, tokenizer
semantics, atomic storage, representative quality or device evidence. Historical
real MiMo discrepancy and all real-profile approval boundaries remain open.
Next bounded slice: discriminate stale-file and fresh-pin selected-range
pre-yield refusals for either physical source shard, without changing oracles.

## Multi Shard Refusal Checkpoint (2026-10-14)

After `332e901`, each paired two-shard BF16/F32 actual-loader fixture mutates
either physical source file after successful native prefill: changed byte,
restored byte plus original mtime, or identical-byte replacement copied in
<=64 KiB chunks. Each requires nonempty withheld output, then the exact source
identity error, cleared handles, closed public model descriptor and baseline
registry. The score-parse counter cannot advance on exit refusal. Disposable
data is restored between controls; source caches and frozen codes/scales remain
untouched. This covers **12 post-prefill shard/effect/dtype combinations**, not
nonzero model quality or atomic storage.

Verification: **2 multi-shard paired controls** pass in 25.13 s, peak RSS
310,156 KiB; **2 preserved single-shard owner controls** pass in 11.15 s, peak
RSS 310,084 KiB; swaps zero. Gate **553 default passes, 74 optional skips** in
16.14 s; Ruff, whitespace and diagnostics clean. Initial two-case run failed
before policy creation: a nested copy output variable shadowed the enclosing
GGUF destination path (`UnboundLocalError`). Renamed only that handle and reran
the same focused controls successfully before widening validation; no assertion,
tolerance, deadline or byte/token/tensor-bound relaxation. Scratch removed.

No production/native changes, rebuild/CTest/sanitizer, real inputs/reads/hashes/
conversion/execution/quality, downloads/refits/new candidates, push/branch or
subagents. User settings/caches untouched. Next bounded integration: write
physical shards for the existing nonzero seven-role source variant and preserve
its independent full/peer/max128 oracles and output-withholding controls. Do not
infer that integration from these zero-source paired fixtures. All real-profile
approval, issuer/frozen equivalence, recurrent nonzero-state and quality limits
remain unchanged.

## Multi Shard Exit Checkpoint (2026-10-14)

After user-pushed `81d583a`, extended the existing lifecycle fixture to one/two
tiny physical source shards. First/last-shard changes, restored bytes/mtime,
identical replacement, symlink/missing files, caller failures, path/inventory/
cwd snapshot isolation and default opt-out retain free-before-close behavior.
A deterministic clock charges 20 seconds per real bounded hash under one
unchanged 30-second deadline: one shard succeeds, two hashes exceed the shared
phase budget and refuse. No per-shard deadline reset or tolerance relaxation.

The existing synthetic prefill fixture now optionally creates two headers with
local offsets, index entries and profile pins; single-shard defaults preserve
original ordering/names/zero bytes. Both layouts retain all seven native record
geometries and frozen codes/scales. Two new BF16/F32 paired actual-loader modes
verify 28 source/converted ranges, selected metadata and exit checks, count both
physical shards/exact aggregate bytes <=16 MiB, and preserve restored source and
fresh-pin converted corruption refusals. Zero dense source arithmetic is not
nonzero quality evidence. No production/native source changes in this increment.

Verification: **30 lifecycle controls** in 0.86 s; **44 focused fixture/record/
lifecycle controls** in 1.21 s. Actual **2 new paired controls** in 9.19 s, peak
RSS 309,484 KiB; **18 preserved paired modes** in 36.19 s, peak RSS 309,560 KiB;
swaps zero. Gate **553 default passes, 74 optional skips** in 12.28 s; Ruff,
whitespace and diagnostics clean. Pylance confirms all 20 resolved fixture
callers compatible. No failed checks. Scratch removed; no rebuild/CTest/
sanitizer, real inputs/reads/hashes/conversions/execution/quality, downloads/
refits/new candidates, push/branch or subagents. User settings/caches untouched.
Observed restart resources: 29 GiB RAM, about 22 GiB available, 8 GiB unused
swap and 241 GiB free disk; these are not target-device or job guarantees.

Next bounded integration: changed first/last physical shards after actual
prefill must refuse at exit and discard the paired caller's output buffer.
Existing 49 nonzero single-shard workflows/21 exit refusals remain historical
evidence, not proof of nonzero multi-shard integration or atomic snapshots.
Issuer provenance, frozen equivalence, the historical real MiMo discrepancy,
Defiant Fable execution approval and broader real-model/quality work remain open.

## Source Exit Score Checkpoint (2026-10-13)

After `e4c8b0b`, all 49 nonzero source-gated seven-role workflows opt into source
exit checks. The caller captures native stdout in a disposable temporary file,
requires successful prefill/nonempty <=64 KiB output, restores stdout before
cleanup and releases bytes only after owner/source/runtime closure succeeds
with registry restored/pointers cleared. Body/exit failure discards output; no
buffer is published from a failed owner. Existing file/table-only and unsupported
controls preserve their prior path, limits and results.

Three actual post-prefill source changes per role now refuse at exit: changed
byte, restored byte plus original mtime, and identical-byte inode replacement
using <=64 KiB copy chunks. Each requires native success and withheld bytes
before mutation, then the exact source identity error, baseline registry/cleared
pointers and empty score stdout. Parent restores only disposable source data
between controls; frozen codes/scales and pinned source caches stay unchanged.
These **21 exit refusals** are separate from **14 preserved pre-yield refusals**.

Verification: FFN-down **1 pass** in 53.82 s, peak RSS 310,548 KiB; other six
eligible roles plus unsupported fixtures **8 passes** in 284.78 s, peak RSS
316,144 KiB; swaps zero. All **49 successful source-exit-gated workflows**, 21
post-prefill and 14 pre-yield source refusals pass with unchanged independent
full/peer/max128 oracles and `2e-5` native tolerances. Gate **529 default passes,
72 optional skips** in 13.03 s; Ruff, whitespace and diagnostics clean. No failed
executable checks in this increment. Temporary buffers/shards/GGUF/archives
cleaned.

The documentation patch was interrupted with unknown outcome; worktree/search
confirmed it had not applied before recovery at the user's stop request. One
documentation retry failed on wrapped-line context; the unchanged state was
verified and the exact local context corrected. Only checkpoint documentation/
review/commit proceeded after that request, not the next implementation slice.
No native API/hook changes/rebuild/CTest/sanitizer, real source/model reads/
hashes/conversion/execution/quality, downloads/refits/new candidates, push/branch
or subagents. User settings and source/library caches untouched. Synthetic
FFN/full-attention is not nonzero recurrent state, independent issuer provenance,
frozen quantization equivalence, representative quality or device evidence.
Historical real MiMo discrepancy and real-profile approval boundaries remain.

Next session, after explicit resume: multiple selected shards must share one
exit deadline and every shard identity must be required; do not infer that
coverage from these single-shard controls. Multi-shard checks are not started.
Implementation is paused for today with no running jobs.

## Source Exit Owner Checkpoint (2026-10-13)

User pushed `2290dc7` and explicitly resumed autonomous bounded code-only work.
Added strict opt-in `verify_source_files_on_exit=False` to the verified owner;
it requires existing complete source inputs and preserves defaults/v1. Source
receipts now distinguish pre-yield-only evidence from required entry/exit checks.
For the opt-in, absolute non-resolved paths and verified dev/inode/size/mtime/
ctime identities are snapshotted before yield. Final no-follow/nonblocking
selected-shard hashes and descriptor/path identities use the same bounded helper
as the source table's final recheck. Aggregate source cap stays 16 MiB; <=64 KiB
positional reads use one separate cooperative `maximum_seconds` exit phase
(default 30, cap 600). This does not bound the whole caller/native session or
prove continuous immutability/atomic storage. Exit/caller failure still frees
the policy and clears pointers before model descriptor close.

Verification: **36 focused controls** in 0.85 s include 13 tiny-real-file exit
lifecycle cases and three new flag guards plus preserved refusals. Persistent/
restored source changes, restored mtime, identical-byte replacement, symlink/
missing file and deadline refuse; default pre-yield behavior remains unchanged.
Caller exceptions and mutable path-map/path-like/inventory/cwd inputs exercise
cleanup and snapshot isolation. Actual **2 paired BF16/F32 source-gated native
loader controls pass** in 13.65 s, peak RSS 309,536 KiB, swaps zero; score buffers
are parsed only after successful source-exit closure. Pylance finds all four
resolved callers compatible, syntax/diagnostics clean. Gate **529 default passes,
72 optional skips** in 12.96 s; Ruff/whitespace clean. First extraction passed
105 tests but lint found one obsolete import; removed extraction leftovers and
reran the same 105-test/lint slice successfully before new controls. No numerical
failure or tolerance/deadline/cap relaxation.

No native changes/rebuild/CTest/sanitizer, real model/source reads/hashes,
conversion/execution/quality, downloads/refits/new candidates, push/branch or
subagents. User settings/caches untouched; disposable fixtures cleaned. Locally
generated pins are not issuer provenance; frozen source identity is not
quantization equivalence. Real MiMo discrepancy remains open and Defiant Fable
execution unapproved. Next bounded caller increment: nonzero source-gated native
scores must be buffered until the source-exit gate succeeds; changed/restored
sources after prefill must discard scores and restore registry/pointers. Real
actions still need separate approval.

## Source Gated Workflow Checkpoint (2026-10-13)

After `be25780`, the same nonzero source-gated fixture completes the existing
bounded workflow surface: full, chunked, reset, reordered, parallel, maximum and
kernel-recovery, across all seven eligible roles. Chunk/reset require two
successful dispatches, chunk last-input length two; maximum requires 128 input/
prefilled/last-dispatch tokens and one dispatch. Reordered IDs/labels/token IDs
and logits/scores follow C/A/B. Every mode has one repack, ABI 2, zero generation,
appropriate rejection/error counts and unchanged registry/owner cleanup. The
second writer's full/peer/128-token references all equal the originals exactly;
native comparisons retain `2e-5` tolerances. Complete physical source/converted/
selected metadata and six runtime file gates precede all native calls.

Verification: FFN-down **1 pass** in 42.57 s, peak RSS 310,272 KiB; remaining six
eligible roles plus unchanged full-only/large-state refusals **8 passes** in
243.93 s, peak RSS 315,948 KiB; swaps zero. This covers **49 positive source-gated
role/workflow combinations** and **14 pre-yield physical source refusals**, plus
preserved original native controls. Default gate **513 passes, 72 optional
skips** in 23.75 s; Ruff, whitespace and diagnostics clean. No failures or limit
relaxation in this increment. Disposable files removed on success/failure.

The current code-only source/provenance integration checkpoint is complete for
the existing bounded synthetic hook. Next real-profile comparison still requires
explicit authorization for source reads/hashes, temporary complete conversion/
execution and its budget, with immutable cached sources, sole frozen candidate
and held-out data preserved. Do not implicitly begin it. Nonzero synthetic FFN/
full-attention is not nonzero recurrent state, external issuer provenance,
frozen quantization equivalence, quality/calibration or target-device performance.
The historical MiMo discrepancy remains unresolved; Defiant Fable is offline
inventory only, not execution-approved. No production/native/cache/settings
changes, rebuild/sanitizer/CTest, real source/model inputs, downloads/refits/
retained candidates, push, branch or subagents in this increment.

## Source Gated Refusal Checkpoint (2026-10-13)

After `d8f5315`, every eligible nonzero source-gated role adds two physical source
refusals. A changed byte with the original file pin must fail file identity.
A selected embedding byte changed by the synthetic writer, with freshly computed
shard/header/profile pins and matching converted metadata, must instead fail
selected-range equality against the unchanged archived operand/digest. The
fresh-pin case preserves geometry and frozen codes/scales/source identity; this
is not a stale hash or metadata-only refusal. Both fail before owner yield and
native prefill, restore the registry baseline and emit no scores. All changes
are in one disposable directory, removed on success/failure.

Verification: FFN-down **1 pass** in 34.39 s, peak RSS 310,268 KiB; remaining roles
**6 passes** in 210.54 s, peak RSS 315,700 KiB; swaps zero. All **14 source
role/refusal combinations** and existing **21 positive source-gated role/mode
combinations** pass. Gate: **513 default passes, 72 optional skips** in 24.41 s;
Ruff, whitespace and diagnostics clean. No failed checks or relaxed limits.
No production/native/settings/cache changes, rebuild/sanitizer/CTest, real
source/model reads/hashes/conversion/execution, downloads/refits or new candidates.

These checks discriminate declarations/authenticated local bytes/selected
operand equality before synthetic runtime use; they do not authenticate an
external issuer, frozen quantization equivalence, recurrent nonzero state,
representative quality or target performance. Historical real MiMo discrepancy
and approval boundaries remain unchanged. Next bounded integration: source-gated
chunk/reset/reordering and the existing 128-token cap, using unchanged hooks and
oracles. No push, branch or subagents.

## Source Gated Recovery Checkpoint (2026-10-13)

After `7b3bc11`, the same nonzero source-gated variant runs the unchanged full,
parallel and kernel-recovery hooks across all seven eligible roles. Complete
source/converted/selected metadata and six runtime file gates precede every
call. Parallel requires three successful dispatch/prefill calls and one repack,
with independent primary/peer logits. Recovery requires one rejected call and
kernel error, then one successful dispatch and the unchanged reference scores.
Typed IDs 11/17/23, labels A/B/C, inspect/edit/ask, three input/prefilled tokens,
ABI 2, zero generation and registry cleanup are explicit in all three modes.
Both second-writer reference sequences equal the original exactly. Tolerances,
deadlines, budgets and hooks stay unchanged.

Verification: FFN-down **1 pass** in 31.00 s, peak RSS 310,112 KiB; other roles
**6 passes** in 159.50 s, peak RSS 316,132 KiB; swaps zero. This covers **21
source-gated role/mode combinations** in addition to preserved existing checks.
Default gate **513 passes, 72 optional skips** in 11.08 s; Ruff, whitespace and
diagnostics clean. No failures in this increment. Disposable scratch removed;
no production/native/cache/settings changes, rebuild/sanitizer/CTest, real
model/source reads, conversions/execution, downloads/refits or new candidates.

Evidence remains nonzero synthetic FFN/full-attention, not nonzero recurrent
state, external issuer provenance, frozen quantization equivalence, representative
quality, calibrated confidence or device performance. Historical real MiMo
discrepancy is unchanged. Next local check: changed physical source bytes must
refuse before nonzero native scoring, restoring registry and emitting no scores.
Real actions still need separate approval; no push/branch/subagents.

## Nonzero Source Prefill Checkpoint (2026-10-13)

After `0624bcb`, the existing seven-role nonzero native fixture adds a separate
physically pinned source variant when `PRISM_PARITY_PYTHON` is selected. Its 27
dense operands and distinct zero-valued original frozen source are written into
a disposable BF16 safetensors shard, with complete header/offset coverage and
updated disposable file/header/profile/manifest pins. The second GGUF's selected
identity strings match those pins; frozen PQ2 codes and independent FP16 scales
remain unchanged. No existing fixture or runtime policy is replaced.

Before full native prefill, the existing verified owner checks the complete
source/converted table and 25 selected metadata records, plus six actual runtime
file pins. Descriptor access, source recheck, registry lifetime, one genuine
BitNet dispatch/repack, zero generated tokens and nonuniform oracle logits/
conditional scores are required. Second-writer logits must equal the original
oracle exactly; native tolerances remain `2e-5`. Scratch shard/GGUF/archive are
removed on success/failure; receipts/scores are evaluated after successful owner
closure. All seven eligible roles pass; unsupported fixtures are unchanged.

Verification: FFN-down **1 pass** in 24.89 s, peak RSS 310,508 KiB; remaining six
roles **6 passes** in 154.72 s, peak RSS 314,084 KiB; swaps zero in both runs.
Required gate: **513 default passes, 72 optional skips** in 13.64 s, Ruff and
whitespace clean; touched diagnostics clean. Initial focused run reached policy
creation but failed on a test-only receipt `KeyError`: descriptor access is in
the nested converted receipt, and the source recheck key is singular. Corrected
those assertions and reran the same focused check before expanding coverage.
No assertion, tolerance, budget or deadline relaxation.

This proves nonzero synthetic FFN/full-attention integration with locally
generated physical source pins, not independent HF issuer provenance, nonzero
recurrent-state coverage, source-to-frozen equivalence or representative quality.
No production/native changes, rebuild/sanitizer/CTest, real source reads/hashes,
conversion/execution, downloads, fitting or retained candidates. Historical real
MiMo discrepancy remains open. User settings/caches untouched; no push/branch/
subagents. Next bounded increment: source-gated parallel/context recovery with
the unchanged hook and independent oracle. Real actions still need approval.

## Nonzero Source Permutation Checkpoint (2026-10-13)

After `8e913e6`, four more cases extend the same physical paired-file fixture to
raw BF16 QKV row/output-column permutations and BF16/F32 convolution reorder/
squeeze. Independent literal toy indices preserve 128 Q/K rows, then reorder
four 32-wide value heads as 0/2/1/3; columns use the corresponding value indices.
Distinct finite source bits make omitted head layout changes observable. Raw
BF16 encoded output preserves bits, avoiding float quantizer round trips. Actual
source shards are repinned only in disposable synthetic profiles; header
geometry and frozen codes/scales remain unchanged. Fresh-pin unpermuted outputs
refuse together with source restored writes and converted one-bit changes.

Verification: **513 default passes, 72 optional skips** in 12.22 s; Ruff,
whitespace and touched diagnostics clean. **All eight nonzero paired controls
pass** in 12.03 s, peak process RSS 245,404 KiB, swaps zero, including the four
prior vector cases. Scratch empty after cleanup. No failures, production API/
native changes, prefill/build/sanitizer/CTest, real inputs, downloads, refit or
new retained candidate. User settings and existing caches remain untouched.

This closes the nonzero selected operand/range gap for vector, matrix and
convolution families, not native nonzero source-gated prefill, complete-model
parity, independent issuer provenance, frozen quantization equivalence or quality.
Next bounded integration: attach genuinely pinned disposable source bytes to
the existing nonzero native fixture and independent score oracle, preserving
all numerical, deadline, token/tensor and hook limits. Real actions still need
separate approval.

## Nonzero Source Vector Checkpoint (2026-10-13)

The user resumed after pushing `3e32444`; starting `main` matched `origin/main`,
with only the untouched editor directory untracked. Four new cases reuse the
existing paired-file control for actual nonzero BF16/F32 source vectors and
selected typed GGUF ranges. The disposable source shard is changed and repinned;
synthetic profile/inventory/manifest identity is updated, header geometry and
frozen payload remain unchanged. Independent F32 norm-add and negative-exp
expectations use fixed four-head ordering 0/2/1/3, not the verifier's transform
implementation. Fresh-pin no-op norm and unpermuted exp outputs refuse alongside
existing restored-write and converted-bit controls.

Verification: **513 default passes, 68 optional skips** in 33.66 s; Ruff,
whitespace and touched diagnostics clean. **Four nonzero paired controls pass**
in 13.75 s, peak process RSS 245,524 KiB, swaps zero. Scratch empty after cleanup;
no failures, source implementation/API changes, native prefill/build/sanitizer/
CTest or unrelated optional reruns. Cached pinned Prism revision and existing
dense environment verified. Host observation before checks: 29 GiB RAM, 19 GiB
available, 8 GiB swap with 1.5 MiB used, 242 GiB free disk; these are not target
device or per-job memory guarantees. No real weights, downloads, refit or retained
candidate; source pins change only on disposable synthetic fixtures.

This discriminates selected nonzero vector arithmetic/ranges, not nonzero native
prefill, whole-model parity, independently issued provenance, frozen approximation
or quality. Source-gated native controls remain zero-source historical evidence.
Next bounded control: nonzero physical matrix row/column head permutations and
convolution squeeze/ranges. Real-model and quality approvals remain unchanged.

## Source Gated Prefill Checkpoint (2026-10-13)

After `125221e`, both existing `complete-owner-*` controls run actual native
prefill under the same source-gated verified owner using the unchanged bounded
test hook. Synthetic tokenizer metadata is added; no model repository code or
real tokenizer inputs execute. Complete pinned source/converted and selected
policy metadata checks precede yield. The hook borrows the live policy and
destroys native users before returning. Provisional native JSON is captured in
<=64 KiB disposable scratch and interpreted only after successful owner exit.
Exact registry restoration, cleared pointers and closed-path refusal are checked.

Verification: **513 default passes, 64 optional skips** in 29.33 s; Ruff,
whitespace and touched diagnostics clean. **Two actual source-gated native
prefill controls pass** in 14.01 s, peak process RSS 310,240 KiB, swaps zero.
Both BF16 and F32-vector/convolution source variants verify all 28 physical
ranges and 25 selected metadata records, then report positive genuine BitNet
dispatch calls matching prefill calls, exactly one weight repack, typed
inspect/edit/ask IDs 11/17/23 and labels A/B/C, and zero generated answer tokens.
Zero logits/uniform conditional scores agree at unchanged 2e-5 tolerances.
Actual restored source writes and fresh-pin converted corruption still refuse
before prefill. Scratch empty after cleanup; original frozen codes/scales and
native hooks/caches/libraries remain unchanged. No other optional gates,
all-profile rerun, rebuild/sanitizer/CTest, real source/model operations,
downloads, refit or retained candidate in this increment. User settings untouched.

Failed-check history: initial prefill arithmetic controls passed in 12.62 s.
Adding registry assertions exposed embedded-string indentation errors in both
subprocesses before launch; a whitespace-only repair did not change the line.
An equivalent assertion replacement then landed inside the still-live owner,
so both cases failed at the wrong lifetime boundary after successful prefill.
Moving the check onto the existing post-exit block restored both passes without
relaxing pointer, registry, dispatch, arithmetic or generation requirements.

These are zero-dense-source synthetic controls, not independent HF/issuer proof,
nonzero transform discrimination, frozen quantization equivalence, calibrated
confidence, representative decision accuracy or edge performance. Historical
nonzero operand/native controls remain separate evidence; the real MiMo numerical
discrepancy remains unresolved. Receipts are pre-yield source evidence and
provisional until successful owner exit, not immutable cross-file storage or
loaded-code/tokenizer semantic attestation. Further real-model/source conversion,
execution, quality or device checks still require the separate approvals and
resource budgets below. Keep any further autonomous work code-only and synthetic.

## Source Gated Owner Checkpoint (2026-10-13)

After `3fb20a8`, the verified owner explicitly accepts `source_paths` plus
`frozen_source_values` alongside the existing reader/unfrozen arrays/digest maps.
Partial or conflicting frozen inputs refuse; frozen digest stays manifest-owned.
Complete source snapshot/preflight occurs before model access, with aggregate
source file default/cap 16 MiB and the existing tensor charge cap. Source-gated
deadline includes early preflight, model hashing and composition before yield.
Descriptor-anchored source/converted evidence is retained at immutable
`source_conversion_table`; unchanged converted receipt stays at `tensor_table`,
with separate top-level source statuses. Selected metadata opt-in remains
independent. Defaults/file-only/table-only behavior is unchanged. Pylance
confirms all three resolved caller signatures remain compatible.

Verification: **513 default passes, 64 optional skips** in 17.26 s; Ruff,
whitespace and touched diagnostics clean. **89 focused owner/table cases** pass
in 7.44 s, including ten new preflight/lifetime cases: missing/partial/conflicting
inputs, source budget and early-preflight deadline, source failure/change/
deadline/snapshot receipts and free-before-descriptor-close. **Two actual source-
gated native policy-owner controls pass** in 24.79 s, peak process RSS 298,848
KiB, swaps zero: complete pinned BF16/F32-vector/convolution synthetic shards,
28 converted ranges, 25 selected metadata records, restored inter-stage source
write and fresh-pin converted corruption refusal, normal owner exit and cleared
pointers/post-exit path refusal. These create/free policy but do not run prefill.

The unchanged-default **FFN-down native prefill control passes** in 25.59 s,
peak process RSS 293,404 KiB, swaps zero. This is separate from source-gated
policy creation and does not establish source-gated prefill. No full nine-profile
rerun, native build/sanitizer/CTest, real source/model conversion/execution,
downloads, refit or new retained candidate. All files are disposable pinned
zero-source synthetic fixtures; scratch empty after cleanup. User settings
untouched. Source identity is pre-yield, not immutable source storage throughout
scoring, atomic multi-file snapshot, issuer, quantization equivalence or quality.
Receipts remain provisional until successful owner exit; runtime approval is
not granted. Keep files quiescent and destroy native users before exit.

Failed-check history: both optional owner cases first failed before factory
dispatch due to a guessed shared-library basename. Corrected to the existing
controls' verified `libprism_group_scale.so` and reran the same two cases; no
cache rebuild or tolerance/budget change. Next bounded control: source-gated
native prefill using existing hooks, synthetic budgets and zero-generation
scores, without widening hooks or claiming representative model quality.

## Complete Source Conversion Checkpoint (2026-10-13)

After `d202b99`, `verify_profile_gguf_source_tensor_table()` composes all selected
source and converted ranges through shared read-only snapshots, keeping frozen
original source digest and canonical PQ2 identity separate. Existing source
snapshot/converted layout rules are extracted without changing prior public
receipts. Both preflights, exact reader table and optional live owner identity
are checked before source dispatch. One cooperative deadline covers source,
converted and source recheck, including restored inter-stage write refusal.
Operand cap 64 MiB charges the maximum of source table and converted table plus
three retained original frozen-source extents; this is not peak RAM. Selected
source shard aggregate and separate converted file each cap at 16 MiB.

Verification: **503 default passes, 62 optional skips** in 19.31 s; Ruff,
whitespace and touched diagnostics clean. **46 focused source/converted cases**
pass in 4.68 s, including 11 new composition cases for shared arrays/digests,
caller mutation, source recheck/deadline and early source/converted/owner/
frozen/budget/reader refusals. **Eight actual paired-file controls pass** in
39.00 s, peak process RSS 285,028 KiB, swaps zero: six prior selected controls
plus complete 28-range BF16 and F32-vector/convolution tables. Every unfrozen
source digest is shared with its converted range; frozen source/payload hashes
remain separate, with quantization equivalence unchecked. Fresh-pin converted
corruption and actual restored source writes refuse; scratch empty after cleanup.

The existing **same-owner FFN-down native prefill control passes** in 26.45 s,
peak process RSS 292,896 KiB, swaps zero, verifying the shared converted preflight
still supports table/selected metadata, typed zero-generation scores and prior
recovery/lifetime controls. It does not run the new actual source-file table gate
inside the owner. No full nine-profile rerun, native rebuild/sanitizer/CTest,
real-model conversion/execution/source reads, downloads, refit or new retained
candidate. No failures in this increment; user editor files untouched.

This API remains a standalone composition, optionally anchored to a live owner,
not automatic production scoring integration. Zero-source synthetic files do not
prove independent HF/issuer provenance, nonzero transform discrimination,
quantization quality or representative decision accuracy. Metadata/tokenizer,
loaded/transitive code and runtime approval remain separate, and files must be
quiescent rather than atomically immutable. Next bounded caller work: explicitly
opt the verified owner into complete source/converted evidence while preserving
all defaults and provisional-until-successful-exit receipts. Real-model actions
still require separate approval.

## Complete Source Table Checkpoint (2026-10-13)

After `6c34a97`, `verify_profile_safetensors_tensor_table()` verifies every
selected conversion-plan source, including the declared frozen original tensor,
through exact tensor/digest/shard maps and copied read-only operands/payload.
Complete dtype/shape/layout/digest and aggregate preflight precedes dispatch.
Aggregate operand default/cap 64 MiB charges three source plus three canonical
payload extents; per-operand cap stays 16 MiB. Aggregate selected shard files,
not just each file, cap at 16 MiB. One cooperative 30 s/cap 600 s deadline covers
all checks and final shard rereads. Inode/extent/timestamps must match across
operands and completion, including restored writes. Plan/header memory is not
peak-RAM bounded; files must be quiescent, not an atomic/concurrent snapshot.

Verification: **492 default passes, 60 optional skips** in 11.73 s; Ruff,
whitespace and touched diagnostics clean. **22 focused table cases** pass in
3.65 s: 16 complete-map/preflight/routing controls and six actual pinned shard
controls, with complete 28-range BF16 and F32-vector/convolution variants,
caller array/map mutation isolation, restored inter-operand/final writes and
deadline exhaustion. Frozen codes/scales remain byte-identical; nested frozen
source identity explicitly does not imply quantization equivalence. Disposable
shards are zero-source synthetic fixtures, not independent HF/issuer or quality
evidence. Scratch is deleted. No failures, optional native/build/sanitizer/CTest
reruns, real source reads, downloads, refit or new retained candidate in this
increment. No new peak-RAM measurement; user editor files untouched.

This standalone API does not bind converted bytes or integrate into the verified
native owner/scoring path. Source identity, converted table parity, tokenizer/
metadata semantics, frozen approximation/quality and runtime approval remain
separate. Next code-only caller preparation: compose source/converted evidence
with explicit shared snapshots/deadlines, preserving these distinctions. Real
model operations still require separate approval.

## Frozen Source Identity Checkpoint (2026-10-13)

After `58d13d0`, `verify_profile_frozen_source_tensor()` reuses the bounded
source shard/header/range verifier for the manifest's original BF16 tensor and
source digest. It records frozen payload identity separately and explicitly
leaves `source_to_frozen_quantization_verification` unchecked. There is no
quantization, refit, dense/frozen equivalence or quality claim. Unfrozen calls
still refuse frozen inputs; malformed manifests refuse explicitly. Real frozen
candidate, manifest, codes and independent FP16 row/group scales are unchanged.

Source receipts now retain immutable device/inode/size/mtime/ctime identity,
allowing combined post-conversion rechecks to reject restored writes between
stages, not only changed bytes. Final pathname identity uses no-follow `lstat`.
This is not atomic cross-file storage or a concurrent-writer guarantee.

Verification: **470 default passes, 60 optional skips** in 13.71 s; Ruff,
whitespace and touched diagnostics clean. **44 affected source cases** pass in
4.59 s, including eight frozen identity/layout/digest/file/budget/schema cases.
**Six paired-file controls pass** in 13.45 s: the existing five unfrozen pairs
plus distinct frozen source/PQ2 checks. Unfrozen pairs reject an actual restored
write between stages; all pairs refuse freshly pinned converted corruption.
These are zero-source, four-tensor disposable synthetic fixtures, not complete
models, independent HF source/issuer evidence or representative quality. No
native model execution/build/sanitizer/CTest, real weights, downloads, refit or
new retained candidate. Scratch directories empty after cleanup; no new peak-
RAM measurement in this increment. User editor files remain untouched.

Failed-check history: after retaining source identity, all six paired subprocess
cases initially failed before launch because a nested byte literal became a
null byte in the `-c` argument. Replaced it with child-runtime `bytes(...)`
construction; the same focused check then passed without relaxing assertions.
Next bounded provenance preparation: whole selected-source table identity under
explicit aggregate budgets, keeping source identity, converted parity and frozen
quantization evidence separate. Real-model operations still need approval.

## Combined Source Range Checkpoint (2026-10-13)

After `b0db7a8`, `verify_profile_gguf_source_operand()` binds one unfrozen
source-file range and typed converted GGUF range through one copied read-only
operand/digest, rechecking the source afterward under one cooperative deadline.
Each file remains capped at 16 MiB with existing per-kind operand limits;
retained snapshot/plan/header memory is not a peak-RAM guarantee. Optional live
owner anchors the converted file; wrong owner/dtype/shape refuses before source
dispatch. Immutable receipt nests unchanged source and converted receipts and
explicitly marks `cross_file_atomic_snapshot` as `not_guaranteed`. It does not
upgrade converted-only claims, frozen replacement/source equivalence, whole
tables, issuer, metadata semantics, imported/loaded code or model quality.
Quiescent files remain a prerequisite; no concurrency/runtime approval.

Verification: **462 default passes, 59 optional skips** in 14.78 s; Ruff,
whitespace and touched diagnostics clean. **36 focused source/composition cases**
pass in 1.51 s, covering shared snapshots/digests, caller mutation, post-conversion
source rejection, shared deadline and early owner/dtype/shape/identity/budget/
frozen refusals plus existing actual source-file cases. **Five actual paired-file
controls pass** in 4.27 s, peak process RSS 234,528 KiB, swaps zero: BF16/F32
vectors/convolution and BF16 matrix. Each uses a genuinely pinned disposable
zero-filled safetensors shard and a four-tensor selected GGUF, with exact
fresh-pin converted-bit corruption refusal. Temporary directories empty after
cleanup. No native model execution/build/sanitizer/CTest or complete-model
conversion was performed in these controls; historical same-owner/native gates
remain separate evidence. No failures in this increment, real-model/source reads,
downloads, refit or retained candidate. User editor files remain untouched.
Next bounded source work: verify the declared frozen source digest from a small
profile-pinned source range without refitting/changing codes, independent scales
or transforms, and keep source identity separate from quantization/quality.
Whole-source/issuer/metadata semantic/reader-code/loaded-page/transitive runtime
and real-model quality gates remain open; real-profile approvals unchanged.

## Source Operand File Checkpoint (2026-10-13)

After `aaa72fd`, `verify_profile_safetensors_operand()` binds one unfrozen
BF16/F32 caller array to a real profile-pinned source shard/header/range. It
reuses the checked conversion plan and inventory header parser, not the MiMo-only
row reader. Full size/hash must equal the profile pin; header size/hash and
selected dtype/shape/storage bytes must match. Copied array digest, positional
range bytes and before/after file hashes are checked under no-follow/nonblocking
regular descriptor access, <=64 KiB reads and stable device/inode/size/mtime/
ctime/path identity. Changed/restored writes still refuse. Default/cap file
16 MiB, operand charge three source extents with default/cap 16 MiB, header
1 MiB plus length; these are not peak process/header/plan memory. Cooperative
30 s deadline/cap 600 s is not a hard I/O timeout. Frozen replacements and
unknown operands refuse; no source-to-frozen equivalence/refit is claimed.
Receipt immutable, selected range only; converted file, pin issuer and runtime
approval remain unchecked/not granted. No real 9B shard fits the file cap.

Verification: **451 default passes, 54 optional skips** in 8.28 s; Ruff,
whitespace and touched diagnostics clean. **25 source cases pass** in 1.29 s:
BF16/F32 vectors/convolution and BF16 matrix positives, twelve no-open layout/
identity/cap/frozen/unknown refusals, differing arrays with fresh array digests,
changed file bytes, header/data changes during reads, deadline, symlink/FIFO and
restored-byte writes. The helper's optional `pin_zero_source` constructs genuine
whole-file pins only for disposable zero-payload fixtures; its default preserves
all previous profile/manifest bytes. Source files are deleted in test finally
blocks. The relevant FFN-down complete-stage/native regression passes in 20.88 s,
peak process RSS 293,400 KiB, swaps zero; the unchanged nine-role metadata gate
is the historical previous checkpoint, not rerun here. No failures in this
increment, native build, sanitizer/CTest, real source hashes/reads/conversions/
execution, downloads, refit or retained candidate. User editor files untouched.
Next bounded increment: connect selected source-file receipts to converted
operand/file parity using the exact same source digest, then expose combined
evidence explicitly rather than changing table-only claims. Whole-source/frozen
source proof, issuer identity, tokenizer/config semantics, reader-code/loaded-page/
transitive dependency proof and real-model quality remain open and approvals
unchanged.

## Integrated Policy Metadata Checkpoint (2026-10-13)

After `7834982`, `verified_profile_model_policy(..., verify_policy_metadata=True)`
requires the selected-policy byte gate inside the same descriptor owner, after
complete tensor parity and before yield. Strict boolean defaults to false;
full reader/source-array/pin inputs are required. The immutable metadata receipt
is nested in owner verification with a top-level selected-policy status. Table
receipts continue marking metadata unchecked; metadata receipts mark only their
selected records. Remaining pre-yield time is shared, with no hard interruption
of native/I/O operations. Failures free the policy before descriptor closure or
caller access; no default-caller changes or new execution approval.

Verification: **426 default passes, 54 optional skips** in 10.19 s; Ruff,
whitespace and touched diagnostics clean. Forty focused lifecycle/preflight
cases cover strict options, incomplete reader inputs, snapshots, immutable
receipts and pre-yield metadata failure/mutation/deadline cleanup. Pylance
reports all three resolved Python call sites signature-compatible; embedded
subprocess callers are verified by execution, not that static result. All
**nine profile/native controls pass** in 129.10 s, peak process RSS 299,228 KiB,
swaps zero. Each supported role runs actual verified full/parallel/recovery
with 28 tensor ranges and 25 selected records in its own prefill owner; precise
type/copied-part metadata refusals and the three existing table refusals emit
no scores and restore the registry. Both gates read the original descriptor
after pathname replacement, but owner exit still refuses changed inode metadata.
Normal integrated owner exit succeeds. Vector-only FFN-down compatibility
passes in 7.51 s. No failures in this increment and no tolerance/budget/hook
relaxation, native rebuild, sanitizer/CTest, real-model reads/conversions/execution,
downloads, refit or new retained candidate. Disposable file mutations are
restored before prefill and fixtures cleaned; user editor files untouched.
Next bounded provenance work must preserve the distinction between selected
policy bytes, genuine source-file operands, tokenizer/config semantics, imported
code/loaded pages, issuer identity and model quality. Real-profile gates and the
historical numerical differences still require separate approval/evidence.

## Policy Metadata API Checkpoint (2026-10-13)

After user-pushed `78af44e`, the bounded standalone
`verify_profile_gguf_policy_metadata()` binds 25 selected policy declarations
to typed little-endian GGUF records: eight architecture/provenance/execution
strings, three root geometry fields and the existing 14 numerical fields.
Mapped parts must be contiguous/read-only/shared and exactly serialize the
declaration. Independent positional rereads compare actual file bytes, with
before/after caller file hashes. A bounded metadata-chain walk checks unique
keys, supported scalar/non-nested array types, lengths and tensor-table boundary,
anchoring the selected offsets; unselected values/tokenizer semantics are not
interpreted. Caps remain 16 MiB file, 256 metadata keys, 64 KiB key/read and
8 KiB selected records. Deadline is cooperative 30 s (cap 600 s), not caller
parser construction or hard I/O. Optional owner access uses its borrowed
descriptor without changing its offset/lifetime or ordinary symlink refusals.
Receipt is immutable and marks only `selected_policy_records_only`; source
authenticity, full metadata semantics, reader-code/loaded-page/runtime approval
remain absent. Table-only receipts and policy callers are unchanged.

Verification: **420 default passes, 54 optional skips** in 4.84 s; Ruff,
whitespace and touched diagnostics clean. Seven invalid input/owner cases
refuse before parser import. All **nine complete-stage/native profiles pass**
in 98.19 s, peak process RSS 299,412 KiB, swaps zero. Each supported role checks
the 25 records plus ten precise missing/type/copied-part/offset/value/header/pin/
deadline/mid-read/unselected-type refusals. Changed values and headers with
fresh file pins still refuse; malformed unselected type tags are rejected by
the independent walk despite a stale reader. Mutations are restored before
native prefill; existing all-range/owner/refusal/typed-score/zero-generation
controls remain. No new source/environment/library/model downloads, real-model
reads/conversions/execution, retained candidate, native build, sanitizer or CTest.
Initial focused preflight passed but Ruff caught a missing local `sys` import;
it was repaired and the exact focused command passed before optional gates.
Next bounded increment: make this selected-policy check an explicit opt-in
pre-yield gate inside the same verified owner used for synthetic prefill. Real
source/model proof, tokenizer/configuration semantics, reader-code/loaded-page/
transitive dependency proof, historical numerical differences and quality stay
open; no real-profile approval. User `.vscode/` remains untouched.

## Integrated Table Caller Checkpoint (2026-10-13)

After `a9b9d0e`, `verified_profile_model_policy()` optionally requires
`tensor_reader`, `source_values` and `expected_source_sha256` together and runs
the descriptor-bound complete-table gate before yield. File cap remains 16 MiB;
`maximum_tensor_bytes` defaults to/caps at 64 MiB. Maps/pins are snapshotted
before model open. The immutable table receipt is nested in owner verification;
without these inputs, file/runtime-only calls are unchanged and gain no table
claim. The cooperative pre-yield deadline checks elapsed file/runtime/native
creation time and the table stage, but does not interrupt native or blocking
I/O; caller parsing/archive work and prefill are outside it. Gate failures free
the policy before closing its descriptor and before caller access. Receipts and
scores are still publishable only after successful owner exit.

Verification: **413 default passes, 54 optional skips** in 5.48 s; Ruff,
whitespace and touched diagnostics clean. Twenty-seven owner lifecycle cases
cover incomplete inputs, caps, pin/map snapshots, immutable receipts and
pre-yield parity/mutation/deadline cleanup. All **nine integrated profile/native
controls pass** in 94.01 s, peak process RSS 298,648 KiB, swaps zero. For each
of seven supported roles, verified full/parallel/recovery now check all 28
tensor ranges inside the same owner used for actual prefill. Three pre-yield
refusals require exact source-key, matrix-byte-parity and aggregate-budget errors,
emit no scores and restore the registry. Typed scores, zero generation,
independent scales, genuine dispatch and unsupported-profile refusals remain.
The vector-only FFN-down compatibility control passes in 6.80 s.

Failure history: the first integrated FFN-down run failed on a fixture import
because cached `dense-venv` has no pytest. The driver now receives the validated
synthetic fixture on bounded stdin, reusing the parity protocol instead of
installing pytest or changing caches. The same check passed after repair.
Final small test changes made the clock deterministic and released caller
reader/operand references before owner exit: the same 27 lifecycle cases and
integrated FFN-down pass (0.62/13.90 s); the source gate below is rerun for the
commit with **413 passes/54 optional skips** in 8.28 s, Ruff and whitespace clean.
No real-model reads/conversions/execution, downloads, retained candidate,
native build, sanitizer or CTest gate ran. Source archives remain same-producer
synthetic operands, not independent HF authentication. Model metadata semantics,
reader-code/loaded-page/transitive dependency proof, historical real numerical
differences and representative quality remain open; no runtime approval is
implied. User editor files remain untouched. A real-profile comparison requires
separate approval and the recorded resource/provenance prerequisites.

## Descriptor Table Checkpoint (2026-10-13)

After `43611ce`, the complete-table API accepts an optional live
`VerifiedModelPolicy`. Its profile ID/hash, file byte count/hash, ABI and frozen
payload pin must match. The selected file is hashed and independently reread
through its descriptor, never a NumPy-resolved original pathname. The explicit
descriptor-only hasher duplicates a regular descriptor, uses <=64 KiB positional
reads within its initial extent, checks identity/deadline, and closes only the
duplicate. It refuses invalid/nonfinite/boolean budgets/deadlines and pipes.
Owner offsets and lifetimes remain intact; ordinary no-follow paths, typed
mapping checks, all caps and v1 refusals are unchanged. The immutable table
receipt identifies descriptor versus reader-filename access; it grants no new
source/metadata/reader-code/loaded-page proof or runtime approval.

Verification: **403 default passes, 54 optional skips** in 6.56 s; Ruff,
whitespace and touched editor diagnostics clean. Twenty-three focused
descriptor/table cases pass. All **nine complete-stage/native controls pass**
in 60.16 s, peak process RSS 298,516 KiB, swaps zero. Each of seven supported
roles checks all 28 tensor ranges through a stable owner and after replacing
the original pathname. The latter still reads the original descriptor, but
owner exit detects the changed inode metadata and refuses publication;
closed/mismatched owners refuse and registry cleanup returns to baseline.
Normal unchanged owner exit succeeds. Existing synthetic prefill modes,
independent scales, BitNet dispatch, typed scores and zero generation remain.
Full-only/large-state refusals are unchanged; no native hook widening.

Failure history: a finite-deadline follow-up introduced one indentation error
in the hasher; pytest refused collection. The exact block was inspected and
repaired, and the same focused preflight/FFN-down command then passed before
the full gates. No assertions, tolerances or deadlines were relaxed. Disposable
model/path replacement files were restored and cleaned; no source weights,
downloads, conversions, retained candidates, builds, sanitizer or CTest runs.
Next bounded increment: integrate the table gate into the verified-policy
caller before exposing a policy for prefill. Real source/model authentication,
configuration/template/metadata semantics, reader-code/loaded-page and transitive
dependency proof, historical numerical differences and quality remain open.

## Complete Table API Checkpoint (2026-10-13)

After `8e11f7a`, `verify_profile_gguf_tensor_table()` provides a reusable bounded
API for every declared tensor range. Exact unfrozen HF source-name keys and
caller array digests are required; layouts, per-kind limits, aggregate budget
and the minimum payload/file extent are checked before source copies or reader
import. Source mappings/pins are snapshotted; copied operands and frozen blocks
are read-only. File cap is 16 MiB; aggregate default/cap is 64 MiB, charged as
source plus twice output bytes for each dense tensor and three frozen payload
extents. This is not peak process RAM or separate projection validation memory.

One cooperative 30 s deadline (cap 600 s) starts before plan construction and
covers snapshots and all range checks, not caller parser construction or hard
I/O/native timeouts. Existing typed range validation now also handles PQ2:
dimensions/type/encoded rows, v3 table record, mapped range and exact canonical
payload are checked without refitting codes/scales. Public vector/matrix/
convolution APIs and v1 policies remain unchanged. The immutable table receipt
contains a tuple of immutable per-tensor receipts; source files, model metadata
and reader code remain `not_checked`, with no runtime approval.

Verification: **392 default passes, 54 optional skips** in 6.82 s, Ruff,
whitespace and editor diagnostics clean. Twelve pre-import batch cases cover
source keys/pins/dtypes/shapes, invalid aggregate/file/time budgets, impossible
file extents and singleton convolution kernels. Both reviewed pinned-source
operand/file controls passed in **4.27 s**, peak process RSS 290,088 KiB, no
swaps. The complete fixture now uses the API: **nine native cases in 53.79 s**,
peak process RSS 282,584 KiB, no swaps, across the unchanged seven roles/two
refusals. It checks immutable receipts and missing-source, aggregate-budget,
duplicate-table and shared-deadline refusals. A changed frozen code byte with
an updated file digest still refuses exact canonical parity; the independent
changed-embedding probe and native modes also pass. Timing variation is not an
optimization or target-performance claim.

No failing checks occurred in this increment. The impossible-file/kernel
preflights were added after local review and the same focused controls passed.
Producer-generated source archives remain synthetic, not authenticated HF
weights; sparse recurrent limits and independent nonzero toy evidence remain
as below. No native rebuild, new sanitizer/CTest, real weight/candidate reads,
downloads, conversion/execution, fits, retained artifact, quality experiment,
push, branch or subagent. Next: descriptor-path compatibility with the existing
verified owner; real-profile/source/quality approvals remain separate.

## Complete Synthetic Tensor Checkpoint (2026-10-13)

After `7f21b1e`, `PRISM_PARITY_PYTHON` explicitly enables the complete bounded
synthetic tensor stage before existing native prefill. Each supported two-layer
profile has 28 tensors: ten F32 vectors, sixteen unfrozen BF16 matrices, one
F32 convolution and one frozen PQ2 tensor. Typed file gates verify the 27 dense
ranges; frozen dimensions/type/length and canonical codes/scales match exactly.
The two unsupported profiles still refuse before this stage, and the native
hook, numerical tolerances and zero-generation/dispatch assertions are unchanged.

The fixture writer emits a disposable uncompressed NumPy source-operand archive,
capped at 16 MiB. The stage checks its producer bytes/SHA-256 and uncompressed
entry-size sum before `allow_pickle=False` loading. Its keys must match the exact
unfrozen table. Dense operands are checked within each API's 16 MiB maximum;
the model file stays within the existing 16 MiB synthetic bound. The archive,
model and payload are removed on every fixture exit. `PRISM_VECTOR_PYTHON`
retains the previous vector-only mode when the complete-stage variable is unset.

This archive is **producer-generated synthetic evidence**, not independently
authenticated Hugging Face source weights. BF16 matrix operands are the writer's
encoded synthetic bits; the nonzero identity projections/embeddings and sparse
attention drive the existing oracle. Recurrent projection/convolution values
remain zero here; nonzero reorder/convolution evidence is the separate reviewed
upstream toy control. Source-file/reader-code attestation, complete real-model
conversion parity, quality and runtime approval remain absent. Native factory
metadata/tag validation is separate from this tensor-byte stage.

Verification: **380 default passes, 54 optional skips** in 5.51 s, Ruff,
whitespace and editor diagnostics clean. Legacy vector-only FFN-down passed in
**6.70 s**. The complete opt-in native slice passed **nine cases in 92.79 s**,
peak process RSS 254,468 KiB, no swaps: seven supported roles check **189 dense
ranges plus seven frozen payloads**, followed by existing native modes and
refusals. A changed embedding bit with an updated file hash refuses exact matrix
parity before loading; restoring the original hash is checked before prefill.
Host observation: 29 GiB total/20 GiB available RAM, 8 GiB unused swap and
286 GiB free disk; these are not target-device or peak-host guarantees. No
leftover source archives were found in pytest scratch.

Failed-check history: embedded-script imports were over-indented, so the first
complete FFN-down check failed in its parity subprocess. Inspecting and repairing
the full import block made the same check pass; the later corruption/cleanup
check passed in 7.29 s. No tolerance, assertion, deadline or bound was relaxed.
No native rebuild, new sanitizer/CTest, real weight/candidate reads, downloads,
conversions/execution, retained artifact, fits, quality experiment, push, branch
or subagent. Next bounded prerequisite: reusable complete-table receipt/API,
preserving per-tensor/source provenance limits; real-profile gates stay open.

## Convolution File Checkpoint (2026-10-13)

After `daf92f1`, `verify_profile_gguf_convolution()` binds declared F32
convolution views to caller file pins through the existing shared checker.
Its 3D BF16/F32 source must match the 2D reader output after reorder/squeeze.
Public vector/matrix APIs and their limits remain unchanged; the private helper
uses an explicit operand kind. Convolution defaults to a 512 KiB operand budget,
16 MiB file budget, and the existing cooperative 30 s deadline (cap 600 s).
The selected v3 table entry and mapped payload are independently reread; source
layout/budgets refuse before parser import, file hashes or copying the view.

Verification: **380 default passes, 54 optional skips** in 4.09 s, Ruff,
whitespace and editor diagnostics clean. **37 focused file/convolution cases**
passed in 0.71 s: invalid source pins/layouts/nonfinite values, singleton kernel,
byte budgets and all three APIs' pre-import safety gates. Both reviewed
pinned-source controls passed in **3.84 s**, peak process RSS 296,108 KiB,
no swaps. They include actual disposable F32 convolution GGUF views from BF16
and F32 sources, wrong file/source pins, copied views, writable readers, geometry,
deadline and changed payload-bit refusals. Existing vector/matrix controls pass.
The vector-enabled complete FFN-down path passed in **6.50 s** after refactoring;
no new full nine-role native result is claimed.

Failed-check history: the new parametrized test import and setup block were
over-indented, causing two collection failures. The complete local block was
inspected and repaired, then the same command passed. No failure was hidden;
no tolerance, assertion, resource cap or native hook was relaxed.

Receipts verify only the selected range in the caller-pinned file. Source-file
authentication, reader-code attestation, complete-table/tag/tokenizer parity,
immutable storage, concurrent-writer safety and runtime approval remain absent.
Temporary GGUFs are removed. No native changes/rebuilds, new sanitizer/CTest,
real source/candidate reads, retained artifacts, downloads, conversion/execution,
fits, quality experiments, pushes, branches or subagents. Next: complete bounded
synthetic-table verification before native prefill, keeping real-profile gates.

## Convolution Operand Checkpoint (2026-10-13)

After `c783c2f`, `verify_profile_converted_convolution()` checks the pinned
3D source to 2D F32 convolution transformation. BF16 sources are raw
little-endian uint16; F32 sources/outputs are little-endian float32. Checked
profile geometry, finite values, contiguous layout and caller source digest
are required. It preserves exact F32 bits while reordering only V channels
and squeezing the singleton axis. Singleton kernels refuse because upstream
squeeze would no longer match the declared 2D table. Other tensor kinds refuse.

Default `maximum_bytes` is 512 KiB, capped at 16 MiB, bounding
`source.nbytes + 2 * converted.nbytes`, not process RAM or projection validation.
Immutable receipts leave source/converted files `not_checked`; there is no
source issuer, whole-model parity or runtime approval claim.

Verification: **363 default passes, 54 optional skips** in 3.80 s, Ruff,
whitespace and editor diagnostics clean. Four focused cases compare explicit
channel index maps for equal/unequal heads and BF16/F32 sources, rejecting
changed output bits. The two reviewed pinned-source controls passed in
**3.77 s**, peak process RSS 295,796 KiB, no swaps, with nonzero convolution
operands and changed sign-bit refusal alongside existing vector/matrix checks.
No failed checks or numerical tolerance changes occurred in this increment.

No native changes/rebuilds, new sanitizer/CTest evidence, real source/candidate
weight reads, downloads, model conversion/execution, retained artifacts, fits,
quality experiments, pushes, branches or subagents. Existing integration
evidence remains at the checkpoints below, not an unchanged rerun. Next:
extend the shared typed GGUF range checker for F32 convolution, then prepare
bounded complete synthetic table provenance; real-profile work stays gated.

## BF16 Matrix File Checkpoint (2026-10-13)

After `ef51266`, `verify_profile_gguf_matrix()` binds raw BF16 matrix views to
caller file byte/SHA-256 pins. It shares `_verify_profile_gguf_operand()` with
the unchanged public vector API. Matrix storage must be the reader's uint8
encoded rows, with reversed GGUF dimensions, exact byte count, shared read-only
mmap/range and a matching v3 tensor-info record. The raw little-endian uint16
view is copied only after source geometry/dtype/pin syntax/contiguity and operand
budget preflight. The shared file cap is 16 MiB (matrix default 16 MiB, vector
default 64 KiB), with the existing cooperative deadline and two file hashes.
Matrix `maximum_bytes` remains the 1 MiB default/16 MiB operand cap.

Receipts bind one selected tensor range and retain `source_file_verification`
and `gguf_reader_verification: not_checked`. Parser construction remains outside
the wrapper budget. These are not complete-model table/tag/tokenizer parity,
source issuer authentication, immutable storage, concurrent-writer safety or
runtime approval. Frozen PQ2/F32 convolution remain separate and refused.

Verification: **359 default passes, 54 optional skips** in 4.48 s, Ruff,
whitespace and editor diagnostics clean. **37 focused matrix/file cases**
passed in 0.75 s. The two pinned controls passed in **3.70 s**, peak process RSS
295,260 KiB, no swaps: all 16 previously verified matrix operands are written
to disposable <=16 MiB BF16 GGUF and checked through actual reader views.
Wrong file pins, operand budgets, copied mappings, reversed/wrong shapes and
changed payload bits refuse. Existing vector name/type/range, metadata,
deadline and mutation controls continue to pass. The vector-enabled complete
FFN-down native control passed in **6.61 s** after the shared refactor; the
full nine-role result remains at `7326a14`, not a new all-role rerun.

Failed-check history: review found operand budget validation after the mapped
copy. Two new pre-import regressions reproduced `ModuleNotFoundError` rather
than early `ValueError` for oversized inputs. Moving source/budget preflight
ahead of reader import, file hashes and copies repaired both; the same checks
and complete focused slice passed. No assertion, tolerance or deadline was
relaxed. Temporary GGUFs are removed; no data is retained in Git or caches.

No native changes/rebuilds, new sanitizer/CTest evidence, source/candidate
weight reads, real model conversion/execution, fits, quality experiments,
downloads, pushes, branches or subagents. Next code-only prerequisite: F32
convolution squeeze/V-channel operand parity, then synthetic complete-table
provenance integration. Real-source/runtime work still needs separate approval.

## BF16 Matrix Operand Checkpoint (2026-10-13)

After complete-vector integration `7326a14`,
`verify_profile_converted_matrix()` adds exact BF16 matrix operand comparison.
Raw little-endian uint16 source/output arrays must match checked 2D geometry,
contiguous layout, finite BF16 bits and the caller source SHA-256. The default
`maximum_bytes` is 1 MiB, capped at 16 MiB, bounding
`source.nbytes + 2 * converted.nbytes`; it is not peak process RAM or the separate
projection validation budget. Copies preserve ordinary caller ownership.

Only declared BF16 matrix bit-preserving transforms execute: identity, V-only
QKV row reorder, Z/alpha/beta head row reorder and output-projection column
reorder. Equal K/V head counts require no reorder. Frozen PQ2 replacements,
F32 vectors/convolution, unsupported dtypes and oversized operands refuse;
codes/scales are never regenerated. The immutable receipt explicitly leaves
source-file and converted-file verification `not_checked` and grants no runtime
approval. No GGUF matrix/file binding or whole-model parity follows yet.

Verification: **350 default passes, 54 optional skips** in 5.64 s, Ruff,
whitespace and editor diagnostics clean. **21 focused matrix cases** in 0.47 s
cover exact identity, independent explicit head-index maps for five roles with
equal/unequal heads, unreordered-output rejection, invalid source pins/budgets/
layouts/types, nonfinite bits and signed-zero changes. The modified pinned-source
controls passed **two cases in 3.52 s**, peak process RSS 268,340 KiB, no swaps.
The BF16-source case additionally checks all 16 unfrozen matrix operands in the
synthetic profile against the reviewed upstream Torch transform methods and
rejects changed sign bits. Existing vector/file controls also remain covered.
No failing checks or relaxed tolerances occurred in this increment.

Native prefill remains the prior nine-case/70-vector result at `7326a14`; it was
not rerun unchanged for operand-only work. No native rebuild, new sanitizer/
CTest, real source/candidate weight reads, retained arrays/artifacts, model
conversion/execution, fitting, quality experiment, downloads, push, branch or
subagents. User settings and caches remain untouched. Next bounded step: reuse
the existing file-range checker for BF16 matrix views, retaining the vector API
and its refusal behavior; F32 convolution and complete-table/source proof remain
separate prerequisites.

## Complete Synthetic Vector Checkpoint (2026-10-13)

After `5240c26`, the existing complete synthetic profile-prefill fixture now
optionally checks all ten one-dimensional F32 tensors against generated BF16
source operands before native loading. Enable this stage explicitly with
`PRISM_VECTOR_PYTHON` pointing at the existing dense Torch environment. Without
that variable the native control still runs, but provides no new vector-stage
evidence. The full-only/large-state refusal profiles remain refused before this
stage; no native test-hook bound was enlarged.

The source operands are synthetic zeros, except recurrent norm ones. The writer
now emits `ssm_a = -1`, matching the pinned `-exp(0)` transform; its previous
zero constant could not represent a finite zero A_log operand. Recurrence
projection weights in this fixture remain zero, so this does not add recurrence
quality evidence or alter its independent prefill oracle. A focused FFN-down
check verified that correction in 5.22 s; vector-enabled FFN-down passed in
6.66 s. `_profile_parity_fixture()` reuses one <1 MiB stdin serializer across
the toy and complete-model controls. No source or converted array is retained.

Verification: **329 default passes, 54 optional skips** in 3.51 s, Ruff,
whitespace and editor diagnostics clean. The vector-enabled complete gate
passed **nine cases in 44.03 s**, peak process RSS 232,924 KiB, no swaps:
ten vector ranges for each of seven projection roles (70 checks), followed by
unchanged full/chunked/reset/reordered/parallel/maximum/error-recovery and
verified-owner modes; both refusal profiles still refuse. Typed options, zero
answer generation, actual BitNet dispatch, exact FP16 scales and 2e-5 prefill
oracles remain unchanged. The shared serializer's two BF16/F32 toy file controls
passed in **2.73 s**. No failed checks occurred in this increment.

Before the serial gates the host reported 23,296 MiB available RAM, no used swap
and 286 GiB available overlay disk. Process peak counters are not host or target
device guarantees. Fixture cleanup removes the generated GGUF/projection files;
cached sources/libraries, frozen candidate and user settings are untouched.
No native sources/libraries/hooks changed, no new sanitizer/CTest evidence,
source-weight hashes, real-model conversion/execution, fitting, quality tests,
downloads, pushes, branches or subagents. The vector receipt still does not
authenticate source files or reader code, or prove other tensor bytes.
Next bounded prerequisite: exact BF16 matrix operand transforms and refusals;
full-table/model parity and real-source/runtime approval remain open.

## GGUF Vector Binding Checkpoint (2026-10-13)

The explicit resume now adds `verify_profile_gguf_vector()` beside the unchanged
operand-only comparator. A read-only little-endian `GGUFReader` view must match
the profile's F32 vector name/type/shape/byte count and share the reader's mmap
at the declared offset. The wrapper checks v3 header counts, alignment/table end
and the selected tensor-info record against actual file bytes, then checks the
selected payload range. Cached name/alignment fields alone are not trusted.
Two bounded unbuffered whole-file hashes must match the caller's byte/SHA-256
pins around exact operand parity. Default file budget is 64 KiB, cap 16 MiB;
one cooperative deadline defaults to 30 s and is capped at 600 s.

This binds **one vector range**, not the complete converted table, model tags,
tokenizer, matrices/convolution or source files. The caller constructs the reader
first in the separately pinned environment: this wrapper does not bound parser
construction or attest its imported code. The immutable receipt now marks
`converted_file_verification: caller_sha256_and_reader_range_match`, retains
`source_file_verification: not_checked`, and explicitly adds
`gguf_reader_verification: not_checked`. File pins' issuer, immutable storage,
concurrent writers and hard I/O/native deadlines are not proven. No runtime
or real-profile execution approval follows.

Verification: **329 default passes, 54 optional skips** in 3.32 s, Ruff,
whitespace and editor diagnostics clean. **22 focused plan/vector/file cases**
passed in 0.45 s. The two reused BF16/F32 pinned-source controls write seven
vectors to a disposable <64 KiB GGUF and consume actual mapped tensor views.
Both controls pass in **2.82 s**, peak process RSS 241,092 KiB, no swaps.
They refuse wrong digests/budgets, writable readers, missing/duplicate names,
wrong types/offsets, copied views, deterministic deadline expiry, stale name/
alignment metadata, changed file payloads and mid-parity mutation. No failed
checks occurred in this increment; no tolerance or native bound was relaxed.

The cached converter/reader source files are clean at the pinned Prism revision.
Temporary GGUFs are removed by `TemporaryDirectory`; no payload or report was
retained. No source/candidate weight bytes, real checkpoint, native library,
v1 factory or prefill hook was changed/read/rebuilt. No downloads, conversions
of a real model, quality experiments, subagents, push or branches. Native
actual-loader, sanitizer and CTest evidence remains at the prior checkpoints.
Next local integration: exercise vector binding against the existing complete
synthetic profile model before bounded prefill; no hook expansion is needed.

## Conversion Plan And Vector Parity Checkpoint (2026-10-12)

Following bounded runtime hashing `c18e90f`, `profile_model_conversion_plan()`
reuses the checked complete-model table and declares the pinned source-to-output
mapping, arithmetic/storage dtypes, normalization +1, negative exponential,
convolution squeeze and grouped-to-tiled V-head row/column transforms. It carries
profile/source/converter declarations and the exact frozen payload hash. The
selected PQ2 tensor is explicitly a frozen replacement, not dense conversion.
The plan does not execute transforms or verify any source/converted bytes.

`verify_profile_converted_vector()` adds a separate operand-array gate for
one-dimensional F32 outputs from declared BF16/F32 sources. BF16 input is raw
little-endian uint16; F32 source/output arrays must be little-endian float32.
Exact shapes, contiguous layout, finite operands and caller source SHA-256 are
checked before importing Torch. The operand/reference budget is
`source.nbytes + 2 * converted.nbytes <= maximum_bytes`, default/cap 64 KiB;
it is not a process-RAM bound or the separate PQ2 validation budget. Copied
operands are transformed with Torch under inference mode and compared as bytes,
including signed zero. An immutable receipt explicitly leaves source-file and
converted-file verification `not_checked` and grants no runtime approval.

Verification: **322 default passes, 54 optional skips** in 6.86 s, Ruff,
whitespace and editor diagnostics clean. **15 focused default cases** cover
the plan, equal-head/no-reorder profiles, all seven frozen projection roles and
pre-import vector refusals. The reused pinned-source AST toy control now checks
the plan plus seven exact vector roles for both BF16 and F32 sources: **two
cases in 2.58 s**, peak process RSS 237,400 KiB, no swaps. It verifies exact
normalization, recurrent norms, A_log and dt-bias bytes and rejects sign-bit or
one-bit changes. Existing toy tolerances were not changed. Pinned upstream
methods are exercised on synthetic operands only; model-repository code is
never imported. The earlier nine-role actual-loader gate remains at `c18e90f`,
43.53 s; it was not rerun unchanged for this operand-only increment.

Failed-check history: the first plan test omitted the fixture's unequal K/V
head reorder; its expectations were corrected against the fixture's 2/4 heads.
Seven ownership cases then treated tuple inventory shapes as lists; a tuple
snapshot of the matching source row repaired the assertion. The cached Torch
environment lacks pytest, so the controller passes a verified synthetic fixture
over bounded stdin rather than installing software there. Both optional cases
now pass; no failures were hidden or numerical tolerances relaxed.

No source weights, real converted checkpoint or frozen candidate bytes were
read/refit. No new artifact was retained; synthetic stdin is <1 MiB. Native
libraries/factories/hooks, caches and user settings are untouched. No downloads,
push, branch, subagent, new sanitizer/CTest evidence or model-quality claim.
Next: bind this bounded vector gate to actual synthetic GGUF tensor views;
matrix/convolution parity, source-file authentication, full-model parity and
production scoring remain open. Real-profile work still requires new approval.

## Bounded Runtime Hashing Checkpoint (2026-10-12)

Following symbol-origin checkpoint `4973863`, the strict runtime pin checker
now uses `bounded_native_file_record()` rather than hashing to EOF after
preflight. Each Linux regular-file descriptor is opened nonblocking/no-follow,
with unbuffered reads of at most 64 KiB. Only its initial extent, capped by the
corresponding caller pin, can be read. Descriptor and path device/inode/size/
mtime/ctime must still match after hashing. Truncation, growth, replacement,
FIFO and unexpected symlink inputs refuse. Resolved linker aliases remain
supported through the defining-symbol paths.

`maximum_seconds` defaults to 30 and is capped at 600; one cooperative deadline
covers all three or six runtime files per check. Invalid, Boolean and nonfinite
budgets refuse. The 64 MiB aggregate ceiling remains unchanged. This is not a
hard timeout on blocked I/O or native calls, nor immutable-storage, loaded-page
or transitive-dependency attestation. The model-file owner also now reads
unbuffered; legacy `native_backend_dependencies()` behavior is unchanged.

Verification: **307 default passes, 53 optional skips** in 11.71 s, Ruff and
whitespace clean; **56 focused cases** in 0.59 s cover read ceilings, partial
reads, mutation, aggregate/final deadlines, invalid budgets and owner cleanup
on runtime deadline refusal. Editor diagnostics are clean. Pylance signature
validation reports all five checker call sites compatible. Its first report
flagged two heterogeneous test keyword dictionaries; explicit keyword arguments
in the expanded deadline cases resolved that test-inference limitation.
There were no failing runtime checks in this increment.

The unchanged synthetic native caller gate passed **nine cases in 43.53 s**,
peak process RSS 110,640 KiB, no swaps. Exact FP16 scales, 2e-5 oracles,
generation/dispatch assertions and v1 refusals remain unchanged. No native
source/library rebuild, new sanitizer/CTest evidence or real-profile run is
claimed. Fixtures removed generated models/payloads; cached sources/libraries,
frozen candidate and user editor files were untouched. No downloads, push,
branches, source-weight hashing or subagents.

Next bounded provenance prerequisite: independent converted-tensor equality
against caller-provided synthetic source operands. Real source authentication,
conversion/parity, production scoring and quality/runtime approval stay open.

## Symbol Origin Checkpoint (2026-10-12)

The user pushed `1426bce` and resumed continuous autonomous code-only
caller/provenance preparation. `_model_runtime_library_paths()` now resolves
the defining objects of known C symbols through libc `dladdr`, using the ABI
verified in the installed system header. Bridge identity follows the v2 factory
address, not mutable `CDLL._name`; CPU/base origins are cross-checked against
the unchanged bridge witness. An alias-request regression reproduced the old
filename lookup and now passes.

With `scorer=...`, the standalone checker requires six exact records: bridge,
GGML CPU/base, bounded prefill scorer, llama and GGML registry. All six paths
come from defining symbol addresses. The owner accepts `runtime_scorer=...`,
retains it through the context and requires corresponding runtime pins. It
records `runtime_file_scope` and all six digests, rechecking on entry/yield/exit.
Without a scorer the three-module mode remains available. The 64 MiB aggregate
preflight ceiling is unchanged; cached six-module files total 7,439,312 bytes.
No native source, ABI, factory, hook bound or library was changed/rebuilt.

These are **selected inference-module file identities**, not transitive ELF
dependency closure or loaded-page attestation. System/runtime libraries, pin
issuer authenticity, concurrent writes, source weights and converted-tensor
parity remain outside this proof. Runtime hashing still uses the existing EOF
hasher after byte preflight; bounded reads/deadlines are the next local repair.
The scorer witness refers to the existing synthetic test library, not a new
production scoring API. No real-profile execution is authorized.

Verification: **283 default passes, 53 optional skips** in 13.44 s, Ruff,
whitespace and editor diagnostics clean. Thirty-two focused owner/runtime cases
cover alias independence, both module scopes, incomplete pin sets and scorer
without pins. Actual native FFN-down checks first verified the three-module
ABI in 6.69 s, then six modules in 7.04 s. The complete synthetic caller gate
passed **nine cases in 42.12 s**, peak process RSS 110,908 KiB, no swaps.
Existing 2e-5 oracles, generation/dispatch checks and native refusals remain.
No new sanitizer, CTest or full 25-control result is claimed; the earlier Python
leak-check limitation remains.

Failed-check history: the alias regression failed with `FileNotFoundError` at
the requested filename before the actual-origin repair. Two positive six-module
unit cases then omitted their scorer argument and hit the intended strict-key
refusal; forwarding that argument repaired the same checks. No production
model file or source/candidate weight payload was read. Only cached runtime
files and bounded synthetic fixtures were used; generated model/payload files
were deleted by fixtures. User editor files and immutable caches remain
unchanged; no push, branch, download, conversion, fit or subagent.

## Runtime File Pin Checkpoint (2026-10-12)

Following verified owner `e6f6824`, the same approved code-only scope now adds
`verify_model_runtime_files()` and optional `expected_runtime_files` owner
inputs. Strict caller records declare bytes/SHA-256 for `bridge`, `ggml_cpu`
and `ggml_base`, with a 64 MiB aggregate preflight ceiling. The existing
`native_backend_dependencies()` helper hashes actual GGML paths returned by
the bridge's `dladdr` witness; the bridge file uses the supplied CDLL's resolved
filename. Regular-file sizes are checked before that existing hasher runs, and
observed paths/bytes/digests must match. Pin records are copied so caller
mutations cannot alter the active comparison.

When requested, the owner checks these files before policy creation, before
yielding and at exit, and records flat bridge/CPU/base digests in its immutable
receipt. Exit-time runtime refusal still frees the policy and closes its model
descriptor. Omitting runtime pins explicitly records `runtime_file_verification:
not_checked`; it never becomes inferred approval. The seven-role fixture binds
the independently recorded bridge digest and unchanged pinned GGML hashes in
verified full/parallel/recovery modes.

This is **partial file provenance**, not complete loaded-runtime attestation.
It does not prove loaded memory pages equal current disk bytes, attest bridge
symbol origin independently of the requested CDLL filename, bind llama/registry/
scorer binaries, authenticate the expected pins' issuer or enforce immutable
libraries. The byte ceiling is preflight validation around the existing hasher,
not a hard read/deadline limit under concurrent file growth. Concurrent library
changes are unsupported. Full dependency closure, independent converted-tensor
equality, source payload authentication and production scoring remain open;
there is no new real-model execution authorization.

Verification: **272 default passes, 53 optional skips** in 8.37 s, Ruff,
whitespace and editor diagnostics clean. **21 focused owner/runtime cases**
cover digest/size/budget/missing-path/path mismatch, exit-time failure cleanup
and mutable caller pins. The updated synthetic caller gate passed **nine cases
in 42.75 s**, peak process RSS 110,872 KiB, no swaps. Existing oracle tolerances,
hook bounds, native code and all v1 paths are unchanged. No runtime rebuild,
new sanitizer/CTest or full 25-control gate is claimed. The prior Python leak
limitation remains.

Failed-check history: the initial runtime-pin fixture raised `NameError` for a
missing parent `hashlib` import before native execution; adding the local import
repaired the same FFN-down check, which passed in 5.54 s. A patch initially listed
driver fragments out of file order and was refused; the reordered patch applied.
No source-weight inputs, downloads, conversions, fits, new candidates or quality
experiments ran. Synthetic model/payload files were deleted by fixture cleanup;
only small existing runtime-library files were read for hashes. User editor
files and immutable cached libraries remain unchanged; no push/branch/subagent.

## Verified File Owner Checkpoint (2026-10-12)

Committed as `e6f6824`. The user resumed from pushed `af62d7e` and explicitly approved code-only
production-caller/provenance preparation using bounded synthetic fixtures,
without real conversions/execution, downloads or test-hook widening.
`verified_profile_model_policy()` is a separate context-managed preparation
API beside the unchanged bare-handle factory. It requires caller-supplied exact
file bytes and SHA-256, defaults to a 16 MiB hashing budget, and refuses budgets
above the checked specification's whole-file maximum. Hashing uses 64 KiB reads
with a cooperative 30-second default deadline, capped at 600 seconds; this is
not a hard timeout for blocked filesystem I/O or native execution.

The Linux-only owner rejects symlinks/nonregular inputs without blocking on
FIFOs, verifies the opened descriptor's size/digest and device/inode/size/time
identity, and passes `/proc/self/fd/N` to the unchanged native factory. It keeps
the descriptor, library, copied read-only PQ2 blocks and ctypes record alive.
Context exit frees the policy before closing the file, clears handle/override
pointers, and refuses public path/record access afterwards. File changes during
creation or use refuse; replacement cannot redirect the already-open descriptor.
The immutable receipt binds profile/revision, converter declaration, expected
model-file identity and projection digest, while explicitly marking source
payload verification as declared and runtime approval as not granted.

This authenticates file bytes against the caller's expected digest, not the
digest's provenance, source weights, conversion correctness, loaded binary
identity or tokenizer behavior. It does not make the file immutable, support
concurrent writers or enforce cleanup of arbitrary caller-created contexts.
Destroy every model/context/buffer before leaving the owner and publish scores
only after successful context exit. There is still no production scoring ABI;
the compiled test hook remains limited to synthetic geometry and files.

Default tests cover wrong digest/size/budget/boolean/deadline, nonregular and
symlink inputs, caller exceptions, creation-time changes, in-place changes,
replacement, record-construction failure, cleanup ordering, immutable receipts,
retained copied payload and post-exit guards. The existing seven-role fixture
adds producer-recorded file hashes and verified full/parallel/recovery modes.
Legacy refusal modes remain separate and unchanged. Each compiled call releases
all native users before the owner; registry restoration and unchanged 2e-5
arithmetic/conditional-score tolerances pass with zero generated tokens.

Verification: **264 default passes, 53 optional skips** in 12.96 s, Ruff,
whitespace and editor diagnostics clean. Thirteen focused owner cases pass.
The joint caller gate passed **nine cases in 42.62 s**, peak process RSS
110,540 KiB, no swaps; after the final post-exit property guard, a focused
FFN-down replay passed in 6.72 s. No native code/library was changed or rebuilt,
and no new sanitizer/CTest success is claimed. Prior Python-loaded leak-check
limitations remain. Editor test discovery found no tests, so direct pytest was
used without changing user settings. The first focused run passed six tests
but Ruff rejected a local lambda (E731); a named helper repaired the same slice
and the same checks then passed. No numerical or safety assertion was relaxed.

Synthetic files were deleted by fixture cleanup. No new weights/source copies,
real conversions/inference, source-payload hashes, retained candidates, fits,
promotion or quality experiments ran. User `.vscode/` remains untouched; no push,
branch or subagent. Next caller/provenance work must independently bind converted
tensor payloads and loaded runtime provenance; a whole-file receipt alone cannot
establish v1/v2 tensor parity or authorize a real profile's execution.

## Metadata Budget Checkpoint (2026-10-12)

Following planning commit `7cbc1ff`, the user explicitly approved a code-only
v2 metadata-budget correction using synthetic fixtures and cached metadata,
with v1/test-hook bounds unchanged and no real-model execution authorization.
Python and the native header now declare **32 MiB metadata/table/alignment
overhead and at most 256 keys**. The whole-file cap remains declared tensor
bytes plus that allowance, with <=19 GiB of tensor data. A pinned GGUF callback
limits parser reads to 32 MiB in <=64 KiB chunks; magic/version, exact tensor
count and key count are checked in the first 24 bytes before parsing. Padded
data offset must also fit 32 MiB. No v1 factory body or prefill-hook bound changed.

The existing synthetic policy control accepts metadata over 1 MiB, near-32-MiB
metadata, exactly 256 keys and the exact whole-file ceiling. It refuses 257 keys,
metadata beyond the parser budget even when the whole file fits, one-byte file
overflow and enormous header counts. Refusals clear both outputs and preserve
the registry baseline; prior table, provenance-declaration and canonical-payload
checks remain. The Python test recomputes tensor bytes and verifies cap agreement.

The cached 19,989,325-byte MiMo tokenizer JSON was authenticated against its
profile pin, without reading weight or frozen-artifact payloads. Static source
token strings encode to 4,848,741 bytes and merges to 5,086,368; 248,320 token
types add 993,280. This 10,928,389-byte subtotal excludes fallback vocabulary
entries, other metadata, tensor information/alignment and converter-specific
token handling. It supports headroom planning, not a measured final GGUF or
native tokenizer compatibility result. The
[research audit](research-audit.md#real-v2-planning-limits) preserves the original
blocker and accounting limits.

Verification: **251 default passes, 53 optional skips** in 5.19 s, Ruff,
whitespace and editor diagnostics clean. The focused metadata/native-Python
pair passed two cases; the applicable compiled gate passed **23 synthetic
controls in 41.57 s**, including unchanged v1 complete-model refusals, twelve
tiny dense/v1/v2 prefill combinations and all seven caller roles plus two
resource/architecture refusals. Peak process RSS 197,284 KiB, no swaps. The
full 25-control frozen-projection command was not rerun. Disposable GCC
ASan/UBSan/leak early/late controls and **four tiny v2 prefill cases** passed
in 6.30 s; peak replay RSS 86,284 KiB, no swaps. Project bridge/control were
instrumented, cached llama/GGML were not. Large metadata boundary cases were
not sanitizer-replayed, and the prior Python-loaded leak limitation remains.
No new CTest result is claimed.

Failed-check history: the initial >1-MiB regression reproduced native status 6
at the old cap; the callback/budget correction passed that same check. A writer
inspection probe requested a nonexistent `get_metadata` method; confirmed
`kv_data` counting and controlled filler sizes were used instead. The first
tokenizer planner treated `ProfileFile` as a dictionary and refused before any
file read; verified `bytes`/`sha256` attributes repaired the check. No numerical
tolerance or existing v1/test-hook safety gate was relaxed. Existing GCC
`fclose` deleter-attribute warnings remain.

Only isolated project wrappers were rebuilt. Upstream source revisions, clean
worktrees and both immutable GGML hashes were rechecked. Synthetic policy files
were removed by pytest's temporary-directory retention cleanup; prefill fixtures
deleted their files. Disposable sanitizer binaries were removed after recording
the results. No downloads, source copies, real conversion/inference, fit, new
candidate, promotion or quality scoring occurred; user `.vscode/` is untouched.
Next: separately scoped production caller/provenance preparation and explicit
real MiMo comparison approval. Never widen the bounded test hook for this.

## Projection Role Checkpoint (2026-10-12)

Committed as `fab3cc0`, following caller checkpoint `d653efc`.
Synthetic profile/inventory/manifest-to-
prefill checks cover FFN gate/up/down and full-attention Q/K/V/output. Runtime
code, bounds and v1 factories are unchanged in this increment. All seven records
verify exact retained payload bytes after garbage collection. Each role passes
full/chunked/reset/reordered/parallel/128-token/recovery and caller-refusal modes,
with typed conditional scores, one repack and zero generated answer tokens.
Attention roles include a nonzero sparse attention reference and a distinct
peer-prefix reference; recurrence remains zero in these profile fixtures and is
covered separately by the existing nonzero tiny controls.

Q/K/V use 1,024-wide input and 32-dimensional heads, with sparse per-head codes;
attention-output uses hidden 128, FFN 256 and 16 x 64 query heads so its 1,024-wide
input satisfies unchanged rotation eligibility. No bound was raised. FP16 cache
matmuls round rotated query and probability operands to FP16 as well as stored
K/V; pinned CPU traits confirm the F16 dot operand type. BF16 dense operands and
group-A8 BitNet arithmetic retain their separate contracts. These are bounded
integration/arithmetic controls, not whole-model quantization or quality evidence.

Verification: **251 default passes, 53 optional skips**, Ruff, whitespace and
editor diagnostics. The focused joint role gate passed **16 cases in 32.85 s**:
seven native-record cases, seven prefill roles and two resource/architecture
refusals. Peak process RSS 96,380 KiB; no swaps. The prior unchanged 20-control
native gate and standalone sanitizer evidence remain at `d653efc`; they were
not rerun merely for reassurance. The development command now selects 25 controls
if run in full; that expanded full command is not claimed as measured here.

Failed-check history: Q/K differed by up to about 0.00062 when the reference
rounded only cache values. Pinned F16 CPU dot traits justified rounding query/
probability operands, and the same eight-case role gate passed without changing
2e-5 tolerances. Attention-output's alternate small geometry initially failed a
nonzero-FFN assertion for one random toy coordinate; setting its gate-driving
embedding coordinate explicitly nonzero repaired the fixture, not the assertion.
All model/payload files were deleted by fixture cleanup. No raw traces, new runtime
builds, conversions of real weights, downloads, fits, retained candidates,
held-out scores or quality experiments were introduced. User editor files,
cached pins and the sole frozen candidate remain untouched; no push/branch/subagent.

### Next Approval Boundary

All agreed bounded synthetic P4 prerequisites are now implemented. A meaningful
next runtime claim requires a separately approved real-profile comparison, not
more repetitions of unchanged synthetic gates. P2 and P5-P7 are not implicitly
authorized. Before requesting a real run:

- Select the exact profile/revision and authorize runtime execution separately
  from file verification. MiMo alone has existing execution evidence; Defiant
  Fable still needs its own dense/tokenizer/native reference gates and weights.
- Define a production caller beside the bounded test hook; never raise its
  16 MiB/1,024-width limits to smuggle in a real checkpoint. Bind source-file
  identity, converter flags, typed option IDs and artifact provenance explicitly.
- For MiMo v1/v2 comparison, use the sole frozen codes/scales without refitting.
  The two policies require different metadata tags, so approved temporary files
  must have independently verified identical tensor payloads; native declarations
  alone do not authenticate them. No another-profile artifact is approved.
- Write numerical acceptance against the same frozen mixed-model execution,
  separating v1/v2 parity from the unresolved dense-reference gap. Keep dispatch,
  zero generation, exact independent scales and error/recovery checks mandatory;
  do not relax existing tolerances to conceal a difference.
- Budget conversions/files, serial inference, measured host RAM/swap/disk and
  cleanup before launch. Recheck current headroom and exact artifact byte costs;
  per-process counters from synthetic controls are not whole-model estimates.
  No new conversion, download, refit or retained candidate may begin implicitly.

No new real-model approval has been granted. The preparation below identifies
a historical metadata-budget blocker, now corrected by the current checkpoint;
production-caller, provenance and real-execution approvals remain open.

## Real Comparison Preparation (2026-10-12)

After `fab3cc0`, metadata-only planning exposed a real v2 prerequisite: both
the Python specification and native factory allow only 1 MiB beyond tensor
bytes. MiMo's full vocabulary needs 1,986,560 bytes for GGUF string-length
prefixes alone, before token text or other metadata. The
[research audit](research-audit.md#real-v2-planning-limits) records the pinned
writer check and exact static BF16/F32/PQ2 byte accounting. This cap was not
changed, and no real model was loaded, converted or payload-hashed.

Two differently tagged mixed-model files have a **33.197 GiB payload floor**;
retaining a dense base as well gives **49.877 GiB**, before metadata, alignment
and staging. These are not final disk budgets. The F32 small-tensor conversion
adds 2,122,752 bytes to the older source-dtype plan. Existing frozen-array sizes
match their manifest; this size check is not fresh payload authentication.
Current host observations were about 22 GiB available RAM, 8 GiB unused swap
and 291 GiB available disk. Neither those values nor synthetic RSS guarantees
whole-model runtime headroom.

At planning commit `7cbc1ff`, next approval needed a bounded v2 metadata-budget correction using
synthetic fixtures and cached tokenizer metadata only, keeping v1 and the
16 MiB/1,024-width prefill hook unchanged. It must bound header bytes/counts,
check the revised Python/native agreement and preserve oversized-file refusals;
do not merely remove the cap. Real conversion/execution remains a separate
later approval, with a production caller beside the test hook.

For that later MiMo comparison, keep only the two necessary tagged files if
possible, reuse cached source and frozen codes/scales, authenticate their tensor
payload equality independently, and run conversion and inference serially.
Set exact final disk/RAM/token/deadline budgets before launch; monitor host
availability and swap growth rather than accepting per-process RSS as proof.
On a budget or numerical refusal, stop and diagnose, not rerun seeking green.
Delete approved temporary models/staging after verifying the small report;
preserve cached sources, libraries and the frozen candidate. Defiant Fable's
weights/P2 execution and all quality experiments remain separately gated.

Planning verification: static geometry/dtype accounting and frozen file sizes
were checked without reading their payloads. The first in-memory writer probe
omitted its required `add_vtype` argument and refused with `TypeError`; inspecting
the pinned method and supplying `False` verified the eight-byte array-element
prefix. No real-model failed check or new sanitizer result is claimed. The
unchanged source gates at `fab3cc0` were not repeated for this documentation-only
preparation; scoped documentation/link/table/whitespace checks apply instead.

## Profile Caller Checkpoint (2026-10-12)

Following `c825cfc`, Python profile/header-inventory/manifest validation now
reaches actual BF16/PQ2 model loading and typed zero-generation prefill.
`native_model_specification()` retains its ctypes backing arrays; default tests
delete the original payload, collect garbage and verify its exact retained hash.
The project-owned `libprism_bitnet_prefill_test.so` accepts the checked record,
borrowed policy and overrides. The caller frees its policy only after every
model/context/buffer is destroyed. C++ exceptions do not cross this test-only C
boundary. Registry restoration passes success, refusal and distinct-policy cases.
No generic CLI or per-model execution approval was introduced.

The synthetic hybrid fixture has two layers, hidden/FFN widths 1,024, vocabulary
32 and 28 text tensors. Layer-0 FFN-down/up each dispatch the grouped BitNet-derived
kernel with exact independent FP16 scales. Ordinary matrices use the pinned
GGUF BF16 codec, with F32 norms/convolution/state parameters. Declared profile
A/B/C IDs match its toy vocabulary. Rotation-1024 eligibility remains unchanged.
Attention/recurrence weights are zero in this fixture; existing nonzero tiny
controls remain separate arithmetic coverage. Full/chunked/reset/reordered/
parallel/128-token/recovery modes match grouped-A8/BF16 references at unchanged
2e-5 relative/absolute tolerance, with one repack and zero generated tokens.
BF16 dense references round both weights and inputs, matching the pinned CPU
dot type; this does not resolve historical real-model numerical differences.

The hook caps files at 16 MiB, layers at four, hidden/FFN/target dimensions at
1,024 and vocabulary at 64. Loaded embedding/FFN/vocabulary agreement and
attention/SSM bounds are checked before context allocation: heads <=16/4,
head dimensions <=256, convolution <=4, inner <=4,096, state <=128 and rank/
groups <=32. Both layer types must exist. Full-attention-only and oversized-SSM
fixtures refuse before context creation. Null arguments, wrong-policy buffers,
shape/hidden/FFN/vocabulary mismatches, oversized files/widths, unknown modes and
kernel errors refuse without scores. Policy tags/digests remain declarations,
not authentication of source bytes or untouched dense weights.

Verification: **246 default passes, 48 optional skips**, Ruff, whitespace and
editor diagnostics. The expanded pinned native gate passed **20 controls in
43.29 s**, peak process RSS 946,360 KiB, no swaps. The final FFN-agreement guard
then passed four focused caller cases in 9.71 s and the refreshed default gate.
Earlier local checks covered 11 adapter/native-policy cases and two retained-
storage cases. Only isolated project wrappers/test libraries were rebuilt;
no new CTest result is claimed.

Sanitizer results are separated: Python-loaded prefill computed the expected
report but exit-time leak detection failed with 1,345,553 bytes in 1,239 CPython/
NumPy allocations. The no-prefill baseline had exactly the same counts, and
neither report had project-source allocation frames; this is not a leak-clean
Python gate. A standalone driver mechanically derived from the identical checked
native specification passed nine full/chunked/reset/reordered/parallel/maximum/
recovery/error/unknown modes with ASan, UBSan and leak detection enabled. Bridge
and test library were instrumented; cached llama/GGML were not. Compile peak RSS
was 569,700 KiB; the last native replay peak was 100,232 KiB. No leak check was
disabled or suppressed. That replay preceded the final simple FFN-agreement
guard, whose compiled refusal checks are recorded above.

Failed-check history: 128/256 widths failed unchanged rotation-1024 eligibility;
eligible bounded widths repaired the fixture. The GGUF writer omitted an empty
merge array and Prism refused it; the existing one-merge vocabulary repaired it.
GDB traced the all-attention fixture's abort to an empty recurrent-state input,
not symbol visibility; valid hybrid data and pre-decode refusal avoid that pinned
runtime limitation without upstream edits. A weights-only BF16 oracle left
0.003-0.005 logit differences; pinned CPU traits proved BF16 input conversion,
and correcting the oracle passed without tolerance relaxation. The disposable
sanitizer generator replaced a macro substring and emitted an invalid float
literal; explicit template placeholders/float formatting repaired it. Python
allocator failures remain recorded, not rewritten as successes.

Synthetic models/payloads were deleted by fixtures/probes. Disposable instrumented
libraries, generated driver, allocator reports and raw traces were removed after
recording the evidence. No download, source copy, real-model conversion/inference,
fit, retained candidate, promotion, held-out scoring or quality experiment ran.
Frozen candidate and user `.vscode/` remain untouched; no push, branch or subagent.
Next: remaining eligible FFN-gate/full-attention roles through the same synthetic
caller boundary. Real P4/P2 and P5-P7 remain separately gated.

## Synthetic V2 Prefill Checkpoint (2026-10-12)

The user resumed autonomous bounded work from user-pushed `06a1fa2`/`6abb44c`.
The owned v2 policy now feeds the actual `llama_model_load_from_file` and
`llama_decode` path in the existing compiled control. The fixed native synthetic
specification declares all 53 F32/PQ2 tensors and 14 typed metadata requirements
for the four-layer, hidden-32, FFN-256, vocabulary-64 fixture. A separately
supplied single-projection GGUF provides the exact independent FP16 row/group
scales and codes. Each file is capped at 1 MiB before parsing. The target remains
`blk.3.ffn_down.weight`; this is a synthetic integration proof, not a generic
execution CLI or the Python profile/manifest adapter's BF16 plan end to end.

`prism_bitnet_cpu_tensor_last_input_tokens_v2` requires the owning policy,
preserving the v1 diagnostic refusal. The shared prefill control uses v2 status
only when it owns that policy; every old CLI route remains intact. Policy teardown
follows all model/context/buffer teardown, including failure paths. Reports add
`model_policy_abi: 2` only for the v2 route, with zero generated answer tokens.
The existing NumPy oracle and 2e-5 relative/absolute tolerances are unchanged.
Full, chunked, reset, reordered, parallel-context and 128-token modes pass with
all four recurrent/full-attention combinations. Traced/untraced scores agree;
kernel failure emits no score report and recovery retains one weight repack.
Missing/truncated/oversized references, oversized models, wrong profile tags,
changed scales and corrupt PQ2 refuse; the MiMo v1 model factory rejects the v2
tags with no report. Independent canonical bytes remain owned by the policy;
synthetic digest strings are declarations, not source-weight authentication.

Verification: the default source gate passes **244 tests, 44 optional skips**,
Ruff and whitespace. Editor diagnostics are clean. The pinned bounded native
command passes **16 controls in 34.30 s**, including twelve dense/v1/v2 synthetic
prefill combinations and the unchanged complete-model v1 policy control.
Peak process RSS was 944,816 KiB, with no swaps; this is not a host/device memory
guarantee. GCC ASan/UBSan/leak early/late controls and all four v2 model-prefill
cases pass (5.84 s for the latter). Both our bridge and full-runtime control were
instrumented; cached llama/GGML libraries were not. Instrumented compile peak
RSS was 569,188 KiB and the prefill gate peak was 84,524 KiB. GCC's existing
`fclose` deleter-attribute warning pattern remains. Manual compiled controls
were used; no new CMake Tools/CTest result is claimed.

Failed-check history: the first expanded refusal run failed four test cases
because its v1 CLI invocation omitted the required prompt and returned usage
status 2, not policy status 27. The actual route was inspected, the invocation
was corrected, and the valid model was restored after the oversized-file probe
before testing kernel failure. The same four-case gate then passed. No tolerance,
bound, numerical assertion or runtime refusal was relaxed.

Restart/end checks: pinned upstream sources were clean; Prism and BitNet revisions
and GGML CPU/base hashes matched. Only project-owned wrappers were rebuilt.
Pylance selected `/opt/venv/bin/python`; existing converter/dense environments and
the sole frozen candidate were reused. RAM available remained about 23 GiB,
swap unused and disk available 290 GiB. Disposable sanitizer binaries and the
synthetic raw trace directories from the selected native and sanitizer runs were
removed; generated model/reference files were deleted by the fixture. No source
copy, download, real-model conversion/inference, fitting, retained candidate,
compensation promotion, held-out scoring or quality experiment occurred. User
`.vscode/` remains untouched; nothing was pushed or branched and no subagent ran.

Next concrete step: bounded synthetic caller integration from the Python
profile/manifest-derived specification through actual model prefill, including
BF16 converted matrices and a non-layer-3 eligible target. The fixed tiny F32
native fixture does not establish that boundary. Real-profile P4/P2 and broader
P5-P7 work remain separately gated; the historical MiMo numerical gap is unchanged.

## Profile Model Policy Checkpoint (2026-10-09)

Source checkpoint: `6abb44c`, following documentation audit `037b51b`.
The user resumed bounded synthetic P4, then requested a commit and stop for today.
Implementation is now paused; do not start the next integration slice without
an explicit resume. Nothing was pushed and no branch or subagent was introduced.

`prism_bitnet_cpu_model_override_from_gguf_v2` adds an owned ABI-2 policy beside
the unchanged MiMo v1 factories. P3's manifest validator supplies the exact
canonical PQ2 target bytes and profile-derived eligibility/shape. The new Python
adapter derives the entire text tensor table, pinned BF16/F32 conversion policy,
and 14 typed attention/SSM/rotary metadata requirements from the verified profile
and inventory. It refuses altered identities, missing/extra/duplicate tensors,
geometry/dtypes, unpinned runtime/converter rules and unsupported depth/patterns.
SSM convolutions become F32, as required by the pinned converter; source dtype
and converted dtype remain distinct. All 427 names for each cached MiMo and
Defiant Fable profile match the pinned converter's static filter/name map.
This comparison used metadata/headers and synthetic payloads, not model loading.

Native validation checks GGUF v3, profile/fingerprint/revision/converter/source
digest declarations, the separate v2 model tag, identity execution contract,
every expected tensor name/type/shape, geometry metadata and nonoverlapping
in-file ranges. It rejects ordinary/other PQ2, transforms, explicit recurrent
layer overrides and changed target bytes. Reads compare the target in <=64 KiB
chunks. Bounds remain <=32 layers, 427 tensors, 19 GiB declared weights plus
1 MiB metadata/alignment reserve, and P3's <=50,331,648-weight projection bound.
Malformed factory calls clear both outputs. The v1 complete-model factory's
body is byte-for-byte unchanged; no v1 refusal or numerical tolerance widened.

The exact-name/shape v2 buffer reuses the existing immutable/chunked upload,
canonical readback, FP16-block repack and BitNet-derived A8 kernel lifecycle.
Upload bytes must match the policy even after factory validation. Separate
policy-bound v2 status queries leave all v1 diagnostics strict. The control
hosts an 8,192-row synthetic FFN projection, checks exact arithmetic at 1/2/128
tokens, one repack, canonical readback, wrong-name refusal, non-finite-input
no-result/recovery, changed-upload refusal and early/late runtime initialization.
The policy, override strings and shared library must outlive every associated
model/context/buffer; free the policy only after their teardown. Registry
restoration passes repeated creation/free, not general lifecycle concurrency.

Verification: 10 focused default adapter cases pass. The final required source
gate passes **244 tests, 40 optional skips**, Ruff and whitespace; editor
diagnostics are clean. The expanded bounded native command in the development
guide passes **12 controls in 28.53 s**, including 25 v2 synthetic file refusals,
unchanged v1 exact policy, the P3 frozen projection and eight dense/BitNet
recurrent/full-attention zero-generation typed-prefill combinations. Those
prefills remain the existing v1/synthetic route, not v2 model-prefill evidence.
GCC ASan/UBSan/leak early/late controls pass; cached GGML is uninstrumented.
Compilation peak RSS was 570,884 KiB; early-control peak RSS was 56,240 KiB.
Existing `fclose` deleter warnings remain. CTest discovery is empty and its tool
cannot configure this workspace; no CTest result is claimed.

Failed-check history: the first adapter probe used a nonexistent `ModelProfile`
source accessor; removing that assumption and using the strict parser repaired
it. The pinned PQ2 enum was verified as 142 before native dispatch. One test
oracle incorrectly doubled the synthetic gated-query rows; the fixture's
8 x 64 x 2 geometry corrected it to 1,024. The GGUF writer initially received
logical rather than encoded PQ2 row-byte shapes; its documented raw dtype path
repaired the fixture. A rejected tensor was reused for another allocation,
triggering GGML's buffer assertion; fresh refusal tensors repaired the control.
The static mapping probe first lacked `gguf-py`, then Transformers, then the
converter's static block/MTP fields; the cached dense environment and explicit
static fields resolved these, without installation or model initialization.
The F32 convolution omission was found in pinned-source review and repaired
before the final gates. The interrupted metadata patch was verified absent
before applying it. No failed check was retried without diagnosis or relaxation.

Inputs/resource check: cached source revisions and GGML CPU/base SHA-256 values
match the development pins; tracked upstream sources are clean, and only our
isolated wrapper was rebuilt. Entry headroom was about 20 GiB RAM available,
8 GiB swap and 290 GiB disk. Editor Python is `/opt/venv/bin/python`; cached
converter/dense environments were reused. Sanitizer scratch and the 591,872-byte
fixture left by the diagnosed abort were removed. Synthetic pytest files remain
ordinary disposable test scratch outside Git. No new conversion/download/fit,
retained candidate, compensation promotion, held-out scoring or quality run
occurred; the sole frozen artifact and user editor directory remain untouched.

Limits: native profile/source digests are declarations, not authentication of
source tensor bytes or untouched dense weights. The Python boundary checks
profile/header identity and hashes the supplied PQ2 payload, not source weights.
The policy retains an additional canonical reference byte array for upload
comparison; compact resident tensor blocks do not imply zero total policy
overhead or a whole-model 2.125-bpw budget. This is policy preparation plus
isolated loader-graph proof, not complete-model v2 hosting, Defiant Fable
inference, numerical acceptance, quality or edge performance. After resume,
the next step is bounded synthetic actual-loader/typed-prefill integration
through v2; real model inputs, conversion and scoring still need separate approval.

## Projection Geometry Checkpoint (2026-10-09)

The P3 resident checkpoint was committed as `cedaeec`; the second P3 increment
is `db002c1`. It adds `prism_bitnet_registered_projection_create_v2` beside unchanged
v1 creation/file/loader/complete-model factories. A typed C specification declares
ABI 2, exact tensor name, rows/columns and canonical reference PQ2 bytes. Creation
checks payload length and every byte against that reference, binds the trait to
the exact name/shape, and retains one compact owned block array. Existing
compute/free/storage diagnostics are reused. Failed creation clears the handle.

Bounds: 1-128 fixed-shape tokens, rows/columns at most 12,288, group-128 columns,
and at most 50,331,648 logical weights (the largest current 9B-class projection).
The name gate admits only bounded `blk.N` FFN gate/up/down and full-attention
Q/K/V/output names, not recurrent or unknown projections. The new path supports
8,192 x 4,096 Q and 12,288 x 4,096 FFN gate/up synthetic weights with exact
direct-kernel parity. V1 still rejects these larger rows. No v1 name, shape,
file/model count, token bound, policy or numerical tolerance was widened.

The [Python adapter](../embedded_jev/prism_codec.py) validates a strict in-memory
v2 manifest before native access: exact profile ID/fingerprint and inventory
identity, independently profile-derived geometry/eligibility/dtype, canonical
HF-to-GGUF name, identity transform, group size, source-tensor digest declaration,
exact contiguous payload shape/size/SHA-256 and valid trits/FP16 scales. Codec
validation uses existing <=4,096-row slices; the old codec's bounds remain intact.
`create_profile_projection()` creates one native owned handle, not a model loader
or saved candidate. It explicitly reports the source digest as declared, not
rehashed by the adapter. A supplied expected byte array is caller provenance,
not native authentication of a source profile or artifact manifest.

Verification: 14 focused profile-projection tests passed, covering altered
profile/report geometry, names, transforms, hashes, unknown fields and rejection
before dispatch. The default gate passed with **234 tests, 39 optional skips**,
Ruff, whitespace and editor diagnostics clean. After refreshing only our isolated
runtime wrapper, all **11 selected compiled controls** passed in 32.34 s: larger
v2 projection/refusal cases, exact frozen v2 adapter and v1 parity, PQ2 decoder,
owned-module reuse, tagged loader/file refusals, unchanged complete-model exact
policy and eight synthetic dense/BitNet recurrent/attention prefill combinations.

The transient frozen v2 manifest used the existing artifact and a real SHA-256
of its 100,663,296 BF16 source-tensor bytes, read from the verified cached snapshot
in <=1 MiB chunks. It was not saved. No whole shard was rehashed, new fit/codes/
scales retained, model inference or held-out scoring run. The actual source
digest does not close the existing model-level numerical discrepancy. Weight
resident bytes remain 34 per 128 values with zero auxiliary weight arrays.

Failed-check history: the first opt-in frozen v2 bridge exposed a test-local
`unpack_group128_codes` import below its new use. Moving the import into the
existing import block repaired it; the same check passed afterward. No retry
without diagnosis, relaxed assertion or changed runtime policy followed.
GCC ASan/UBSan/leak early/late runtime controls pass with the added v2 lifecycle,
mixed v1 graphs, readback and teardown; cached GGML remains uninstrumented.
The existing deleter-attribute warnings remain. Sanitizer scratch was removed.
CMake Tools/CTest remain unavailable for this workspace configuration as recorded
below; no CTest result is claimed. User editor settings and pinned upstream
source/libraries remain unchanged.

Limits and next step: this completes P3's bounded owned-projection proof, not
whole-model ternary hosting, a generic discoverable loader buffer, native
Defiant Fable execution, an accepted numerical tolerance, calibrated quality or
edge performance. P4 must introduce a separately versioned complete-model
policy whose targets and exact artifacts are profile-bound; real checks and P2
download/conversion still require separate approval. No push, branch or subagent
was introduced; user `.vscode/` and the frozen candidate remain untouched.

## Resident Weight Checkpoint (2026-10-09)

This continuation began after the user's push of `1df6458`, with `main` matching
`origin/main` and only the untracked user `.vscode/` directory. The first P3
increment removes resident weight duplication from the registered trait and v1
loader. Each owned 34-byte row/input-group block retains its original
little-endian FP16 scale bits and repacks only its 32 code bytes to BitNet's
separated high-bit-first lanes, in place. No full-size code copy or expanded
FP32 scale array remains. Scale conversion occurs exactly at the grouped dot;
no scale refit, code change or arithmetic-order change is introduced.

Canonical PQ2 readback reconstructs bytes on demand, including arbitrary
one-byte reads. File-policy validators remain non-mutating, preserving the v1
complete-model factory's exact canonical target-byte comparison. Chunked upload,
one repack, immutable weights, failed-input/no-score recovery, mixed read-only
graphs and owned handles retain their existing contracts. New storage diagnostics
report weight-tensor payload bytes separately from auxiliary weight arrays;
these are not total process RSS or graph/A8 allocator measurements.

The sole 4,096 x 12,288 frozen projection reports 13,369,344 resident weight
payload bytes and zero auxiliary weight bytes, versus the prior 27,525,120-byte
payload/duplicate-array basis. This is 2.125 bpw for those weight blocks, not
the complete model, a 1.75-bpw format, edge fit or a speed claim. Original
artifact files, independent FP16 row/group scales and all v1 model/tensor/name/
shape/refusal bounds remain unchanged. Larger trait shapes and manifest-driven
selection are the next P3 increment, not implemented by compaction alone.

Verification: 17 standalone kernel tests passed, four optional checks skipped;
the new three FP16-block cases match expanded-scale output exactly at 256,
4,096 and 12,288 input widths, including zero/subnormal/maximum FP16 scales and
transactional malformed-block refusal. The focused shared Prism control passed.
The two full-size registered/PQ2 decoder gates passed in 59.00 s with the existing
frozen artifact and owned-module reuse. A complete-model policy case initially
skipped without its runtime variable; after rebuilding only our isolated wrapper,
all nine synthetic dense/BitNet recurrent/attention prefill and v1 exact-policy
cases passed in 5.50 s. No real complete-model inference or conversion ran.

The default gate passed with **220 tests, 39 optional skips**, Ruff and whitespace
clean; editor diagnostics are clean. Cached Prism/BitNet source revisions and
GGML CPU/base SHA-256 values are unchanged; tracked upstream source is clean.
GCC ASan/UBSan/leak controls pass early/late initialization, chunked upload,
canonical readback, mixed graphs and teardown. Cached GGML is not instrumented;
GCC emitted the existing `fclose` deleter attribute warnings. Disposable
sanitizer binaries were deleted after verification.

CMake Tools returned no targets/tests and could not configure this workspace,
so the already-configured isolated runtime was rebuilt with `cmake --build`.
The CTest tool likewise could not configure; no CTest result is claimed. The
same early/late runtime behavior passed in the compiled pytest/sanitizer controls.
No editor settings were changed to work around tool configuration. The cached
CLI bridge and pinned upstream libraries were not overwritten; only the
project-owned isolated wrapper targets were rebuilt.

Next: geometry/manifest-gated larger projections beside unchanged v1 factories.
No additional download, full-model conversion, fitting, retained candidate,
held-out inference, push, branch or subagent is authorized or introduced.

## Template Policy Checkpoint (2026-10-09)

P0.3 was committed locally as `8d3556e`. P0.4 now refuses profile-declared
control syntax anywhere in bounded untrusted JSON fields before template
rendering. It scans dictionary keys/values and nested message content, including
case IDs/groups and option IDs/descriptions, without stripping or rewriting
evidence. Scanning is capped at 4,096 nodes, depth 32 and 1 MiB text, checking
container expansion and character bounds before allocation; invalid Unicode
refuses. Marker refusal raises an explicit error and returns no score report.

Defiant Fable's profile conservatively refuses the exact `{REASON:` prefix in
all untrusted fields, for both source and embedded-template probes. It pins
non-thinking `reasoning_effort=medium`. The separately cached `plusIQ` template
is 17,057 UTF-8 bytes, SHA-256
`1aca4ae064eff63b6fae32f9e260a1b641be87e49667f046629d6a299be89954`;
its separate GGUF repository/revision/file and remote whole-file hash are
declared, never mistaken for the safetensors template or a locally authenticated
GGUF. `probe_label_boundary(..., template_source="gguf_embedded")` requires
that separate record. Source/embedded template hashes, origin and all render
arguments are recorded. This does not enable GGUF weight ingestion or assert
source/GGUF full-weight equality. The MiMo record declares no GGUF template;
both record fingerprints changed with the explicit provenance field, not source
files, frozen candidate or numerical policy.

Verification: 12 focused template cases and all **217 default tests** passed,
with 39 optional skips, Ruff, whitespace and editor diagnostics clean. In the
existing dense environment, both cached Defiant Fable templates render identical
default prompt bytes/IDs at pinned non-thinking medium: 49 tokens, A-P 32-47,
the same `299d512e...` prompt hash as P0.3. All five source-template engineering
cases pass contextual label boundaries. The three real-template probes with
`{REASON:ispoon}`, `{REASON:xhigh}` and `{REASON:unknown}` refuse before
rendering. MiMo's `325474ab...` prompt hash remains unchanged. No model weights
or GGUF file were read, no answer generated and no native/full-model gate run.

Limits: this is reference-renderer policy/provenance evidence. Native rendering
and tokenization for Defiant Fable, complete GGUF/source tensor comparison,
dense/native inference and model quality still need their own gates and any
separately approved inputs. No hidden mode selection from untrusted text is
accepted merely because a template is pinned. Next is P3 kernel geometry and
resident-storage work, using synthetic fixtures and the unchanged MiMo artifact.
No further download, conversion, candidate, fitting or held-out scoring is
authorized; no push, branch or subagent was introduced. User `.vscode/` and
existing caches remain untouched.

## Packaging Checkpoint (2026-10-09)

P0.2 was committed locally as `c0c8547`. P0.3 now validates declared architecture
and static text-config paths, text/vision/MTP prefixes, index-listed shard names,
exact tokenizer/processor classes, every text/MTP shape, per-class float dtypes,
text count and explicit projection eligibility. Offline prefix/streamed planners
use the same profile rules; real scoring remains MiMo-only. Bound records can
declare a recomputed index total, whose exact difference is reported, not hidden.
MTP presence, missing names, complete/incomplete status and explicit text
exclusion are separate accounting fields; unexpected names/shapes/status refuse.

The separate [Defiant Fable record](../embedded_jev/profiles/defiant-fable.json)
comes from the approved preflight cache, not another download. It declares
`TokenizersBackend`, `Qwen2VLImageProcessorFast`, no processor-config/merges file,
the restored MTP shard, complete MTP and the base-template suffix. Its native
policy is null: metadata inventory/tokenization is not a dense/native execution
or quality claim. Complete shard SHA-256 values are remote LFS pins, not locally
verified payloads. The MiMo record gained exact metadata-class declarations;
its canonical fingerprint changed, not its source files or numerical policy.
No real v2 capture had been retained; legacy v1 captures are unchanged.

The inventory CLI supports explicit `--profile` and `--header-dir` with
`--local-dir`, never another remote download. Sidecars contain raw JSON; the
reader reconstructs the eight-byte safetensors length prefix and checks its
size/hash, reporting `profile_pinned_remote_sizes` rather than `local_stat`.
Prefix hashes in the new record therefore differ from the historical preflight
table's raw-JSON hashes. No weight file needs to exist for sidecar inspection.

Verification: 114 focused inventory/label/data/evaluation cases passed, and the
default gate passed with **206 tests and 39 optional skips**, Ruff, whitespace
and editor diagnostics clean. Synthetic controls cover multimodal/text-only,
padded/alternate/unpadded/restored shards, F32 norms, stale totals, absent/complete/
incomplete MTP, explicit eligibility and undeclared-variation refusals. Cached
Defiant Fable reconciles 775 tensors (427 text, 333 vision, 15 MTP),
19,306,208,736 tensor bytes and a 67,108,864-byte index shortfall. Every text/MTP
shape matches its own configuration; all 11 recorded nonweight files verify.
The historical preflight's `774` total was a prose/count typo, not another
tensor layout; its category counts already sum to 775 and its bytes agree.

The cached tokenizer-only check passes 49 tokens and A-P 32-47, prompt SHA-256
`299d512e3217660f75b5fe9d384f5b0357b205e2dbf9c3aaace5a351c3a4b612`,
zero generation. MiMo still reconciles 760 tensors, 427 text and
18,819,627,488 tensor bytes. No weight payload was read, model run, refit,
conversion, new candidate or native gate followed.

Failed-check history: the first cached-source probe refused the then-hardcoded
tokenizer/image classes. The actual cached declarations were read, pinned into
records and retested; no broad unknown-class allowance was added. Two early
scoring-refusal tests then exposed a changed error message when undeclared
shards were rejected earlier. The specific profile-mismatch error was preserved
and the same slice passed. Numerical tolerances and resource bounds are unchanged.

Next: P0.4 template-control refusals and separate GGUF-template provenance;
then P3 geometry/resident-storage controls. No further download, full-model
conversion or quality experiment is authorized. Nothing was pushed, branched
or delegated; the user's `.vscode/` directory and caches remain untouched.

## Verified Identity Checkpoint (2026-10-09)

P0.1 was committed locally as `2f659d0`. P0.2 now derives inventory identity
from an exact profile match. Local readers and pinned remote-header fetches
refuse unmatched metadata/shard sets, sizes or hashes before payload reads.
Raw analytical header/tokenizer fixtures remain explicitly unbound (`model` and
`revision` are null), never implicitly labelled MiMo. Explicit synthetic second
profiles report only their own identity; the existing MiMo Prism-width entry
point rejects them, beside a new verified-profile width mapper.

The real prefix/streamed scorers require the MiMo record and all 13 nonweight
file hashes before model imports or trace creation. Summaries carry the profile
ID/fingerprint. Label probes bind exact templates, rendering arguments, suffixes,
special IDs and contextual label IDs; CLI file fingerprints are checked before
loading. Existing model/candidate/native-policy names and numerical contracts
are unchanged. Header identity remains a cheap check, not full payload
authentication; full shard hashes are not rerun here.

New calibration manifests are v2: raw arrays are unbound unless the caller
supplies the verified profile. Bound captures cannot be transplanted to another
profile; fitting consumers refuse unbound captures, and balanced banks refuse
mixed identities. Legacy v1 MiMo captures remain readable without rewriting or
retroactively claiming a profile fingerprint. Paired evaluation requires both
reports' profile fingerprints and freezes the profile record across the run.
The evaluator still requires the sole MiMo candidate; no other-model scoring
or candidate inheritance is introduced.

Verification: 102 focused inventory/label/data/evaluation cases passed; the
default gate passed with **194 tests and 39 optional skips**, Ruff, whitespace
and editor diagnostics clean. In the existing CPU dense environment (Torch
2.10.0+cpu, Transformers 5.12.1), the cached tokenizer's bound/unbound prompt
bytes and IDs are identical: 46 tokens, A-P 32-47, prompt SHA-256
`325474ab1f04150179209c0bd4c8c34bac284ffccda18788b481b4c6f92208be`,
zero generated tokens and no weight loading. All four cached v1 training
captures load read-only; their balanced 128 x 12,288 array still hashes to
`7e49c6d52dd93a2386425d0850a2680d56f2222aff765373422f40dacaaf9866`.
The existing frozen projection passes its code/scale/manifest hash checks with
unchanged 4,096 x 12,288 geometry and source revision. No fitting or writes to
those caches followed. Optional native/full-model inference gates were not run.

Failed-check history: the full inventory slice initially exposed a tiny fixture
that relied on false MiMo identity to enter the legacy width mapper. It now
uses its own verified profile, while the MiMo refusal remains strict. The local
guard then exposed an incorrect expected first-refusal message and an HTTP
transport fixture without a registered synthetic record. Both were repaired
without bypassing production checks; the same focused slice passed afterward.
No numerical tolerance, payload/token bound or execution refusal was relaxed.

Next: P0.3 packaging adapters and a record derived from the verified Defiant
Fable metadata/header cache; then P0.4 untrusted template-control refusals and
P3 native geometry/resident-storage work. Current CLI model execution remains
MiMo-only. No download, complete-model conversion, new retained candidate,
held-out inference, push, branch or subagent was introduced; user `.vscode/`
remains untouched.

## Profile Record Checkpoint (2026-10-09)

The continuation began at user-pushed `affedd4`, with `main` matching
`origin/main` and only the untracked user `.vscode/` directory present. P0.1 now
has a bounded strict v1 [parser](../embedded_jev/model_profile.py), immutable
records and the checked-in [MiMo record](../embedded_jev/profiles/mimo.json).
It binds all 17 existing verified files, the four complete shard hash pins and
separate header hashes, canonical text geometry, per-class float dtype policy,
template arguments/suffix/IDs, converter/runtime revisions and the exact existing
native model-policy tag. Unsupported tiers, unknown/duplicate fields, unsafe
paths, invalid numeric types, inconsistent layers and conflicting labels refuse.

The cached MiMo record matches all 13 nonweight files, all four shard sizes and
headers, and every recorded text-geometry field. Full weight payloads were not
rehashed; those digests are the existing onboarding pins. The bounded verifier
explicitly refuses complete weight-shard hashing. No download, inference,
conversion, environment rebuild or candidate change followed. P0.2 report
identity enforcement and P0.3/P0.4 adapters are not implemented by this record.

Verification: 24 focused profile cases passed; the default gate passed with
187 tests and 39 optional skips, Ruff and whitespace clean. Editor diagnostics
are clean. VS Code test discovery returned no tests, so focused pytest was used.
The rebuilt container has no `rg`; workspace search with ignored files included
provided the native tag after the default search omitted it. A guessed policy
name was corrected from that source evidence, then focused and default checks
reran for the changed record. No optional native/full-model gate was rerun for
this schema-only increment. The selected interpreter is `/opt/venv/bin/python`,
and the existing Prism source still has its pinned `842b1880` revision.

Next: derive report identity from verified profiles, reject unmatched local
directories, and preserve MiMo's current numerical and artifact contracts.

**The frozen BitNet-derived FFN now runs inside real native 32-layer MiMo,
with zero generated answer tokens. Whole-model ternary export remains open.**
The historical real-model checkpoint below superseded bootstrap-era status.
The October 9 checkpoints above supersede its source/native-projection status,
not its unresolved MiMo numerical evidence. Later sections retain historical
probe results; do not read them as current resume instructions or renewed approval.

**Documentation-only checkpoint, 2026-10-09:** The user-pushed starting point
was `c8d4c5e`. The copied [agent workflow](../AGENTS.md) now uses this project's
documentation paths, Python/native gates, numerical and resource boundaries,
and scoped approval rules. The [public introduction](../README.md) is shortened
to capabilities, maturity caveats, documentation links and licensing; its removed
setup commands are already covered in the development guide. Scoped links,
anchors, fences, Bash syntax, content and whitespace checks passed. No source,
native or model gates were rerun; the implementation results below are unchanged.
The user `.vscode/` directory remains untouched. **Automatic implementation is
paused at the user's request until they explicitly resume it.** Another temporary
complete-model conversion is not authorized by this documentation request.

**Model-scope clarification, 2026-10-09:** Starting from `c8e1e29`, all project
docs and plans now target reusable contracts supporting multiple Qwen3.5
9B-class models. MiMo-V2.6-Distill-Qwen-9B is the first test subject, not a
permanent requirement; a different model may become the next subject before
MiMo's full-model work is complete. The roadmap's
[support gate](roadmap.md#model-scope-and-support-gate) requires per-model
inventory, tokenization, architecture/runtime validation and artifact/evidence
binding. Existing MiMo results, source pins, cache paths, `MIMO_*` commands and
frozen candidate are unchanged; they are not family-wide compatibility evidence.
Checks passed for eight docs, 138 local links/anchors, balanced fences, 48 Bash
blocks, unchanged command examples, scope and whitespace. The initial wording
check falsely rejected the capitalized "First test subject" label; correcting
the validator's case handling resolved it without changing compatibility criteria.
Editor diagnostics are clean. No source/native/model gates were rerun.
No successor model has been selected, downloaded or converted. Implementation
remains paused until the user resumes it; the broader goal is not authorization
for another model experiment.

**Family design review, 2026-10-09:** Starting from user-pushed `bc8e616` (local
`main` matched `origin/main`), the user requested a deep review of the plan to
support Qwen3.5 9B-class models generally. Research was metadata-only: base
`Qwen/Qwen3.5-9B` Hub metadata, configuration, index, tokenizer configuration and
template at `c202236235762e1c871ad0ccb60c8ee5ba337b9a`; the fine-tune listing and
one text-only derivative's index; upstream heads; local MiMo metadata/headers; and
the pinned Prism converter source. No other model's weights, tokenizer or processor
were downloaded, and no code, tests, caches, candidate or user `.vscode/` changed.
Findings:

- MiMo and the base share the 9B-class geometry but differ in shard names,
  `mtp.*` tensors, 3,840 F32 parameters, template separators and non-thinking
  suffix, `tokenizer.json` bytes and licence; text-only exports also change the
  architecture and tensor prefix. See the
  [family review](research-audit.md#qwen35-9b-class-family-review).
- Current code derives model identity from constants. Other packagings are refused
  only incidentally, while a MiMo-packaged checkpoint would be mislabelled as MiMo.
  This is the first generalization item, documented with every other coupling in
  the [development guide](development.md#mimo-coupling-inventory).
- Native trait bounds and the v1 policy cover only the layer-3 down projection.
  FFN gate/up and `q_proj` need 12,288/8,192 rows, and resident storage keeps PQ2
  blocks, repacked lanes and FP32 scales (about 4.4 bpw versus 2.125 on disk).
- Microsoft BitNet, SemIf and MiMo heads still equal our pins; Prism published two
  newer releases, not adopted. A third-party Jev-style Qwen3.5-9B release with
  adapters and a decision head exists; its claims are unverified.

Documentation now defines the 9B-class geometry, packaging axes, profile record,
identity rule and shared/per-profile split in the
[design](design.md#model-profiles-and-portability), an ordered
[work queue](roadmap.md#generalization-work-queue), source observations and an
[upstream check](sources.md#upstream-status-check), plus an agent safety rule.
The README is unchanged. Implementation remains paused; recommending the base
model as an early second profile does not select or authorize it. Checks passed
for eight docs: 181 local links/anchors, balanced fences, consistent columns in
33 tables, editor diagnostics and `git diff --check`. No source, native or model
gates were rerun because no code changed.

**Derivative release review, 2026-10-09:** Starting from local commit `e95ddcb`
(one commit ahead of `origin/main`), the user asked about supporting DavidAU's
Defiant Fable releases, especially `plusIQ-NEO-MAX-MTP-bf16.gguf`, possibly
ZDTaichu5.0-9B, and most fine-tuned, merged or expanded Qwen3.5-9B derivatives.
Research read Hub APIs, configurations, indexes, licence files and model cards
only, plus the pinned Prism converter and loader source. Findings:

- The Defiant Fable safetensors source (head `7af0a9c4`) has 9B-class
  configuration, all-BF16 tensors, the base template blob and complete MTP, with
  mergekit packaging quirks: an extra `model-mtp-restored.safetensors` shard and
  a stale index total. Current inventory code refuses both.
- The `plusIQ` BF16 GGUF size fits 32 text layers plus one MTP layer; whether its
  weights or template differ from the source is unknown. Its card describes
  in-band `{REASON:mode}` controls, an injection surface for decision evidence.
- GGUF records no value-head layout or converter revision, so a GGUF-only source
  needs an independent anchor; the converter only permutes eligible projections.
  The pinned loader skips MTP unless requested and accepts an explicit
  recurrent-layer array that the pinned converter cannot write.
- ZDTaichu5.0-9B wraps a 9B-class decoder in custom code with mixed licences;
  48-layer DavidAU depth variants exist with a regular pattern.

Documentation now defines support tiers A-E with float-GGUF, wrapped-decoder,
depth-variant, template-control and refusal-removal rules in the
[design](design.md#support-tiers-for-derivatives), the
[derivative review](research-audit.md#derivative-release-review) and its
[source observations](sources.md#derivative-release-observations), queue items
P0.4 and P5-P7, three verification rows, new coupling rows and an agent rule
against executing model-repository code. The README now names fine-tuned and
merged derivatives in its scope. Implementation remains paused; no release was
selected, downloaded or converted, and no code, cache or user `.vscode/` changed.
Checks passed for eight docs: 209 local links/anchors, balanced fences, consistent
columns in 36 tables, recomputed parameter and byte arithmetic, editor
diagnostics and `git diff --check`. No source, native or model gates were rerun
because no code changed.

**First non-MiMo profile selected, 2026-10-09:** Starting from local `b0c9089`,
the user selected DavidAU's Defiant Fable safetensors source at
`7af0a9c4e221e01b246b3c577fbb7110b79823e8` as the first non-MiMo profile, approved
the bounded preflight, and asked that implementation stay paused here because
another model will implement the queue. The preflight did not run: `curl` to the
Hub failed certificate verification because the network presents a TLS-inspection
chain whose root is not in the container trust store (see
[environment pitfalls](#13-environment-pitfalls-already-encountered)). Verification
was not bypassed and nothing was downloaded. Hub pages read through the external
web tool confirmed the source file listing, with no `merges.txt` or
`processor_config.json` shown, and the exact `plusIQ` file name, size and
Hub-displayed SHA-256. The development guide now holds the exact
[preflight procedure](development.md#defiant-fable-preflight-procedure); the
design, roadmap, research audit, source register, coupling inventory and this
handover record the selection, approval, blocker and suggested implementation
order. Checks passed for eight docs: 222 local links/anchors, balanced fences,
consistent columns in 37 tables, repeated identifiers and byte figures, editor
diagnostics and `git diff --check`. No source, native or model gates were rerun
because no code changed; README and AGENTS.md needed no change.

**Host certificate provisioning, 2026-10-09:** At the user's request, and modelled
on another of the user's projects, both devcontainer profiles now run
[prepare-build.sh](../.devcontainer/prepare-build.sh) on the host through
`initializeCommand`. It exports the host's trusted roots into a git-ignored
`.devcontainer/host-ca-certificates.crt`. The Dockerfile installs readable
entries with `update-ca-certificates` before any download and points
`SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE` and `NODE_EXTRA_CA_CERTS`
at the system bundle. Checks in this container: JSON parsing of both profiles,
`bash -n` and ShellCheck, Git ignore rules, the script's Linux fallback in a
temporary directory (121 certificates), and the Dockerfile step run in a sandbox
with fake installation. That step split CRLF input, dropped a corrupt block,
produced a bundle Python's `ssl` loaded with 121 roots, and handled an empty file.
The Windows and macOS export paths, an actual image build and verified HTTPS were
not tested here; the user must rebuild the container. AGENTS.md now forbids
committing or publishing the generated file or images built with it. The default
gate passed (163 passed, 39 skipped, Ruff and `git diff --check` clean), and the
eight docs passed link/anchor, fence and table checks (231 local links, 37 tables).

**Rebuild verified and preflight run, 2026-10-09:** The user rebuilt the container
from `5a58cd8`. It holds 110 exported host roots, 231 certificates in the system
bundle and the four CA variables; `curl` and Python both reached the Hub with
verified HTTPS (200), and the external cache, MiMo snapshot, frozen candidate and
dense environment survived. The approved Defiant Fable preflight then ran with
two throwaway scripts: 291,402,916 bytes in 49 requests and 48 s, plus a 5.6 s
offline tokenizer probe. Findings, detailed in the
[development guide](development.md#preflight-results):

- The source has 774 BF16 tensors (427 text, 333 vision, 15 MTP); the text names,
  shapes and dtypes equal MiMo's. The index total omits exactly `mtp.fc.weight`,
  alone in the restored shard; there is no `processor_config.json` or
  `merges.txt`.
- The base template renders the expected suffix, and A-P are tokens 32-47 as
  for MiMo; `vocab.json` matches MiMo's while `tokenizer.json` differs by 18 bytes.
- The `plusIQ` BF16 GGUF has 442 tensors including 15 MTP tensors. Its sampled
  FFN, embedding and output bytes equal the source, and its linear-attention rows
  match in tiled order. Its template renders the same default prompt, but a
  `{REASON:ispoon}` marker silently expands the prompt from 202 to 4,219
  characters.

Retained: 47.2 MB of verified files, headers and the `plusIQ` template under
`models/defiant-fable-7af0a9c4-metadata/` in the external cache, a 93 KB report
and the two scripts beside it. Sampled tensor bytes stayed in memory; the
temporary directory was deleted. No repository code, frozen candidate or user
`.vscode/` changed, and no further download is authorized. In the rebuilt
container the smoke check reported `ok` and the default gate passed (163 passed,
39 skipped, Ruff clean). The eight docs passed link/anchor, fence and table checks
(237 local links, 38 tables) with the checker recreated after the rebuild, which
a self-test showed still detects a broken anchor.

## Current Checkpoint (2026-10-08)

The last tested implementation commit is `9408f3f` (2026-10-08),
`test: cover fused sigmoid and softplus recurrent gates`.
This continuation started clean at `445c8d7`, with `main` and `origin/main`
matching after the user's push. Local commits `d95ffa4` (repeated graph),
`e2378e4` (owned handles), `4efa59b` (streamed backend), and `cb4fe74` followed,
with documentation checkpoints at `79608a8` and `03b183d`. The latest
continuation began clean at the user-pushed `03b183d`; `b1f306a` (versioned CPU
buffer) and `9a1fbfa` (explicit loader route) followed. The assistant made no push
or branch change. The resource-plan continuation started clean at `c390c0d`,
adding `ecfcff5` (header-only native budgets) and `3c4fa81` (full runtime build
and no-weight tokenizer preflight). Recheck Git state in each session.
After reboot, the caches and packaged runtime remained intact. This continuation
started at the user-pushed `1a7b08a`, preserving the untracked user `.vscode/`
directory. `64c5b1f` added synthetic native prefill and cached converter-index
validation; `50a943b` added nonzero BitNet FFN execution inside that model.
`27cfd63` rejects unknown, missing, and extra control arguments before discovery,
instead of silently returning the unrelated default control's successful report.
The next continuation started at `44fc7df`, still preserving user `.vscode/`.
`6cba251` added nonzero recurrent arithmetic, `14aac4f` added nonzero attention
and explicit FP16 KV caches, `5fff700` checked initialized shared-model contexts,
and `cf16046` added runtime-kernel failure/refusal/recovery controls.
`76cc0b2` added contextual tokenizer-verified prompt scoring; `ecc6cbd` measures
actual BitNet input batches and checks 128-token model prefill. No push or branch
change followed.
The user then approved one temporary text-only BF16 GGUF and bounded native
prefill. `462dc12` adds its opt-in real-model contract regression. Conversion,
direct native prefill, streamed comparison, and cleanup completed; no additional
bulk conversion or full-model BitNet policy was authorized at that checkpoint.
The October 6 continuation began at user-pushed `ed5fea0`. The user approved one
new temporary complete model containing the existing frozen layer-3 projection.
`dda784a` adds declared complete-model policy plus exact target-byte binding;
`669ed7b` adds real prefill, no-score kernel refusal, and cleared-context recovery
regressions. Conversion, measurements, comparison, and deletion are complete.
No refitting, new retained candidate, held-out inference, push, or branch change
followed; the user `.vscode/` remains untouched.

The user then approved one more temporary mixed model solely for precision
diagnostics, without refitting or held-out inference. `63e8b4c` adds bounded
native/streamed traces, `2bc3ce0` compares validated trace stages, and `6e8386c`
adds a fixture-only BF16-RHS experiment based on the pinned CPU trait.
`5b7e559` gates the real trace with a bounded 1,200-second native limit; other
real cases keep 600 seconds. The approved diagnostic and cleanup are complete.
The user's push was observed through `63e8b4c`; the assistant made no push.

The next continuation began at user-pushed `771663e`. `a072049` adds a native
Q/K norm graph and fixture-only streamed Q/K diagnostic; `e03b48e` checks the
actual native gated-delta output/state and tiled head contract; `9408f3f` adds raw
sigmoid/softplus gate fusion. All are bounded controls, not a production-policy
promotion. The existing frozen projection is unchanged and no new complete
model was converted. Q/K-only alignment worsens the selected-logit gap from
0.004558563232421875 to 0.028759002685546875 on the same 80-token fixture.
Raw streamed traces/compiler scratch are deleted; only 44 KiB of reports remain
at `native/mimo-qk-contract-report-20261008-HNEVga` under the cache.

Current gates: **163 default tests passed, 39 optional tests skipped; all seventeen
pinned Prism controls passed** with full-size PQ2, reused weight/graph, owned
handle, two-forward module, tagged toy GGUF import, and isolated versioned CPU
discovery, mixed concurrent graphs, and actual pinned loader selection/upload
enabled, including the sole full-size frozen projection, packaged dependency
provenance, guarded vocabulary-only native tokenizer parity, and synthetic
dense/BitNet hybrid model prefill plus model-policy/CLI refusals, Q/K normalization,
sixteen activated/raw-gate native recurrence cases, and eight pinned Torch
recurrence/state cases (51.67 s).
The real one-BitNet prefill regression passed separately in 221.62 s; its two
real kernel controls passed in 733.45 s combined. All five real-model cases
were skipped in the seventeen-control invocation, avoiding another large model
load or conversion. The prior dense-only regression passed at `462dc12`.
The real trace case passed separately in 746.54 s at `5b7e559`, peak RSS
17,611,040 KiB; its temporary files and all raw activations are deleted.
The isolated full llama build also passes two CTests.
The 32-layer direct/callback/registered, reordered synthetic typed-option, and
signed-Hadamard full gate passed at `4efa59b` (138.36 s); these unchanged model
controls were not rerun after the later native file/runtime additions. The new
four-/32-layer fixture-only Q/K probes pass, without closing runtime parity.
Ruff, Python editor diagnostics, explicit C++ compilation, and `git diff --check`
passed. The C++ editor lacks the external GGML include path; user editor settings
remain untouched. GCC ASan/UBSan and leak checks
passed for the bridge runtime controls; cached GGML is not instrumented. See the
development guide for flags.

- The complete pinned BF16 snapshot is verified and cached outside Git. The
  streamed text-only reference executes all 32 decoder layers and scores
  context-verified selected head rows with **zero generated answer tokens**.
- Exactly one hash-checked searched-FP16 layer-3 FFN-down fixture is retained:
  4,096 x 12,288, group size 128, about 13 MB. Saved loading bypasses that BF16
  projection. Direct BitNet-derived AVX2, actual Prism `MAP_CUSTOM2`, and
  native-A8 callback paths reproduce the same final hidden hash and scores.
  The new `prism_ggml_registered` backend also reproduces them through a native
  weight tensor/`MUL_MAT` trait, with exactly one counted grouped kernel call,
  one upload/repack, native A8, and explicit handle release after the layer.
- Matching signed-Hadamard weight/input rotation (seed 773) passes dense and
  native-reference controls. It is a reversible in-memory experiment, refuses
  the saved identity artifact, and has no saved candidate or quality promotion.
- Separate PQ2_0 conversion preserves every frozen code/FP16 scale exactly;
  the actual native decoder and PQ2 tensor/`MUL_MAT` controlled outputs pass.
  This is **Prism PQ2 dispatch, not BitNet dispatch or GGUF loader proof**.
  Native PQ2 uses Q8_K when input width is divisible by 256, otherwise Q8_0
  with FP16 activation-scale rounding; our callback uses group-128 A8 instead.
- A separate scoped `JEV_BITNET_GROUP128` CPU buffer/tensor trait now registers
  a PQ2-storage weight tensor and dispatches `MUL_MAT` through the BitNet-derived
  grouped kernel, with native group-128 A8 and one counted call per successful
  evaluation. One graph/weight tensor now handles two changed input batches
  with one weight repack; late failures leave the entire caller output intact.
  Reusable native create/compute/free handles own copied weight bytes, graph,
  repacked codes, and scales between caller invocations. Two live toy handles,
  failed-input recovery, fixed token shape, and registry cleanup pass.
  Explicit adjacent/low-bit-first PQ2 bytes are repacked to BitNet's separated,
  high-bit-first lanes; independent FP16 row/group scales are expanded exactly.
  One-/two-token controls and the sole full-size frozen projection match the
  direct kernel exactly. Invalid +2 codes, negative/nonfinite scales, nonfinite
  inputs, and excess tokens are rejected without changing caller output.
  This is a pinned internal-ABI, single-threaded tensor proof, not a new GGUF
  type, loader integration, persistent registration, or model quality evidence.
- Registered and direct execution give identical typed IDs, labels, conditional
  scores, and final hidden hash on the existing reordered synthetic option case,
  with zero generation. This is an engineering smoke, not calibrated confidence
  or representative quality. The frozen evaluator still excludes the new mode.
- Native `prism_bitnet_registered_projection_create_from_gguf` now imports one
  explicitly tagged, identity-only PQ2 ternary weight through the pinned GGUF
  parser into the same owned handle. A tiny generated fixture has exact counted
  BitNet parity after its source file is deleted. Missing/wrong execution tags,
  transforms, name/type/count/geometry mismatches, malformed payloads, truncation,
  and excess file size are refused. This is standalone one-weight file import,
  not Prism model-loader selection or a MiMo GGUF load. No actual projection
  was initially converted to a file; the later full-size loader control below
  uses a transient exact encoding, without retaining another candidate.
- `prism_bitnet_cpu_runtime_init_v1` exposes a stable `JEV_BITNET_LOADER_V1`
  buffer through actual early CPU discovery, checks the declared source/ABI and
  storage layout, is idempotent, and refuses initialization after discovery is
  cached. Its owned tensor has exact BitNet parity at 1, 2, and 128 tokens with
  one repack, chunked upload checks, invalid-input recovery, immutable packing,
  and invalid-payload refusal. Default loader dummy probes are intentionally
  rejected: ordinary PQ2 must not be automatically reassigned. Library lifetime
  must cover all CPU users; initialization must precede discovery on one thread.
  Two independent BitNet backend graphs sharing a weight also pass concurrently
  with ordinary PQ2 execution, with eleven total calls and one repack. This is
  initialized, read-only steady-state use, not registry mutation/lifecycle race
  safety. This new path is not yet used by the streamed module.
- `prism_bitnet_cpu_loader_override_from_gguf_v1` validates the bounded tagged
  file and ternary payload before returning exactly one anchored public
  `tensor_buft_overrides` rule. A temporary executable compiled from the actual
  pinned loader/support/registry sources verifies `create_tensor`, allocation,
  `load_all_data`, and two native BitNet `MUL_MAT` calls with one repack. The
  full 4,096 x 12,288 frozen projection passes this route via a temporary GGUF
  encoding of its unchanged codes/scales, deleted in `finally`. Untagged PQ2
  remains ordinary CPU even when this buffer is a candidate. This is genuine
  one-tensor loader selection/upload/dispatch, not a complete MiMo GGUF/model
  load or 32-layer native hosting. Factory callers must load the same unchanged
  validated file and keep the library alive; no full-model policy is authorized.
- Header-only `native_reference_plan` reconciles 427 text tensors with
  17,907,606,528 source bytes (16.678 GiB), excluding vision/MTP and retaining
  the full embedding/head. The largest source+FP32 staging basis is 5.684 GiB;
  encoded output and runtime allocations are extra. The development guide
  records actual RAM/disk observations and the staged conversion plan, not a
  native-fit or converter-peak guarantee.
- The versioned [native build wrapper](../native/CMakeLists.txt) compiles the
  full pinned llama library in a separate approximately 15.3 MB cache directory,
  importing unchanged GGML CPU/base dependencies. Existing build/dispatch
  controls pass. A guarded temporary vocabulary-only GGUF has zero weight
  tensors and loads via real native `vocab_only` API; exact non-thinking prompt
  and A/B/C token IDs match HF with zero generation and source-weight access
  forbidden. The existing dense environment supplies the full converter CLI;
  no installs, source duplication, or cached library overwrite followed.
- A temporary four-layer Qwen3.5 model now exercises actual public native model
  loading, CPU context construction, and prompt-only `llama_decode`. Eight cases
  cross dense/BitNet FFNs with zero/nonzero recurrent and attention outputs.
  Layer 0 exercises causal convolution and gated-delta state with fixed half
  decay/beta; layer 1 exercises grouped-query attention with one rotary plane
  and explicit FP16 KV caches. Both enabled paths must change hidden values
  materially; token-by-token NumPy references check their prefix dependence.
  The other recurrent/attention outputs remain zero. A nonzero layer-3
  FFN has a 32 x 256 down projection with two independent FP16 groups per row.
  Dense FP32 and packed BitNet/A8 final logits match independent NumPy references
  within 2e-5. Fixed synthetic typed IDs and numeric option-token IDs remain
  correctly paired after reordering, with stable conditional softmax and zero
  generated answer tokens. Full/chunked/reset executions check actual hybrid
  memory positions, one BitNet call per successful decode, and one repack per model load.
  The additive `prism_bitnet_cpu_tensor_last_input_tokens_v1` diagnostic measures
  actual projection batches: three rows for full prefill, two for the final chunk,
  and 128 for the maximum-batch control. Final-layer masking does not reduce
  these tested batches to one row; prompt length alone was not used as evidence.
  The diagnostic is zero before computation, records attempted batches even on
  kernel failure, returns tensor status, and uses the existing per-weight mutex.
  Two initialized contexts sharing one model run on separate threads with
  different effective histories; both independent references pass, with three
  successful BitNet calls and one repack. Loading/initialization/freeing is not
  concurrent, and this is not a registry/lifecycle race guarantee or speed claim.
  Unsupported rounding after loading yields failed tensor status and NaN logits
  even though `llama_decode` succeeds, with no counted kernel call. That path
  emits no scores; restoring rounding and clearing state recovers the reference
  on the same model/context with one successful call and no additional repack.
  Illegal +2 codes, negative/nonfinite scales, nonfinite head weights, and unknown
  modes return no score report. Files are removed in `finally`.
  Dense `--prefill-text-control` tokenizes an already-rendered prompt and A/B/C,
  requiring distinct single-token labels and unchanged prompt token prefixes
  when each label is appended. Two- and 128-token toy prompts match independent
  references; empty/oversized prompts and contextual BPE merges are refused.
  It does not apply a chat template or choose a thinking policy. All eight cases
  cover 60 successful native invocations and 76 refusal checks.
  The BitNet route is a test-only explicit override, capped at 1 MiB and exact
  four-layer/32 x 256 geometry; it does not call or expand the public one-tensor
  factory. This proves synthetic architecture/dispatch compatibility, not real
  MiMo weight loading, real-model contextual label scores, general gate/head/rotary
  correctness, native MiMo parity, model quality, or a retained candidate. See the
  [synthetic native gate](development.md#bounded-synthetic-native-prefill).
- One explicitly approved temporary text-only BF16 GGUF now proves real MiMo
  dense native load/context/80-token prefill through all 32 text layers, with
  finite full-vocabulary logits, typed A/B/C IDs 32/33/34, and zero answer tokens.
  It contained 427 tensors (250 BF16, 177 F32), 17,920,693,472 file bytes, both
  full vocabulary matrices, and no vision/MTP. Conversion used lazy disk spill:
  5m40.42s, peak RSS 10,266,840 KiB. Direct one-thread native load/prefill took
  4m37.03s, peak RSS 17,667,176 KiB, with retained memory position `[79,79]`.
  Independent identical-prompt streamed BF16 scores have maximum logit gap
  0.09715080261230469 and conditional-score gap 0.0005860534409651841. Exact
  parity and a broader acceptance tolerance remain open. Native F32 graph
  outputs versus streamed BF16 hidden states are a concrete precision difference,
  not full attribution. Process swap counters were zero, but host swap use grew;
  these are observations, not edge fit/performance or calibrated quality evidence.
  The opt-in regression checks metadata and exact selected embedding/head rows
  without performing conversion itself. The GGUF and temporary spill directory
  have been deleted; 156 KiB of reports remain outside Git. Candidate hashes,
  source/library pins, and the production one-tensor override are unchanged.
  This is **dense hosting, not native MiMo BitNet dispatch**. See the
  [measured reference](development.md#approved-dense-native-reference).
- The separately approved October 6 mixed model now proves **real native MiMo
  with one BitNet-derived projection**: 427 tensors, 249 BF16/177 F32/1 PQ2,
  17,833,399,744 file bytes, exact frozen codes/FP16 scales, full vocabulary,
  and no vision/MTP. The additive complete-model policy leaves the old one-tensor
  factory unchanged. It validates declarations and exact target identity, not
  every source tensor or architecture completeness; load the same unchanged
  files and retain the library. The native loader checks actual structure.
  On the same 80-token fixture, one counted dispatch/one repack/80 actual input
  rows yield finite typed scores and zero answer tokens. Native peak RSS was
  17,605,340 KiB. Real no-score kernel refusal and cleared-context recovery pass.
  Native/streamed maximum logit/conditional gaps are 0.06760978698730469 and
  0.0009491202828953993, not exact parity or a new tolerance/quality claim.
  Both temporary GGUFs and spill files were deleted; only 160 KiB of reports
  remain at `native/mimo-one-bitnet-report-20261006-WHaMyP` under the cache.
  The sole candidate hashes are unchanged. Further conversion needs approval.
  See the [one-projection run](development.md#approved-single-projection-native-model).
- The separately approved precision model has the exact same whole-file hash as
  that mixed model. Its native 80-token trace preserves the earlier score report
  exactly, with one dispatch/repack and zero generation. All three streamed
  modes match embeddings and first diverge at `l_out-0`, before the frozen FFN.
  Native-input projection replay is bit-for-bit exact; its independent reference
  error is 3.6182e-7. BF16-RHS rounding improves early agreement, not every later
  metric. Same-native-state BF16-RHS head replay has at most 1.9074e-6 logit error.
  Exact runtime parity remains open, with no accepted tolerance or quality
  claim. Its real trace regression passes source-row/payload/token/stage contracts.
  All GGUFs, spill and raw captures under `native/mimo-precision.GiDmcd` are deleted;
  only 27 small reports (240 KiB) remain in
  `native/mimo-precision-report-20261006-GiDmcd` under the cache. Candidate hashes
  and pinned tracked source worktrees are unchanged. Further conversion needs
  approval. See the [precision diagnostic](development.md#bounded-prefill-precision-diagnostic).
- Q/K normalization and recurrence now have separate operation-level controls.
  Native L2 uses a clamped norm, unlike the additive-epsilon Torch helper. The
  fixture-only Q/K mode preserves prior defaults and refuses dataset/calibration
  use. Sixteen native 128-wide cases cover 1/7/64/80 tokens, zero/nonzero initial
  state, activated/raw gates, tiled two-to-four-head broadcast, and unchanged input
  state; outputs/final state match an independent FP64 reference. The converter
  reorders grouped V weights/gates to tiled order, so this kernel mapping alone
  is not a model bug. No new tolerance, candidate or quality acceptance follows.
  See the [recurrent diagnostics](development.md#recurrent-operation-diagnostics).
- Split-aware datasets, calibration-only hashed captures, and frozen paired
  evaluation are implemented. The attributed CC-BY-3.0 CLINC150 four-choice
  proxy has four cases per split and four training captures. Held-out proxy
  inference remains untouched; it is not representative agent/tool quality.
- Independent-block compensation is not full GPTQ and has negative wider
  validation evidence: relative error 0.426 versus RTN 0.406 with one training
  context, and 0.456 with four balanced contexts. No compensated candidate,
  bulk fit, or policy promotion followed.
- Full-model ternary GGUF loading for MiMo or another model, broader model
  compatibility, production registry lifecycle safety,
  deployment-grade native services, calibrated decision quality,
  vision, edge performance, and ARM/RISC-V validation remain open.

### Reusable Local Paths

All paths below are outside Git under `$HOME/.cache/huggingface/embedded-jev`.
Check their existence after container/cache changes; do not redownload or
rebuild the environments merely because the chat history changed.

| Cache-relative path | Purpose |
| --- | --- |
| `models/mimo-2367e865d009c13ac81713a2878291d33ab28177` | Complete pinned BF16 source; 17 verified files |
| `quantized/layer3-ffn-down-rtn-searched-fp16` | The sole retained packed projection fixture |
| `dense-venv/bin/python` | CPU Torch 2.10.0, Transformers 5.12.1, safetensors 0.7.0 |
| `native/prism-source` | Prism commit `842b1880415d6f508f03b789e5ce70194def7bfd` |
| `native/prism-build/bin` | Pinned GGML CPU/base libraries |
| `native/jev-prism-runtime-v1-build/bin` | Separate versioned full llama/registry/bridge build and control; no model weights |
| `native/bitnet-source` | Pinned Microsoft BitNet checkout and llama.cpp submodule |
| `native/converter-venv/bin/python` | Isolated CPU converter environment; use the pinned `gguf-py` path |
| `datasets/clinc150-828f809-proxy-seed902` | Attributed public proxy, provenance, and four training captures |

The base interpreter is `/opt/venv/bin/python`; use the terminal for environment
work, as requested by the user. Set `OMP_NUM_THREADS=4` for dense model probes;
forcing `OPENBLAS_NUM_THREADS=1` or `MKL_NUM_THREADS=1` previously made them
dramatically slower. The cached `native/prism_group_scale_probe.so` includes
the signed-Hadamard path, but its PQ2 and registered-tensor exports have not been
refreshed: these controls compiled a fresh bridge in pytest temporary storage.
Rebuild only if an intended caller needs those exports, using the development
guide's command, including the two pinned internal-header include paths.

### Resume Contract

**Currently paused:** this session's October 9 requests authorized documentation,
planning and the bounded Defiant Fable preflight, which is complete. The user
plans to have another model implement the generalization queue; that session
starts work only on the user's instruction. The selected first non-MiMo profile
is the Defiant Fable safetensors source at `7af0a9c4`; its full download,
conversion and any artifact still need separate approval.

The user authorized routine bounded experiments, the existing full source
download, quantization scratch space, and periodic **local commits**. Keep only
one saved quantized candidate, do not push or create branches, preserve user
changes, and do not launch subagents without explicit authorization. Do not
bulk-quantize from these diagnostics or consume held-out data while tuning.
Once resumed, do not stop solely because representative domain data is absent if
authorized bounded runtime work can still proceed. Section 16 gives the existing
MiMo implementation context; the roadmap also defines new-model onboarding.
The registered-tensor step did not fit a policy, create another saved candidate,
run validation/held-out inference, or promote compensation.

## 1. User Intent

The user wants an **embedded Jev-like decision engine**, built from this chain:

`Selected Qwen3.5 9B-class model -> ternary quantization -> BitNet-capable CPU inference -> SemIf-style decisions`

MiMo-V2.6-Distill-Qwen-9B was chosen as the first test subject, not as a permanent
dependency. The user expects the system to become more general, hopes to support
most Qwen3.5-9B derivatives, and on 2026-10-09 selected DavidAU's Defiant Fable
safetensors source as the first non-MiMo profile. Generic profile support is not
delivered yet. Preserve MiMo-specific evidence and apply the model support gate
before claiming another checkpoint is compatible.

The original motivation is to use a knowledgeable, agentically fine-tuned model
on an edge computer, eventually with camera/sensor evidence. Outputs should be
typed decisions and probabilities without a generated answer sequence. Potential
targets are an Intel N100-class machine, an ARM board, or RISC-V hardware.

The user supplied a long conversation with another AI containing a proposed
implementation. The user explicitly asked us to double-check its physics,
mathematics, and research rather than trust its claims. Many critical statements
and scripts in that conversation are wrong or incomplete.

The user also requested:

- A devcontainer before starting the model implementation.
- Detailed checked-in research, design, and planning documentation.
- This detailed handover, because the next session may lose chat history.
- Local commits for the work so the user can push it themselves.

BitNet is a real requirement. Do not deliver ordinary llama.cpp inference and
call the BitNet part complete. Conversely, do not pretend that ternary MiMo is a
model trained natively with the BitNet b1.58 recipe; the same distinction applies
to post-training conversion of another dense checkpoint.

## 2. Read These Documents

| Document | Why it matters |
| --- | --- |
| [AGENTS.md](../AGENTS.md) | Stable agent workflow, verification, approval and safety rules |
| [README.md](../README.md) | Public purpose, capabilities, maturity limits, documentation links and licensing |
| [docs/research-audit.md](research-audit.md) | Technical audit, equations, actual formats, native API and cache risks, memory accounting |
| [docs/design.md](design.md) | Proposed artifact contracts, quantizer, genuine BitNet integration, decision semantics, caching, and vision |
| [docs/roadmap.md](roadmap.md) | Ordered implementation milestones, proposed modules, tests, and benchmark protocol |
| [docs/development.md](development.md) | Environment contents, Docker/Podman usage, dependency updates, and unverified platforms |
| [docs/sources.md](sources.md) | Primary source URLs, immutable revisions where resolved, and limits of the evidence |

These documents supersede unsupported claims from the supplied AI conversation.
They are not substitutes for reading the selected runtime implementation before
adapting its converter, kernels, graph, or bindings. For the first non-MiMo
profile, read the [support tiers](design.md#support-tiers-for-derivatives),
[derivative review](research-audit.md#derivative-release-review) and
[preflight procedure](development.md#defiant-fable-preflight-procedure).

## 3. Repository State

The repository originally contained a one-line README and a license. A
user-created ignore rule and the conversation PDF were also present locally.
No existing quantizer, training code, inference service, package scaffold, or
native runtime needed to be preserved or extended.

Delivered files:

| File | Implemented role |
| --- | --- |
| [.devcontainer/devcontainer.json](../.devcontainer/devcontainer.json) | CPU devcontainer, cache volume, extensions, post-create smoke command |
| [.devcontainer/Dockerfile](../.devcontainer/Dockerfile) | Ubuntu/Python/C++ toolchain and installation of locked numerical dependencies |
| [.devcontainer/requirements.in](../.devcontainer/requirements.in) | Four direct numerical/test/lint package pins |
| [.devcontainer/requirements.lock](../.devcontainer/requirements.lock) | Generated transitive lock with package hashes |
| [.devcontainer/smoke.py](../.devcontainer/smoke.py) | Numerical solve, C++17/OpenMP compile/run, tools and permissions check |
| [docs/test_research_math.py](test_research_math.py) | Nine small executable checks of audit assumptions |
| [embedded_jev/inventory.py](../embedded_jev/inventory.py) | Bounded pinned metadata/header retrieval, deterministic inventory and memory estimator |
| [tests/test_inventory.py](../tests/test_inventory.py) | Offline fixtures for valid, incomplete, inconsistent, unsupported, and range-ignoring sources |
| [embedded_jev/label_probe.py](../embedded_jev/label_probe.py) | Pinned text-only MiMo template and A-P token-boundary check without model weights |
| [tests/test_label_probe.py](../tests/test_label_probe.py) | Offline tokenizer, processor, and fixture contract cases |
| [tests/fixtures/agent_tool_smoke.json](../tests/fixtures/agent_tool_smoke.json) | Five synthetic option-mapping cases, never calibration or benchmark data |
| [.gitignore](../.gitignore) | Existing PDF exclusion preserved; generated Python bytecode caches ignored |

This table records the bootstrap delivery only. Later modules and native sources
are described in the development guide; its
[coupling inventory](development.md#mimo-coupling-inventory) maps the current
files and their MiMo-specific assumptions.

The documentation above is also delivered. The existing [LICENSE](../LICENSE)
was not modified. Third-party model, data, and code licenses remain separate.

The original PDF is deliberately not part of the commits. Do not force-add it.
All essential project decisions are captured in tracked Markdown instead.
No legacy devcontainer backup was edited. If backups appear later, leave them
alone unless the user explicitly requests otherwise.

## 4. Entering the Environment

The original host workspace is `/home/vscode/AI/embedded-jev`. The observed CLI
mount inside the container is `/workspaces/embedded-jev`. Use the current workspace
root rather than assuming that host-specific absolute path in scripts.

In VS Code, use **Dev Containers: Reopen in Container**. The host needs the Dev
Containers extension and a working container engine. With Podman, configure the
host setting `dev.containers.dockerPath` as `podman`.

The bootstrap session built and started the container through the CLI without
reopening the editor. A previous container/image may therefore already exist.
Reuse or rebuild through Dev Containers; do not delete unrelated containers or
prune caches. A hard-coded historical container ID is not a durable interface.

Host-side commands, from this repository:

```bash
devcontainer read-configuration --workspace-folder . --docker-path podman --log-level info
bash .devcontainer/prepare-build.sh
devcontainer build --workspace-folder . --docker-path podman --image-name embedded-jev-research:dev
devcontainer up --workspace-folder . --docker-path podman
```

Use `docker` instead of `podman`, or omit `--docker-path`, on a Docker host.
These are host commands. The development image intentionally does not mount a
host container socket or provide nested container management. The
[certificate export](development.md#host-certificates) runs automatically for
`up` and VS Code; the explicit command covers a separate `build`.

Inside the container, run:

```bash
pwd
python --version
python .devcontainer/smoke.py
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider docs/test_research_math.py
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_inventory.py
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_label_probe.py
ruff check --no-cache .devcontainer/smoke.py docs/test_research_math.py
ruff check --no-cache embedded_jev tests
git status --short
```

Expected Python interpreter: `/opt/venv/bin/python`. The smoke check must print
`"status": "ok"`; the research suite must report nine passing tests. No model
downloads or credentials are needed for these checks.

From outside, use `devcontainer exec --workspace-folder . --docker-path podman`
before an inside-container command. Do not try to fix a missing host NumPy by
installing the research dependencies globally; use the container.

The opt-in [Podman/WSL NVIDIA profile](../.devcontainer/gpu/devcontainer.json)
reuses the CPU image and adds `nvidia.com/gpu=all`, `/dev/dxg`, and the WSL
driver-library mount from the user's working container. The default config
remains CPU-only. Its post-create command checks `/dev/dxg` and `nvidia-smi`;
see [docs/development.md](development.md) for host-side launch commands and
prerequisites. A WSL/Podman host build and post-create check succeeded on
2026-09-28, reporting an RTX A3000 Laptop GPU (12,288 MiB, driver 595.95)
inside the container. The base image contains no GPU Torch; a separate isolated
GPU venv passed a tiny CUDA matmul (below). Model weights and native BitNet
were not exercised by that GPU smoke; later CPU checks used cached weights.
An optional [CUDA driver probe](../embedded_jev/gpu_probe.py) can now run from
the already-started GPU container without a rebuild or Torch installation;
it successfully enumerated one RTX A3000 Laptop GPU at compute capability 8.6
and 12,884,377,600 driver-reported memory bytes. Its opt-in four-byte
`--memory-round-trip` mode passed on the same WSL/Podman container, confirming
context creation, allocation, host/device copies, and cleanup. No GPU kernel
was executed by this driver-only probe.
An optional [GPU Torch compute probe](../embedded_jev/torch_probe.py) has offline
fail-closed tests and a 4x4 FP32 CUDA matmul. It passed on the RTX A3000 with
isolated `torch==2.10.0+cu128` (CUDA 12.8); the base research venv remains
unchanged. NumPy was not installed in the isolated GPU venv, producing a warning
but not affecting this pure-Torch smoke. This GPU-only check does not validate
model loading, quantization quality, or BitNet CPU execution; later CPU evidence
is recorded separately below.
The compatible `torch==2.10.0+cu128` CPython 3.12 Linux wheel is 916,856,347
bytes by HTTP header, plus dependencies. [docs/development.md](development.md)
has the opt-in isolated install and run commands. The first GPU-container uv
attempt failed before installation: top-level `.cache` is root-owned, so uv's
default cache directory was not writable. Use the documented writable volume
child and explicit `UV_CACHE_DIR`; do not change ownership or rebuild for this.
WSL `df` does not establish physical free space on the Windows drive containing
the virtual disk. No Torch package was added to the lightweight research base;
the separate GPU venv and cache are not a locked full-model environment.

## 5. Initial Environment Validation

The initial checks ran on Linux x86-64 with rootless Podman, Dev Containers CLI
0.87.0, and Node.js 24.19.0 on the host. This is not evidence for an ARM or RISC-V
build, nor a benchmark of any proposed edge board.

Observed versions inside the container:

| Component | Version |
| --- | --- |
| Python | 3.12.3 |
| Clang | 18.1.3 |
| CMake | 3.28.3 |
| Ninja | 1.11.1 |
| Git | 2.49.0 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| pytest | 8.4.2 |
| Ruff | 0.13.1 |
| uv | 0.8.22 |

The following succeeded:

- Devcontainer configuration resolution, image build, and startup.
- Post-create smoke check as the configured non-root development user.
- A numerical Cholesky solve using NumPy/SciPy.
- Compilation and execution of a small C++17/OpenMP program.
- Writable workspace and Hugging Face cache paths.
- `pip check`, with no broken package requirements.
- Nine mathematical/accounting checks and Ruff on the two Python files.
- Eight offline inventory tests, Ruff/editor diagnostics, and live pinned-header
  reconciliation without downloading weight payloads.
- Opt-in WSL/Podman GPU profile image build and post-create CPU smoke; inside
  the container `nvidia-smi` reported RTX A3000 Laptop GPU, 12,288 MiB VRAM,
  driver 595.95. No CUDA kernels or model inference were exercised.
- Local Markdown link and code-fence checks during documentation preparation.

The nine tests cover matching rotations, Hessian spectrum/condition-number
invariance, repeated-activation rank, inability to replace group scales with one
tensor scale, format bit rates, INT16 FWHT overflow, vocabulary matrix memory,
cache/Hessian/activation accounting, and selected-head conditional probabilities.

They do **not** cover a GPTQ implementation, full-model loading, tokenizer parity,
native codecs, BitNet kernel dispatch, camera processing, or real decision quality.
The inventory tests also do not test those behaviors.

## 6. Base Environment Scope and Implementation History

The base contains compiler tools and a lightweight numerical Python environment.
It deliberately does not install Torch, Transformers, datasets, safetensors,
gguf, SemIf, llama-cpp-python, BitNet, Prism, or CUDA. No inference server runs,
no ports are forwarded, and no camera or actuator is exposed.

The header-only pinned inventory fetched 172,461 metadata/header response-body
bytes. Later, a separate opt-in reader fetched 2,048 BF16 weight bytes from four
rows and 256 columns of the pinned `layers.3.mlp.down_proj.weight`. On
2026-09-29 the user authorized and downloaded the complete pinned BF16 snapshot
into `$HOME/.cache/huggingface/embedded-jev/models/mimo-2367e865d009c13ac81713a2878291d33ab28177`.
All 17 files passed pinned Hub size/content checks, and all four safetensors
shards passed LFS SHA-256 verification. Offline header/index reconciliation
counts 760 tensors and 18,819,627,488 logical weight bytes; the four shard
files occupy 18,819,720,848 bytes including headers. A local 2 KiB sample
matched the prior pinned remote ranges and the opt-in native toy slice test
passed. A second bounded reader checks four complete layer-3 FFN-down rows
(98,304 BF16 bytes) from the local shard. On fixed synthetic Gaussian inputs,
max-abs, searched-scale, and signed-Hadamard searched-scale ternary RTN had
relative output RMSE of 0.685, 0.426, and 0.446, respectively. The signed
rotation itself preserves the dense output; rounding does not. Do not select
a full-model quantization policy from four rows or synthetic inputs. No
quantized model copy was kept; see [docs/development.md](development.md) for
the reproducible command. A second CLI streams the entire 4,096 x 12,288
FFN-down tensor in 64-row batches: relative **weight** RMSE is 0.770 for
max-abs RTN, 0.461 for searched FP16 RTN, and 0.454 for signed-Hadamard
searched RTN. This is one full projection, not representative activations or
a saved/loadable ternary model. Do not bulk-convert MiMo from these results.
An opt-in native test streams and packs this entire projection in memory,
then executes a BitNet-derived AVX2 group-scaled matvec: all 4,096 outputs
match the portable integer/group-scale reference (max error 1.67e-6).
Relative error versus dense BF16 output for one synthetic A8 input is 0.467.
This is neither the stock I2_S model path nor a MiMo loader/quality proof.
A separate CPU-only Transformers 5.12.1 environment now materializes the
verified embedding and first four text layers plus norm (55 BF16 tensors,
3,764,136,064 bytes) under a 4 GiB source-weight budget. The 13-token
text-prefix forward reaches the layer-3 FFN-down input. On those actual
model-path activations, unrotated searched-scale ternary RTN has 0.423 relative
output RMSE versus the BF16 FFN output, without A8. This one prompt is neither
calibration nor decision quality; the probe produces no final model logits,
no generated tokens, and no retained quantized candidate. The same captured
last-token activation, after dynamic A8, fed the isolated BitNet-derived
AVX2 group-scale kernel: 4,096 outputs matched portable integer arithmetic
(maximum difference 3.23e-8), while relative error versus dense BF16 was
0.433. This is a single projection with a real model-path activation, not a
registered Qwen3.5/BitNet tensor type or full-model decision result.
An opt-in non-thinking chat-template variant validates distinct A-P label
continuations and runs a 22-token prompt with zero generated tokens; the
last-token native/reference max difference is 1.10e-7 and its BF16-relative
error after A8 is 0.422. This is still synthetic engineering smoke, not
representative calibration.
The pinned BF16 text decoder can now execute all 32 layers sequentially,
materializing at most one 436,814,208-byte layer at a time after the
2,034,237,440-byte embedding. The four-layer activation and final-norm hashes
match the previous full-prefix control exactly. A bounded untied LM-head reader
then scored only A/B (16,384 BF16 bytes): FP32-accumulated logits 19.481 and
20.478, conditional probabilities 0.269/0.731, zero generated tokens.
Independent safetensors BF16 row slices and Torch BF16 rounding passed parity
(max rounding gap 0.022). The full-vocabulary label mass is unknown, and this
single engineering prompt is not calibrated confidence or a decision benchmark.
The five versioned synthetic agent/tool fixture cases (80-87 tokens) now pass
through the same streamed text scorer and return typed option IDs, descriptions,
labels, and A-C conditional scores with zero generated tokens. All five selected
IDs matched fixture expectations; reordered labels and a missing-permission
case are gated in an opt-in test. This is engineering smoke only, not an owned
held-out benchmark, representative calibration, or a confidence claim.
A subsequent Python-hosted staged text run replaces only layer 3's real
FFN-down matmul with the BitNet-derived searched-FP16 ternary/A8 grouped AVX2
kernel. The BF16 and substituted runs share the same pre-FFN activation hash;
native/portable last-token outputs agree to 1.10e-7. Final A/B conditional
scores on the 22-token prompt move from 0.2695/0.7305 to 0.2709/0.7291.
No quantized candidate was saved. This is a proof of one in-model Python
adapter, not evidence of acceptable quality, a registered GGUF type, or a
loadable BitNet/Qwen3.5 runtime.
Subsequently **one** searched-FP16 layer-3 FFN-down projection was saved outside
Git as hash-checked packed codes, FP16 row/group scales, and a JSON manifest
(about 13 MB) after verifying the pinned source shard SHA-256. Reloading it
through the same native adapter reproduces the fresh-RTN 32-layer final hash
and A/B scores exactly. [docs/development.md](development.md) records its path,
hashes, and opt-in test. This is not GGUF or an approved model-loadable format;
it does not address the large ternary error or missing calibration/holdout data.
An opt-in full-vocabulary normalizer streams the 2,034,237,440-byte BF16
output head in bounded row batches. On the 22-token smoke, selected A/B mass
was 0.873 for BF16 and 0.882 after the one-layer native substitution; B was
the top token both times. This clarifies the earlier conditional-only score
for this prompt, but is not confidence calibration or a decision-quality gate.
The 2026-09-30 post-restart review reverified all source and candidate hashes.
It fixed actual-size-before-hash checks and ambiguous projection manifests,
FP32 numerator/denominator inconsistency in full-vocabulary mass, and missing
native dimension/output-byte overflow guards. The staged-prefix regression
now compares directly with the normal Transformers forward. The cached
candidate's bytes and its unrotated searched-scale policy were not changed.
The saved-projection path now bypasses BF16 FFN-down tensor materialization;
an instrumented full-model regression rejects any attempt to load that source
tensor and confirms identical saved/fresh final hidden hashes and option scores.
A separate dataset contract now supports explicit calibration, validation, and
held-out splits with declared source/license provenance. Case IDs, groups,
and whitespace/option-order-equivalent prompts cannot cross splits. The
streamed scorer requires an explicit split and retains its digest and purpose;
it never relabels the existing synthetic fixture as representative data.
The dataset-to-32-layer smoke passes using temporary synthetic split data.
Representative agent/tool calibration/holdout cases are still absent; scoring
performs no tuning.
The dataset path can now save the actual layer-3 FFN-down input in a bounded
calibration-only array plus digest/provenance manifest. Evaluation splits and
native substitutions cannot request captures; the writer checks the original
dataset digest again before saving. A real-prefix temporary capture reproduces
the model's activation hash exactly, but retains its synthetic-smoke purpose.
No representative activations or additional quantized candidate were created.
A bounded paired evaluator now compares explicit cases with BF16 and the
single frozen saved native projection. It records immutable data/candidate/
kernel/source/runtime bindings, choices, conditional-score deltas, and expected
matches without fitting. Changed inputs or inconsistent paired reports abort
the comparison. Real BF16/native paired execution passes on temporary synthetic
split data; this is evaluation infrastructure, not representative quality proof.
A small public CLINC150 four-choice proxy is now cached outside Git, pinned at
`828f8093932c8fe6ca7936c3d2e52903b1c523de`, with the original CC BY 3.0
license and attribution README. Its 4/4/4 source train/validation/test subsets
retain those roles and use deterministic distractor shortlists, excluding OOS.
One validation case matched the gold intent in BF16/native runs (conditional
total variation 0.00354), and one training FFN capture reloads correctly.
Proxy held-out data remains unscored and no new candidate was fitted. This is
not official CLINC150 performance, agent/tool representativeness, or proof
against training contamination; see the development guide for provenance.
The 256-column compensated toy was fit using 75 tokens from the public train
capture and scored on 70 live validation tokens. For four rows/two groups,
searched RTN relative error is 0.417 calibration / 0.423 validation, versus
0.212 / 0.401 for searched compensation; max-abs compensation gives 0.970
on validation. Code/scale hashes cannot depend on validation inputs, and
compensated native A8 arithmetic parity passes. No candidate was saved or
replaced, held-out data was not observed, and these local errors do not prove
full-width GPTQ or final-model quality.
The bounded compensation factorization is reusable across row batches, with
read-only factors tied to calibration digest/shape and damping settings.
Fresh/cached codes and scales match exactly; reuse with changed inputs or
settings is refused. Its original 256-column width bound remains intact.
A separate independent-256-column-block approximation was tested on four
complete 12,288-input rows. It discards cross-block curvature/error propagation
and is not full GPTQ. Searched compensation has 0.194 calibration / 0.426
validation relative error, versus 0.390 / 0.406 for searched RTN on the same
public proxy pair. Native parity passes for all 96 groups, but this validation
result is worse; no wider fit, policy promotion, or second artifact was saved.
The four public training FFN captures are now available. A bounded loader
balances 32 evenly spaced tokens from each into a read-only 128-token sample,
retaining source indices/hashes and refusing mixed datasets or repeated IDs.
On the same validation case, full-width four-row searched compensation becomes
worse (0.456 versus the unchanged RTN 0.406), although native parity passes.
No merged capture or compensated candidate is saved, and held-out inference
remains untouched. Further policy promotion needs a different justified
approximation and representative quality evidence, not more training-score wins.
The original frozen RTN projection now also executes through a real pinned
Prism CPU GGML `MAP_CUSTOM2` node. Its full 32-layer hidden hash and selected
scores match direct native execution exactly. The bounded graph bridge checks
one callback and releases its graph/backend buffers after each call. Actual
GGML dependency paths/hashes are recorded and frozen by paired evaluation.
This is one native graph-scheduled projection with Python-hosted model/A8
preparation and callback-owned packed weights, not a registered GGUF format or
whole-model native runtime. No new candidate or speed claim follows.
A third `prism_ggml_f32` backend now performs production-batch A8 in the native
GGML callback itself. Its codes/scales and final 32-layer selected scores
match the direct and Python-prepared graph paths exactly. An instrumented
test rejects Python preparation of that batch; only the one-token reference
check remains Python-side. Frozen artifact/model hosting and format limits
are unchanged, and no speed or quality claim follows.
An additional reversible `prism_ggml_hadamard128` experiment rotates weights
in 64-row batches with signs (seed 773) and matching native input signs/FWHT/A8.
Identity artifacts are refused on that path. A four-row dense FP32 probe
passes before rounding, and the 32-layer/public validation smoke remains
finite, zero-generation, and native-reference consistent. The public case
still selects `calories`; no quality promotion or rotated artifact followed.
A separate native codec control now proves that the frozen signed codes and
FP16 scales convert exactly to pinned PQ2_0 bytes (different packing order
from BitNet). Every full-projection native-decoded weight matches exactly,
and a real PQ2 tensor/MUL_MAT matches controlled activation oracles. Native
Q8_0/Q8_K preparation differs from our group-128 A8 contract. No conversion
file or second candidate was saved, and this is Prism PQ2 dispatch, not
BitNet dispatch or full GGUF/model-loader compatibility.
Separately, an MIT-licensed 1,187,801,280-byte BitNet control GGUF was
downloaded, SHA-256 verified, and loaded in the pinned native fork. See
the development guide for both bounded tracks. No vision path or registered
MiMo ternary/BitNet model runtime has been validated, and no owned MiMo quality,
perplexity, latency, energy, or RSS result exists. The host GPU
is an RTX A3000 Laptop GPU with 12,288 MiB VRAM. On 2026-09-28, WSL showed
29 GiB RAM, 21 GiB available, and 8 GiB swap; the user reported 48.3 GB free
on the Windows drive backing the virtual disk. These point-in-time readings do
not establish RAM for full-model conversion: 18,819,627,488 BF16
weight bytes alone occupy about 17.53 GiB of memory before caches and scratch.
On 2026-09-29, the cache filesystem had 319 GB free after the download; this
does not establish working RAM or a safe conversion scratch budget. Keep only
one quantized candidate at a time and measure/offload before full-model work.

Do not run the full-model commands from the original PDF: their modules do not
exist here, and several use incompatible layouts or nonexistent APIs.

## 7. Immutable Source Anchors

Use these as the audit's starting points, not as automatically compatible
dependencies:

| Component | Inspected revision |
| --- | --- |
| MiMo HF checkpoint | `2367e865d009c13ac81713a2878291d33ab28177` |
| Microsoft BitNet parent | `0b341e582afbf9e1011f24744b554c96a3477eb5` |
| BitNet llama.cpp gitlink | `390c307752ab78fd8189f359d6954c9ba1be74af` in `isHuangXin/llama.cpp` |
| SemIf | `23cf1f39fc9534fe81437200959b6dfc7106e45a`, branch `master` |
| Prism source | `842b1880415d6f508f03b789e5ce70194def7bfd` (`prism-b10735-842b188`); build-specific library hashes are in the development guide |

Some broader documentation was read on moving branches. The source register
distinguishes those observations from immutable pins. Pin all selected runtime
submodules, converter packages, and datasets for actual experiments.

SemIf's inspected dependency metadata pins Torch 2.10.0, Transformers 5.17.0,
and llama-cpp-python 0.3.35 for its optional native backend. MiMo's configuration
records Transformers 5.12.1. Do not assume these and BitNet's build dependencies
should be installed into one environment.

## 8. Critical Findings to Preserve

### The Requested Checkpoint Is Real, but the Proposed Footprint Is Not

The pinned model is `Qwen3_5ForConditionalGeneration`, with 32 text layers,
24 linear-attention and eight full-attention blocks, hidden width 4,096,
intermediate width 12,288, and vocabulary size 248,320. Its embeddings are untied.
HF reports 9,409,813,744 BF16 parameters. The pinned header inventory independently
counts 760 tensors, 9,409,813,744 parameters, and 18,819,627,488 tensor bytes.
It attributes 912,020,960 bytes to vision; the config declares one optional MTP
layer but no MTP tensors appear in the weight index or headers.

Each 248,320-by-4,096 BF16 vocabulary matrix is 2,034,237,440 bytes. Input embedding
plus output projection therefore requires about 4.068 GB before the transformer.
The conversation's 150 MB output head and complete under-3-GB BF16-preserved model
are not possible. Q8_0 would be about 1.081 GB per such matrix, subject to actual
converter support and quality validation.

The configured SSM dtype is FP32. The inspected Prism hybrid-cache constructor
also uses FP32 recurrent state. Preserving a few recurrent *parameters* in BF16
does not establish the dtype or size of runtime recurrent *state*.

### Ternary Values Are Not a Storage or Execution Standard

- `PQ2_0`: 128 values, 34 bytes, 2.125 bpw.
- `PTQ1_0`: 128 values, 28 bytes, 1.75 bpw; dense trits and a scale.
- `Q2_0`: 64 values, 18 bytes, 2.25 bpw.
- `TQ1_0`: 256 values, 54 bytes, 1.6875 bpw; cannot merge arbitrary two-group scales.

Those rates are for the respective blocks, not total resident model memory.
I2_S, TL1, TL2, codebook IQ formats, and Prism formats have different contracts.
Never relabel bytes as another ggml type or write them as `I8` and claim native
ternary inference.

### Genuine BitNet Integration Is Necessary Work

The inspected I2_S quantizer uses a tensor-level FP32 scale and architecture/build
dependent packed ordering. It is not the PDF's per-group FP16 layout. A MiMo
group-scaled artifact cannot pass through it unchanged without losing scales.
Its MAD kernels use integer multiply-accumulate; LUT kernels are a separate path.
BitNet does not mean that every transformer operation becomes addition-only.

An isolated [group-scale AVX2 fixture](../native/bitnet_group_scale.cpp) now
executes a pinned BitNet I2_S-style packed-code MAD block on x86-64. The 128
codes occupy 32 bytes; one separate FP32 scale per output row/group adds four
bytes (2.25 bpw for that weight representation, not PTQ1_0's 1.75 bpw).
Because BitNet packs ternary codes as 0/1/2, the kernel subtracts each group's
signed A8 activation sum before applying row/group and activation scales.
Seven golden tests compare compiled outputs against an independent scalar
reference for zero weights, negative/extreme activations, unequal scales, and
serial three-token batches at 256/4,096/12,288 input widths. The
[BitNet MIT notice](../native/BitNet-LICENSE.txt) is retained separately.
The test-only [activation preparation](../embedded_jev/activation.py) applies
the same signed normalized Hadamard transform to activations and weight inputs,
then rounds rotated activations to symmetric A8 with one FP32 scale per
token/input group (all-zero groups use scale 1). Toy integrated tests at 128-
and 1,024-point transforms compare the compiled multi-token output with an
independent quantized reference and bound its difference from rotated dense.
This does **not** execute the Microsoft fork's GGML graph, run transforms/A8
inside the runtime, process an actual model block, run fused GEMM, or prove
runtime dispatch, quality, ARM/RISC-V support, or a speedup. Those remain gates.

Separately, the pinned Microsoft BitNet parent and llama.cpp gitlink were
checked out **outside the workspace** (about 208 MB source). Clang 18 built
only the `ggml-cpu` library after setting `CCACHE_DIR` to a writable path.
The build graph includes `ggml-cpu-i2s.c`, fork `quants.c`, and BitNet's LUT
source, but not `ggml-bitnet-mad.cpp` under these flags. The compiled fork
exports `ggml_vec_dot_i2_i8_s`; optional
[native-control tests](../tests/test_bitnet_native_control.py) call this real
symbol. On AVX2, an all-zero ternary group encoded as 1 returns the signed
activation sum, so the adapter subtracts it **per group** before applying
row/group and activation scales. Golden vectors pass. The pinned fork builds
with warnings including enum-initializer overrides; these model-free tests
alone did not establish GGUF loading or model inference. A separate
[model-free GGML graph smoke](../native/bitnet_ggml_graph_smoke.cpp) did execute
its I2_S path with four packed rows, yielding [0, 128, -128, 0] for an all-ones
input. This verifies that **one tiny graph** dispatches, not that any Qwen3.5
projection or group-scaled MiMo model does.
A second mode reuses that GGML graph serially for two group-128 blocks, then
combines their partial outputs with different per-row/group scales outside the
graph. It returns [32, -320, -256, 64] on a fixed four-row fixture. This
proves group scales can survive execution through genuine fork I2_S graphs in
a toy serial bridge; it is not a native packed group-scale type, fused GEMM,
MiMo model loader, or measured speedup.
The same graph also passed two independent token columns with +1.0 and -2.0
F32 inputs, verifying different per-token dynamic A8 scales in this fixture.
No hybrid sequence state or real model activations were exercised.

Later, the separately pinned official
[BitNet control GGUF](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T-gguf/tree/a1f2f1c765812aa8af3f6eda4a313707064bba15)
(1,187,801,280 bytes; SHA-256
`4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`)
loaded via the pinned fork's `llama` API. Its 22-token CPU prompt returned
128,256 finite final-position logits with **zero sampled/generated tokens**.
A batch debugger stopped at `llamafile_sgemm_i2s` from
`ggml_compute_forward_mul_mat`, establishing actual I2_S execution for that
control model. One load and prefill took about 410/439 ms on this machine;
these are not latency benchmarks. The loader also reported CPU_REPACK buffer
fallback, which needs profiling before any optimized-path or speed claim.
The control prompt's A-C labels each extend the exact prefix by one distinct,
non-special token (IDs 32/33/34). Direct final-logit softmax selected A with
conditional probability about 0.553, but the allowed labels carried only
about 0.000070 of the full-vocabulary mass and calibration status is unknown.
This is a structural **uncalibrated** SemIf-style readout on a different model,
not MiMo tokenizer parity, MiMo graph dispatch, group-scale preservation in
MiMo, SemIf backend integration, or decision correctness.

Use a native BitNet checkpoint as a tooling control. The preferred proposed MiMo
direction is to integrate a real BitNet-derived kernel into a Qwen3.5-capable
runtime, preserving group scales, rotations, activations, and hybrid operations.
This is a hypothesis awaiting a small integration spike. Extending BitNet's own
fork is an alternative to compare, not an already ruled-out possibility.

The inspected top-level build options are `BITNET_ARM_TL1` and
`BITNET_X86_TL2`. Kernel generation, model shapes, layout, and dispatch must still
match. RVV support for these paths is not established by generic ggml RVV flags.

### Rotations Must Preserve the Function

For row-vector activations and weights with input features on the last axis:

$$
XW^T=(XR)(WR)^T,\quad RR^T=I.
$$

Both operands need the same complete transform. Explicit signs can make a
Hadamard-based rotation nonsymmetric. Do not interchange inverse and forward
order. Preserve bias and do not move rotations through gates/nonlinearities
without proving equivalence.

Curvature is collected in that same basis:

$$
G=2X^TX/N,\quad G_R=R^TGR.
$$

Orthogonal similarity preserves eigenvalues, rank, and spectral condition number.
It changes coordinate incoherence, not conditioning by 10-100x as claimed in the
PDF. Damping changes eigenvalues. More tokens do not guarantee full rank, and
duplicating activation samples does not add new covariance information.

### The Provided GPTQ Loop Is Not Correct

The PDF computes a scale across output rows per column, changes it while walking
the group, and saves only the last scale. That is not one retained scale per
output-row/input-group. Its fallback from inverse-Cholesky to `pinv` changes the
meaning of the update. Its replay drops actual decoder arguments and rotates
weights without the matching activations.

Start from a reviewed implementation and small reconstruction tests. Store codes
and exact representable scales explicitly; never infer them again from BF16
dequantized tensors. Keep processing-block size separate from scale-group size.
Act-order permutations require matching runtime/group mapping and are initially
best left disabled.

The toy [ternary RTN baseline](../embedded_jev/ternary.py) now saves one FP16
max-abs scale and -1/0/+1 codes per output row and contiguous group. Code
assignment uses the **stored representable scale**, handles zero groups, and
rejects nonfinite, overflowing, underflowing, or tail groups. Reconstruction
uses only those saved arrays. Toy golden tests also cast the saved scales to
FP32 for the isolated BitNet-derived AVX2 kernel and compare its output with
the saved-artifact reconstruction at 256 and 4,096 input values. This is not
GPTQ, an exporter/codec, a model-weight quantization result, or native graph
integration; quality and format gates remain open.

The same module also has a separate **256-column-bounded GPTQ-style toy**:
token-normalized curvature, positive bounded damping retries, inverse-Cholesky
error updates, and FP16 row/group scales fixed at group entry. Processing blocks
are separate from scale groups and must be group-aligned; tested smaller blocks
reproduce the full-width toy's saved codes and scales. A correlated
four-input toy improves its local reconstruction error relative to RTN, while
diagonal-curvature, zero-group, factorization-failure, and native saved-artifact
tests pass. It follows the algorithmic update in
[pinned IST-DASLab GPTQ](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py)
under this repository's Apache-2.0 license. No pseudoinverse or act-order
fallback is used. It is not reviewed full-model GPTQ, a proof of model quality,
or ready for multi-thousand-width factors or the MiMo hybrid graph.

The [toy artifact container](../embedded_jev/ternary_artifact.py) uses bounded
uncompressed ZIP entries containing non-pickle NumPy codes/scales and a versioned
JSON manifest with hashes. Identity and explicit signed-Hadamard transforms are
validated, including sign width, block size, and sign-before-Hadamard order.
Tests reject changed hashes, object arrays, malformed transform metadata, and
oversized data; loaded artifacts feed the native AVX2 kernel in toy cases.
The archive is capped at 1 MiB and marked as toy/no-model-weights, not PTQ1_0,
PQ2_0, GGUF, or a complete source/calibration/runtime provenance artifact.
A separate toy Prism v1 metadata check requires full logical input widths,
rejects unsupported foldable names, and refuses conflicting block sizes or
sign vectors for the same input width. A conservative candidate mapping now
derives widths from the pinned MiMo inventory for exact HF projection paths;
it still rejects `ssm_out`: the inventory does not carry the key/value-head
counts needed to validate Prism's grouped-V loader geometry.
layer-3 FFN down is 12,288 inputs and FFN gate/attention Q/K are 4,096.
It rejects assigning the 256-wide slice to the full FFN down projection.
The pinned Prism Qwen3.5 name map returns no direct match for MiMo's exact
`model.language_model.layers.*` names, but the converter's shared filter removes
`language_model.` before mapping. A model-free opt-in check of all 760 pinned
index names found 333 vision-side names filtered from the text pass and all
427 retained text names mapped after that filter, including 24 `.dt_bias`
names renamed to `.dt_proj.bias` by the pinned Qwen converter source. Name
coverage alone does not verify tensor values. A separate pinned source-method
test with isolated CPU Torch 2.10.0 executed small QKV/Z/alpha/conv1d value-head
row reorders against independent NumPy ordering, and checked A-log `-exp` and
dt-bias reorder/rename. Unrotated `out_proj` permutes value-head columns;
Hadamard-folded `out_proj` keeps training order and sets the grouped-V runtime
permutation flag. Executing the pinned converter's `add_hadamard_metadata`
method on a toy folded `ssm_out` manifest also records a typed
`prism.hadamard.gdn_v_grouped=true` writer call only when that flag is set.
The pinned GGUF writer/reader round-trips that bool and folded weight name in
a no-tensor metadata file. This does **not** cover MiMo-sized tensors, model
tensor or vision-projector export, or native loader parity. The inspected
Prism release tag is fully pinned at
`842b1880415d6f508f03b789e5ce70194def7bfd`; no converter or Prism model
loader was exercised by this check.
An opt-in native test also processes the bounded 2,048-byte real MiMo projection
slice through toy signed rotation, searched FP16 code/scale storage, reload,
synthetic dynamic A8, and the standalone AVX2 BitNet-derived batch kernel.
The original and rotated dense outputs agree; native output matches the saved
artifact reference. This is only four weight rows with synthetic activations,
not a full attention/recurrent block, actual graph execution, or quality proof.

### Prism Has a Concrete Transform Schema

The inspected `prism.hadamard.*` metadata specifies version, one block size,
transform identity, input axis, tensor names, and identity/explicit signs.
Explicit sign vectors are keyed by input width, not independently by tensor.
Optional `gdn_v_grouped` changes the value-head ordering at `ssm_out`.
The pinned MiMo `config.json` is 2,784 bytes and declares 16 linear key heads,
32 linear value heads, and 128 dimensions per value head, so Prism's declared
repetition count is two. This metadata alone does not prove folded `ssm_out`
tensor shape/value parity or safely extend our reject-only toy exporter.

The generic matmul helper applies optional permutation, signs, Hadamard, matmul,
and scale in that order. Inverse embedding lookup uses Hadamard then signs.
Version 2 is tied-output mode; our MiMo checkpoint is untied. Begin with compatible
version-1 semantics and unrotated embeddings/head.

A richer internal manifest must not silently export incompatible per-tensor
block sizes or signs into this schema. The pinned Prism fork at
`842b1880415d6f508f03b789e5ce70194def7bfd` built CPU `llama`/GGML libraries.
An opt-in [native FWHT test](../tests/test_prism_native_control.py) executed a
two-token signed 128-point Hadamard graph with zero matrix data and agreed with
independent dense math (max error about 7.2e-7). A separate toy now schedules
the BitNet-derived AVX2 grouped dot as Prism's `MAP_CUSTOM2` node after FWHT
and dynamic A8 (max transform/output errors about 3.8e-7/9.5e-7).
An optional grouped-V mode first permutes a two-key-head/two-repetition,
64-wide tiled activation to grouped feature order, then applies grouped signs,
128-point FWHT, A8, and that same dot. Independent scalar parity on both
tokens passed (max transform/output errors about 5.7e-7/7.2e-7).
Two consecutive graph evaluations also passed when both input and sign leaves
were restored before each run; the toy graph allocator may overwrite those
buffers during compute. This is a fixture-owned callback, **not** a loadable
group-scale GGUF type, Qwen3.5 model hook, proof of MiMo head geometry or
folded-weight export, safe sequence-state implementation,
codec test, other CPU/ARM backend, or speedup.

### Integer Arithmetic Needs Bounds

A 1,024-point unnormalized FWHT of INT8 inputs can reach 130,048, exceeding INT16.
Use FP32 rotation first, then dynamic A8 and INT32 accumulation. Integer FWHT is
a later measured optimization requiring proven intermediate scaling or width.
Quantizing before rotation changes the error relative to quantizing after it.

Group-scaled ternary execution must retain
$y_r=\sum_g s_{r,g}a_g\sum_{i\in g}t_{r,i}z_i$.
Row-dependent group scales cannot generally be folded into a single shared input
scale. Measure preprocessing, lookup construction, unpacking, and rescaling.

### Jev Is an Interface Inspiration, Not a Recovered Architecture

TypeSafe describes its own architecture, parallel sampler, and RLCD training.
SemIf explicitly reproduces an open interface pattern with frozen models, not
that undisclosed stack. Direct logits do not provide generated chain-of-thought
reasoning for free and do not eliminate semantic errors.

Public Jev exposes Choice, Score, and Noul. Choice selects a categorical option;
Score is an expectation over ordered descriptive levels; Noul is probability of
yes, not a degree measurement. Confidence is a separate distribution statistic.
Our initial scope is SemIf's 2-16 fixed-label choices, with later typed adapters.
Use `max_option_probability` and explicit calibration status rather than pretend
to duplicate TypeSafe's confidence computation.

### The Existing SemIf Backend Is Useful

SemIf already includes a llama.cpp CPU backend. It uses a pinned HF tokenizer,
verifies GGUF tokenization, marks the final batch position for logits, and reads
`llama_get_logits_ith`. It snapshots/restores the entire hybrid sequence state.
Its CPU shared path iterates restored suffixes serially; it does not evaluate
arbitrarily many questions in constant time.

Map semantic option IDs/descriptions to context-verified one-token labels such
as A-P. Never take the last token of an arbitrary multi-token option string.
Preserve exact prompt prefix IDs and isolate KV, recurrent, convolution, and
position state. Fit decision calibration separately after quantization is frozen.

### There Is a Source-Disclosed Recurrent Cache Risk

The pinned Prism `build_rs_cache_view` comments describe possible overwrite of a
main state row during extra-row relocation in multi-sequence execution. This was
not reproduced or fixed here. Start with `n_seq_max=1`; concurrency requires a
tested fix or verified conservative path. `GGML_GDN_STATE_GATHER=1` selects the
legacy gather path in the inspected graph, but still needs parity/performance tests.

### The Native API Must Match the Build

BitNet's pinned llama header contains real `llama_*` lifecycle, batch, logits,
tokenizer, and state APIs, not the PDF's `bitnet_eval_prompt` shim. Calling
`llama_decode` on prompt tokens is appropriate for prefill even if no token is
sampled. Names are not execution semantics.

An arbitrary llama-cpp-python wheel does not automatically link to a separately
built fork. Use a verified compatible binding or compile a small native adapter
against the exact headers. A versioned JSONL executable is the first proposed
boundary when fork ABIs differ. Copy transient logits, check errors, and release
contexts/models explicitly. Do not load incompatible native forks into one process.

## 9. Selected-Label Head: Bounded Reader Implemented

The Python-hosted streamed reference now reads only the head rows required by
the context-verified option labels and computes their logits from the exact
final normalized hidden state. Sixteen BF16 rows of width 4,096 occupy 128 KiB.
Conditional softmax over those logits is algebraically equal to conditioning
the full vocabulary distribution on the same labels. The optional full-head
normalizer remains available as a bounded reference diagnostic.

This could remove most output-head storage/work in a decision-only product.
It does not remove input embeddings. A native selected-head artifact still
requires deliberate loader/graph support;
simply omitting the output tensor can activate tied-embedding fallback and change
the model. It also removes full-vocabulary perplexity, allowed-label-mass
diagnostics, and general generation from that artifact. Keep a full-head reference
and compare actual native outputs before adopting the optimization.

## 10. Calibration and Measurement Discipline

There are three distinct calibration activities: PTQ activation statistics,
A8 clipping/scales, and decision probability calibration. Use separate manifests
and disjoint final evaluation data. Split by conversation/repository/session
before chunking. Do not train or calibrate on a benchmark's held-out answers and
later report that benchmark as untouched.

The MiMo template renders real `reasoning_content` and structured `tool_calls`
itself. Use it directly. Do not invent reasoning traces, substitute Qwen2.5's
tokenizer, silently drop tool schemas, or repeat two records until the token
counter looks large. For deployment, include representative short non-thinking
decision prompts; for vision, include actual image-derived activations.

No damping value, data mixture, target perplexity gap, or temperature in the
supplied conversation is a proven universal optimum. Local MSE is a screening
metric. Evaluate whole-model language and decision behavior on held-out data.

Report cold load, warm prefill, suffix processing, complete-state copying, vision
encoding, score readout, p50/p95 latency, peak RSS, and joules/decision separately.
Zero generated answer tokens does not mean zero prompt compute or data movement.
Never infer prefill latency from a decode memory-bandwidth bound. Keep original
7B/3B T-MAC and BitNet results attributed to their models and hardware.

Physical safety constraints belong outside the model. Initial sensor integration
is advisory/log-only, with freshness checks, watchdogs, bounded queues, abstention,
and deterministic limits. Typed output does not prove safe actuator behavior.

## 11. Delivered Inventory and Remaining Native Gate

Run `PYTHONDONTWRITEBYTECODE=1 python -m embedded_jev.inventory` to reproduce
the full sorted JSON inventory and estimates. A live run reconciles the pinned
index to all four safetensors headers: 760 tensors, 9,409,813,744 BF16 parameters,
18,819,627,488 tensor bytes, and 93,360 header-overhead bytes. Retrieved metadata
and headers (not payloads) total 172,461 response-body bytes; metadata and header
SHA-256 hashes, revision, Python/tool versions, and per-tensor policy reasons
appear in the report. Eight offline tests cover incomplete/mismatched inputs,
unsupported metadata, and refusal of full-body shard responses.

The policy initially allows only geometrically compatible FFN and full-attention
projections: 5,301,600,256 candidate parameters. It retains embeddings/head,
vision, sensitive/recurrent weights, and fused/linear-attention projections.
Estimated on-disk weights (not BitNet I2_S or resident allocations) are
9,376,152,032 bytes for PTQ1_0 group-128 with BF16 vocabulary, or
7,469,054,432 bytes if both vocabulary matrices are **hypothetically** Q8_0.
A selected-16 head estimate requires a custom loader; it is not implemented.
The report keeps FP16 KV and FP32 recurrent-state examples separate from weight
payload. It cannot detect identical payloads stored in different shards.

The inventory command covers these metadata-only requirements:

1. Read the model config, tokenizer/processor metadata, weight index, and bounded
   safetensors headers for the pinned revision. Use an established HF/safetensors
   metadata facility or HTTP range requests with explicit byte limits. If the
   server ignores a range, stop rather than downloading an entire shard.
2. Produce exact tensor names, shapes, storage dtypes, parameter counts, byte
   counts, shard locations, and categories: language projections, sensitive
   parameters, input embedding, output head, vision, and optional MTP.
3. Detect tied/shared data and duplicate accounting. Reconcile sums against the
   index/metadata and explain any mismatch instead of hiding it.
4. Produce an explicit initial quantization allowlist and exclusion reasons.
   Check actual dimensions for candidate group/rotation sizes.
5. Estimate separate byte totals for retained BF16/FP32, Q8 embeddings/head,
   group-128 ternary blocks, vision, and an optional selected-label head.
   Keep runtime cache/scratch estimates separate from stored weights.
6. Record source revision, retrieved-file hashes, tool versions, assumptions,
   and transfer bytes. Do not deserialize arbitrary pickle or enable remote
   model code by default.
7. Add offline fixtures for valid, missing, inconsistent, and unsupported
   metadata. Unit tests should need neither network nor GPU.

Acceptance: deterministic reconciled inventory, explicit unsupported cases, no
bulk weight download, and a credible memory budget. The current HF API total is
not a substitute for this implementation.

The processor checks, full pinned BF16 download, streamed dense text reference,
one-projection native substitution, and bounded rotation/codec controls are now
complete at the scopes described above. Native MiMo tokenizer/loader parity and
representative owned decision data remain separate open gates. Continue from
Section 16 rather than repeating bootstrap or downloading the model again.
Measure memory and bound temporary copies before full-model conversion; do not
create an empty module tree or copy the numbered scripts from the original PDF.

## 12. Questions Before Large Jobs

Ask the user for these once they become necessary:

- Which machine performs quantization? CPU/architecture, RAM, GPU model/VRAM,
  host driver, storage location/free space, and whether the GPU is accessible
  from containers.
- Which device is the first deployment target? N100, a specific ARM board, or a
  specific RISC-V SoC; RAM, OS, cooling, and power budget matter.
- Is the first deployment text/sensor-only or does it need raw camera frames?
  What input token/image resolution and criteria-per-state budgets are realistic?
- What decision accuracy, false-positive/negative cost, abstention rate, p95
  latency, and memory ceilings are acceptable?
- Are selective higher precision, QAT recovery, and a decision-only output head
  acceptable, or must arbitrary text generation remain available?
- What downloads and long-running compute are authorized? Avoid assuming the
  current lightweight container is permission to fetch tens of gigabytes.

No hardware purchase was recommended or priced during this phase. Old pricing
tables and expected tokens/s from the PDF are not verified evidence. For Finland,
obtain dated delivered quotes with taxes, shipping, and required accessories
only after a workload/target is chosen.

## 13. Environment Pitfalls Already Encountered

- Two WSL/Podman GPU `devcontainer up` attempts failed at `apt-get update`, first
  with a `noble-backports/multiverse` Hash Sum mismatch and then with one in
  `noble-security/restricted`. The backports-only workaround was disproven and
  replaced by HTTPS for the existing Ubuntu sources, preserving all pockets
  and APT integrity checks. Both failing by-hash URLs returned the expected
  SHA-256 over HTTPS from the CPU container. A later WSL/Podman `devcontainer up`
  built successfully and passed post-create `nvidia-smi` with driver 595.95.
  Do not bypass TLS or checksum verification if a future mirror issue recurs.
- On 2026-10-09, `curl` from this container to `huggingface.co` failed with
  "unable to get local issuer certificate": the network presented an enterprise
  TLS-inspection chain whose root certificate is not in the container trust
  store. A similar issuer failure appeared after an earlier reboot. Pages read by
  the external web tool are text, not byte-exact downloads. Do not disable
  verification or change trust settings as a workaround. At the user's request,
  the devcontainer now trusts the host's exported roots
  ([host certificates](development.md#host-certificates)). After the rebuild on
  the same day, `curl` and Python reached the Hub with verified HTTPS.
- Dev Containers CLI 0.87.0 accepts `info`, `debug`, and `trace` log levels,
  not `error`. The initial unsupported flag was corrected.
- Direct rootless Podman bind mounts used for one-off tooling needed
  `--userns=keep-id` and the host UID/GID to preserve workspace permissions.
  Do not compensate by recursively changing the repository owner.
- The scratch uv image lacked Python/libc discovery tools and a useful writable
  environment for this resolution step. The successful bootstrap resolver image
  was `ghcr.io/astral-sh/uv:0.8.22-python3.12-bookworm-slim`. Future lock updates
  can simply use the uv already installed in the devcontainer.
- Podman printed harmless unconsumed BuildKit-argument/OCI-shell warnings while
  the image build and startup still succeeded. Judge the exit status and smoke
  results, not those warnings alone.
- The SemIf repository's branch is `master`; fetching its README from `main`
  returned 404. The repository itself is available.
- Verify research paper titles, not remembered arXiv numbers. The checked
  QuaRot identifier is `2404.00456` and QuIP# is `2402.04396`.
- The user prefers avoiding `rg` commands unless explicitly requested.

The Python research lock is hash-checked, but Ubuntu apt packages and the tagged
base image can change. This is not a bit-reproducible OS build. Record image
digests and full build/package provenance before publishing performance results.

## 14. Git and Delivery

The bootstrap began on `main` at `3e846bb` (`Initial commit`). The user requested
periodic local commits, not a push or a new branch. Implementation through
`6d74ae8` is committed, and the user has independently pushed checkpoints.
Use `git log` to identify any later commits; this document cannot embed its own
containing commit hash. Before committing or handing work back, verify:

```bash
git status --short
git log -2 --oneline
git diff --check
```

Do not revert unrelated user work or treat an earlier clean-worktree observation
as a guarantee about a later session. The ignored conversation PDF stays local.
Container images, running containers, and cache volumes are not stored by Git
and must be recreated or reused through the documented environment workflow.

## 15. Pinned Text-Only Label Boundary

The [label probe](../embedded_jev/label_probe.py) checks the actual MiMo template
with `enable_thinking=False`, matches rendered and tokenized prompt IDs, and
requires each A-P label to add one distinct non-special token. The sample has
46 prompt tokens and label IDs A-P = 32-47. The pinned inventory reconciles
standalone and nested image/video processor metadata. With approved isolated
CPU-only Torch 2.10.0, Torchvision 0.25.0, and Pillow 12.1.1, real
`Qwen3VLProcessor` text inputs matched the tokenizer IDs and mask, with all-zero
multimodal token types. That initial metadata-only cache was about 979 MB;
the complete model snapshot was downloaded later on 2026-09-29.
[docs/development.md](development.md) has the commands.

The [synthetic fixture](../tests/fixtures/agent_tool_smoke.json) has five
agent/tool cases: inspect-first with an irrelevant-context perturbation,
authorized README edits with reordered options, and missing upload permission.
Their expected semantic IDs map to A/A/B/C/C; the pinned text-only processor
passed all five prompt/label boundary checks. Six offline label/processor tests
pass. The CLI records hashes and versions, not model predictions. This is **not**
calibration data, a held-out quality benchmark, image processing, native backend
tokenization parity, or a trained decision service.

## 16. Restart Brief

Implementation is paused for today after the user's commit-and-stop request.
P0.1-P0.4 and bounded P3 are complete through `db002c1`; bounded P4 policy
preparation is committed as `6abb44c`. Read the
[current checkpoint](#profile-model-policy-checkpoint-2026-10-09) before resuming.
The next instructed coding chat should complete bounded synthetic actual-loader/
typed-prefill integration through v2 beside strict v1 factories, not run another
real model. Read the [work queue](roadmap.md#generalization-work-queue) and updated
[coupling inventory](development.md#mimo-coupling-inventory), not an older
paragraph's suggested next step.

The first non-MiMo subject is DavidAU's Defiant Fable safetensors source at
`7af0a9c4`: approved [preflight](development.md#preflight-results), offline
inventory and reference tokenization pass; dense/native inference and quality
do not. Use its own [support gate](roadmap.md#model-scope-and-support-gate).
Existing MiMo controls/artifacts must not be transplanted. P2, any real-model P4
check, another MiMo precision conversion, new download/fit/candidate or held-out
experiment needs separate approval. This documentation request supplies none.

For a fresh coding session:

1. Read [AGENTS.md](../AGENTS.md), the
  [latest checkpoint](#projection-geometry-checkpoint-2026-10-09), the profile/
  support-tier design, updated coupling inventory and work queue. Check actual
  Git state, interpreter, cache files/source pins and RAM/disk before acting.
2. Preserve genuine BitNet-derived CPU execution, independent FP16 scales per
  output row/input group, and typed conditional option scores with zero
  generated answer tokens. PQ2 storage/dispatch alone does not meet BitNet.
3. Continue P4 from `prism_bitnet_cpu_model_override_from_gguf_v2` in
  [native/prism_group_scale.cpp](../native/prism_group_scale.cpp), its contract in
  [native/prism_bitnet_runtime.h](../native/prism_bitnet_runtime.h), and
  `test_pinned_prism_complete_model_policy_binds_exact_projection` in the
  [existing controls](../tests/test_prism_native_control.py). Reuse P0 verified
  profiles and P3's [manifest adapter](../embedded_jev/prism_codec.py), but do
  not confuse its owned v2 handle with a discoverable loader/model policy.
  Keep the separately versioned model policy and strict v1 comparator. Preserve exact
  target-byte checks, ordinary-PQ2/dummy-probe refusal, canonical readback,
  one repack, no-score failure/recovery, context reset/isolation and library/
  file lifetime requirements. Synthetic policy/hosting checks must stay bounded.
  Source digests supplied by callers are not native source authentication.
  The prior dense/mixed/precision models were deleted; do not recreate them
  without approval. Runtime numerical acceptance remains open. Scoped registry
  calls are not general lifecycle/concurrency safety or deployment evidence.
4. Keep fitting on calibration only, diagnostics on validation, and held-out
  inference untouched. Do not promote the negative independent-block
  compensation approximation or save another candidate. No full-model
  conversion, benchmark, speed, or calibrated confidence claim follows.
5. Validate each edit locally, update these documents, and make periodic local
  commits. Only after the user resumes, continue autonomously within the
  authorized bounded scope; ask only for a genuine blocker or permission to
  expand scope. Use the terminal for environments, no subagents, no push, and
  no branch change.

The last source gate at `6abb44c` was 244 passed/40 optional skips; 12 selected
compiled P3/P4 controls and GCC ASan/UBSan/leak checks also passed. CTest integration
was unavailable. The [development guide](development.md#current-checkpoint-and-gates)
contains the default test/Ruff/whitespace gate and exact bounded native command.
Use focused checks after changes and the required gate before source commits;
do not rerun successful expensive gates merely to reconstruct history. Keep
editor settings and pinned caches intact when tool integration is unavailable.