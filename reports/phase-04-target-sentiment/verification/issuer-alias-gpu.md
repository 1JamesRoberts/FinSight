# Reviewed-alias GPU release verification

The reviewed Dell/Boeing/Toyota rules were installed as a new pinned inference
release and verified using real CUDA BERT NER and the saved RoBERTa target
sentiment model. This is an invented-input integration check, not another news
accuracy evaluation. No model training, target masking, checkpoint selection or
Phase 5 implementation occurred.

## Source, runtime and artifact provenance

Source manifest:
`f36459ecf9743b434df509b84963aa676bc135d4d025dcfc173758373ecc264a`.
The release contains 142 allowlisted source/config/evidence/script files.
The 591,404-byte transferred archive has SHA-256
`ff4a336ab5ea5cbc5f136669528ccb532fe5d83f264ceb5ba90e31c774d25302`.
All extracted bytes matched the manifest; the archive was then deleted.
No credentials/contact configuration, `.env`, model weights, news records or
reviewed references were transferred. Existing local SEC and model files were
reused, with artifact integrity checks during loading.

Runtime: NVIDIA GeForce RTX 4070 SUPER, PyTorch `2.11.0+cu128`, CUDA `12.8`.
NER revision: `d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc`.
RoBERTa artifact:
`2f742f69854f097b5b1351e4a1de126b54c28fe2c900edb3c2e41526cd32e855`.
Deployment SHA:
`e5092415a03e442852e43d503c5cab52ac3d9387d3287ef8221b499c8b6e34da`.
Alias catalog SHA:
`6567153dfa3bcb007c78585c91d429cc377553811d2679853f37448717c3e5c5`.
SEC snapshot:
`d76a4e6fd5d7cc540b3ef6974ece2ca508b3fd16360e8d9eca7b94ddce9f528b`.
Resolver version is `organization-resolution-v4`. The model input remains the
original target/headline pair and probability outputs remain uncalibrated.

## Real CUDA checks

The primary invented input is “Toyota reported higher profits. Boeing reported
a loss. Dell announced higher sales.” No `--confirmed-company` is supplied.

| Case | Company classification | Issuer identity | Sentiment routing | Check |
|---|---|---|---|---|
| Toyota with company context | Company | CIK 0001094517 | Real sentiment; grouping eligible | Passed |
| Boeing with company context | Company | CIK 0000012927 | Real sentiment; grouping eligible | Passed |
| Dell, default no roll-up | Company | Unresolved, null issuer fields | Real sentiment; grouping ineligible | Passed |
| Dell, explicit parent roll-up | Company | Parent CIK 0001571996 | Real sentiment; grouping eligible | Passed |
| Dell, missing company context | Uncertain | Unresolved | Empty sentiment; loader not called | Passed |
| NASA, unknown company type | Uncertain | Unresolved | Empty sentiment; loader not called | Passed |
| No detected organization | No candidate | No identity | Empty sentiment; loader not called | Passed |

**Attributes:** Case specifies an invented input/policy combination. Company
classification records the rules' company/uncertain decision, not independent
real-world type verification. Issuer identity is the accepted current CIK or
abstention under that policy; the Dell roll-up CIK identifies the listed parent.
Sentiment routing records whether real inference should occur and whether the
identity is eligible for later grouping. Check means assertions passed, not that
sentiment labels are human gold or that stock aggregation exists. Rows overlap
in input text and are not independent accuracy observations. Context and
roll-up remain explicit caller assertions.

All three original target strings were passed unchanged to the model. Strict
and parent-roll-up outputs matched public direct `--target` CLI predictions for
the same headline, target batch/order, labels, model version and probabilities.
Maximum absolute probability difference was **0.0** under a `1e-6` tolerance.
Identity policy changes metadata and eligibility, not text or sentiment scores.
Actual model parameters were confirmed on CUDA. For skipped cases, real CUDA
NER ran while sentiment loading was replaced with a function that raises if
called; it was never called. No neutral fallback was produced.

The initial suite took 58.20 seconds, including subprocess startup, repeated
model/file verification, loading and direct comparison. This is not production
per-headline latency or throughput. Only invented inputs were used; no frozen
sentiment or news/reference records were opened.

## Launcher installation and handoff

After module-level CUDA checks passed, the two existing project-root launchers
were switched to this release. Their old byte hashes were checked against the
previous receipt and exact copies preserved under `previous-launchers` in the
new release. A failed launcher verification would restore both prior files.
The prior source release remains intact.

The organization launcher now supplies pinned SEC and alias paths, while
`--company-context` and `--allow-parent-rollup` remain explicit. Both launcher
help commands were checked. Real strict and explicit-parent CUDA runs through
the organization launcher were compared with the already verified CLI outputs.
The direct-target launcher selects the same release without company alias flags.
[PowerShell commands and rollback](../../../docs/phase-04-target-sentiment/issuer-alias-gpu.md).

[Complete GPU receipt](issuer-alias-gpu.json) includes outputs, policy checks,
launchers' hashes, rollback metadata and source/model provenance. The ordinary
tracked GPU checkout and index were preserved. No Git commit/push, historical
alias-source reinterpretation or further rule changes were part of deployment.
The separate independent entity test collection and other roadmap tasks remain
open; this closes only the reviewed-alias inference-release verification.

## Final verification

All 811 local tests pass. A separate audit confirms the installed source/catalog
identity, strict/roll-up CIK behavior and identical sentiment across module and
launcher runs. The protected registry, deployment selection, benchmark freeze,
locked final target result and approved fresh reference retain their hashes.
Both tracked GPU checkout/index differences remain empty; the two existing
untracked launchers are the only project-root changes. Full Ruff checks still
flag only the existing scratch notebook (nine lint errors and formatting); lint
and format checks excluding it pass. `git diff --check` passes.
[Final verification receipt](issuer-alias-gpu-verification.json).
