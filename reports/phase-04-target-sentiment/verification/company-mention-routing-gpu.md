# Real CUDA company-mention pipeline verification

The updated Phase 4 routing was installed and verified on the user's NVIDIA
GeForce RTX 4070 SUPER. Both BERT organization extraction and the selected
RoBERTa target sentiment checkpoint executed on CUDA. This supersedes the earlier
stub-only verification limit for this release; historical receipts remain intact.
This is an invented-input integration smoke check, not a news accuracy benchmark.

## Installation and provenance

The primary checkout was clean but lacked the new organization CLI. To preserve
it, 140 explicitly selected source/evidence files were installed under the
project's ignored `data/interim/organization-release` directory. Two project-root
launchers select that code and shared local model files. No weights were
redownloaded and no existing source, model, reference, dataset or Git branch was
overwritten. The transferred 766,567-byte ZIP was deleted after verified extraction.

Source manifest:
`3be40da98596b9b065358bbc01bb529d04db1720e1d8b7080163362698517924`.
Archive SHA-256:
`c1d45a3f34a49e7096c49df779dd0ec24365d8e313ad55ec1c14c49252bde454`.
Every allowlisted file matched its manifest hash. The manifest's hash names the
release; the exact smoke script is included in that inventory. Credentials and
contact configuration were not copied. The small SEC snapshot was separately
installed to the shared data location with byte-equality checks.

Runtime: PyTorch `2.11.0+cu128`, CUDA runtime `12.8`, NVIDIA GeForce RTX 4070 SUPER.
NER revision `d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc`, RoBERTa artifact
`2f742f69854f097b5b1351e4a1de126b54c28fe2c900edb3c2e41526cd32e855`,
deployment SHA `e5092415a03e442852e43d503c5cab52ac3d9387d3287ef8221b499c8b6e34da`,
and SEC manifest `d76a4e6fd5d7cc540b3ef6974ece2ca508b3fd16360e8d9eca7b94ddce9f528b`.
The model loaders verified local artifact bytes. No target masking was applied.

## Verified cases

| Case | Company type / identity | Expected routing | Result |
|---|---|---|---|
| Toyota, explicitly confirmed | Company / unresolved | Real sentiment, null issuer fields, ineligible for issuer grouping | Passed |
| Microsoft, registry match | Company / linked | Real sentiment, preserved registry identity | Passed |
| Tesla, SEC lookup with context | Company / linked | Real sentiment, CIK and listing associations retained | Passed |
| NASA, unconfirmed ORG | Uncertain / unresolved | No company sentiment and no sentiment loading | Passed |
| No detected organization | No company candidate | Empty company sentiment and no sentiment loading | Passed |

**Attributes:** Case is an invented input scenario, not a labeled news example.
Company type describes the policy's company/uncertain decision; identity records
whether an issuer link exists. Expected routing states whether the actual
sentiment model should execute and whether later issuer grouping is allowed.
Result means the corresponding assertions passed, not that predicted sentiment
is ground truth or real-world identity is independently verified. Toyota and
Microsoft share one headline; rows are overlapping scenarios, not independent
accuracy observations. Eligibility is metadata only; aggregation remains a
separate roadmap phase.

The organization CLI's Toyota/Microsoft results matched the public direct-target
CLI for the same complete headline, target order, labels, model version and all
three probabilities. Tesla also matched direct Python prediction. Maximum
absolute probability difference was **0.0**, within the predeclared `1e-6`
tolerance. This establishes routing equivalence for these inputs, not sentiment
accuracy or calibration.

For uncertain/empty cases, actual CUDA NER ran through the CLI entry point while
sentiment loading was replaced with a function that raises if called. It was
never called. No neutral fallback was emitted. The initial integration run took
56.67 seconds including subprocess startup, repeated file verification/model
loading, and direct comparisons; this is not per-headline latency or throughput.

## Launcher handoff

Both launcher help commands were verified. A further actual CUDA prediction
through `finsight-organizations.cmd` verified Windows argument quoting, the pinned
source path, existing environment/model locations, and confirmed-unlinked Toyota
output. The two launchers are `finsight-organizations.cmd` and
`finsight-predict.cmd` in the primary project root. They work offline and reuse
local artifacts. [PowerShell commands](../../../docs/phase-04-target-sentiment/prediction-gpu-release.md).

[Complete GPU receipt](company-mention-routing-gpu.json) includes invented-input
predictions, skip results, launchers' hashes and model/source provenance. No news
or frozen sentiment records were read. No training, source acquisition, protocol
tuning, model selection, alias additions or historical listing inference occurred.

Final post-installation audit reverified all 140 allowlisted files; tracked GPU
checkout/index differences remain empty, and only the two new launchers are
untracked at project root. Protected local benchmark/deployment/registry result
checksums remain unchanged. All 795 local tests pass. Full Ruff checks still
report only the existing scratch notebook issues (nine lint errors and
formatting); excluding that notebook passes both checks. `git diff --check` passes.
[Final verification receipt](company-mention-routing-gpu-verification.json).
