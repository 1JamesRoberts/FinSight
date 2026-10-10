# Frozen external sentiment evaluation contract

Release: `sentfin-independent-v1`, frozen 2026-10-10 before model training or
inference. Scope: Phase 1 benchmark curation only. The authoritative membership,
native labels, duplicate groups, source revisions, and model reference are pinned
by [the freeze receipt](../configs/benchmark-freeze.json). The
[aggregate report](../reports/frozen-benchmark.md) records the measured cohorts.

## Selection and curation

Use the already assigned seed-42 SEntFiN `test_candidate` groups from the pinned
provisional split. Do not resample to balance classes, improve coverage, or favor
a model. The existing policy excludes conflicts, target-coverage ambiguities,
cross-corpus overlap, and holdout overlap. Equivalent copies retain lineage;
near duplicates stay in the same group. Higher-agreement PhraseBank subsets
are nested in the master corpus and do not supply independent test examples.

The curator checked schema/gold validity, duplicate-group boundaries, target
multiplicity, literal target-name presence, and descriptive company coverage.
Two source annotations contain malformed target names, rather than merely
canonical names absent from the text: SEntFiN source IDs `3652` and `6788`.
Their complete groups are excluded by `configs/benchmark-policy.json`; no gold
label or target is repaired. They remain withheld from development. All 1,360
remaining target names occur literally after the documented normalization.
Literal occurrence is a diagnostic, not proof of correct entity identity.

The final release retains 1,029 headlines in 1,028 duplicate groups and all their
1,360 native target annotations: 333 negative, 548 neutral, and 479 positive.
The 766 single-target headlines form the sentence-model control cohort. The
263 multi-target headlines include 110 with mixed target sentiments. Cohorts
overlap as declared: mixed-target is a subset of multi-target, and all includes
every example. Never add their counts as independent samples.

This is source-label curation, not fresh human annotation. Apart from the
target-name exceptions and the earlier ambiguity review, the agent has not
adjudicated every gold label. No model predictions were used for selection.
Target types remain source financial entities, including entities beyond
companies. There is no verified company-only or per-ticker benchmark yet.

## What independence means

This release is held out from FinSight training, validation, threshold selection,
calibration, and model selection. The accessible PhraseBank/FiQA/SEntFiN texts
pass the recorded exact and conservative near-duplicate checks. It is an external
source relative to PhraseBank supervision, with no known accessible-corpus overlap.

The [pinned FinBERT card](https://huggingface.co/ProsusAI/finbert/blob/4556d13015211d73dccd3fdd39d39232506f3e43/README.md)
confirms PhraseBank fine-tuning. Complete sample-level training and pretraining
manifests for that checkpoint are unavailable. Consequently, absence of SEntFiN
text or syndication exposure during pretraining cannot be certified. Use the
phrase **“held-out SEntFiN evaluation; no known accessible-corpus overlap;
pretraining exposure unknown”**, not “proven unseen by FinBERT.” Any other
checkpoint, including a SEntFiN-fine-tuned model, requires its own provenance audit.
This freeze does not resolve the existing model-weight license question or
authorize downloading weights.

The [SEntFiN paper](https://arxiv.org/html/2305.12257) describes human entity
annotations for Indian-market news from Economic Times, with labels established
by consensus/curation. Headlines and financial entities differ from full articles
and the five-company US-market MVP. Exact normalized target-name coverage in the
freeze is two Apple matches and zero for Microsoft, Nvidia, Amazon, and
Google/Alphabet under the predeclared name list. This is a descriptive lexical
check; it does not establish ticker identities or unseen aliases. These counts
cannot support company-specific performance claims. A separately licensed,
independently annotated US-company collection is later work.

Local noncommercial research remains within the recorded
[SEntFiN release assessment](sentfin-asset-review.md). Frozen text and labels
remain ignored; the repository publishes aggregate statistics, metadata, and
digests only. No public headline redistribution is authorized by this freeze.

## Evaluation protocol for later phases

- Prediction unit: one native `(headline, target)` annotation. Preserve all
  targets in a headline and their group assignment. Never invent a single
  headline gold label for mixed-target examples.
- Primary target metric: macro F1 across the fixed class order negative,
  neutral, positive on all target annotations. Give each annotation one vote;
  this is pooled target F1, not an average of F1 over companies. Report class
  support, accuracy, per-class precision/recall/F1, and a confusion matrix.
  Set undefined precision/recall/F1 to zero and disclose the rule.
- Sentence-model comparison: report the same metrics separately on the frozen
  single-target cohort. VADER, TF-IDF, and article-wide FinBERT have one prediction
  per headline. On multi-target cohorts, copying that prediction to every target
  is an explicitly named **broadcast baseline**. It is not target-aware sentiment.
- Target-aware comparison: report all, single-target, multi-target, and mixed-
  target cohorts separately on the same frozen membership. Empty/missing classes
  in a cohort retain the same class order and report support. Per-target or
  per-company breakdowns with tiny support are descriptive, not reliable claims.
- Uncertainty: when implemented, use a paired bootstrap over duplicate groups,
  retaining every headline/target in each sampled group, with 1,000 replicates,
  seed 42, and percentile 95% intervals. Do not treat target annotations from a
  shared headline as independent observations. Report latency with batch size,
  hardware, warm-up, and input lengths separately from accuracy.
- Fix thresholds, calibration, preprocessing, token limits, prompts, and model
  settings using training/validation only. Model class IDs must map explicitly
  to the named labels. Failed predictions must be counted/reported, never silently
  dropped. No label-based exclusion is permitted after seeing predictions.
- Freeze every compared model configuration before running final evaluation.
  Store code revision, experiment settings, model revision, receipt digests,
  predictions, and metrics. Final-test error analysis follows locked comparison;
  changes inspired by it are exploratory and require a new untouched benchmark
  for subsequent confirmatory claims. Do not iterate model choice on final scores.

This document declares the later evaluation protocol; no metric computation,
model inference, target extraction, or training is implemented in this task.

## Reproduce and use safely

```sh
uv run --locked python -m finsight.data acquire
uv run --locked python -m finsight.data freeze-benchmark
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
```

On a fresh checkout, the committed receipt verifies reconstruction of the same
freeze. `freeze-benchmark` supports `--manifest`, `--data-dir`, `--split-policy`,
`--benchmark-policy`, `--freeze-receipt`, and `--report-path`. Its receipt defaults
to `configs/benchmark-freeze.json`. Alternate receipt paths are for synthetic
tests or separately versioned research; official experiments must use this
committed receipt, not bypass it to regenerate favorable splits.

Local outputs are under `data/processed/benchmarks/<benchmark-manifest-sha256>/`:
`benchmark-manifest.json` and `test-records.jsonl`. Existing files and receipts
must match exactly; changes fail without overwriting the release. The receipt
pins policy and record bytes and binds the freeze to its original parent split.
Running `prepare-splits` with changed inputs or seed after freezing fails.

For later development, use
`finsight.data.benchmark.load_development_records(data_dir, receipt_path)`.
It verifies frozen and parent checksums and returns **only train/validation**
records. All other provisional test candidates remain withheld too. The original
split artifacts stay available for lineage and reproduction, but loading their
entire JSONL directly is not an authorized training path. Checksum guards do not
physically prevent a developer from opening local test files; the evaluation
contract is also a research discipline requirement.

Any future correction to a frozen label or membership needs a documented,
explicitly versioned release. Preserve this receipt and report results against
their original release; never silently replace it.
