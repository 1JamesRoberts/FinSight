# Financial PhraseBank — FinSight dataset card

Maintainer: FinSight project. Reviewed 2026-10-10. Scope: pinned local
noncommercial sentiment research. This card describes our selected artifact,
not a new dataset or a claim of independent FinBERT supervision.

## Identity, provenance, and rights

Financial-news sentence corpus attributed to Malo et al. Selected source:
[pinned takala release](https://huggingface.co/datasets/takala/financial_phrasebank/tree/8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964).
Revision: `8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964`.
Archive SHA-256: `0e1a06c4900fdae46091d031068601e3773ba067c7cecb5b0da1dcba5ce989a6`.
Exact files and notices are recorded in `configs/asset-revisions.json`.

The source declares CC BY-NC-SA 3.0. Local noncommercial acquisition/research
is supported by the [recorded terms review](../acquisition/phrasebank-asset-review.md).
Retain attribution, license, and modification notices. Public source-text demos,
commercial use, and resulting weight redistribution need separate review.
No dataset text is committed; this repository contains aggregate findings only.

## Schema and supervision

ISO-8859-1 sentence files in a ZIP, with the final `@` separating text and label.
One record is a sentence with negative/neutral/positive investor-impact sentiment.
It does not provide a separate gold label per company target. Typed records
preserve sentence, raw label, named label, agreement variant, member filename,
line-based source ID, and revision. No model integer class IDs are inferred.

| Agreement subset | Raw sentences |
| --- | ---: |
| >=50% | 4,846 |
| >=66% | 4,217 |
| >=75% | 3,453 |
| Unanimous (`all`) | 2,264 |

These subsets overlap. `all` means all annotators agreed, not all sentences.
Individual votes/rater identities are not available in our source format, so
we cannot reconstruct per-example vote fractions or calculate individual-rater
agreement statistics. Subset membership supplies only agreement-level metadata.

## FinSight preparation and partitions

The >=50% master collection supplies source rows; subsets are never concatenated.
The [split policy](../benchmark/split-policy.md) applies text normalization, context-aware
conflict quarantine, equivalent-copy deduplication, and grouped near-duplicate
assignment. Raw files and original labels remain unchanged.

| View | Sentences |
| --- | ---: |
| Train | 3,785 |
| Validation | 518 |
| Withheld test candidates | 527 |
| Quarantined source rows | 10 |
| Redundant equivalent source rows | 6 |

The 4,303 development sentences have 537 negative, 2,554 neutral, and 1,212
positive labels. Their word-length median is 21 and P95 is 42; counts use
whitespace splitting, not model tokens. Development text/label matches in
agreement subsets are 4,303 / 3,754 / 3,087 / 2,023 respectively. These are nested
membership counts after filtering, not independent samples or raw source counts.

## Intended use and limitations

Use designated training/validation records for sentence-level baselines and
development experiments. The withheld candidates are not yet a separately
curated final PhraseBank test release. Pretrained ProsusAI/finbert used PhraseBank;
its scores here cannot establish independent generalization. Use the frozen
SEntFiN benchmark according to the [evaluation contract](../benchmark/evaluation-contract.md).

Eight exact text groups include two source-label conflicts. Near pairs can contain
different events or amounts; grouping does not establish semantic equivalence.
Quarantine preserves unresolved cases rather than adjudicating them automatically.
Neutral-heavy labels can make accuracy misleading. Sentences do not establish
full-article quality, target-level sentiment, five-company coverage, or market
prediction usefulness. No reliable prediction-time timestamps are standardized.

Reproduce with the commands in [Phase 1 findings](../eda/phase1-findings.md). Source
statistics appear in `reports/label-analysis.md`; development-only EDA and exact
provenance digests are in `reports/phase1-eda.md` and `reports/eda-summary.json`.
