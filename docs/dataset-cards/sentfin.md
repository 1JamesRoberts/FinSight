# SEntFiN 1.0 — FinSight dataset card

Maintainer: FinSight project. Reviewed 2026-10-10. Scope: local noncommercial
entity-level sentiment research and qualified external evaluation.

## Identity, provenance, and rights

Authors: Ankur Sinha, Satishwar Kedas, Rishu Kumar, and Pekka Malo.
Source: [pinned release](https://github.com/pyRis/SEntFiN/tree/eba43610fe3cb57a3e5773cd6c037da1f8992ad5).
Revision: `eba43610fe3cb57a3e5773cd6c037da1f8992ad5`.
Main CSV SHA-256: `570af7ba58f724f3fcc414dd5ba80da5b1a3b342134d76c6ebdf53a32c54a92f`.
Auxiliary entity CSV, README, and license identities are pinned in
`configs/asset-revisions.json` and retained locally.

The [authors' paper](https://arxiv.org/html/2305.12257) describes Economic Times
headlines from 2011–2015 with human entity annotations, consensus, and curator
resolution. The repository-wide MIT grant supports local research under the
[recorded release interpretation](../sentfin-asset-review.md), with attribution
and notices retained. Underlying publisher rights/public headline display are
not cleared. No source text, annotated example text, or dataset CSV is committed.

## Schema and supervision

CSV fields: `S No.`, `Title`, `Decisions`, `Words`. Each headline has a dictionary
of financial-entity targets and negative/neutral/positive labels. Preserve native
title, raw decision string, targets, raw/named labels, source word count, source
ID, and revision. All targets remain grouped under their headline. Opposite
labels for different targets are valid; they do not define a sentence gold label.

The source contains 10,753 headlines and 14,405 annotations: 3,814 negative,
5,517 neutral, and 5,074 positive. There are 2,847 multi-target headlines, including
1,233 with mixed target labels. The observed neutral count is one higher than
the paper's table. The entity CSV has 1,010 rows versus the paper's described
1,009. Retain those artifact/publication discrepancies. Targets include financial
entities beyond companies; no verified entity-type mapping is supplied by FinSight.

## Preparation, curation, and use

The [split policy](../split-policy.md) quarantines unresolved shared-target
conflicts, preserves different target coverage in development, collapses equivalent
copies with lineage, and groups exact/near duplicates across corpora.
There are 67 exact groups with seven same-target gold conflicts; 13 malformed
dictionary strings require the narrow, non-executing apostrophe parser.
Our word-count rule differs from the supplied field on 88 source headlines.

| View | Headlines |
| --- | ---: |
| Train | 8,598 |
| Validation | 1,053 |
| Frozen final benchmark | 1,029 |
| Withheld malformed-target candidates | 2 |
| Quarantined source rows | 16 |
| Redundant equivalent source rows | 55 |

Development comprises 9,651 headlines / 12,957 target annotations. It includes
2,571 multi-target headlines and 1,116 with mixed sentiments. Median whitespace
length is 10 words and P95 is 14. Source counts and processed counts differ
because of grouping, quarantine, and equivalent-copy removal.

The immutable `sentfin-independent-v1` release contains 1,029 headlines and
1,360 native target labels in 1,028 duplicate groups. Two malformed-target groups
are excluded without repairing test annotations. The
[freeze receipt](../../configs/benchmark-freeze.json) pins membership and labels;
the [evaluation contract](../evaluation-contract.md) defines shared model
evaluation, single-target controls, broadcast baselines, and multi-target cohorts.
Final-test labels must never guide training, thresholds, calibration, or selection.

## Limitations

Independence means held out from FinSight development, with no known overlap
against accessible PhraseBank/FiQA texts under the recorded lexical screen.
Complete FinBERT pretraining exposure cannot be certified. SEntFiN-fine-tuned
checkpoints require their own training-provenance audit. This is source-label
curation, not new human annotation or full re-adjudication of gold labels.

Indian-market headlines are not a representative US-company/full-article test.
The frozen release has two exact Apple name matches and none for the other four
MVP companies under the predeclared name list; those counts do not support ticker
performance claims. Entity resolution remains future work. Source-year ranges
do not supply reliable per-record prediction cutoffs. This card does not authorize
stock modeling, news scraping, or publication of trained weights.

Reproduction and development-only figures: [Phase 1 findings](../phase1-findings.md).
Detailed source review examples stay in ignored local data.
