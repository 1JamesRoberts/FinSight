# FinSight provisional split preparation

Policy: approximate 80/10/10 group partitions, seed 42; SHA-256 bucket
assignment without stratification or seed searching. Development-only
groups renormalize the train/validation weights. Counts can
differ from the requested ratios because connected groups stay intact.

| Corpus | Raw sources | Retained | Redundant | Quarantined | Unscored | Train | Validation | Test candidate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fiqa | 1305 | 1112 | 0 | 1 | 192 | 907 | 102 | 103 |
| phrasebank | 4846 | 4830 | 6 | 10 | 0 | 3785 | 518 | 527 |
| sentfin | 10753 | 10682 | 55 | 16 | 0 | 8598 | 1053 | 1031 |

Exact equivalent copies retain one processed representative and all source
references. Near pairs form transitive groups but are not automatically
merged. Unresolved exact/near gold differences quarantine whole groups.
Scored groups touching original unscored holdouts are also quarantined.
Cross-corpus overlap, target-coverage differences, and FiQA snippet flags
restrict a group to development. Mixed labels across different targets
remain intact. FiQA continuous scores are never converted to classes.

Flagged FiQA annotations excluded from snippet supervision: 48.

## Class counts after preparation

PhraseBank counts sentences; SEntFiN counts target annotations. FiQA
retains continuous scores, so its class-count dictionaries are empty.

| Corpus | Partition | Negative | Neutral | Positive |
| --- | --- | ---: | ---: | ---: |
| phrasebank | train | 468 | 2257 | 1060 |
| phrasebank | validation | 69 | 297 | 152 |
| phrasebank | test_candidate | 67 | 314 | 146 |
| sentfin | train | 3078 | 4395 | 4074 |
| sentfin | validation | 385 | 523 | 502 |
| sentfin | test_candidate | 334 | 550 | 479 |

## Reproduction and limitations

Run `uv run --locked python -m finsight.data prepare-splits`.
Source files are verified before processing and remain unchanged.
Processed records and the full lineage manifest are ignored local data.
Existing content-addressed artifacts must match exactly; new policies
produce new directories rather than silently replacing prior splits.

Split manifest SHA-256: `a9a67c231903bc2e3e71afead1d6daaf6b7201211fc25a05efd2bb2ee2d5e898`.

Known exact text, connected-group, and near-edge partition overlap
checks pass. This covers the current lexical screen, not all semantic
overlap or inaccessible model-training corpora. PhraseBank test
candidates remain unsuitable as independent pretrained FinBERT evidence.
SEntFiN candidates still need independent-benchmark curation and review.
No final test is frozen by this provisional preparation command.

The subsequent [cohort label-distribution supplement](split-label-distribution.md)
compares train/validation/test proportions for PhraseBank and single-target
SEntFiN, distinguishing these parent candidates from the final frozen cohort.
The separate benchmark receipt and `docs/evaluation-contract.md` define
the subsequent frozen release. No model training or tuning is performed.
