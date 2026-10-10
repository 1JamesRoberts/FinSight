# Phase 1 findings and completion record

Reviewed 2026-10-10. Phase 1 is dataset preparation and descriptive analysis;
no learned preprocessing, sentiment model, or stock prediction is implemented.
The selected assets support local noncommercial research; public source-text
redistribution and FinBERT weight terms remain unresolved.

## Deliverables

| Deliverable | Evidence |
| --- | --- |
| Audited ingestion and typed schemas | `src/finsight/data/`, `configs/asset-revisions.json`, offline synthetic tests |
| Dataset cards | [PhraseBank](dataset-cards/phrasebank.md), [SEntFiN](dataset-cards/sentfin.md), [FiQA copy](dataset-cards/fiqa.md) |
| Source integrity/label/length/annotation review | Existing aggregate reports and `docs/annotation-review-findings.md` |
| Leakage-aware partitions and source lineage | `docs/split-policy.md`, ignored content-addressed split manifest |
| Frozen external holdout | `configs/benchmark-freeze.json`, `docs/evaluation-contract.md` |
| Development EDA notebook | `notebooks/01_dataset_eda.ipynb` |
| Exported figures and aggregates | `reports/phase1-eda.md`, `reports/figures/phase1/`, `reports/eda-summary.json` |

Earlier source-wide reports document pre-freeze quality inspection. The final
EDA notebook/figures use only the verified train/validation view. Every test
candidate, unscored input, quarantined row, redundant copy, and frozen/curation
group stays out of these figures. Frozen-test descriptive counts remain in the
already declared benchmark report; no additional test exploration is performed.

## Development findings

Development totals 14,963 texts: 4,303 PhraseBank sentences, 9,651 SEntFiN
headlines, and 1,009 FiQA texts. Units differ by task; do not pool their labels.

- PhraseBank is 59.4% neutral, with 537 negative versus 2,554 neutral sentences.
  Majority-class accuracy alone will be weak evidence. The planned macro F1
  and per-class metrics matter; no model result follows from these counts.
- SEntFiN development has 12,957 target annotations and 2,571 multi-target
  headlines. Of these, 1,116 have mixed sentiments: about 43.4% of multi-target
  examples. One article-wide label cannot express those annotations. Grouped
  sampling and target-aware evaluation are necessary for the intended comparison.
- Median whitespace lengths are 21 words for PhraseBank, 10 for SEntFiN, and
  11 for FiQA; P95 values are 42, 14, and 23. These demonstrate domain differences,
  not tokenizer lengths. Choose any future model token limit from training-only
  tokenization, without inspecting the final test.
- PhraseBank agreement membership falls from 4,303 development matches in
  >=50% to 2,023 unanimous matches. The four overlapping subsets are not four
  independent datasets. Any agreement-based development experiment must retain
  the original group/partition boundaries and record the changed sample selection.
- FiQA development contains 1,068 continuous target scores over 395 headlines
  and 614 posts. Its score median is 0.237; 64.7% have positive numeric signs.
  Sign counts and fixed histogram intervals do not define sentiment classes.
  Scores remain continuous; no neutral threshold has been chosen.
- The 48 flagged FiQA annotations remain available for reviewed target-score
  research, with snippet supervision disabled. Suspect scores are not made
  trustworthy by ingestion or by a well-formed distribution.

## Reproduction

Run from the repository root:

```sh
uv sync --locked --group eda
uv run --locked python -m finsight.data acquire
uv run --locked python -m finsight.data freeze-benchmark
uv run --locked --group eda python -m finsight.data eda
uv run --locked --group eda ruff check .
uv run --locked --group eda ruff format --check .
uv run --locked --group eda pytest
uv build
```

Only acquisition needs network access. The committed freeze receipt reconstructs
and verifies the same holdout on a fresh checkout. `eda` checks frozen/parent
digests and pinned raw bytes, then exports five aggregate PNG figures and JSON.
Use `--data-dir`, `--manifest`, `--freeze-receipt`, and `--report-dir` for explicit
paths. Analysis version, source/freeze digests, and a development-record digest
are included in the aggregate JSON. No source text or identifiers are exported.

Matplotlib and ipykernel are confined to the optional `eda` dependency group;
base ingestion/runtime dependencies remain unchanged. Open the notebook in an
existing Jupyter-compatible editor and select `.venv/bin/python` as its kernel.
The notebook calls the same tested EDA functions. It is committed without output
cells to avoid stale artifacts; the exported report contains the reviewed output.
No Jupyter server is required for the CLI. PNG appearance may vary across OS/fonts;
aggregate values reproduce from the same pins, and repeated local renders match.
CI installs the EDA group and checks synthetic plotting without dataset downloads.

## Acceptance and remaining risks

Raw checksums/schema validation, deterministic split reproduction, grouped
duplicate checks, immutable benchmark reconstruction, train/validation-only
development access, and EDA unit-count/plot checks have passed locally. Figures
were visually inspected and the notebook code was executed. Hosted GitHub Actions
results still need checking after push; local checks do not establish hosted CI.

Phase 1 is complete within its documented scope. This does not resolve every
annotation ambiguity or establish zero semantic/model-training overlap. Unresolved
gold conflicts stay quarantined. FinBERT pretraining exposure is unknown; its
PhraseBank supervision is known. The frozen SEntFiN release lacks representative
coverage of the five US companies. FiQA copy provenance, unverified scores, and
class-conversion questions persist. Dataset cards describe these limits explicitly.
Prediction-time timestamps, US-company gold data, public-text rights, and weight
permissions need separate future work. Phase 2 can begin with traditional NLP
baselines using the designated training/validation data and frozen contract.
