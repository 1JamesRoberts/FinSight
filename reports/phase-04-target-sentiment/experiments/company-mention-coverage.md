# SEntFiN development company detection

Run and reviewed 2026-10-19, using unchanged `registry-mentions-v1` and
`companies-v1`. [Aggregate JSON](company-mention-coverage.json) records input,
registry and implementation hashes. Only the locked native train/validation
views were loaded; frozen test records were not opened. No sentiment model was
run, trained or selected in this task.

## Headline counts

| Measure | Train | Validation |
| --- | ---: | ---: |
| Unique headlines scanned | 8,598 | 1,053 |
| Native target annotations | 11,547 | 1,410 |
| Headlines with any registry candidate | 29 | 2 |
| Headlines with a resolved company | 7 | 2 |
| Headlines with an ambiguous candidate | 23 | 1 |
| Resolved mention spans | 7 | 2 |
| Ambiguous mention spans | 24 | 1 |
| Headlines with multiple resolved companies | 0 | 0 |

Resolved and ambiguous headline counts overlap. One training headline repeats
Apple, so mention-span counts also differ from headline counts. These counts
describe the **five-company registry**, not all financial entities in SEntFiN.

| Company | Train resolved / ambiguous headlines | Validation resolved / ambiguous headlines |
| --- | ---: | ---: |
| Apple | 2 / 20 | 0 / 0 |
| Microsoft | 5 / 0 | 2 / 0 |
| NVIDIA | 0 / 0 | 0 / 0 |
| Amazon | 0 / 2 | 0 / 1 |
| Alphabet, including Google candidates | 0 / 1 | 0 / 0 |

The fixed default policy leaves bare Apple/Amazon/Alphabet unresolved and requires
explicit Google-to-Alphabet rollup. No context flag was inferred from corpus
membership or native labels.

## Comparison with existing target annotations

Native target strings were separately resolved with explicit financial-entity
context and without subsidiary rollup, matching the earlier strict coverage
definition. This comparison does **not** feed context into detection.

| Native registry target | Train: resolved / ambiguous only / no candidate | Validation: resolved / ambiguous only / no candidate |
| --- | ---: | ---: |
| Apple | 2 / 19 / 0 | 0 / 0 / 0 |
| Microsoft | 4 / 0 / 0 | 2 / 0 / 0 |
| Amazon | 0 / 2 / 0 | 0 / 1 / 0 |
| NVIDIA and direct Alphabet | 0 / 0 / 0 | 0 / 0 / 0 |

Every strict registry-matched native target headline has at least a corresponding
resolved or ambiguous company candidate. This is an annotation-coverage check,
not a general recall estimate. Most Apple/Amazon company mentions intentionally
remain ambiguous under the fixed policy. Google has one training candidate and
is excluded from strict direct-Alphabet target counts.

## Qualitative review

All 31 candidate headlines were inspected by the assistant. This is exploratory
review, not independent human adjudication or exhaustive NER annotation. Full
headlines, source IDs, annotations, spans and review notes stay in the ignored
`data/interim/company-mention-review/` directory.

- One training Apple candidate appears to concern imported fruit prices. Its
  native target is outside the strict Apple company mapping. The ambiguity gate
  prevents this from automatically becoming an Apple sentiment prediction.
- One resolved Microsoft span occurs within a regional Microsoft entity name.
  The native target names that regional entity and does not match the registry's
  whole-target parent identity. The difference needs an explicit regional-rollup
  decision; it does not establish a false positive or justify changing the label.
- One Google training mention is held for explicit parent-rollup permission.
- The remaining candidate spans appear to denote the named companies. Native
  targets are not exhaustive mention annotations, so unannotated candidates must
  not be counted automatically as false positives.

The validation candidates comprise two Microsoft headlines, one of which also
contains an ambiguous Amazon mention. There are no default-policy multi-company
resolved headlines to exercise multi-company automatic sentiment here. Three
companies have no validation support. No precision, recall, per-company F1 or
five-company performance claim is warranted from this review.

## Reproduce and next step

From the repository root, with existing prepared development files:

```bash
uv run --locked python -m finsight.sentiment.experiments.company_mentions
```

The command verifies development file hashes, deduplicates annotations by headline
ID, runs the unchanged detector, writes the private review cases and public
aggregate JSON. A rerun regenerates `cases.jsonl` and the aggregate; qualitative
`review.json` notes are maintained separately. No GPU or weights are required.
Four synthetic tests cover headline counting, ambiguity, unannotated matches,
reserved-partition rejection and inconsistent headline identity.

Verification: all 643 pytest tests pass. Required full Ruff checks were run;
their only failures are the pre-existing nine lint issues and formatting in the
untracked `notebooks/experiment.ipynb`. Excluding that scratch notebook, lint
and format checks pass for all 315 Python/notebook files. The notebook was left
untouched. Source and private-packet hashes verify; the public frozen contract
and final target-evaluation result retain their previous checksums.

This completes the bounded Phase 4 development detection run. Detector rules,
native labels, frozen cohorts and previous experiment results remain unchanged.
Next, explicitly review any context/rollup policy changes using development data,
and obtain independently labeled mention examples before claiming extraction
accuracy. Broader NER and the company-labeled test collection remain open roadmap
work; Phase 5 is not implemented here.
