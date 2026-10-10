# FinSight label analysis

Phase 1 descriptive analysis of pinned local artifacts. Raw source rows
are counted without deduplication. No thresholds or new splits are assigned.
FiQA analysis uses original training annotations only; unscored test inputs
are excluded. PhraseBank and SEntFiN are candidate corpora, not frozen tests.

## Source class distributions

PhraseBank counts sentences; SEntFiN counts target annotations. These
represent different supervision and must not be pooled as one task.

| Source | Total | Negative | Neutral | Positive | Largest/smallest |
| --- | ---: | ---: | ---: | ---: | ---: |
| PhraseBank 50 | 4,846 | 604 (12.46%) | 2,879 (59.41%) | 1,363 (28.13%) | 4.77 |
| PhraseBank 66 | 4,217 | 514 (12.19%) | 2,535 (60.11%) | 1,168 (27.70%) | 4.93 |
| PhraseBank 75 | 3,453 | 420 (12.16%) | 2,146 (62.15%) | 887 (25.69%) | 5.11 |
| PhraseBank all | 2,264 | 303 (13.38%) | 1,391 (61.44%) | 570 (25.18%) | 4.59 |
| SEntFiN targets | 14,405 | 3,814 (26.48%) | 5,517 (38.30%) | 5,074 (35.22%) | 1.45 |

Class percentages use the stated row/annotation denominator. The ratio is
largest class count divided by smallest; absent classes make it undefined.
Majority shares describe this corpus, not measured held-out model accuracy.
PhraseBank variants overlap and must not be concatenated or split apart as
independent corpora. Higher agreement indicates annotator consensus, not
verified truth; subset composition can change alongside agreement.

## Target-specific label differences

SEntFiN contains 10,753 headlines and 2,847 with multiple targets.
1,233 headlines have different labels across targets:
11.47% of all headlines and 43.31% of multi-target headlines.
These differences support target-aware evaluation; they do not by themselves
indicate annotation disagreement or errors. Keep each headline group intact.

## FiQA continuous scores (original training annotations)

| Source | Texts | Scores | Mean | Stddev | Min | Q25 | Median | Q75 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| headline | 438 | 503 | 0.0908 | 0.3756 | -0.9380 | -0.1890 | 0.1850 | 0.3640 | 0.9750 |
| post | 675 | 675 | 0.1431 | 0.4139 | -0.7450 | -0.3010 | 0.3180 | 0.4765 | 0.8140 |
| combined | 1,113 | 1,178 | 0.1208 | 0.3988 | -0.9380 | -0.2502 | 0.2375 | 0.4380 | 0.9750 |

| Source | Score < 0 | Score = 0 | Score > 0 |
| --- | ---: | ---: | ---: |
| headline | 169 (33.60%) | 13 (2.58%) | 321 (63.82%) |
| post | 234 (34.67%) | 1 (0.15%) | 440 (65.19%) |
| combined | 403 (34.21%) | 14 (1.19%) | 761 (64.60%) |

Signs describe numbers, not three-class gold labels. Exact zero does not
define a neutral class or justify choosing a neutral band. Any future
classification conversion must be specified using development data only.
Quartiles use inclusive linear interpolation; standard deviation is
population standard deviation. Each target annotation has equal weight:
texts with multiple annotations contribute more than single-target texts.
The JSON report also contains fixed histogram bins [-1, -0.5),
[-0.5, 0), [0, 0.5), [0.5, 1]; these are descriptive bins, not labels.

## Reproduction and boundaries

Run `uv run --locked python -m finsight.data analyze-labels` after acquisition.
The machine-readable report is `data/interim/label-analysis.json` (ignored).
This Markdown report contains only aggregates. All manifest files are
verified before analysis; byte mismatches fail instead of being accepted.

Manifest SHA-256: `e937655fe70b8db4576b9e1652a6d574e4c172dcf4a80f2b759fd17540df4793`.

- financial_phrasebank: `8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964`
- sentfin_1_0: `eba43610fe3cb57a3e5773cd6c037da1f8992ad5`
- fiqa_task1_original: `acd0f4cfafa6421774bc8f009e8fa3e0ec441357`

Lengths, duplicate/ambiguity investigations, per-target coverage, individual
annotator agreement, and split design remain open Phase 1 tasks. No
sentiment model is trained and no independent-test claim is made.
