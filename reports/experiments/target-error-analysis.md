# Phase 4 target-sentiment error analysis

Saved validation predictions only: 1,410 native target annotations from 1,053 headlines; DeBERTa Base, RoBERTa Base and FinBERT; seeds 42/43/44.
Both matched headline-only and target-pair arms are checked (18 trials). No new inference, training, calibration fitting, ensemble or test scoring.

## Shared errors and disagreement

All nine target-pair runs get **1117** annotations correct and **51** wrong. At least one run misses **293** annotations.
These are annotation counts, not nine independent copies of the data.

| Cohort | Annotations | Headlines | All nine wrong | Any run wrong |
| --- | ---: | ---: | ---: | ---: |
| all_targets | 1410 | 1053 | 51 | 293 |
| single_target | 781 | 781 | 39 | 170 |
| multi_target | 629 | 272 | 12 | 123 |
| mixed_label_headlines | 283 | 122 | 5 | 52 |
| seen_native_target | 1147 | 891 | 41 | 233 |
| unseen_native_target | 263 | 243 | 10 | 60 |
| nonliteral_target | 8 | 8 | 3 | 4 |

Cohorts overlap. Seen/unseen refers to exact native target strings, not verified company identities. Nonliteral uses the existing whole-token matching diagnostic and does not automatically imply an alias error.

| Backbone | Wrong in 1/3 seeds | Wrong in 2/3 | Wrong in 3/3 | Prediction varies | Persistent fixes | Persistent harms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deberta_base | 46 | 43 | 112 | 92 | 74 | 5 |
| roberta_base | 67 | 46 | 101 | 118 | 67 | 9 |
| finbert | 66 | 47 | 106 | 118 | 79 | 6 |

Prediction variation includes changing wrong labels. A recurring error is wrong in all three paired seeds, even if its wrong label changes. A persistent fix requires all three headline seeds wrong and all three paired seeds correct within one model; persistent harm reverses this.

## What conditioning fixes and breaks

| Backbone | Seed | Fixed | Harmed | Both correct | Both wrong |
| --- | ---: | ---: | ---: | ---: | ---: |
| deberta_base | 42 | 143 | 39 | 1110 | 118 |
| deberta_base | 43 | 149 | 29 | 1111 | 121 |
| deberta_base | 44 | 147 | 39 | 1102 | 122 |
| roberta_base | 42 | 141 | 37 | 1104 | 128 |
| roberta_base | 43 | 165 | 34 | 1100 | 111 |
| roberta_base | 44 | 159 | 44 | 1099 | 108 |
| finbert | 42 | 155 | 49 | 1092 | 114 |
| finbert | 43 | 147 | 54 | 1095 | 114 |
| finbert | 44 | 146 | 30 | 1117 | 117 |

Each matched-seed row partitions the same 1,410 annotations. These correctness counts are not macro F1 and should not be pooled as independent observations. Every model/seed fixes more examples than it harms.

## Pairwise target-model disagreements

| A | B | Seed | Predictions differ | A only correct | B only correct | Both wrong |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| deberta_base | roberta_base | 42 | 110 | 57 | 49 | 108 |
| deberta_base | roberta_base | 43 | 97 | 43 | 48 | 102 |
| deberta_base | roberta_base | 44 | 103 | 43 | 52 | 109 |
| deberta_base | finbert | 42 | 125 | 62 | 56 | 101 |
| deberta_base | finbert | 43 | 128 | 70 | 52 | 98 |
| deberta_base | finbert | 44 | 111 | 46 | 60 | 101 |
| roberta_base | finbert | 42 | 125 | 58 | 60 | 105 |
| roberta_base | finbert | 43 | 112 | 65 | 42 | 103 |
| roberta_base | finbert | 44 | 101 | 44 | 49 | 103 |

Matched seed numbers are descriptive comparisons, not a formal significance test. No voting or new ensemble prediction is constructed.

## Qualitative review

Reviewed **36 annotations from 36 headlines**. Category tags overlap; counts are descriptions of the selected sample, not corpus prevalence.

| Review tag | Sample annotations |
| --- | ---: |
| target_binding | 14 |
| alias_or_nonliteral_target | 5 |
| comparison | 5 |
| neutral_boundary | 25 |
| conflicting_cues | 9 |
| financial_context | 25 |
| event_interpretation | 12 |
| insufficient_context | 3 |

**18** selected cases have suspected gold/context ambiguity. This is exploratory Codex judgment, not independent human adjudication, an estimated annotation-error rate, or a reason to drop/relabel examples after observing model failures. All gold stays unchanged.

Observed patterns, paraphrased without source text:

- Conditioning often separates the affected business from a quoted adviser, regulator, transaction vehicle or location that is only mentioned factually.
- It can also introduce sentiment into factual mentions: advice, legal events and financing substitutions can be interpreted more strongly than the source label. There are persistent harms in every backbone.
- Relative underperformance need not mean an absolute loss; resilience need not imply a price increase. Contrasting actors, historical declines, current reversals and negated advice remain difficult.
- Neutral-versus-directional labels are sometimes unclear from a headline alone. Some headings omit the actual recommendation; some source labels differ from the apparent price direction. Agreement among models does not adjudicate the gold label.
- Nonliteral targets in the sample include singular/plural differences. Some mismatches occur in both input formats, so they are not evidence that an alias caused the error. An unexpanded abbreviation also leaves identity uncertain. Mechanisms suggested in notes are hypotheses, not causal tests.

## Method, privacy and next decisions

Up to five annotation examples per declared stratum, distinct headline duplicate groups within a stratum; SHA-256(target-review-v1:42:example_id) order. Deduplicate example IDs across strata, preserving all selection reasons. All nine wrong = all three target-pair seeds of all three models wrong. Persistent fix/harm requires all three headline seeds wrong/correct and all three paired seeds correct/wrong within at least one model. Comparison regex and nonliteral matching select review candidates only, not automatic error tags. Selected annotation examples may share headlines across strata. Purposive, not representative; no error-category prevalence estimate. No ensemble model.

Sampling covers shared failures, stable inter-model disagreement, seed instability, persistent fixes/harms, mixed-label errors, comparison-regex candidates, nonliteral errors and neutral-boundary errors. Review uses all three seeds plus sibling target labels. Native text, names, IDs, row predictions and free-text reasoning remain in ignored local storage.
The selection is purposive and the tags describe apparent difficulties, not proven model mechanisms. Comparison-regex hits are reviewed manually; a contrastive conjunction is not automatically a comparison category.

Keep this benchmark unchanged. Record an annotation rubric and seek independent adjudication before any separately versioned label repairs. Next, predeclare target-level model selection and reserved-test comparisons; this review does not select a best seed, unlock tests or establish a winner.
Existing validation F1/calibration/architecture comparisons remain in [the backbone report](target-backbones.md). Five-company coverage and automatic entity identification are still unresolved.

Receipt verification covers all 18 prediction hashes, native membership/gold, 108 cohort metric sets and artifact manifests. Metric roundoff tolerance is 1e-12; prediction bytes must match exactly. Full weight bytes are not loaded in this analysis; prior Windows training verified them.

Private packet SHA-256: `ab1caea2b89a25e0f8d4d412517147190c4e020bafd08bd4b36898ff4f0d1f6b`.

Private annotations SHA-256: `2e85e33ae9e33d72d4b9e8c48b80afd7442709386d3a3cc175cf44f9d943d644`.
