# Phase 2 qualitative validation-error review

The selected Logistic Regression and SVM models make many of the same errors.
Review suggests failures involving financial context, numeric direction,
conflicting cues, and the neutral-label boundary. These are qualitative hypotheses
about the observed texts, not measured causal explanations of model coefficients.
Gold labels and model configurations remain unchanged.

## Scope and method

Only the 518 PhraseBank and 781 single-target SEntFiN **validation** rows were
used. The existing tuned LR model and separately selected SVM model were reviewed
for each corpus, alongside saved VADER predictions. Original run, prediction,
model-artifact, validation-membership, gold-label, and metric checks pass before
joining predictions to text. No model is retrained, reloaded for scoring, or tuned
in this task; no frozen-test examples are inspected.

The primary sample is **LR errors**, because LR is the overall grid selection for
both cohorts. For each off-diagonal gold/predicted cell, take up to five rows
ordered by SHA-256 of `42:record_id`. This gives 28 PhraseBank examples (only three
positive-to-negative errors exist) and 30 SEntFiN examples. All six error directions
are represented. The sample is deterministic and deliberately stratified, not
representative of the full error distribution. It does not separately sample
VADER errors or SVM-only failures.

Codex reviewed all 58 texts and their SEntFiN target where present, assigning one
primary category, a possible-gold-ambiguity flag, and a local explanation. Categories
can overlap conceptually; the primary category is a reviewer judgment. This is
agent-assisted exploratory review, not independent human adjudication or a new
gold-label dataset. Source text, identifiers, targets, and individual notes remain
in ignored local storage. This public report contains aggregate counts only.

## Complete-cohort paired error counts

| Measure | PhraseBank | SEntFiN single-target |
| --- | ---: | ---: |
| Validation rows | 518 | 781 |
| Selected LR errors | 115 | 141 |
| Selected SVM errors | 113 | 143 |
| VADER errors | 230 | 411 |
| Both LR and SVM correct | 396 | 633 |
| Only LR correct | 7 | 7 |
| Only SVM correct | 9 | 5 |
| Both LR and SVM wrong | 106 | 136 |
| LR and SVM labels disagree | 16 | 15 |
| All three models wrong | 63 | 78 |

The first four paired correctness rows partition each validation cohort. Label
disagreement is a different measure: both classifiers can disagree while both
are wrong. The large shared-error counts suggest that changing the linear
classifier alone leaves much of the difficulty unresolved. This does not prove
that a contextual model will solve those cases. PhraseBank SVM has fewer errors
(higher accuracy) despite LR's slightly higher macro F1; selection uses macro F1.

## Reviewed sample categories

| Primary category | PhraseBank sample | SEntFiN sample | Interpretation |
| --- | ---: | ---: | --- |
| Financial context | 5 | 9 | Meaning of liquidity, default, recommendations, commodity/target perspective, or financial idioms |
| Numeric comparison | 6 | 1 | Change relative to another value, direction, or whether reduction is beneficial |
| Mixed cues | 4 | 7 | Favorable and adverse clauses, multiple entities, or opposing consequences |
| Event interpretation | 4 | 5 | Delivery, personnel, regulatory, issuance, acquisition, or other event implications |
| Neutral boundary | 8 | 6 | Factual reporting versus evaluative language, missing comparison/context, or fragments |
| Negation scope | 1 | 2 | Avoidance, reversal, or time-limited denial of an otherwise directional cue |
| Total reviewed | 28 | 30 | One primary category per sampled LR error |

These counts describe the reviewed sample only. They must not be interpreted as
percentages of all errors or compared as population rates between datasets.

## Findings and implications

Financial polarity depends on what changes and for whom. A reduction can be
beneficial for costs or errors, while higher prices can benefit a producer and
hurt a buyer. Linear word/bigram features may associate the same word with one
class without recovering the underlying relationship. Numeric tokens are features,
not an arithmetic comparison module. TF-IDF also uses local word/bigram patterns,
not a general representation of clause roles or target identity.

The review includes cases where a favorable cue is cancelled by a later adverse
clause, and cases where avoiding an adverse event is itself favorable. These
motivate checking contextual and target-aware models in later phases. VADER
occasionally gets a selected linear-model error right, but its complete validation
metrics are substantially lower; cherry-picked examples do not justify replacing
the stronger baselines or creating an untested ensemble.

Some source labels are difficult to establish from text alone. We flagged **8 of
28 PhraseBank** and **12 of 30 SEntFiN** sampled cases for possible gold ambiguity.
Examples of the underlying issue include absolute profit/loss without a comparison,
planned positive-sounding events labeled neutral, opposing movements in one
sentence, and unclear beneficiary perspective. A flag means a plausible alternate
reading or missing context, not a confirmed annotation error. No case is dropped,
repaired, relabeled, or excluded from metrics. Human review would be needed before
any separately versioned label correction.

## Reproduction and provenance

```bash
uv run --locked python -m finsight.sentiment.error_analysis
```

The command verifies development inputs and saved predictions, then prints the
ignored packet path. To reproduce the existing aggregate review using its saved
local annotations:

```bash
uv run --locked python -m finsight.sentiment.error_analysis \
  --annotations data/interim/phase2-error-review/38e8776caf64b7b4c39aea49dab0125396c5876fb24d024fe3406ad72828ad63/5746ebf0e64f92009c76e3bd719c68c91aa3db4bf747c7f2bfc0e965ae481103-annotations.json
```

The CLI supports `--data-dir`, `--freeze-receipt`, `--tuning-report`, and
`--report-path`. Local annotations are a JSON list containing `record_id`,
`category`, `gold_ambiguous` (boolean), and a nonempty `note`. The six allowed
categories are declared in the module; the list must cover the exact packet once.
Preparing a packet is automatic; qualitative annotation requires inspecting the
texts. A fresh checkout cannot reconstruct the review judgments without the local
annotation artifact. No source text or annotation notes are redistributed.

The accompanying [aggregate JSON](phase2-error-analysis.json) pins tuning report,
freeze receipt, packet, annotations, and review-implementation digests, plus the
selected original run IDs. Those runs pin model/config/code/dependency identities.
See the [consolidated Phase 2 report](phase2-comparison.md) for metrics, architecture,
selection, scope, and remaining limitations.
