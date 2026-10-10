# FiQA 2018 Task 1 participant copy — FinSight dataset card

Maintainer: FinSight project. Reviewed 2026-10-10. Scope: local noncommercial
target/aspect sentiment research. This is not the gated classification benchmark
mirror or FiQA Task 2 question-answering retrieval data.

## Identity, provenance, and rights

Original challenge: FiQA 2018, Task 1 sentiment scoring; cite
[Maia et al., WWW 2018](https://doi.org/10.1145/3184558.3192301).
Selected accessible source: [participant repository](https://github.com/dayanfcosta/fiqa-2018-task1/tree/acd0f4cfafa6421774bc8f009e8fa3e0ec441357).
Revision: `acd0f4cfafa6421774bc8f009e8fa3e0ec441357`.
All four JSON file SHA-256, Git blob IDs, sizes, and the retained README are in
`configs/asset-revisions.json`. Organizer downloads were unavailable during
review; byte equivalence to the organizer's files is unverified.

Local research relies on the original organizer's noncommercial-use terms, not
a participant software-license inference. Retain challenge attribution and the
[terms/provenance review](../fiqa-asset-review.md). Public text redistribution and
commercial use are not cleared. The gated TheFinAI mirror's broader MIT claim
and different split sizes are not treated as overriding the selected copy's terms.
No gated access form was submitted and no source text is committed.

## Schema and supervision

Nested JSON maps opaque source IDs to `sentence` and an `info` annotation list.
Each annotation retains target, aspects, snippets, continuous sentiment score in
[-1, 1], and original serialized strings. Records preserve source type, original
split, filename, and revision. IDs can contain symbols/target names. Restricted
AST inspection parses list literals without executing code.

| Original source | Texts | Annotations | Scored |
| --- | ---: | ---: | ---: |
| Headline train | 438 | 503 | 503 |
| Post train | 675 | 675 | 675 |
| Headline test | 93 | 93 | 0 |
| Post test | 99 | 99 | 0 |

Training has 1,113 texts / 1,178 scores. Original test inputs have no gold scores
or aspects; they cannot produce supervised evaluation metrics. Missing scores
stay `None`, not neutral or zero. Scores are never converted to three classes;
numeric signs and histogram bins are descriptive, not class definitions.

## FinSight preparation and development EDA

The [split policy](../split-policy.md) keeps target annotations grouped with text.
One original train/test exact overlap quarantines the scored counterpart. The
near screen also finds one FiQA–SEntFiN overlap, restricted to development.

| View | Texts |
| --- | ---: |
| Train | 907 |
| Validation | 102 |
| Withheld scored test candidates | 103 |
| Quarantined scored source row | 1 |
| Original unscored test inputs | 192 |

The 1,009 development texts have 1,068 scored annotations: 395 headlines and
614 posts. Median whitespace length is 11 and P95 is 23 words. Development score
median is 0.237 and mean is approximately 0.1213. Positive numeric scores account
for 64.7%; this is not a three-class positive-label share.

Four source snippet values are missing, and 49 snippet values across 45
annotations do not match normalized literal spans. The union flags 48 annotations
(one annotation has both conditions). All 48 remain in development with review
flags; snippet supervision is disabled and scores remain unverified pending
review. Do not infer score correctness from the presence of a literal snippet.
The retained processed view has 1,112 scored texts; raw bytes remain intact.

## Intended use and limitations

Use development records for continuous target-level research. Adapting them to
three-class training requires a separately justified development-only protocol.
The 103 scored candidates are withheld but have no separately curated final
regression release yet. FiQA is not the shared three-class frozen benchmark.

FinBERT's original paper reports FiQA regression experiments, but exact exposure
of the published classifier is unresolved. Participant-copy provenance, suspect
snippet contexts, mixed headline/post domains, and unscored original tests limit
comparability with other FiQA releases. The schema does not establish full-article
validity, reliable prediction-time timestamps, or coverage of all five MVP companies.
No thresholds, models, or target extraction are implemented in Phase 1.

See [Phase 1 findings](../phase1-findings.md) for reproducible development figures
and `reports/label-analysis.md` for original-source score statistics.
