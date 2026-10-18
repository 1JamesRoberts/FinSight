# Duplicate and ambiguous-annotation findings

Phase 1 review on 2026-10-10, against the revisions and manifest hash in
[the reproducible aggregate report](../../../reports/annotation-review.md). Source
records and annotations are unchanged. This document records inspection findings
and recommendations, not new labels or a partition manifest.

## What was inspected

The automated review covers PhraseBank >=50%, SEntFiN, and the original-format
FiQA participant copy. Agreement variants are used only for agreement metadata,
not pooled as additional examples. FiQA train/test text is compared for leakage;
unscored test inputs are excluded from score and snippet-quality indicators.

Qualitative inspection covered all 14 PhraseBank/SEntFiN exact groups with
different annotation sets (including the nine groups with conflicting gold),
all 25 near-duplicate candidate pairs, all 49 FiQA nonmatching snippet values,
and the four annotations with missing snippets. The FiQA exact train/test match
was checked through source/annotation metadata. Illustrative inspection also
covered the first five lowest-subset-match PhraseBank examples, first five
SEntFiN mixed-target examples, and first five small-magnitude FiQA training
annotations per source type in release order. These samples are deterministic,
not random or representative; no ambiguity prevalence is inferred from them.
Full raw text, source IDs, and contexts stay in the ignored local review JSON.

## Exact duplicates and conflicts

| Corpus | Duplicate groups | Extra rows beyond first | Groups with inconsistent gold |
| --- | ---: | ---: | ---: |
| PhraseBank >=50% | 8 | 8 | 2 |
| SEntFiN | 67 | 67 | 7 |
| FiQA train + test inputs | 1 | 1 | 0; test gold absent |

Both PhraseBank conflict groups assign neutral and positive to identical
normalized sentences. The boundary between factual reporting and favorable
investor implications is a plausible source of disagreement, but the released
labels do not establish why these repeated entries differ. Neither label can be
chosen solely from duplicate frequency or a model prediction.

All seven SEntFiN gold-conflict groups assign neutral and negative to the same
normalized headline and same normalized target. These are annotation
inconsistencies in the selected artifact, rather than mixed sentiments about
different companies. Five additional exact groups have differing target sets or
target scope without a conflicting label on a shared target. Extra or omitted
entities and broader/narrower market targets need a target-coverage decision;
we must not automatically alias them or treat them as sentiment errors.

The FiQA duplicate occurs across original headline train/test inputs. The test
copy has no gold score/aspects, so the annotation sets differ without a gold
conflict. It is still a leakage issue if those source partitions are reused.
No cross-corpus normalized exact match was found.

## Near duplicates

The fixed lexical screen finds 18 PhraseBank pairs, six SEntFiN pairs, and one
FiQA–SEntFiN pair. Inspection found recurring material with metadata/date
wrappers, punctuation or spelling changes, possessive variants, wording changes,
attribution changes, and added context. One PhraseBank pair contains generic
publisher boilerplate rather than financial event sentiment. Some truncated or
poorly formatted sentences also deserve contextual review.

The cross-corpus pair is a near-identical financial headline with an additional
source-name suffix in one copy. It is direct evidence that the two selected
corpora are not completely text-disjoint. This does not establish which samples
were used by the published FinBERT checkpoint, but an evaluation must account
for this overlap if FiQA is development/training material.

Four candidate pairs disagree on a shared gold context: three neutral/positive
PhraseBank pairs and one neutral/negative SEntFiN pair. Different text can change
interpretation, so these are potential label inconsistencies to adjudicate,
not automatic errors. Five pairs have different extracted numeric tokens;
inspection found date wrappers or added unit/context information. A numeric
mismatch alone cannot decide whether two records describe the same event.

All 25 pairs remain overlap candidates. None is automatically merged, removed,
or used to infer corrected labels. Lexical screening misses paraphrases and
texts below six words. Generic statements and repeated templates can also look
similar without referring to the same event. This is a bounded duplicate review,
not exhaustive proof that no leakage remains.

## Ambiguity and annotation quality

SEntFiN's 1,233 headlines with mixed target labels are not automatically
ambiguous gold. Reviewed examples include positive and negative outcomes for
different instruments, and neutral attribution entities beside affected targets.
These support retaining target-level labels and whole-headline groups.

PhraseBank subset membership supplies limited confidence evidence. Matching each
>=50% text/label pair to the strongest available subset gives 629/763/1,190/2,264
source rows for 50/66/75/all. Repeated text/label pairs inherit the strongest
match, so these are **matching-subset proxies**, not recovered individual vote
percentages. Inspected lower-consensus examples include mixed business outcomes,
forward-looking language, and contextual/formatting issues. Lower agreement does
not by itself mean incorrect annotation. Individual annotator votes are not
available in the acquired release; kappa or per-item vote distributions cannot
be calculated from it. SEntFiN's consensus labels and FiQA scores likewise do
not provide individual raters' vote tables.

FiQA has 95 training target scores with absolute value at most 0.1. This is a
fixed descriptive review marker, not a chosen neutral band. Examples show mild
or mixed financial cues and aspect-specific judgments; small magnitudes alone
do not prove uncertainty or errors. Original continuous scores remain intact.

Four FiQA training annotations have missing snippet values. Separately,
49 nonmissing snippet values across 45 annotations do not occur as normalized
literal spans of their source text. Inspection found a mixture of paraphrases,
ellipses/transcription differences, annotation comments/placeholders, and clearly
unrelated topics or companies. A failed span check is not uniformly an error,
but the unrelated cases are a concrete annotation-quality concern. These
observations apply to the **pinned participant copy**; they cannot be attributed
to the unavailable organizer files or the unacquired benchmark mirror.

## Recommendations for the next bounded task

- Create a partition policy using whole-text groups. Keep exact duplicates and
  reviewed near-overlap candidates together, including the FiQA–SEntFiN pair.
- Exclude unresolved exact gold conflicts and suspicious annotation contexts
  from final evaluation until independently adjudicated. Preserve raw records
  and log any later exclusion or adjudication decisions.
- Decide how to handle SEntFiN target-coverage differences before any merging.
  Do not silently union target annotations or resolve aliases from string similarity.
- Avoid FiQA snippet/aspect supervision until the mismatched fields are traced
  to a reliable release or reviewed individually. These findings do not justify
  changing all associated target scores or dropping all flagged examples.
- Keep near-zero scores continuous. Any future class conversion must use
  development data and be specified before final evaluation.

Split creation, final-test freezing, model comparisons, broader semantic overlap
checks, and annotation correction are not implemented by this review.
