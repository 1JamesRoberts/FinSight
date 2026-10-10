# FinSight duplicate and annotation review

Exact identity: Unicode NFKC + casefold + whitespace collapse, retaining
numbers and punctuation. PhraseBank uses >=50% only; agreement subsets
are not pooled. FiQA original train/test inputs are screened for duplicates,
with test inputs excluded from score/ambiguity analysis. No rows are removed
or relabeled and no final evaluation split is selected.

| Corpus | Records | Exact groups | Extra rows | Conflicting gold contexts | Different annotation sets |
| --- | ---: | ---: | ---: | ---: | ---: |
| fiqa | 1305 | 1 | 1 | 0 | 1 |
| phrasebank | 4846 | 8 | 8 | 2 | 2 |
| sentfin | 10753 | 67 | 67 | 7 | 12 |

Cross-corpus exact text groups: 0.

## Near-duplicate screening

Candidate unique-text pairs: 25.
Pairs with different numeric tokens: 5.
Pairs with different gold for a shared context: 4.
Candidates use whitespace-word trigram sets, Jaccard >=0.80, and at least
six words per text. An inverted index enumerates shared-shingle pairs and
an exact Jaccard check accepts candidates. Exact matches are collapsed first.
This lexical method misses short, paraphrased, and lower-similarity overlap.
Candidate counts do not establish semantic equivalence or model independence.

- fiqa/sentfin: 1 candidate pairs
- phrasebank/phrasebank: 18 candidate pairs
- sentfin/sentfin: 6 candidate pairs

## Annotation ambiguity indicators

Gold conflicts require the same normalized text and annotation context.
For PhraseBank the context is the whole sentence; for SEntFiN it is the
normalized target name. FiQA also requires matching aspects and snippets.
Targets are not alias-resolved. Different target/aspect/snippet annotations
are not automatically inconsistent. Missing test gold is not a neutral label.

Within-record conflicting contexts: 0.
SEntFiN mixed-target label headlines: 1233.
Mixed sentiments about different targets can be correct; they are examples
for target-aware review rather than evidence of annotator disagreement.
FiQA training annotations with missing snippets: 4.
FiQA nonmatching snippet values: 49 across 45 annotations.
A nonmatching snippet is not a normalized literal span of its source text.
This includes possible paraphrases, annotation notes, and mismatches; it
is a review flag, not proof that the target score is wrong.
FiQA training scores with abs(score)<=0.1: 95.
The fixed small-magnitude marker is descriptive, not a neutral-band decision
or proof of annotation uncertainty. Continuous scores remain unchanged.

PhraseBank strongest matching text/label subset (proxy, not per-row votes):

- 50: 629 >=50% source rows
- 66: 763 >=50% source rows
- 75: 1190 >=50% source rows
- all: 2264 >=50% source rows

Repeated text/label pairs inherit the strongest matching subset; this
cannot distinguish duplicate occurrences or recover individual vote counts.
The source release does not support computing per-item agreement percentages
or an inter-annotator statistic such as kappa. Consensus subsets are preserved.

## Reproduction

Run `uv run --locked python -m finsight.data review-annotations`.
Ignored `data/interim/annotation-review.json` contains local source text,
IDs, annotations, exact groups, near candidates, and five deterministic
examples per selected ambiguity indicator. These are candidates, not an
exhaustive manual adjudication. Aggregate-only Markdown is committed.
See `docs/annotation-review-findings.md` for the completed qualitative review.

Manifest SHA-256: `e937655fe70b8db4576b9e1652a6d574e4c172dcf4a80f2b759fd17540df4793`.

- financial_phrasebank: `8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964`
- sentfin_1_0: `eba43610fe3cb57a3e5773cd6c037da1f8992ad5`
- fiqa_task1_original: `acd0f4cfafa6421774bc8f009e8fa3e0ec441357`
