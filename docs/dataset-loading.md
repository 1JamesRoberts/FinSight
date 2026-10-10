# Reviewed dataset acquisition and validation

Scope: Phase 1, Financial PhraseBank, SEntFiN, and an original-format FiQA
challenge participant copy reviewed for local noncommercial research. Provisional
split preparation is now available as a separate command; no sentiment model or
prediction function is implemented. The runtime uses Python's standard library.

## Reproduce from the repository root

```sh
uv sync --locked
uv run --locked python -m finsight.data acquire
uv run --locked python -m finsight.data validate
uv run --locked python -m finsight.data prepare-splits
```

`acquire` downloads the eleven pinned archive/CSV/JSON/notice files listed in
`configs/asset-revisions.json`. It verifies sizes and SHA-256 before atomically
saving each file; SEntFiN CSVs and FiQA JSONs also match their Git blob identities. A valid cache
is reused without network access. A corrupted cache fails rather than being
overwritten: inspect/remove the affected ignored artifact, then reacquire.
The checked-in manifest contains no source text or credentials and downloads no
model weights. A partial run can be rerun; completed verified files are reused.

These commands accept `--manifest PATH` and `--data-dir PATH`. The default data
root comes from `FINSIGHT_DATA_DIR` (otherwise `data`). Run from the repository
root or pass the manifest's absolute path. For local environment overrides use
`uv run --locked --env-file .env python -m finsight.data ...`.

Files are stored below `data/raw/financial_phrasebank/` and `data/raw/sentfin/`, plus `data/raw/fiqa/`.
The acquisition receipt is `data/raw/acquisition-receipt.json`; the validation
summary is `data/interim/dataset-integrity.json`. These are ignored artifacts.
The validator verifies every manifest file again before parsing; changing source
bytes prevents validation. Summary output contains aggregate counts, no headlines.

`prepare-splits` applies the [documented filtering/grouping policy](split-policy.md)
and writes content-addressed local processed records with source lineage. Its
`--split-policy PATH` defaults to `configs/split-policy.json`; `--report-path PATH`
overrides the aggregate Markdown report. It revalidates sources independently.
The [measured report](../reports/split-preparation.md) distinguishes quarantine,
redundant copies, unscored originals, and provisional supervised partitions.
The subsequent [benchmark freeze](evaluation-contract.md) reserves curated SEntFiN
groups; loading or provisional splitting alone does not freeze a final test.

## Source formats and preservation

PhraseBank is read directly from the verified ZIP; arbitrary extraction paths and
the historical dataset loader are not used. Agreement variants are separate,
not concatenated. ISO-8859-1 text is split at the final `@`, preserving sentence,
original label, member name, line-based source ID, agreement, and repository revision.
Empty text, missing delimiters, unknown labels, empty sources, and missing/duplicate
requested members fail validation.

SEntFiN's exact CSV schema is `S No.`, `Title`, `Decisions`, `Words`. Each source
headline remains one typed record with its entity sentiments attached; targets are
not treated as independent split units. Preserve the original decision dictionary
string, original word-count field, title, source ID, and revision. Header mismatch,
missing/extra cells, duplicate numeric IDs, invalid counts, empty/duplicate targets,
and unknown labels fail with source identifiers, without echoing private text.

The release has 13 decision strings with unescaped apostrophes in entity names.
An AST inspector handles normal string dictionaries without executing them. Only
on syntax failure, a narrow quoted-pair parser accepts the observed apostrophe
format, requiring dictionary boundaries and comma separators. It rejects expressions
and invalid labels. No records are dropped or source files rewritten. Tests use
invented text, including apostrophe and executable-input cases.

The entity database retains its unnamed numeric index, `Symbol`, and `Entity`
strings. No ticker aliases or company identities are inferred at this stage.
Common labels are string enum values `negative`, `neutral`, and `positive`; original
labels remain available. Do not transfer integer label IDs between datasets and
models: FinBERT's configured integer order differs from PhraseBank's historical order.

## FiQA source format

The selected participant copy uses original nested JSON: each source ID maps to
`sentence` and an `info` annotation list. IDs are opaque strings; test IDs include
target names/tickers. The typed loader preserves groups, targets, aspects,
snippets, continuous scores, revision, filename, type, and original split.
Duplicate JSON keys, empty sources/fields, incorrect schemas, invalid lists, and
nonfinite/out-of-range scores fail validation. Source test inputs have no gold
scores or aspects and retain `None`. No sentiment classes are derived.
Four bare `nan` snippet values map to missing values with raw strings preserved.
The parser inspects constant lists without executing expressions.

See [FiQA asset review](fiqa-asset-review.md) for measured counts, source terms,
and the distinction from the gated 1,173-row benchmark mirror. The audit reports
one original train/test normalized exact match and zero exact matches with
PhraseBank/SEntFiN. No partitions are repaired or frozen during loading.

## Initial integrity findings (2026-10-10)

| Source | Loaded rows | Entity annotations | Normalized duplicate rows |
| --- | ---: | ---: | ---: |
| PhraseBank >=50% | 4,846 | Not target-labeled | 8 |
| PhraseBank >=66% | 4,217 | Not target-labeled | 6 |
| PhraseBank >=75% | 3,453 | Not target-labeled | 5 |
| PhraseBank unanimous | 2,264 | Not target-labeled | 5 |
| SEntFiN headlines | 10,753 | 14,405 | 67 |
| SEntFiN entity database | 1,010 | Not sentiment examples | Not audited for text duplication |

All higher-agreement PhraseBank normalized texts occur in the >=50% collection.
Two normalized PhraseBank text groups have conflicting source labels. These must
be handled explicitly in split/data-quality decisions, not silently relabeled.
SEntFiN has 2,847 multi-entity headlines and 1,233 with conflicting sentiments.
Its artifact has one more entity annotation (neutral) than the paper's reported
14,404, and the entity database has 1,010 rows rather than the described 1,009.
Use measured artifact counts and retain these discrepancies in research reports.

Cross-corpus normalized exact matches: **0**. The normalization is Unicode NFKC,
casefolding, and whitespace collapse; punctuation and numbers are retained.
Duplicate rows count repetitions beyond the first. This check neither detects
near duplicates nor establishes that the model never saw these texts. PhraseBank
is known FinBERT fine-tuning material; model-pretraining manifests are unavailable.
Final-test selection and freezing remain pending. Full EDA, length distributions,
annotation-quality assessment, fuzzy duplicates, and company coverage are later
bounded tasks. Do not tune models using a subsequently reserved test partition.

## Usage scope

See [PhraseBank asset review](phrasebank-asset-review.md) and
[SEntFiN asset review](sentfin-asset-review.md), and
[FiQA asset review](fiqa-asset-review.md). Local noncommercial research is the
reviewed scope; preserve source notices. Dataset contents and public headline demos
are not included in this repository. This workflow does not settle underlying
publisher rights, commercial deployment rights, or FinBERT weight-license scope.

## Verification

Unit tests cover source preservation, restricted parsing, schema failures,
integrity mismatches, cache behavior, and rejected paths. The integration test
uses synthetic files to exercise acquisition, validation, duplicate counting,
and overlap reporting without remote downloads. Real-artifact validation is a
separate reproducible local check; a fresh-directory download and validation also
passed for the six initial pinned files. The FiQA follow-up acquired and
validated five additional pinned files; a fresh download and validation passed
for all eleven files. Ruff lint/format, 77 pytest cases, and
package builds passed. CI never downloads these research corpora.

## Label analysis

```sh
uv run --locked python -m finsight.data analyze-labels
```

This verifies pinned files and writes `data/interim/label-analysis.json` plus the
aggregate [label report](../reports/label-analysis.md). Use `--report-path PATH`
to choose the Markdown destination; `--manifest` and `--data-dir` work as above.
No network access or new dependencies are needed. Repeated runs produce the same
outputs for the same manifest and source bytes.

PhraseBank distributions are reported separately by agreement subset. SEntFiN
counts target annotations and reports mixed-label headline shares. FiQA uses only
original training scores, with mean, population standard deviation, inclusive
quartiles, fixed histogram bins, and numeric sign counts. No neutral band or
class conversion is chosen. Raw duplicate rows remain in these descriptive
counts; downstream deduplication and grouping decisions are still pending.

PhraseBank's neutral share (59.41% at >=50% agreement) means overall accuracy can
hide weak negative-class performance. Later evaluation should include macro F1
and per-class precision/recall alongside accuracy. SEntFiN's 43.31% mixed-label
share among multi-target headlines illustrates why a single headline label can
lose target-specific information. Higher PhraseBank agreement subsets remain
neutral-heavy; consensus filtering alone does not balance the classes. FiQA's
positive numeric skew is descriptive and does not determine a classification
threshold. These findings guide later development; no model settings are tuned.

Label-analysis verification: 86 pytest cases and Ruff lint/format checks pass.
Two repeated analyses produced identical aggregate reports; source-class counts
also matched the earlier real-artifact integrity audit. Tests cover percentage
and annotation denominators, absent classes, histogram endpoints, inclusive
quartiles, invalid scores, single-value inputs, and training-only FiQA totals.

## Text-length analysis

```sh
uv run --locked python -m finsight.data analyze-lengths
```

The command verifies pinned files and writes `data/interim/length-analysis.json`
and the aggregate [length report](../reports/length-analysis.md). Use
`--report-path PATH` to choose the Markdown destination; the default is separate
from the label report. It works offline and adds no dependencies.

Character length counts original Unicode code points including whitespace.
Word length counts whitespace-separated units, not transformer subword tokens.
Summary statistics include mean, median, minimum, P90/P95/P99, maximum, word
histograms, and fixed <=3/>50/>100-word flags. Percentiles use inclusive linear
interpolation. Empty label groups explicitly have zero records and null length
statistics. No flags are used to filter or modify records.

PhraseBank agreement subsets and labels remain separate. SEntFiN overall counts
headlines once; per-label groups include each headline once for each distinct
label across targets. Mixed-label groups overlap. Its supplied `Words` field
differs from our counting rule for 88 headlines; both the original value and
original text remain unchanged. FiQA counts original training texts once each,
regardless of annotation count, and reports headlines/posts separately. Test
texts are not parsed for this analysis and no FiQA class thresholds are chosen.

The median word counts are 21 (PhraseBank >=50%), 10 (SEntFiN), 9 (FiQA training
headlines), and 13 (FiQA training posts). PhraseBank has 35 sentences over 50
words; the other sources have none. FiQA training posts have 22 texts of at most
three words. These deserve later contextual review, not automatic rejection.
PhraseBank >=50% class medians are 22/21/23 words for negative/neutral/positive;
SEntFiN label-group medians are all 10. These descriptive differences do not
establish model bias or label quality. Model truncation requires later tokenizer
measurements rather than a conversion from these word counts.

Length-analysis verification: 92 pytest cases, Ruff lint/format, and package
builds pass. Repeated real-artifact analyses produced identical reports; source
row counts and histogram totals were checked. Tests cover preserved whitespace,
Unicode code points, percentile interpolation, histogram/flag boundaries, empty
and single-row groups, SEntFiN label grouping, FiQA training-only totals, source
word-count differences, and separate CLI report destinations.

## Duplicate and annotation review

```sh
uv run --locked python -m finsight.data review-annotations
```

The command verifies pinned artifacts, writes detailed ignored
`data/interim/annotation-review.json`, and produces aggregate-only
[review Markdown](../reports/annotation-review.md). The local JSON includes source
text and contexts for inspection; keep it ignored. `--report-path PATH` can
choose an aggregate Markdown destination. No new dependencies or network calls
are required. No source rows are modified or removed.

Exact grouping uses the existing NFKC/casefold/whitespace identity plus SHA-256.
Gold comparisons require the same corpus, normalized text, and annotation unit:
whole sentence for PhraseBank, normalized target for SEntFiN, target/aspects/
snippets for scored FiQA. Absent test gold and different annotation units are not
conflicting labels. PhraseBank uses >=50% only for duplicate counts; overlapping
agreement subsets are metadata rather than additional examples.

Near-candidate screening uses exact Jaccard >=0.80 on sets of whitespace-word
trigrams for texts of at least six words. It retains numbers and punctuation,
collapses exact text identities before searching, and indexes shared trigrams
rather than comparing every pair. Numeric differences and gold differences on
shared contexts are review flags, never automatic merge or relabel decisions.
Fixed scope/thresholds are logged in the JSON; lexical screening cannot establish
complete semantic disjointness or model independence.

FiQA test inputs participate only in duplicate checks. Training-only indicators
cover missing/nonmatching snippets and abs(score)<=0.1; the latter is a descriptive
marker, not a neutral class band. A nonmatching snippet may be a paraphrase or
comment, so inspect it before concluding that the field is erroneous. See
[qualitative findings](annotation-review-findings.md) for completed inspection,
source-quality concerns, and recommendations for partition design/adjudication.

Review verification: 102 pytest cases, Ruff lint/format, and package builds pass.
The inverted-index near search matches a brute-force check on synthetic texts.
Tests cover deterministic grouping, retained numeric/punctuation distinctions,
same-context versus different-target conflicts, missing gold, signed numeric
zeros, nonliteral snippets, and an offline CLI workflow. Repeated real-artifact
reviews matched saved JSON/Markdown. The detailed review JSON is ignored by Git.
