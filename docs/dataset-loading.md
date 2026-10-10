# Reviewed dataset acquisition and validation

Scope: Phase 1, Financial PhraseBank and SEntFiN only. FiQA is deferred while its
terms remain unresolved. No sentiment model, prediction function, or split is
implemented. The runtime still uses only Python's standard library.

## Reproduce from the repository root

```sh
uv sync --locked
uv run --locked python -m finsight.data acquire
uv run --locked python -m finsight.data validate
```

`acquire` downloads the six pinned archive/CSV/notice files listed in
`configs/asset-revisions.json`. It verifies sizes and SHA-256 before atomically
saving each file; SEntFiN CSVs also match their Git blob identities. A valid cache
is reused without network access. A corrupted cache fails rather than being
overwritten: inspect/remove the affected ignored artifact, then reacquire.
The checked-in manifest contains no source text or credentials and downloads no
model weights. A partial run can be rerun; completed verified files are reused.

Both commands accept `--manifest PATH` and `--data-dir PATH`. The default data
root comes from `FINSIGHT_DATA_DIR` (otherwise `data`). Run from the repository
root or pass the manifest's absolute path. For local environment overrides use
`uv run --locked --env-file .env python -m finsight.data ...`.

Files are stored below `data/raw/financial_phrasebank/` and `data/raw/sentfin/`.
The acquisition receipt is `data/raw/acquisition-receipt.json`; the validation
summary is `data/interim/dataset-integrity.json`. These are ignored artifacts.
The validator verifies every manifest file again before parsing; changing source
bytes prevents validation. Summary output contains aggregate counts, no headlines.

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
[SEntFiN asset review](sentfin-asset-review.md). Local noncommercial research is the
reviewed scope; preserve source notices. Dataset contents and public headline demos
are not included in this repository. This workflow does not settle underlying
publisher rights, commercial deployment rights, or FinBERT weight-license scope.

## Verification

Unit tests cover source preservation, restricted parsing, schema failures,
integrity mismatches, cache behavior, and rejected paths. The integration test
uses synthetic files to exercise acquisition, validation, duplicate counting,
and overlap reporting without remote downloads. Real-artifact validation is a
separate reproducible local check; a fresh-directory download and validation also
passed for all six pinned files. Ruff lint/format, 55 pytest cases, and package
builds passed. CI never downloads these research corpora.
