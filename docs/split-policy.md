# Provisional sentiment split policy

Phase 1 prepares reversible filtering decisions and provisional partitions from
the pinned local research inputs. This describes the parent split. The subsequent
[evaluation contract](evaluation-contract.md) freezes curated SEntFiN candidates;
that committed receipt takes precedence for official evaluation and development.
The machine-readable policy is `configs/split-policy.json`; the measured output
is [the split preparation report](../reports/split-preparation.md).

## Reproduce

After the documented acquisition and validation steps, run:

```sh
uv run --locked python -m finsight.data prepare-splits
```

Use `--manifest`, `--data-dir`, `--split-policy`, and `--report-path` for explicit
paths. The command rechecks pinned raw bytes and recomputes the duplicate review;
it does not depend on a potentially stale cached review. Raw files stay unchanged.
The report and `data/interim/split-preparation.json` contain aggregate counts.

Detailed outputs are ignored local artifacts under
`data/processed/splits/<split-manifest-sha256>/`:

- `split-manifest.json`: policy, source manifest digest/revisions, every source
  decision, duplicate groups, near edges, lineage, and processed-record digest.
- `prepared-records.jsonl`: retained native records, partition, source references,
  group IDs, review reasons, and FiQA annotation eligibility flags.

The same inputs and policy yield the same bytes. Existing artifacts must match;
corruption fails without overwriting them. A changed seed or supported partition
weight configuration produces a separate directory. Algorithm changes must bump
the manifest's algorithm version. Unsupported policy changes fail explicitly.

## Filtering decisions

| Condition | Action |
| --- | --- |
| Equivalent normalized text and annotation contexts/values within one corpus | Keep the lexicographically first source reference; retain every equivalent source reference in lineage. |
| Conflicting gold for the same annotation context | Quarantine the whole connected duplicate group pending adjudication. |
| Near pair with different gold in a shared context | Conservatively quarantine the group pending review; similarity alone is not proof that either label is wrong. |
| Near pair without a gold conflict | Preserve both examples, assign the same partition. |
| Same SEntFiN text with different target coverage | Preserve distinct annotation sets, keep the group in training/validation. |
| Different labels for different targets in one headline | Preserve the complete headline and target annotations. |
| FiQA–SEntFiN or other cross-corpus overlap | Keep the connected group in training/validation. |
| FiQA missing or nonmatching snippet | Keep native text/targets/scores for research, flag the annotation, disable snippet supervision, keep the group in development. Scores remain unverified pending review. |
| Original FiQA test input without gold | Retain an unscored manifest entry, exclude from supervised partitions. Quarantine any scored records in its connected group. |

Quarantine and redundant-copy decisions remove rows from the processed view,
never from raw data. Adjudication is a later explicit review decision, not a
majority vote or automatic relabeling. FiQA continuous scores remain continuous;
no three-class thresholds are introduced. Only PhraseBank's >=50% variant is
used as the master collection; nested agreement subsets are not concatenated.

## Assignment and checks

Normalize text with Unicode NFKC, casefolding, and whitespace collapse, preserving
numbers and punctuation; hash the result with SHA-256. Connect exact text
identities across all corpora and all reviewed near edges transitively. The near
screen uses three-word shingles, at least six words, and Jaccard >=0.8. Target
annotations stay attached to headlines rather than being split independently.

Hash the fixed seed (42) and stable component ID into approximately 80% train,
10% validation, and 10% `test_candidate` groups. Development-only groups use
the train/validation weights renormalized to 80/90 and 10/90. Assignment is not
stratified; row counts need not match exact percentages. Report class counts
without searching seeds against test distribution. Input ordering cannot change
the representative, component IDs, or assignment.

Validation fails if a known text identity, connected group, or near edge spans
partitions, if a development-only record enters a test candidate, if an excluded
source enters processed records, or if representative lineage is inconsistent.
Tests use invented examples and require no downloaded data or network access.

## Limits and next task

These lexical checks cannot prove absence of semantic duplication or overlap with
unpublished model training corpora. PhraseBank candidates are unsuitable as
independent evidence for pretrained FinBERT. SEntFiN candidates remain provisional
until benchmark curation and an evaluation contract are complete. The class
distribution is imbalanced and the SEntFiN targets are broader than our five-company
MVP. FiQA source-copy provenance and suspect annotation limitations still apply.

The subsequent benchmark task selected and froze the qualified external SEntFiN
holdout with a target-level evaluation protocol. All provisional test candidates
remain outside training and tuning. `prepare-splits` verifies its parent fingerprint
against `configs/benchmark-freeze.json` and fails if inputs or seed change.
No models are trained or evaluated by this preparation command.
