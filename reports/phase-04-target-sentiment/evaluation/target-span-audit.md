# Native target-span alignment before the masking experiment

Prepared 2026-10-19. [Machine-readable audit](target-span-audit.json) ·
[Rules, examples and reproduction](../../../docs/phase-04-target-sentiment/training/target-format-study.md).

This is the first preparation task for comparing original target–headline pairs,
target markers and target replacement with RoBERTa. It uses all native SEntFiN
financial targets, not just the five registry companies. It performs no company
detection, tokenizer/model acquisition, sentiment inference or training. Frozen
test records are not opened. No improvement in generalization has been measured.

## Alignment and shared membership

| Measure | Train | Validation |
| --- | ---: | ---: |
| Original headlines | 8,598 | 1,053 |
| Original annotations | 11,547 | 1,410 |
| Annotations with literal boundary matches | 11,456 | 1,402 |
| No literal boundary match | 91 | 8 |
| Annotations overlapping another native target | 20 | 2 |
| Individually eligible annotations | 11,436 | 1,400 |
| Complete eligible headlines | 8,497 | 1,044 |
| Shared eligible annotations across all three views | 11,395 | 1,396 |
| Excluded headlines | 101 | 9 |
| Excluded annotations, including their headline siblings | 152 | 14 |
| Annotations with repeated literal matches, before exclusions | 19 | 3 |

What each measure means:

- **Train / Validation:** the existing training and validation partitions;
  alignment does not move examples between them.
- **Original headlines:** distinct source news headlines before this experiment's
  eligibility checks.
- **Original annotations:** target–sentiment label pairs attached to those
  headlines. One headline can contribute several annotations, one for each target.
- **Annotations with literal boundary matches:** annotations whose target text
  appears as a separate word or phrase in the headline. For example, `Acme`
  matches `Acme rose`, but not `Acmeish rose`. This row checks text presence only;
  it does not yet rule out overlapping target names.
- **No literal boundary match:** annotations whose target cannot be located by
  that rule. Spelling differences, plurals, attached possessives or target typos
  can cause this; it does not prove the entity is absent from the headline.
- **Annotations overlapping another native target:** annotations with at least
  one matched span sharing characters with another annotated target's span.
  For example, `Acme` occurs inside `Acme India`. These are flagged rather than
  guessing which identity to mark or replace.
- **Individually eligible annotations:** annotations that pass all character-span
  checks on their own: a literal match, no overlap with another native target,
  and no reserved marker strings already present in the original input.
- **Complete eligible headlines:** headlines where every target annotation is
  individually eligible.
- **Shared eligible annotations across all three views:** all annotations from
  complete eligible headlines. Original pairs, target markers and target
  replacement use this same membership for a fair comparison.
- **Excluded headlines:** original headlines that fail the complete-headline rule
  because at least one target annotation is ineligible.
- **Excluded annotations, including their headline siblings:** all annotations
  belonging to excluded headlines, including individually eligible annotations
  whose sibling target fails. This equals original minus shared annotations.
- **Annotations with repeated literal matches, before exclusions:** annotations
  whose target text appears more than once in the headline, counted before
  eligibility filtering. Each annotation is counted once, not once per occurrence.
  Repetition alone does not cause exclusion; all matched occurrences are transformed.

All targets within a retained headline must pass the alignment policy. This
explains why shared eligible annotations are fewer than individually eligible
annotations: a safely aligned target is also left out when another annotation
in its headline fails. These exclusions apply only to this new experiment view;
the source datasets, labels, existing models and prior results are unchanged.

No examples required case-insensitive-only matching; no reserved marker collisions
were found. The three transformations were dry-run on every retained annotation,
and each partition/variant's text digest is recorded in the JSON. Original native
target identities and annotation IDs are preserved in private membership.

## Retained label and cohort support

| Measure | Train | Validation |
| --- | ---: | ---: |
| Negative annotations | 3,049 | 379 |
| Neutral annotations | 4,326 | 518 |
| Positive annotations | 4,020 | 499 |
| Multi-target headline annotations | 5,158 | 622 |
| Mixed-label headline annotations | 2,234 | 283 |
| Exact native target seen in retained training | 11,395 | 1,134 |
| Exact native target unseen in retained training | 0 | 262 |

What each measure means:

- **Negative annotations:** retained target annotations with the original negative
  sentiment label.
- **Neutral annotations:** retained target annotations with the original neutral
  sentiment label.
- **Positive annotations:** retained target annotations with the original positive
  sentiment label. These three label rows sum to the shared eligible annotation
  count for each partition.
- **Multi-target headline annotations:** retained annotations from headlines
  with more than one native target. This counts annotations, not headlines:
  a headline with three targets contributes three annotations.
- **Mixed-label headline annotations:** retained annotations from headlines whose
  targets have different sentiment labels, such as positive for Acme and negative
  for Beta. This cohort is a subset of the multi-target cohort.
- **Exact native target seen in retained training:** retained annotations whose
  original target string also appears as an annotated target in the retained
  training subset. Merely appearing somewhere in a training headline is not enough.
- **Exact native target unseen in retained training:** retained annotations whose
  original target string never appears as an annotated target in that training
  subset. Training has zero by definition. Seen and unseen rows partition each
  split's retained annotations; different aliases can count as different strings.

Cohorts overlap. Seen/unseen is exact native target-string membership, not verified
company identity. All 283 original mixed-label validation annotations remain in
the character-aligned view. Labels and splits were not balanced or reassigned.

## Verification

All 660 pytest tests pass, including 17 new span/audit tests for boundaries,
repeated targets, nested-name exclusions, identity removal from both sequences,
complete-headline eligibility, label-independent selection and private output.
The real audit was also repeated with frozen-record paths explicitly forbidden;
all source/private hashes verify. The frozen public contract, final target-test
aggregate and saved RoBERTa deployment config retain their previous checksums.

Required full Ruff checks were run. Only the existing untracked
`notebooks/experiment.ipynb` fails: nine lint issues and formatting. Excluding
that scratch notebook, lint and format checks pass across all 322 files. The
notebook is unchanged.

## Review and limitations

The private review packet contains 120 headlines: 109 training and 11 validation.
It includes every excluded headline and every headline with repeated target text.
The assistant inspected these cases; this is not independent human adjudication
or a gold span benchmark. Main issues include:

- Singular native targets inside plural words, or names followed by an attached
  possessive `s`, fail strict word boundaries. The source text is not repaired.
- Some target strings have whitespace differences, truncations, typos or apparent
  annotation corruption. The existing native targets are retained unchanged in
  the source, rather than silently rewriting them for this study.
- Short targets can occur inside longer annotated entity or sector names. Marking
  every occurrence could confuse identities, so affected headlines are excluded.
- Repeated literal matches can appear in an instrument name or a geographic
  phrase as well as a direct entity mention. This remains a semantic limitation
  of literal masking, even when character alignment succeeds.

Replacing a name does not remove other clues to its identity. These span checks
also do not show that replacement helps sentiment or unseen-target generalization.
The eligible subset may be easier than excluded examples; claims must be limited
to the shared subset. Report original support alongside filtered results.

## Status and next step

Update 2026-10-20: the [tokenizer audit and training lock](target-tokenizer-audit.md)
are now verified, with no additional exclusions. The preparation-time status below
records the original character-audit boundary. The subsequent
[training comparison](../experiments/target-format-comparison.md) is now complete and reviewed.

Character-span alignment and transformation helpers are implemented and tested.
The existing saved predictor and training loop remain unchanged. **Training has
not started, and tokenizer eligibility/truncation is not yet verified.**

Next is the pinned RoBERTa tokenizer audit, followed by a locked, matched training
comparison using a newly trained original-pair baseline and the two transformed
views. The baseline must use the same subset; historical full-data metrics are
not a fair direct control. Do not use the consumed test cohort to choose a variant.
