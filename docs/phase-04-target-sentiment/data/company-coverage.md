# Phase 4 development coverage: method and findings

Verified 2026-10-14. This is a descriptive audit of existing SEntFiN training and
validation records, before constructing any target-aware training view. Full
aggregate tables are in [the report](../../../reports/company-coverage.md), with
machine-readable counts and lineage in [JSON](../../../reports/company-coverage.json).

## Main finding

SEntFiN supports a broad financial target-sentiment experiment, but has too little
coverage to train only on, or reliably evaluate, the five application companies.
The following counts are **unique headlines with a registry-matched target
annotation**, not every literal name occurrence in the text:

| Company | Training headlines | Validation headlines | Training labels: neg / neu / pos | Validation labels: neg / neu / pos |
| --- | ---: | ---: | --- | --- |
| AAPL | 21 | 0 | 11 / 6 / 4 | 0 / 0 / 0 |
| MSFT | 4 | 2 | 1 / 2 / 1 | 1 / 0 / 1 |
| NVDA | 0 | 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| AMZN | 2 | 1 | 1 / 1 / 0 | 0 / 1 / 0 |
| GOOGL: direct Alphabet identity | 0 | 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Optional Google-to-Alphabet grouping | 1 | 0 | 1 / 0 / 0 | 0 / 0 / 0 |

Each company has one annotation per matched headline here. Company rows overlap:
strict matching covers **26 distinct training headlines and two validation
headlines**, with 27/3 target annotations. Optional Google grouping adds one
training headline/annotation: **27 training headlines and two validation
headlines**, with 28/3 target annotations. Adding per-company headline counts
therefore overcounts distinct source headlines.

AAPL has eight multi-target training headlines, MSFT three, and AMZN one; these
counts overlap. Validation has one multi-target MSFT headline and one AMZN
headline, referring to the same source headline. Optional Google coverage is
single-target. None of the five-company matches has a same-company label conflict
or lacks both its literal annotated target and registered aliases.

These are registry-scoped counts, not exhaustive entity resolution. The five
company identities are application targets, not a restriction on the broader
SEntFiN training cohort. No new per-company metric is meaningful from these
validation supports: three companies have no examples, and the remaining two
have only two and one examples. Do not report absent-company performance as zero
F1, silently resplit the data, or fill labels from mention matches.

## Broader target-aware data available

| Split | Headlines | Target annotations | Single-target headlines | Multi-target headlines | Headlines with mixed target labels |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 8,598 | 11,547 | 6,299 | 2,299 | 994 |
| Validation | 1,053 | 1,410 | 781 | 272 | 122 |

The previous sentence-model training used only the single-target rows. The
broader view contains **12,957 target annotations**, including multi-target
examples useful for testing whether different targets receive different labels.
These are potential target/text examples before subsequent eligibility review;
this audit does not construct, filter, or train that view. Source entities are
not all independently verified legal companies, and benchmark geography/style
remain limitations for the five-company US application.

## Fixed audit rules

`configs/company-coverage.json` pins `companies-v1`, the freeze receipt and both
views before execution at code revision
`e45324f`. The implementation is `finsight.data.company_coverage`.

- Load only the verified development records returned by the existing freeze-aware
  loader. Reject reserved partitions before selecting SEntFiN. Verify unique
  record IDs and that duplicate groups do not cross train/validation.
- Resolve the provided target strings as financial-entity context. Strict matching
  excludes subsidiary grouping. The separately labeled optional view permits the
  registered Google-to-Alphabet mapping; no other subsidiaries are promoted.
- Count source target annotations and unique headline IDs separately. Preserve
  every original label and target. Multi-target/mixed-label counts refer to the
  original complete headline annotation list, including entities outside the five.
- Screen literals with Unicode NFKC, case folding, comma spacing, whitespace
  collapsing and escaped whole-token boundaries. Ignore trailing target periods,
  retain internal punctuation, and use no fuzzy matching. This provides no
  semantic identity inference, coreference resolution or character-span API.
- Distinguish literal target matches, registered-alias-only matches and absence
  of both. A short target string can match an ordinary word; literal presence is
  not evidence that a name is semantically the intended company.

A **literal candidate headline** contains any allowed registered name/ticker.
An **unannotated literal candidate** has no target annotation resolving to that
registry company in that view; it can still have another annotated entity or an
unresolved target variant. Candidates never become new labeled examples.

## Ambiguity and nonliteral review

Strict matching leaves 11,519/1,407 train/validation annotations outside the five
company registry. Most represent other financial entities; these are coverage
limits, not errors. One training annotation awaits the explicit Google parent
policy. Two other training targets contain registered-name cues: a regional
Microsoft reference, and an import-related Apple name. They remain unsupported;
neither is silently converted into a canonical company label. The latter does
not provide sufficient company identity from the headline alone.

There are **87 training and eight validation nonliteral target annotations**,
95 of 12,957 (about 0.73%). Aliases, abbreviations and implicit references may
explain these; this diagnostic does not establish missing company mentions or
incorrect sentiment. Preserve them for review rather than dropping automatically.
In total, 98 private review flags cover nonliteral targets, the parent-policy case
and unsupported name cues. Raw IDs, targets and labels stay in ignored storage.

Manual review of the 29 grouped-view headline contexts supported the intended
company identities. This reviewed identity context only, not the correctness of
sentiment labels. The two unsupported cue cases were inspected separately, without
expanding the registry. No headlines, source IDs or per-record labels are published.

## Verification and artifacts

Execution on Mac and Windows independently produces **byte-identical aggregate
JSON and Markdown**. All 20 company/split/view annotation, headline and class
count cells were also verified using direct normalized-name counting independent
of the resolver. Count identities, private output checksums and the unchanged
original final aggregate were verified. Eleven synthetic behavior tests cover
count units, parent grouping conflicts, unsupported cues, literal boundaries,
malformed annotations, duplicate IDs, cross-partition groups and reserved inputs.
Required Ruff checks and **440 tests pass** on both platforms.

Aggregate SHA-256:
`da303fc3ae9ff03b4682a47a3f77630a2caf5e20da1bcd49275d56dec9a68875`.
Private records on both machines are under
`data/interim/company-coverage/<aggregate-sha>/aggregate.json` and `review.jsonl`.
The manual identity review is `manual-review.json`, SHA-256
`0157071a9d2c2483632c7acd19e90a010af4c2143c4e8bd47b3b381063ec14bd`.

To reproduce the audit, use the clean execution revision in a separate checkout
with access to the reviewed local data view and no existing output reports:

```sh
uv run --locked python -m finsight.data.company_coverage --data-dir /path/to/data
```

The runner refuses existing public outputs. Keep the saved reports for routine
inspection. Identical execution with archived public outputs may also encounter
an existing immutable private directory; use a separate reviewed data workspace
for deliberate reruns. No new dependency, data acquisition, split assignment,
training, prediction, relabeling, alias expansion or test analysis is performed.
Reserved bytes are checksummed for integrity; reserved records are not selected
or scored. Frozen policies, receipts and final evaluation results stay unchanged.

## Next task

Define the target-pair development view and eligibility policy on the broader
SEntFiN entity set, preserving headline/duplicate groups and native targets.
Nonliteral targets need not be excluded for target-plus-text inputs, but span
masking would require separate alignment decisions. Keep the reserved multi-target
benchmark untouched until the protocol is fixed. A licensed, separately labeled
collection will be needed before making credible performance claims for the five
application companies. No such collection is acquired in this task.
