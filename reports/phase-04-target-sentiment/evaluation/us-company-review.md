# US-company extraction and linking — development review

**Update:** The user has now bulk-approved the drafts. See the
[reviewed comparison](us-company-review-reviewed.md). The preparation-time
status/counts below are retained as historical evidence.

**Status: sample and runner prepared; reference review and accuracy evaluation
pending.** The user authorized a small US-news diagnostic after the SEC directory
integration. The existing NER model and resolver were run without changing their
rules, aliases, thresholds or weights. No sentiment models were trained/scored.

## Source and selection

English Wikinews provides explicitly reusable text. The
[official copyright policy](https://en.wikinews.org/wiki/Wikinews:Copyright)
assigns CC BY 4.0 to text published after December 16, 2024, CC BY 2.5 to material
between September 25, 2005 and December 15, 2024, and public domain earlier.
Attribution: Wikinews; [CC BY 2.5](https://creativecommons.org/licenses/by/2.5/),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Titles are unchanged;
article publication/revision dates and individual license epochs were not
established. No article bodies, images, third-party publisher text or market data
were acquired. The source-use decision is recorded in [datasets](../../../docs/datasets.md).

Eight namespace-zero category API responses were retained locally on 2026-10-21.
Ford Motor Company, JPMorgan Chase and ExxonMobil returned no articles; their
categories were excluded before any predictions. Each remaining category supplied
four unique titles, ranked by SHA-256 of experiment/seed 42/page identity. Shared
pages belong to the first category alphabetically; colon titles/roundups are
excluded. The frozen sample was selected before assistant drafting and inference.
The prior Kaggle shortlist was not downloaded.

| Sampling category | Selected headlines |
|---|---:|
| Apple Inc. | 4 |
| Boeing | 4 |
| General Motors | 4 |
| Microsoft | 4 |
| Tesla Motors | 4 |
| Total | 20 |

**Attributes:** Sampling category is Wikinews source metadata, not a company
mention or CIK label. Selected headlines counts distinct page IDs after shared
page ownership/deduplication. Categories are disjoint in the selected sample;
the total is their sum. A categorized article may contain no explicit company
name in its headline. Source categories, query URLs, response hashes and retrieval
times are saved in the private manifest. This is a historical, technology-heavy
corporate-news convenience sample, not representative US financial news.

Plan: `configs/us-company-review-plan.json`. Frozen sample manifest:
`0712af66f547a820e4b6a1ac039d613e7239bea4a51ef45edb3cab251cf44fbc`.
The packet manifest also hashes the exact sample, plan and sampling implementation.

## Assistant reference draft — not approved labels

Source-only manual drafts contain 15 company mentions, two other-organization
mentions, and seven headlines with no named organizations. Five headlines include
explicit uncertainty about historical identity, brands or organizational units.
Every row is pending, including unflagged/empty rows. CIK drafts were manually
cross-checked against SEC conformed names; they are not independently sourced
identity gold and need reviewer confirmation. Human approval is not inferred.

The [review instructions](../../../docs/phase-04-target-sentiment/data/us-company-review.md)
link the local 20-row sheet, source URLs and editable JSONL. The reader sees
reference drafts without detector outputs. Important decisions include historical
Chrysler/Saturn/Yahoo brands, current Dell/GM versus historical continuity,
Microsoft Research Asia as a lab, and product/person names that imply companies
without explicitly naming them. No parent roll-up is enabled.

## Actual prediction-only run

Word-aware BERT NER (`dslim/bert-base-NER`, revision
`d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc`) ran on CPU with `first` aggregation.
Both methods receive the same raw NER spans, explicit company context, and the
same reviewed registry; the second adds the previously pinned SEC snapshot.
Legacy registry matches are mapped to CIKs only through unique directory ticker
associations for evaluation bookkeeping; this never accepts an abstention.

| Method | Proposed organization spans | Accepted CIK links | Spans without accepted CIK |
|---|---:|---:|---:|
| NER + five-company registry | 21 | 5 | 16 |
| NER + registry + SEC directory | 21 | 8 | 13 |

**Attributes:** Method specifies identity resolution after identical NER proposals.
Proposed organization spans counts detected occurrences across 20 headlines,
including non-company/error proposals. Accepted CIK links counts occurrences
accepted under the fixed policy, not unique issuers or correct predictions. Spans
without accepted CIK is proposals minus accepted links; it includes unresolved
names rather than verified negative companies. The two methods reuse the same
headlines/spans, so rows overlap and must not be added. The three extra accepted
links demonstrate increased decision coverage, not improved accuracy. No
precision/recall/F1 was calculated against pending drafts.

The broad resolver accepted five registry aliases, one exact ticker and two
suffix-normalized names. Thirteen spans remained unknown to its name policy.
No aliases were added in response to this run. The SEC snapshot is
`d76a4e6fd5d7cc540b3ef6974ece2ca508b3fd16360e8d9eca7b94ddce9f528b`.
[Aggregate run/provenance](us-company-review.json) retains source/model/plan hashes;
per-headline predictions stay in ignored data.

## What remains

After the user reviews all references and resolves uncertainties, run the gated
metrics for both fixed methods. Exact company spans and exact-span-plus-CIK
metrics are separate; wrong issuer links count as both extra and missed links,
and abstentions cannot inflate recall. Review false positives, missed spans,
unknown legal/brand names and unsupported types before proposing changes.
Any rule changes based on this sample make it development evidence, not a final
confirmatory benchmark. Current listings cannot prove historical membership;
unknown pretraining exposure also prevents claims of independent extraction
accuracy. Do not mark the independent entity-labeled test collection complete.

## Verification

All 784 tests pass, including ten new tests for deterministic/category-balanced
sampling, deduplication, prediction independence, quota failure, review gates,
CIK/span validation, wrong-link and abstention accounting, empty rows and packet
tampering. Re-running preparation reproduces the identical sample manifest.
Actual CPU inference completed for all 20 headlines. The original registry,
benchmark freeze, deployment selection and locked target-test result checksums
remain unchanged. Required full Ruff checks still flag only the unchanged scratch
`notebooks/experiment.ipynb` (nine lint errors and formatting); both checks pass
when excluding it. `git diff --check` passes.
[Verification receipt](../verification/us-company-review-verification.json).

Only packet preparation, prediction-only execution and runner verification are
complete. Human adjudication and reference accuracy remain open in the roadmap.
