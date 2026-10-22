# US-listed issuer directory — Phase 4

The resolver can now use a pinned SEC issuer/ticker/exchange snapshot in addition
to the reviewed five-company registry. This expands identity coverage without
training a new NER or sentiment model. The directory does not extract spans:
word-aware BERT proposes organizations, identity resolution checks the complete
span, and the saved RoBERTa scores eligible original target strings.

## Source and inventory

Source: [SEC association file](https://www.sec.gov/files/company_tickers_exchange.json),
retrieved 2026-10-21 at 04:36:16 UTC. The
[SEC access guidance](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data)
describes these associations and their accuracy/scope limitations.
[SEC dissemination policy](https://www.sec.gov/about/privacy-information)
permits copying website information; see the recorded decision in
[dataset source review](../../../docs/datasets.md).

| Attribute | Count |
|---|---:|
| Source association rows | 10,443 |
| Included listing associations | 7,699 |
| Included distinct CIK issuers | 6,071 |
| Issuers held for entity-type review | 531 |
| Remaining issuers eligible for unique matching | 5,540 |

**Attributes:** Source rows counts all source records, including excluded OTC/null
exchange records. Included listing associations counts distinct
(CIK, exchange, ticker) tuples on Nasdaq, NYSE, CBOE and TXSE. Distinct CIK issuers
counts filer identities, not share classes or US-domiciled businesses. Held issuers
have at least one conformed name containing the whole word ETF, Fund or Trust;
this conservative lexical hold is not an authoritative security classification.
Remaining issuers is included issuers minus held issuers; ambiguity and context
checks can still prevent a match. Listing and issuer counts use different units.

The source is 523,851 bytes. Raw SHA-256:
`6c29e2e3db973d4479875164603c2434f81ad22e14c9f1694999aca9cca246cb`.
Selected manifest SHA-256:
`d76a4e6fd5d7cc540b3ef6974ece2ca508b3fd16360e8d9eca7b94ddce9f528b`.
The public pointer is `configs/us-listed-directory.json`; raw bytes are ignored
local data. Retrieval time and extraction exchange policy are part of the hashed
manifest. Loading verifies both manifest and source bytes and rejects a different
exchange policy. No contact address is recorded in public artifacts.

## Resolution policy

1. Apply reviewed registry rules first, including Apple/Amazon context and Google
   parent-rollup policy. Directory lookup cannot bypass a registry abstention.
2. Accept unique complete SEC names after Unicode, case, spacing and punctuation
   normalization. Explicit cashtags/exchange tickers can match without context;
   bare tickers and names shortened only by legal suffixes require explicit
   `--company-context`. An ORG tag alone does not supply this context.
3. Keep name/ticker collisions ambiguous. Hold obvious ETF/Fund/Trust issuers.
   Unknown or ambiguous names receive no default sentiment prediction.
4. Use `sec-cik:<10-digit-CIK>` for new identities and preserve every associated
   listing. No primary ticker is guessed. Reuse a reviewed identity when its
   ticker associations identify the same issuer unambiguously.
5. The existing output status `verified_company` means an identity association
   under this policy. For SEC matches, the output explicitly states that this
   does not verify operating-company or security type.

## Invented-input checks

| Input | Explicit company context | Outcome | Listing associations |
|---|---|---|---:|
| Tesla, Inc. | No | Exact SEC name | 1 |
| Tesla | No | Unresolved: context required | 0 |
| Tesla | Yes | Suffix-normalized name | 1 |
| $GOOG | No | Exact ticker | 4 |
| JPMORGAN CHASE & CO | No | Exact SEC name | 9 |
| Verizon | Yes | Unresolved: unknown shortened name | 0 |
| Imaginary Example Corp. | Yes | Unresolved | 0 |

**Attributes:** Input is a whole proposed span or direct directory lookup, not a
labeled benchmark example. Explicit company context is a caller assertion.
Outcome records the resolution rule or abstention. Listing associations counts
all distinct exchange/ticker associations for a matched CIK, including securities
other than common shares; zero means no accepted identity in this lookup, not
that the company has no listings. Rows repeat issuers and must not be summed as
unique coverage. These checks measure behavior, not accuracy or NER recall.

Actual CPU word-aware BERT inference on an invented Tesla/Microsoft headline
successfully fed both spans through the resolver. Receipt:
[NER/resolver smoke](../verification/us-listed-directory-smoke.json). Reusable sentiment routing
was tested with a stub predictor; this task did not execute GPU sentiment inference.
[Machine-readable inventory and lookup checks](us-listed-directory.json).

## Limitations and next bounded review

This is current US exchange coverage, including foreign issuers and ADRs, not a
historical universe or verified operating-company master. Legal-name suffix
normalization cannot recover brands such as Verizon from Verizon Communications;
no fuzzy names or benchmark-derived aliases were introduced. NER can still miss
organizations or produce incorrect boundaries. Source omissions/stale associations,
fund holds and common-word ambiguities require review. The local SEC contact is
used only for acquisition; offline lookup never requests the network.

A next task can review US-news identity linking on a separately defined sample,
including brand aliases and entity types. No historical financial feature may
use this snapshot as evidence of membership at an earlier prediction cutoff.
No frozen sentiment test records were accessed, and no accuracy improvement is
claimed. Training, prediction and aggregation protocols remain unchanged.

## Verification

All 774 tests passed, including new synthetic checks for listing collisions,
context requirements, legacy ambiguity protection, multiple-listing preservation,
source/manifest tampering and verified-only sentiment routing. Ruff lint and
format checks pass excluding the existing scratch `notebooks/experiment.ipynb`;
the required full checks still report its nine existing lint errors and
formatting differences. The notebook was left untouched. `git diff --check`
passed. Five-company registry, benchmark freeze, deployment selection and
locked target-test results retain their recorded checksums.
[Verification receipt](../verification/us-listed-directory-verification.json).
