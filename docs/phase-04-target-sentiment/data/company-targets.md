# Phase 4: company target registry

Verified 2026-10-14. This first Phase 4 increment maps **explicit target strings**
to the five project companies. It does not identify mentions in headlines,
assign sentiment, train models, or inspect reserved test records.

## Reviewed scope

| Stable company ID | Project ticker | Canonical name | Other recognized ticker |
| --- | --- | --- | --- |
| apple | AAPL | Apple Inc. | — |
| microsoft | MSFT | Microsoft Corporation | — |
| nvidia | NVDA | NVIDIA Corporation | — |
| amazon | AMZN | Amazon.com, Inc. | — |
| alphabet | GOOGL | Alphabet Inc. | GOOG |

The hand-curated registry is `configs/companies.json`, version `companies-v1`.
Each company stores source URLs; the explicit loader also records the SHA-256
of the exact configuration bytes. Registry version and digest accompany every
resolution, including ambiguous and unsupported results. This is a current
research scope mapping, not a historical security master or exchange lookup API.

Issuer identity/ticker references: [Apple FAQ](https://investor.apple.com/investor-relations/faq/default.aspx),
[Microsoft FAQ](https://www.microsoft.com/en-us/investor/FAQ),
[NVIDIA FAQ](https://investor.nvidia.com/investor-resources/faqs/),
[Amazon FAQ](https://ir.aboutamazon.com/faqs/default.aspx), and
[Alphabet issuer release](https://abc.xyz/assets/73/22/28498a0f4c399beecce2c43c9baf/2016-exchange-offer-press-release.pdf).
Google's subsidiary relationship is documented by the Alphabet release and the
[issuer's 2024 annual filing](https://d18rn0p25nwr6d.cloudfront.net/CIK-0001652044/228e3674-05ba-49db-a0c6-486e783dfe86.html).
These sources substantiate identities; alias normalization and ambiguity flags
are our conservative engineering policy, not rules supplied by the issuers.

## Resolution contract

`load_registry(path)` performs explicit local I/O. Imports and target resolution
perform no network requests, dataset loading, or filesystem writes. Frozen typed
records keep company identity separate from a sentiment model and future APIs.
No dependency or trained NER model is added.

`resolve_company(value, registry)` matches the **whole input target**. It returns:

- `resolved`: a company plus the match kind and observed ticker, when applicable.
- `ambiguous`: candidate company metadata and a reason; `company` remains `None`.
- `unsupported`: no company or candidates, for an empty or unknown target.

Unicode NFKC, case folding, whitespace collapsing, comma spacing and trailing
period normalization support controlled name variants. Internal periods remain:
`Amazon.com` is retained. Explicit bare tickers, `$TICKER`, and `NASDAQ:TICKER`
are accepted case-insensitively. No fuzzy, substring, product-name, domain-suffix,
exchange-suffix or free-text mention matching is performed.

Bare **Apple**, **Amazon**, and **Alphabet** require `company_context=True`.
That flag asserts external evidence that the target denotes the company—such as
an explicit user-selected identity or reviewed company annotation. It is not a
context classifier, and callers must not set it merely because an article is
financial. Unknown names remain unsupported even when the flag is true.

**Google** and **Google LLC** are recorded as subsidiary aliases, not legal-name
synonyms. They require the separate `allow_parent_rollup=True` choice before
mapping to the project-level Alphabet identity. The result preserves the original
input and `alias_kind='subsidiary'`. This grouping does not establish that a
subsidiary mention has a particular parent-company sentiment; future experiments
must preserve original targets and document their grouping policy.

`GOOG` and `GOOGL` resolve to the same company identity, but `matched_ticker`
preserves the observed symbol. `primary_ticker='GOOGL'` is our project convention;
callers must not treat it as evidence that `GOOG` denotes that share class or use
this company grouping to merge distinct security-price series.

Unknown aliases, products and other subsidiaries such as AWS, iPhone and YouTube
are deliberately outside this registry. An unsupported result describes registry
coverage, not a claim that the text contains no company or has neutral sentiment.
A whole headline such as `Apple beats Microsoft` is unsupported as a *target
string*. Finding its two mentions belongs to the next roadmap task.

## Use from the repository root

```python
from pathlib import Path

from finsight.sentiment.companies import load_registry, resolve_company

registry = load_registry(Path("configs/companies.json"))

resolved = resolve_company("$NVDA", registry)
assert resolved.status == "resolved"
assert resolved.company.primary_ticker == "NVDA"

ambiguous = resolve_company("Apple", registry)
assert ambiguous.status == "ambiguous"
assert ambiguous.company is None

selected = resolve_company("Apple", registry, company_context=True)
assert selected.company.company_id == "apple"

share = resolve_company("$GOOG", registry)
assert share.matched_ticker == "GOOG"
assert share.company.company_id == "alphabet"

related = resolve_company("Google", registry)
assert related.reason == "parent_rollup_required"
```

Configuration loading rejects unknown fields, unsupported alias kinds, invalid
booleans/tickers, empty names, repeated company IDs, and duplicate/colliding
registered names or symbols. Invalid configuration is an error; an unsupported
user target is an ordinary structured result. Do not mutate aliases based on
reserved test examples; review changes, increment the registry version, and log
its digest in later experiments.

## Verification and next task

39 synthetic behavior tests cover all five primary tickers, alternate ticker
preservation, explicit forms, ambiguous/common names, subsidiary grouping,
unsupported text and invalid configurations. Required Ruff lint/format checks and the full 429-test suite pass on Mac.
The mapping increment is verified independently of the deferred NER/model tasks.

The frozen Phase 1 `benchmark-policy.json` coverage names and all freeze receipts
remain unchanged. The released SEntFiN alias table still preserves source strings
without automatically linking them to these five US-company identities. No
reserved single- or multi-target test headlines were inspected or scored.

Only **Build ticker/name/alias mapping for initial five companies** is complete.
Next: implement mention identification with spans and conservative unresolved
cases, then audit development-only target coverage and lock the target-aware
experiment. Target-label collection, target-aware training, independent target
evaluation and `predict(text, target_company)` remain pending.

The subsequent [development coverage audit](company-coverage.md) now verifies
registry-scoped target counts, class balance, literal-name candidates and private
identity review. It does not implement mention extraction or the sentiment API.
