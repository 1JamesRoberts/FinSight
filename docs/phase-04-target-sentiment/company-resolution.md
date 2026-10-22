# Conservative organization resolution

Opt-in resolver v4 alias matching is documented in [reviewed issuer aliases](issuer-aliases.md).
Its fresh reference is now user-approved and its GPU release verified; see
[updated GPU commands](issuer-alias-gpu.md). Earlier releases are preserved.

The updated pipeline is now verified on the GPU PC with real NER and sentiment
weights. Use the [GPU release launchers](prediction-gpu-release.md); the ordinary
checkout remains older. See the [real CUDA receipt/report](../../reports/phase-04-target-sentiment/verification/company-mention-routing-gpu.md).

This step follows NER and precedes default company-sentiment scoring. It does not
change the model weights or fix missing/badly bounded NER spans.
[Reviewed development evaluation](../../reports/phase-04-target-sentiment/experiments/organization-resolution.md).

## Company type and identity are separate

Resolver v3 reports `company_type` (`company`, `other_organization`, `uncertain`)
with a `classification_reason`. `identity_status` is `linked` or `unresolved`;
`company_id`, `sec_cik` and ticker fields remain null when no link is established.
The legacy `status` field below retains the earlier identity/type decision for
research compatibility. It is not the new sentiment-eligibility gate.

An accepted registry/directory identity supplies company-type evidence under
its existing source policy. Multiple SEC candidates can support company type
without selecting one issuer (explicit company context and no entity-type holds
required). Otherwise an exact detected name may be explicitly confirmed with
`--confirmed-company NAME` (repeatable). This is a caller assertion, not a new
trained company detector or an issuer alias; it cannot override identity rules.
Unknown names remain uncertain. A conflict with an institution rule remains
uncertain and is excluded from default scoring.

## Legacy resolution outcomes

- `verified_company`: the complete detected surface resolves through the existing
  reviewed five-company registry and current context/parent-rollup policy. This
  verifies a registry identity match, not independently the real-world truth of
  the headline or its company relevance.
- `other_organization`: the complete surface matches one of four general title
  patterns: named government department/ministry, named high/supreme/constitutional
  court, securities commission/markets authority, or named stock/commodity exchange.
  These are lexical type rules, not a complete or authoritative institution list.
  Exchanges are other organizations under the existing annotation taxonomy.
- `unresolved`: unknown company, unsupported acronym, ambiguous registry alias,
  unapproved parent rollup, or a span not covered by those rules. Unknown never
  means neutral sentiment or confirmed non-company.

No NER confidence threshold, sentiment score or reference annotation supplies
company context. For example Microsoft resolves; bare Apple remains ambiguous
without explicit caller context; Tesla and Infosys remain unsupported by the
five-company registry. A full `Ministry of Finance` title matches the institution
rule, while a short unfamiliar acronym remains unresolved. The resolver does not
infer aliases from the 50-headline benchmark, trim quarter markers, or split merged
names. Such changes would hide extraction errors and need separate evaluation.

`--company-context` and `--allow-parent-rollup` retain their existing meanings as
explicit caller policy assertions. Do not set them merely because NER emitted ORG.
Every result records the original span, status, reason, registry checksum and any
matched identity/ticker. The registry version/contents are unchanged.

## CLI behavior

Detection plus resolution works on the Mac with the existing local NER files:

```bash
uv run --locked python -m finsight.sentiment.organizations \
  --text 'Microsoft reported profits while Tesla reported losses.'
```

The response retains all raw `mentions` and adds `resolutions`. The expected
statuses for these complete name spans are Microsoft = `verified_company` and
Tesla = `unresolved`.

**`--with-sentiment` scores company-classified mentions**, returning
`company_sentiments` and `sentiment_policy: company_mentions_only`. Each eligible
occurrence receives its own output; unlinked same-name occurrences are never
merged into an issuer. The full headline and original target-first RoBERTa pair
encoding are preserved. Other/uncertain organizations stay visible and receive
no placeholder sentiment. With no company-classified mentions, loading RoBERTa
is skipped. Without additional confirmation or a directory, Tesla still remains
uncertain, so this command works without sentiment weights:

```bash
uv run --locked python -m finsight.sentiment.organizations \
  --text 'Tesla reported higher profits.' --with-sentiment
```

To intentionally reproduce the older raw-ORG diagnostic, opt in:

```bash
uv run --locked python -m finsight.sentiment.organizations \
  --text 'Microsoft reported profits while Tesla reported losses.' \
  --with-sentiment --score-all-candidates --device cuda
```

That mode requires saved sentiment weights, returns `candidate_sentiments`, marks
company identity unverified, and does not bypass the resolution metadata. It is a
raw-target diagnostic, not company verification. `--score-all-candidates` without
`--with-sentiment` is rejected. The standalone registry `predict --detect-companies`
interface retains its original policy and behavior.

## Python

`resolve_organizations(text, mentions, registry)` returns one decision for every
input mention without reading files/models/references. `verified_company_sentiments`
batches only verified companies against the existing predictor. The original
`organization_sentiments` helper remains the explicitly unfiltered diagnostic.
The resolver's code version is `organization-resolution-v1`; extraction remains
`organization-ner-v2` with word-aware first grouping.

## Development evaluation

```bash
uv run --locked python -m finsight.sentiment.experiments.organization_resolution
```

This verifies the reviewed aggregate/receipt, reference bytes, exact execution
cache and predictions before evaluating. It reuses all 50 reviewed headlines,
including empty ones. Decisions are made from spans and registry/rules before
joining reference types. There is no new dataset NER inference or sentiment run.
Cached mentions omit confidence scores, represented as null; resolution does not
use scores in either cached or live operation.

The report distinguishes identity/type decision coverage, exact typed precision
conditional on answering, and full company extraction recall/F1 including missed
or unresolved reference companies. Undefined precision with no decisions remains
null. It also reviews the original four extra spans and eight missed organizations
without changing predictions/references. All 30 proposals remain unresolved on
this sample: it has no registry-supported company names, and detected institutions
are acronyms outside the whole-title rules. This is zero decision coverage, not a
company-recognition improvement. A separately reviewed identity source is needed
for broader coverage; no such source is added in this task.

Source text, decisions and assistant review notes stay under ignored
`data/interim/organization-resolution/<provenance-sha256>/`. Public JSON exposes
aggregates and checksums only. The user-accepted assistant reference remains a
small development sample, not an independent final test collection.

## Score a confirmed company without a CIK

```bash
uv run --locked python -m finsight.sentiment.organizations \
  --text 'Toyota reported higher profits.' \
  --confirmed-company Toyota --with-sentiment --device cuda
```

Use the GPU machine with the saved deployment weights for actual sentiment.
`--confirmed-company` only classifies an exact detected whole name (case-insensitive,
outer whitespace ignored). It cannot create missing spans or match substrings.
`--company-context` is a general context assertion used during identity matching;
it does not confirm every unknown ORG as a company. Neither flag silently selects
a CIK or permits subsidiary roll-up.

The Toyota output can contain sentiment with `company_id`, `sec_cik` and
`primary_ticker` null, `identity_status: unresolved`, and
`aggregation_eligible: false`. Only a resolved identity makes a mention eligible
for later issuer-level grouping/stock mapping; eligibility is not a claim that
aggregation has been implemented. `identity_verification_scope` distinguishes
no identity, registry matches, and SEC associations without security-type proof.

Masking targets does not establish company type or identity. The reviewed
[format experiment](../../reports/phase-04-target-sentiment/experiments/target-format-comparison.md)
favored the original pair formulation; this change does not switch checkpoints
or inject masks at inference.
[Routing implementation and verification](../../reports/phase-04-target-sentiment/experiments/company-mention-routing.md).
