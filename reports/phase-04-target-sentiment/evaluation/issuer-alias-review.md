# Reviewed issuer aliases and disjoint headline development review

**Update:** The user has now bulk-approved the fresh references. See the
[reviewed comparison](issuer-alias-review-reviewed.md). Pending-reference
statements below describe the preserved preparation-time run.

The requested Phase 4 implementation adds opt-in source-backed alias matching,
Co/Company legal suffix support, explicit parent policy and collision guards.
A new disjoint 20-headline packet was selected and manually drafted before NER.
Actual CPU BERT inference and four fixed resolver arms completed. **New user
reference review remains pending; coverage counts below are not accuracy.**
The original 20-headline sample, approved reference and reported results are
preserved. No sentiment model training/inference, target masking or frozen
sentiment access occurred in this task.

## Source review and policy

[Official Boeing](https://www.boeing.com/) identifies the company; the SEC name
is BOEING CO. Add Co/Company to removable trailing legal suffixes, preserving
collisions. Do not drop arbitrary business descriptors or use substring matching.
[Toyota's official statement](https://global.toyota/en/album/images/43839258/)
explicitly uses Toyota for Toyota Motor Corporation, supporting the reviewed
issuer alias to CIK 0001094517. Other Toyota group entities are distinct names.

[Dell's issuer-hosted filing](https://investors.delltechnologies.com/sec-filings/all-sec-filings/content/0001193125-26-271266/d159213d424b2.htm)
identifies Dell Inc. as an indirect subsidiary of Dell Technologies. Classify
Dell/Dell Inc. as company references under explicit company context, but require
`--allow-parent-rollup` before assigning current parent CIK 0001571996. This
corrects the overly broad earlier suggestion that Dell is simply a same-issuer
abbreviation. It does not change the earlier user-approved reference or infer
historical continuity.

`configs/us-company-aliases.json` stores four whole-name entries, source URLs,
kind, context policy and pinned SEC manifest. Exact normalized name collisions
with canonical/short names or tickers remain ambiguous; aliases never select the
first candidate. The existing registry still has priority, including Apple
context and Google roll-up gates. Catalog evidence is a curated factual decision,
not a new learned company-type classifier. The source-use record is in
[dataset decisions](../../../docs/datasets.md). No corporate page text/logo, extra
model weights or SEC contact details were acquired/copied.

## Fresh sample

Four namespace-zero headlines each were selected from Boeing, Dell, General
Motors, Microsoft and Toyota category pools, using the existing deterministic
ranker (seed 43). Prior page IDs and normalized headline titles were removed
before sampling. Dell/Toyota pools were newly acquired from the permitted
Wikinews API; the three control pools reuse retained response snapshots. The
headline pool is targeted at known failures and controls, not representative
US financial news. The sample can contain unnamed companies, institutions,
products, historical brands and multiple targets.

Attribution: Wikinews. [Copyright policy](https://en.wikinews.org/wiki/Wikinews:Copyright),
[CC BY 2.5](https://creativecommons.org/licenses/by/2.5/) and
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Titles are unchanged;
article dates/individual license epochs are not established. No article bodies,
images, publisher news content or market data were acquired.

Sample manifest:
`ad6bb8ddb3553db5412f480a27f0839a3d8254a3df684274a1dbd0e4fd04794b`.
Plan: `configs/issuer-alias-review-plan.json`. Source response/retrieval hashes,
previous sample hash, selection code and plan hash are recorded. Old ID/title
intersection is zero, which does not establish distinct underlying events,
unseen pretraining, independent annotation, or temporal evaluation validity.
Current CIKs do not establish historical listing or legal continuity.

## Actual detection and resolution counts

All methods receive the same 26 word-aware BERT ORG proposals on 20 headlines,
using the same pinned `dslim/bert-base-NER` weights, `first` grouping and explicit
company context. No new name is manually passed via `--confirmed-company`.

| Resolver arm | ORG proposals | Company-classified spans | Accepted CIK links | Company spans without CIK |
|---|---:|---:|---:|---:|
| Archived v3 (previous deployed code) | 26 | 8 | 8 | 0 |
| Current suffix fixes only | 26 | 12 | 12 | 0 |
| Reviewed aliases, no parent roll-up | 26 | 18 | 15 | 3 |
| Reviewed aliases, explicit parent roll-up | 26 | 18 | 18 | 0 |

**Attributes:** Resolver arm specifies the fixed identity/type policy after
identical NER extraction. ORG proposals counts detected occurrences across the
20 headlines; company-classified spans counts those accepted as company under
the policy, not independently correct mentions. Accepted CIK links counts
occurrences with a selected current issuer. Company spans without CIK is company
classification minus accepted links. Every linked span here is company-classified;
unknown/other proposals comprise ORG proposals minus company-classified spans.
Rows reuse the same examples and cannot be summed. Counts do not establish
precision, recall or F1. The final row changes the grouping policy and must not be
scored against a reference that disallows parent roll-up.

Suffix normalization adds four accepted links relative to archived v3. Reviewed
issuer aliases add three more, and the subsidiary alias recognizes three Dell
company mentions while withholding a parent CIK by default. Explicit parent
roll-up allows those three parent associations. These are additional decisions,
not yet verified correct links. No aliases/rules were changed after examining
this fresh inference output.

The archived v3 baseline runs from source release
`3be40da98596b9b065358bbc01bb529d04db1720e1d8b7080163362698517924`
in a separate Python process after all 140 source/evidence hashes are verified.
This keeps original suffix/name logic intact rather than calling new code and
labeling it the old baseline. Original weights/SEC source bytes are unchanged.
The `suffix_only` arm isolates normalization from reviewed aliases. Predictions
remain private, while [aggregate counts/provenance](issuer-alias-review.json)
include execution/source hashes. No GPU deployment was changed by this task.

## Reference review and next evaluation

Manual source-only assistant drafts were written before the first NER run and
remain pending for all 20 rows. Eight rows explicitly flag historical identity,
manufacturer/product-boundary or subsidiary issues. Review empty rows as well.
The default reference policy is current issuer with no implicit parent roll-up;
no CIK is assigned to a company merely because its current owner is listed.
This new reference policy is not an amendment to historical accepted labels.

[Review sheet and instructions](../../../docs/phase-04-target-sentiment/issuer-aliases.md).
After user approval/adjudication, the runner reports raw company-candidate
extraction, company-classification precision/recall/F1, and exact-span-plus-CIK
metrics for the primary three arms. The explicit roll-up arm remains a separate
coverage diagnostic unless a matching reference policy is reviewed. Abstentions
remain missed reference links; wrong issuers count as both extra and missed
links. Pending/uncertain references cannot pass the accuracy gate.

The new sample is a development diagnostic. Choices were informed by the old
20-headline errors; source-backed aliases were frozen before new inference but
there is no independent annotation yet. Do not claim an independent benchmark
or move to Phase 5 from these decision counts alone. Automatic company-type
recognition of unrelated unseen names, security-type adjudication, fresh approved
accuracy and any updated GPU deployment remain separate work.

## Verification

All **811 tests passed**, including 16 new checks covering alias context, issuer
collisions, held fund-name collisions, source/schema/snapshot errors, parent
roll-up, exact-name matching, fresh sample exclusions,
archived source tampering and pending-reference gates. Original registry/freeze/
deployment/final-target results and the approved first news reference retain
their hashes. Cached predictions independently reproduce all counts and identical
NER span membership across arms. Preparation reproduces the same sample manifest.
A final provenance run after formatting-only runner correction produced identical
prediction bytes; no alias policy was changed.

Required full Ruff checks still flag only the unchanged scratch notebook (nine
lint errors and formatting); lint/format pass excluding it. `git diff --check`
passes. [Verification receipt](../verification/issuer-alias-review-verification.json).
Implementation, selection and prediction-only comparison are complete; fresh
reference accuracy and GPU rollout remain open.
