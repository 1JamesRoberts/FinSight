# Separate company type, issuer identity and sentiment routing

**Update:** Real CUDA inference and launchers are now verified on the GPU PC.
See the [GPU follow-up](../verification/company-mention-routing-gpu.md). Earlier stub-only
verification statements below describe the original local implementation task.

Phase 4 now permits sentiment toward a company-classified mention without
requiring a CIK/ticker link. This addresses a routing restriction exposed by the
US-company review. It does not improve BERT's extraction or automatically resolve
the four missed links. No reviewed news aliases were added, no new company-type
model was trained, and the existing SEC snapshot and five-company registry remain
unchanged. This is an interface/routing change, not an accuracy experiment.

## Implemented behavior

Resolver v3 adds `company_type` and `classification_reason`, while the CLI and
sentiment outputs also expose `identity_status`. A company can be recognized
under the existing identity policy, through ambiguous SEC company-issuer
candidates under explicit context, or through an exact `--confirmed-company`
caller assertion. An ordinary ORG tag/high NER score/`--company-context` alone
cannot classify an unknown name as a company. Unknown names remain uncertain.
The explicit confirmation supplies type only; it never bypasses Apple/Google
identity ambiguity or parent-roll-up rules. Known institutional-type conflicts
stay uncertain rather than overriding a rule.

The new `company_mention_sentiments` helper batches original headline/target
strings for company-classified occurrences. It skips other/uncertain mentions,
preserves null identities, and never groups unresolved names together. Even
linked mentions are returned per occurrence in this new path. Repeated equal
surfaces can receive identical pair-model predictions: this is a target-text
model, not an occurrence-position-conditioned model. It does not establish that
equal surface names denote the same company.

Legacy `status` and `verified_company_sentiments` retain their meaning for
historical identity-only studies/callers. The latter still excludes unlinked
names. Default NER CLI scoring uses the new helper and
`sentiment_policy: company_mentions_only`. Diagnostic `--score-all-candidates`
remains an explicit option and does not confirm company type or identity.

Each scored occurrence includes its original span, company type/evidence,
identity status, CIK/ticker/listings/checksums and `aggregation_eligible`. Only a
linked identity sets eligibility true. No cross-headline aggregation or stock
mapping is implemented; later code must enforce the identity gate. SEC
associations still do not independently verify operating-company/security type.

## Why target masking was not substituted

Masking controls the text fed to a sentiment model. It cannot determine whether
an ORG is a company, select a CIK, or make unrelated issuers safe to merge.
The prior [nine-run format comparison](target-format-comparison.md) retained the
original pair; target replacement reduced mean overall macro F1 by 0.86 percentage
points on its fixed development cohort. This is qualified development evidence,
not an independent superiority claim. The selected deployed checkpoint is trained
on original target/headline pairs, so introducing replacement at inference would
also change its input distribution. This task changes neither its configuration,
weights nor input encoding. A masking checkpoint would need its own explicit
selection/deployment contract and would still need identity handling later.

## Verification

New behavior tests cover confirmed/unlinked sentiment, null issuer fields,
aggregation ineligibility, unchanged identity ambiguity/parent policy, no substring
confirmation, conflicting institution evidence, ambiguous SEC issuer choices,
repeated unresolved names, empty detections and model-load skipping. Existing
identity-only helper and CLI diagnostic tests remain in place.

Actual CPU BERT inference on an invented Toyota/Microsoft headline feeds both
spans through the new resolver. A stub predictor verifies that both are routed
with the original text, Toyota remains unlinked/ineligible, and Microsoft remains
linked/eligible. [Smoke receipt](../verification/company-mention-routing-smoke.json). The stub
produces no real sentiment predictions. The saved sentiment weights were not
loaded and the GPU machine was not updated; no new end-to-end GPU sentiment run
is claimed. Existing predictor/encoding tests cover unchanged model input logic.

No news acquisition, new terms, model training, reference changes or frozen
sentiment access occurred. The caller's confirmation is explicit evidence, not
an automatic company classifier: names without reliable source/type evidence
still abstain. The earlier 20-headline results remain historical baseline results;
this task does not reuse them as a fresh confirmatory evaluation.

[Usage and field definitions](../../../docs/phase-04-target-sentiment/company-resolution.md).

All **795 tests pass**, including 11 new routing/classification behavior tests.
Full Ruff checks still flag only the existing scratch notebook (nine lint errors
and formatting); checks excluding it pass. `git diff --check` passes. Protected
registry, benchmark freeze, deployment selection, final target results and the
approved news reference retain their recorded checksums.
[Verification receipt](../verification/company-mention-routing-verification.json).
