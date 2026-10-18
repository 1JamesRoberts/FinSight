# Phase 4 target-error review

This validation-only review compares the saved matched headline and target-pair
predictions for DeBERTa Base, RoBERTa Base and FinBERT, seeds 42/43/44. It checks
18 prediction receipts covering the same 1,410 native financial target annotations
from 1,053 SEntFiN headlines. No model weights are loaded and no inference,
training, calibration fitting or reserved-test analysis occurs.

The [aggregate report](../../../reports/experiments/target-error-analysis.md) contains
shared errors, recurrence across seeds, matched conditioning fixes/harms,
pairwise disagreements, overlapping qualitative categories and limitations.
The companion JSON retains all counts for seven overlapping cohorts, provenance,
source-report identities, environment and hashes of private review artifacts.
The code rechecks 108 saved classification/calibration metric sets with a 1e-12
absolute arithmetic tolerance; native rows, labels and prediction hashes must
match exactly. Artifact manifests are checked locally; full weight bytes were
verified during the earlier Windows training and are not reread here.

## Method

Each seed remains a separate trained run. No votes, probability averaging or
ensemble predictions are created. A recurring error is wrong in all three paired
seeds. A persistent conditioning fix requires all three headline seeds wrong and
all three paired seeds correct for one backbone; persistent harm reverses that.
Changing wrong labels still counts as prediction instability.

The fixed purposive sample takes up to five annotations per stratum, with distinct
headline duplicate groups within each stratum. Candidates are ordered by
SHA-256(`target-review-v1:42:example_id`). Deduplication across strata preserves
every selection reason. Strata cover shared nine-run failures, stable inter-model
disagreement, seed instability, persistent fixes/harms, mixed-label errors,
comparison-regex candidates, nonliteral-target errors and neutral-boundary errors.
The comparison regex only selects candidates; a contrastive conjunction does not
automatically receive a comparison tag. Nonliteral matching reuses the earlier
whole-token coverage diagnostic and does not resolve aliases or company identity.

The final sample has 36 annotations from 36 headlines. Reviewers see the focal
native target/gold, sibling target labels, predictions and confidences from every
seed in both arms. Multiple tags are allowed: target binding, alias/nonliteral
form, comparison, neutral boundary, conflicting cues, financial context, event
interpretation and insufficient context. Tag counts overlap and describe apparent
difficulties, not proven causal model mechanisms. Notes were individually reviewed
by Codex; they are not independent human adjudication.

## Local files and reproduction

The existing verified private target view, saved prediction receipts and selected
artifact manifests must be present. This is a CPU analysis using existing locked
dependencies; no new packages or sources are acquired.

```bash
uv run --locked python -m finsight.sentiment.target_error_analysis
```

This verifies the inputs and writes the private packet/counts under
`data/interim/target-error-review/<packet-sha256>/`. Native text, names, IDs,
row predictions and reasoning stay there, outside Git. The checked-in aggregate
records the packet and annotation hashes; the initial reviewed notes are retained
privately as `annotations.json` in that directory.

For initial publication, the runner accepts `--annotations <private-json-path>`.
Each entry requires an `example_id`, nonempty `categories` list from the fixed
vocabulary, Boolean `gold_ambiguous`, and nonempty `note`. Coverage must match the
exact sampled IDs once. The runner refuses to overwrite an existing public report.
Independent human notes should be retained separately, with any public revision
handled as a new version rather than silently replacing the published review.

To verify published counts and notes without overwriting them:

```python
import json
from pathlib import Path
from finsight.sentiment.target_error_analysis import (
    prepare,
    render_report,
    summarize_annotations,
)

report = json.loads(Path("reports/experiments/target-error-analysis.json").read_bytes())
counts, packet = prepare(Path("data"))
root = (
    Path("data/interim/target-error-review")
    / report["qualitative_review"]["packet_sha256"]
)
annotations = json.loads((root / "annotations.json").read_bytes())
assert counts["cohorts"] == report["cohorts"]
assert summarize_annotations(packet, annotations) == report["qualitative_review"]
assert (
    render_report(report)
    == Path("reports/experiments/target-error-analysis.md").read_text()
)
```

## Verified findings and boundaries (2026-10-17)

All nine company-aware runs correctly classify 1,117 annotations and miss 51;
at least one run misses 293. Of the shared misses, 39 are single-target, 12
multi-target and five mixed-label; the latter two cohorts overlap.
DeBERTa/RoBERTa/FinBERT have 112/101/106 annotations wrong in all three seeds,
and 92/118/118 whose predicted labels vary. Stability is not correctness.
Persistent conditioning fixes are 74/67/79 and harms 5/9/6, respectively; every
matched model/seed corrects more examples than it harms.

The manual sample highlights actor/source attribution, factual-versus-evaluative
neutrality, financial advice, relative-versus-absolute performance and temporal
reversals. Some clear target-attribution fixes coexist with introduced sentiment
on factual mentions. Nonliteral forms are often plural variations; errors in both
formats do not establish an alias-specific failure. An abbreviation may also
leave entity identity unresolved even when the financial event has a clear sign.

Eighteen selected cases have suspected gold/context ambiguity. This number is
not an annotation-error rate: the sample deliberately targets difficult cases,
judgments are exploratory, and the underlying annotation convention may differ
from the reviewer's interpretation. Model agreement does not establish gold truth.
All original labels and benchmark membership remain unchanged. Any future label
repairs need an explicit rubric, independent review and separate versioning.

All 509 tests and required Ruff gates pass on the Mac. Independent NumPy
recalculation verifies integer counts in all seven cohorts, including per-seed
transitions, error recurrence and pairwise model disagreements. All 36 private
reviews, their exact ID coverage, hashes and aggregate-only privacy boundary also
verify. Original model/target plans, data, source comparisons and Phase 3 final
aggregate are unchanged. The unrelated pre-existing notebook edit is retained;
its presence explains the analysis receipt's dirty-working-tree flag. Execution
uses committed analysis source, recorded by code revision and implementation hash.

Only the Phase 4 error-tracking task is complete. Model selection and the
reserved-target evaluation contract remain next; automatic mention recognition,
reliable per-company evaluation and the prediction API remain open. This review
does not establish an independent test winner or five-company generalization.
