# Soft and democratic voting

This bounded Phase 3 experiment combines the existing selected-rate FinBERT,
RoBERTa Base and DeBERTa-v3 Base runs without new training or calibration fits.
See the [results](../reports/experiments/ensemble-voting.md) and
[aggregate JSON](../reports/experiments/ensemble-voting.json).

The subsequent [locked final-test comparison](../reports/experiments/phase3-final.md)
is complete. This document's results and unscored-test notes refer to the earlier
development voting study; its recipe and the supplementary lock remain unchanged.

## Rules fixed before analysis

`configs/ensemble-voting.json` fixes equal weights and matching seeds: one
ensemble each for 42/43/44, with three different models per ensemble. Do not
combine all nine runs or choose the best seed. Join by the audited identical
record order and gold labels, using the saved five-fold out-of-fold calibrated
member probabilities. Member calibration and voting assessment are separated;
earlier epoch/rate selection still used these validation records.

- **Soft voting:** average the three calibrated probabilities for each class,
  then choose the largest. Exact ties use negative/neutral/positive order.
- **Democratic/hard voting:** each member casts one label vote. A majority wins;
  when all three labels differ, return neutral. This tie rule was declared
  before results, not selected by validation performance.

The methods can disagree. Two weak positive votes can lose to one strong
negative distribution under soft voting, while positive wins hard voting.
Temperature scaling leaves each member's class prediction unchanged, so hard
voting also gives the same labels using raw member outputs. Hard-vote counts
remain counts; do not interpret 2/3 as calibrated confidence or report their
NLL/Brier/ECE as though they were predicted probabilities. Soft voting has
probability metrics, but averaging calibrated members does not guarantee a
calibrated ensemble. No ensemble temperature is fitted.

## Results and trade-off

On 781 single-target validation records, mean/sample SD across three matching
seeds: soft macro F1 0.8846 ± 0.0010, hard 0.8859 ± 0.0042, versus DeBERTa Base
0.8810 ± 0.0044. Gains are approximately 0.37/0.49 percentage points, not evidence
of statistically established superiority. Hard voting resolves 3/2/0 three-way
ties across seeds 42/43/44. No alternative tie rule or weights were searched.

Soft voting improves NLL from DeBERTa's 0.3679 to 0.3415 and Brier from 0.1907
to 0.1752, but ECE increases from 0.0186 to 0.0354. Thus it improves those proper
probability scores without improving every calibration diagnostic. Per-seed
class metrics, confusion matrices, confidence, reliability bins, ECE sensitivity,
member agreement and corrected/lost DeBERTa predictions are in the aggregate
report. A stratified qualitative headline review is not part of this task.

Each ensemble contains 418,556,937 total member parameters with zero learned
voting parameters. It reuses three existing checkpoint/tokenizer sets; no new
weights are saved. Serving requires three member inferences per seed. Sequential
loading can limit peak memory, but no new serving latency or throughput is
measured by reusing saved validation outputs. Small validation gains do not
establish that this cost is justified in a deployed application.

## Reproduce

From the repository root with the reviewed development data, all original
run/prediction lineage, the canonical calibration packet and selected manifests:

```sh
uv run --locked python -m finsight.sentiment.ensemble_voting
uv run --locked --group eda python scripts/plot_ensemble_voting.py
```

Windows uses `python -m uv` before the same arguments. No optional transformer
runtime or GPU is required. `--data-dir` changes the reviewed local data root;
`--report <other-path.json>` produces a separate aggregate comparison. The
public checkout alone does not contain private inputs.

The runner first checks the unchanged primary evaluation lock and additive
extension, calibration report/packet hashes and exact metadata/fold membership.
It audits each original raw run and recomputes adjusted probabilities using its
saved fold temperature (numeric tolerance 1e-12 for platform math differences).
This is applying a saved calibrator, not fitting one. Corrupted original or OOF
inputs fail before results are published. Output votes/probabilities stay ignored
under `data/interim/ensemble-voting/<predictions-sha>/predictions.json`; public
files record aggregates, source hashes, code revision, Python and dependency lock.

Tests cover arithmetic and label-vote differences, unanimous/majority decisions,
both tie rules, bad/misaligned probabilities/IDs/gold, private/public separation,
OOF bytes/membership/fold/value corruption, reproducibility and protocol hashes.

Verified: all 362 tests and required Ruff checks pass on Mac and Windows.
Repeated local aggregates/private vote hashes match exactly; Windows agrees
with all 700 numeric result values within 1e-12 and produces identical private
vote bytes. Independent NumPy/scikit-learn checks agree across all 2,343 example
triads, including macro F1, accuracy, NLL and all class decisions. The aggregate
figure was visually reviewed. Both protocol locks and all 65 original model
files reverify on Windows; no frozen record was decoded or scored.

## Supplementary final evaluation v2

The [v2 extension](../configs/final-evaluation-ensembles-v2.json) and
[its lock receipt](../configs/final-evaluation-ensembles-v2-lock.json) were written
before the development voting analysis. They bind the recipe to the original
primary plan. Both methods are predeclared supplementary comparisons; their
test inclusion is not chosen from these validation outcomes. Preserve the
individual-model comparison and all three seeds from v1.

For test inference, use the exact **full-validation** temperatures pinned by the
parent plan, never the out-of-fold temperatures used here. Reuse the parent's
raw/calibrated member predictions and compute votes; no new model fitting or
test-derived calibration. Use the parent's paired group bootstrap draws and
report ensemble mean macro F1 minus DeBERTa mean macro F1 for each method; soft
NLL minus calibrated DeBERTa NLL is secondary. Hard probability metrics remain
unavailable. Log voting overhead separately from member inference.

Verify both protocol locks without test inference:

```sh
uv run --locked python -m finsight.sentiment.ensemble_voting --verify-final-extension
```

The extension preserves original input/model identities and the original lock
bytes. It does not implement a final-test scorer or close Phase 3. Freeze the
future evaluation runner's source revision and execution receipt before any
test inference. No test metrics were computed; multi-target/PhraseBank tests and
DeBERTa Large remain deferred. All results above are reused-validation diagnostics.
