# Three-model probability calibration

This bounded Phase 3 experiment calibrates the nine saved selected-rate SEntFiN
validation runs: RoBERTa Base, FinBERT and DeBERTa-v3 Base, seeds 42/43/44.
It uses no GPU, model retraining, new assets or final-test records. Read the
[results](../../../reports/experiments/temperature-scaling.md) and
[aggregate JSON](../../../reports/experiments/temperature-scaling.json).

## Method and assessment

Temperature scaling learns one positive scalar T per trained model:
`p_calibrated[c] = softmax(log(p_original[c]) / T)`.
For T greater than one, confidence becomes softer; a positive scalar preserves
the class ordering. Thus accuracy, macro F1 and the confusion matrix stay the
same. The aim is to improve probability quality, not fix classification errors.
See [Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html).

`configs/temperature-scaling.json` was written before running this experiment.
It pins both completed studies and the freeze receipt. Minimize NLL using fixed
T bounds [0.05,20]. NLL is convex in inverse temperature beta=1/T; use its
monotone derivative for 80 bisection iterations, with explicit boundary checks.
A completely uniform input has constant objective and retains T=1. No optimizer
dependency is added. NLL is the fitting objective; ECE/Brier are assessment
metrics, not additional objectives used to search or choose temperatures.

Five shared `StratifiedGroupKFold` folds use seed 42 and the existing duplicate
group IDs. Scikit-learn is already pinned to 1.7.2. Groups never cross between
fitting and assessment. These 781 validation records have 781 distinct groups;
this rule also protects grouped inputs. Stratification is approximate subject
to grouping; per-fold class counts are logged. Parent train/validation/test
membership stays fixed. No transformer model is retrained within the folds.

For each trained model/seed, fit T on four folds and apply it only to the fifth.
Restore the original record order and compute NLL, unnormalized multiclass Brier
and 10-bin top-label ECE on all 781 out-of-fold probabilities. Also report ECE
with 5/15 bins. Mean/sample SD describe three individual model seeds; do not
average probabilities into an ensemble or pick the best seed. The before/after
comparison uses exactly the same examples and underlying trained checkpoint.

Separately fit T on all validation records and save it for a future locked final
evaluation. Do not report its in-sample improvement as evidence. Cross-fitted
temperatures assess the procedure; the full-validation temperature is the
candidate artifact that would accompany that exact trained checkpoint.

## Reproduction and input checks

From the repository root, with the reviewed private development assets and all
nine original `run.json`/`predictions.jsonl` pairs present:

```sh
uv sync --locked
uv run --locked python -m finsight.sentiment.temperature_scaling
uv run --locked --group eda python scripts/plot_temperature_scaling.py
```

Windows uses `python -m uv` before the same arguments. The command audits study
and receipt hashes, model/seed/rate, ordered validation membership, gold,
probability distributions, original metrics and split lineage. It writes
`reports/experiments/temperature-scaling.json` and the accompanying Markdown.
Use `--report <other-path.json>` to reproduce without replacing the published
report; `--data-dir` selects another reviewed local data root. No optional
transformer packages are needed. A public checkout alone lacks the private inputs.

Private files live in
`data/interim/temperature-scaling/<private-packet-sha>/`:

- `packet.json`: fold memberships and out-of-fold probabilities for all nine runs.
- `calibrators.json`: full-validation temperatures, class order, probability floor,
  bounds and source-run identities. A temperature belongs to one exact model/seed;
  it must not be attached to a different checkpoint.

The public report records both private-file hashes, the plan, source report and
prediction hashes, dependency lock, implementation hashes, environment and code
revision. Tests cover analytic optimization, boundaries, identity/argmax,
duplicate-group isolation, exact assessment coverage, held-out-label isolation,
input corruption, unchanged classification and private/public output separation.
The plot reads public aggregates only and compares before/after probability
metrics, with sample SD across seeds. Each calibrator adds one scalar; original
model architecture, parameter count, saved weights and classification behavior
are unchanged. Nine full-validation scalar calibrators are saved separately.

An independent SciPy optimization cross-check agrees with all 54 fitted
temperatures (45 cross-fitting fits plus nine full-validation fits) within
1e-6. All nine runs improve out-of-fold NLL, Brier and ECE; no fit reaches a
temperature bound or encounters a probability below the chosen floor.

Mac and Windows both pass Ruff lint/format and all 325 tests. Repeating locally
reproduces all numeric results and private hashes exactly. Windows reproduces
all 1,046 numeric aggregate values within 1e-12; tiny platform math differences
produce different private packet/calibrator hashes, so bitwise cross-platform
identity is not claimed. The exact published Mac packet and calibrators are
also retained in ignored Windows storage for subsequent evaluation against the
published hashes. Frozen-test bytes are unchanged on both machines.

## Limits and next task

Earlier learning-rate and epoch selection already used this validation cohort.
Cross-fitting prevents a calibrator from fitting its assessed labels, but does
not undo that model-selection reuse or earlier qualitative inspection. These are
exploratory development results, not an unbiased final generalization estimate.
Three seeds are not three independent datasets or evidence of significance.

Saved outputs contain FP32 probabilities, not raw logits. For positive values,
log probabilities recover logits up to an irrelevant additive constant. Floor
values below 1e-15 before taking logs; log counts for every fit. Rounding and
flooring lose information, so this is a calibration of the saved probabilities,
not exact recovery of original logits. No source probabilities are overwritten.

The next bounded task is to lock the model/seed/calibration comparison protocol
and resolve remaining Phase 3 tracking/scope decisions before the frozen-test
evaluation. Do not tune temperatures on that test or adjust models after looking
at its results. Company-targeted sentiment follows Phase 3; DeBERTa Large remains
deferred. This task alone does not close Phase 3.
