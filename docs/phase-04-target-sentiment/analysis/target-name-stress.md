# Synthetic company-name sensitivity check

This diagnostic reuses all nine selected checkpoints from the completed
target-format comparison. It changes a supplied native company target and its
literal headline mentions, then measures changes in predictions. No company
detector, new model training, calibration fitting or frozen test is involved.

## Fixed selection and inputs

`configs/target-name-stress-plan.json` fixes a manual allowlist of twelve native
target strings: Infosys, TCS, ONGC, Sun Pharma, SBI, BPCL, HUL, Tata Steel,
Tech Mahindra, ICICI Bank, Wipro and HDFC Bank. This is not a company detector or
an exhaustive entity list. Selection uses exact native targets from the already
locked, span-eligible validation subset, never model scores or error categories.

The resulting subset has 67 target annotations from 64 headlines. The assistant
inspected all selected contexts before inference; membership was not changed.
Some mentions are recommendations, affiliations or company relationships rather
than direct operating outcomes. This review does not create gold entity spans or
counterfactual sentiment labels.

For every annotation, replace all audited literal selected-target occurrences
with each of two fixed strings, `Cedar Group` and `Northstar Holdings`. Change the
requested target too. Keep every other character unchanged, including the other
native targets. Reject name collisions, missing/overlapping spans or failed
tokenization instead of guessing. No alias, product or coreference rewriting is
attempted. All formats share all 134 original/substitution comparisons per seed.

Then apply the exact input format on which each checkpoint was trained:

- Original pair: substituted target first, substituted headline second.
- Markers: substituted target first; mark its literal headline occurrences.
- Replacement: mask the substituted target in both sequences. This must recover
  the exact original masked input. It is a construction/inference control, not
  evidence of independently learned company-name generalization.

Before scoring, all original/substituted views are audited using the pinned
RoBERTa tokenizer. Every input fits in full at length 128; maximum lengths are
33/41/35 tokens for pair/markers/replacement respectively. No extra exclusion is
needed. The packet freezes company support, case-byte hash, source/dependency
hashes and token lengths before model output is examined.

## Scoring and interpretation

Reload each existing selected checkpoint with its original configuration and
verified artifact files. Use the label-free `TargetText` / `score_inputs` path,
with eval mode, inference mode and gradients disabled. The native dataset label
is not passed to the model and is not assumed valid for a synthetic replacement.

Replay original inputs and check their predicted classes and probabilities
against the checksummed saved validation cache (absolute probability tolerance
1e-5). Score original and substituted inputs separately, even when replacement
makes their texts identical. This can expose tiny batch-padding numerical changes.

Measure label flips, mean total variation (half the sum of absolute changes in
the three class probabilities), and the maximum absolute probability change.
For the identical-input replacement control, require zero flips and maximum
change at most 1e-5. This is numerical consistency, not a performance threshold.

No synthetic accuracy/F1, model selection or acceptance threshold for name
sensitivity is computed. Changing a real company name can change factual meaning;
remaining brand/product cues can reveal identity. Changes are sensitivity signals,
not automatically errors or proof of name bias. Two names and a small manually
scoped reused-validation subset cannot establish open-world company generalization.
Results do not alter the deployed checkpoint or the primary macro-F1 comparison.

## Code and private outputs

`finsight.sentiment.experiments.target_name_stress` provides preparation, literal
substitution and label-free probability comparisons. It reuses the existing
span/tokenizer/checkpoint/inference helpers. The public plan is separate from the
immutable training plan.

Prepare from the repository root with existing tokenizer files:

```bash
uv run --locked python -m finsight.sentiment.experiments.target_name_stress \
  --data-dir data \
  --tokenizer-dir data/interim/target-format-tokenizer/e2da8e2f811d1448a5b465c236feacd80ffbac7b
```

This prints the frozen private packet directory under
`data/interim/target-name-stress/<manifest-sha256>/`. Original headlines, synthetic
views, native lineage and row predictions stay there. The GPU execution uses a
byte-verified copy of this runner alongside the preserved original execution
source/dependencies, leaving that historical source snapshot unchanged. Run with
`--run <packet-directory> --root <source-root> --data-dir <primary-data>`.

See the [reviewed results](../../../reports/phase-04-target-sentiment/experiments/target-name-stress.md). Only aggregate
counts, probability diagnostics and provenance may be copied to public reports.
