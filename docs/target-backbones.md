# Matched target conditioning: RoBERTa Base and FinBERT

This Phase 4 supplement tests whether the company–headline representation helps
across backbones. DeBERTa was tried first because it was the selected individual
backbone from Phase 3; that does not establish it as the best target-conditioned
model. RoBERTa and FinBERT provide direct checks before further model selection.
This study reuses the exact native SEntFiN target view verified by
`target_control.verified_inputs`: 11,547 training annotations and 1,410 validation
annotations. No new split, PhraseBank combination, masking, aliases or five-company
filter is introduced. The original DeBERTa plan, preparation lock and reports stay
unchanged. Reserved test examples cannot enter this runner.

`configs/target-backbones-plan.json` freezes source revisions/file identities,
three seeds (42/43/44), both input arms, prior selected learning rates, and lineage
before execution. Original RoBERTa Base receives a seeded new head. Original
ProsusAI FinBERT retains its pretrained financial sentiment head; its output rows
are permuted to negative/neutral/positive without changing named probabilities.
Neither starts from an earlier project-adapted checkpoint.

Each model uses headline-only and native tokenizer-paired target/headline input,
with matched initial weights, RNG state, training examples and common epoch row
orders within each seed. BERT uses `[CLS] target [SEP] headline [SEP]`; RoBERTa
uses `<s> target </s></s> headline </s>`. Special tokens are inserted by the
tokenizer. Right padding and `only_second` truncation preserve the full target;
insufficient target space fails. Both input formats receive a tokenizer audit.

All parameters are updated using the existing tested target trainer: FP32,
deterministic eager attention, batch 16/32, maximum five epochs, patience two,
length 128, AdamW, 10% warmup and linear decay. Previous Phase 3 learning rates
are reused (RoBERTa 2e-5, FinBERT 3e-5; DeBERTa used 1e-5). There is no new
search. Strict all-target validation macro F1 selects a checkpoint, with earliest
exact ties. Every seed is reported. Raw probabilities are not recalibrated.

## Running locally on the configured GPU computer

Install the locked environment and transformer extra from the project root:

```powershell
uv sync --locked --extra transformer
uv run --locked --extra transformer python -m finsight.sentiment.target_backbones
```

The pinned source snapshots and verified private dataset view must already be
present. This command performs no acquisition. It verifies source bytes, trains
12 trials, saves/reloads selected checkpoints and writes the aggregate report.
A single trial can be run with:

```powershell
uv run --locked --extra transformer python -m finsight.sentiment.target_backbones --model finbert --arm target_pair --seed 42
```

`--model` chooses the original backbone, `--arm` chooses the input format, and
`--seed` controls initialization, dropout and training shuffle. All three options
are required together. Completed trials are reused only after checking native
membership, prediction bytes, source/configuration/context and every checkpoint
file. A completed aggregate is never silently overwritten.

Aggregate outputs: `reports/experiments/target-backbones.json` and `.md`; per-trial
reports: `reports/experiments/target-backbones-runs/<model>/<arm>/seed-<seed>.json`.
Private safe checkpoints reuse `data/interim/transformer-finetuned-models/<hash>`.
Private prediction receipts reuse the existing content-addressed
`data/interim/target-deberta/<trial-report-hash>` store; this historical directory
name does not determine the model, which is explicit in every report/manifest.
Progress files similarly reuse the existing target trainer's private directory.
Weights and native text/predictions stay outside Git.

## Interpretation boundaries

The matched arms isolate the input-conditioning change within each backbone.
Cross-backbone results also differ in pretraining, head initialization, tokenizer
and prior learning rate. Validation is reused for epoch selection and development,
so this is not independent test evidence. Cohorts overlap; seen/unseen target means
exact native string, not verified identity. FinBERT has known PhraseBank supervision,
and all sources have uncertain pretraining exposure to SEntFiN. License scope and
private-storage decisions remain documented in `docs/datasets.md`.

## Verified execution and interpretation (2026-10-15)

All twelve trials completed on the Windows RTX 4070 Super. The six FinBERT trials
used 37.67 minutes of recorded training time, and six RoBERTa trials used 36.40
minutes; these exclude preparation, hashing, reload verification and export.
All twelve selected checkpoints passed full file checks and reload scoring.
Mac retains manifests and reconstructed private predictions; full weights remain
on Windows. Ruff and all 491 tests pass on both machines. Independent
scikit-learn/NumPy checks reproduce classification metrics, NLL, Brier and ECE
for all 72 new cohort evaluations, and all prediction bytes reproduce their hashes.
Original source/data/freeze/DeBERTa contracts and Phase 3 final results are unchanged.

The [comparison report](../reports/experiments/target-backbones.md) includes all
three backbones, six cohorts, seed variation, class metrics, architecture/size and
selected epochs. RoBERTa's mean overall F1 is 89.07% ± 0.79 pp, DeBERTa's
88.96% ± 0.42 pp, and FinBERT's 88.65% ± 0.74 pp. The small overall gaps do
not justify a strong winner claim from reused validation. RoBERTa has the highest
observed multi-target/mixed-label means (90.43%/92.60%); DeBERTa has the highest
single-target/seen-native-string means (88.11%/89.70%); FinBERT has the highest
unseen-native-string mean (87.76%). These overlapping cohort rankings are
descriptive; unseen strings do not establish novel-company generalization.

Conditioning improves overall F1 by 8.11 pp for RoBERTa, 7.36 pp for FinBERT,
and 7.77 pp for DeBERTa. Its main gains are on multi-target/mixed-label examples;
single-target gains are small. The matched linear reference remains in
[the TF-IDF report](../reports/experiments/target-tfidf-logreg.md).
Mean overall NLL/Brier improve within each backbone, but overall ten-bin ECE
increases for all three. The probability outputs remain uncalibrated.

BERT pair maxima are 45/37 training/validation tokens; RoBERTa pair maxima are
50/39. Neither tokenizer truncates headlines or produces unknown target tokens.
Headline-only inputs have 995 training/122 validation input patterns with conflicting
labels, versus zero for full target pairs. This diagnoses representational ambiguity,
not a reason to alter the native gold labels or remove the matched headline baseline.

Execution code revision: `c94c87816fa8d544839dbceb9a113b1f2ce040d2`.

Supplement plan SHA-256: `3d614de7ff65314ac7eab57a4c83dab7869afa94a65b27c201974394b4fff62b`.

Aggregate JSON SHA-256: `3f5e32fbfcfe714cb63efd796713ffe4f3d0bf0e4309731e55bc94acca90930c`.
