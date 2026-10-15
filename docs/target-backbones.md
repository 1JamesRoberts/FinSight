# Matched target conditioning: RoBERTa Base and FinBERT

This Phase 4 supplement tests whether the company–headline representation helps
across backbones. It reuses the exact native SEntFiN target view verified by
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
uv run --locked python -m finsight.sentiment.target_backbones
```

The pinned source snapshots and verified private dataset view must already be
present. This command performs no acquisition. It verifies source bytes, trains
12 trials, saves/reloads selected checkpoints and writes the aggregate report.
A single trial can be run with:

```powershell
uv run --locked python -m finsight.sentiment.target_backbones --model finbert --arm target_pair --seed 42
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
