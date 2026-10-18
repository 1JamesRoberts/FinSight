# Transformer probability diagnostics and error review

This bounded Phase 3 task analyzes saved selected-rate RoBERTa Base and FinBERT
validation predictions for seeds 42/43/44. It never imports transformer models,
fits a calibrator, rescans test data or changes labels. Read the
[results](../../../reports/experiments/transformer-analysis.md) and
[machine-readable aggregates](../../../reports/experiments/transformer-analysis.json).

## Inputs and setup

Use the repository root and existing locked environment. A public checkout alone
cannot reproduce private inputs: retain the reviewed development datasets,
parent/frozen split artifacts, and six saved `run.json`/`predictions.jsonl` pairs.
The study index is `reports/experiments/transformer-tuning-runs/study.json`;
its `seed_runs` entries identify the report SHA, seed and selected learning rate.
FinBERT predictions live under
`data/interim/finbert-finetuning/<report-sha>/predictions.jsonl`;
RoBERTa predictions under
`data/interim/transformer-comparison/<report-sha>/predictions.jsonl`.
No model weights or GPU are required.

```sh
uv sync --locked
uv run --locked python -m finsight.sentiment.transformer_analysis
```

This creates a checksummed private packet and preliminary calibration JSON under
`data/interim/phase3-error-review/<packet-sha>/`. It leaves the completed public
report intact when no annotations are supplied. `--data-dir` can change the
private data root; the receipt remains `configs/benchmark-freeze.json`.
The loader reads development records only. Prediction report checksums, private
run bytes, ordered membership, gold, metrics and split lineage must agree.

## Reviewed packet

The completed packet SHA is
`335b6d56ea34daa3eab0e951168b470fbeececd0b4a244228f00857616e2d8b4`.
Its 39 headlines/IDs/targets, six predictions/confidences and selection strata
remain private. `annotations.json` next to it contains one review per source ID:

```json
{
  "record_id": "local-source-id",
  "category": "neutral_boundary",
  "gold_ambiguous": true,
  "note": "Reviewer explanation; not a label correction."
}
```

Allowed primary categories are `financial_context`, `numeric_comparison`,
`mixed_cues`, `event_interpretation`, `neutral_boundary` and `negation_scope`.
Coverage must exactly match the packet once, with a nonempty note and boolean
ambiguity flag. Counts concern a stratified exploratory sample, not prevalence.
A new reviewer can annotate the packet; it cannot reproduce the original
interpretive judgments without the original private annotations.

Recompute all public numeric aggregates and the reviewed category counts:

```sh
uv run --locked python -m finsight.sentiment.transformer_analysis \
  --annotations data/interim/phase3-error-review/335b6d56ea34daa3eab0e951168b470fbeececd0b4a244228f00857616e2d8b4/annotations.json
uv run --locked --group eda python scripts/plot_transformer_analysis.py
```

The first command writes `reports/experiments/transformer-analysis.json` and
copies the validated notes into ignored packet storage. `--report` can choose a
separate aggregate destination. The second plots this aggregate JSON, saving
`reports/figures/phase3/calibration.png`; it supports `--report` and `--output`.
The Markdown report is the accompanying interpretive write-up, not generated
annotations. Windows uses `python -m uv` before the same arguments.

## Architecture and verification

One typed analysis module uses existing development loaders and classification
metrics; one plotting script uses the existing optional Matplotlib group. No
new dependency, training framework or prediction pipeline is introduced.
Checksums pin the study, reports, predictions, freeze receipt, implementation,
review packet and annotations. Empty reliability bins use null values;
probabilities must be finite, normalized and consistent with the saved argmax.
Brier uses all three classes without scaling. ECE is top-label only with fixed
10-bin boundaries and 5/15-bin sensitivity. Mean/sample SD describe individual
seeds; probabilities are never averaged into an ensemble. A training-prior
control illustrates why low ECE alone is insufficient.

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
```

Synthetic tests cover known Brier/NLL/ECE values, the probability-1 endpoint,
empty bins, invalid distributions, deterministic recurring/variable sampling,
private/public review separation and corrupted prediction/membership/gold/metric
or test-scope inputs. Frozen tests remain unscored; no weights, dataset text or
per-record outputs are cleared for public redistribution.
