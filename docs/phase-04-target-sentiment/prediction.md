# Reusable target-aware RoBERTa prediction

Phase 4 exposes local Python inference and a CLI. No HTTP server is involved.
`TargetSentimentPredictor.load()` verifies the selected artifact and loads its
model/tokenizer once; subsequent calls reuse them in memory.

## Which checkpoint?

[Inference configuration](../../configs/target-roberta-inference.json) pins
RoBERTa Base, target-pair arm, **seed 43, epoch 2**. The selection rule is highest
unrounded **all-target validation macro F1**, breaking exact ties by lowest seed.
The saved validation scores are 88.2169%, 89.7683%, and 89.2187% for seeds 42–44.
Selection reads the existing development comparison and three trial reports;
no dataset, private prediction or final-test result is consulted.

This deployment choice was made after final evaluation using validation evidence
only. It does not revise the locked benchmark, whose transformer scores remain
three-seed averages. The selected single checkpoint is not an ensemble; its
performance must not be presented as that averaged benchmark score.

The artifact-manifest SHA-256 is
`2f742f69854f097b5b1351e4a1de126b54c28fe2c900edb3c2e41526cd32e855`.
Model version in each response is this identifier; `deployment_sha256` identifies
selection metadata. Model weights/tokenizer live under
`data/interim/transformer-finetuned-models/<manifest-sha256>/` and remain ignored.
Actual full weights are currently on the Windows GPU computer; the Mac retains
metadata only. Missing local weights fail explicitly, with no download.

## Python workflow

Install the locked transformer extra on a machine with the saved weights:

```bash
uv sync --locked --extra transformer
```

```python
from pathlib import Path
from finsight.sentiment.models.target_predictor import TargetSentimentPredictor

predictor = TargetSentimentPredictor.load(data_dir=Path("data"), device="cuda")

# Caller supplies the intended target. No masking or automatic entity extraction.
result = predictor.predict(
    text="Acme profits rose while Beta reported a loss.",
    target_company="Acme",
)
print(result.label, result.probabilities, result.model_version)

# Paired lists: results follow the input order; the model stays loaded.
results = predictor.predict_many(
    texts=["Acme profits rose while Beta reported a loss."] * 2,
    targets=["Acme", "Beta"],
)
```

These are invented smoke-test examples, not an evaluation dataset. There is no
prescribed label for them. Raw softmax probabilities are **not calibrated**.
`device="cpu"` is an explicit supported override; CUDA never silently falls back
to CPU. `batch_size=...` controls inference batching only. The original training
configuration is still verified before applying runtime overrides.

Inputs must be nonempty strings. The encoder preserves the complete target as
sequence one and the headline as sequence two, using the tokenizer's own special
tokens. Max length is 128; only the headline can be truncated. A target that leaves
no room for headline tokens is rejected. The supplied target spelling is retained.
Scoring does not establish that a target is mentioned, relevant or unambiguous.

## Optional company registry guard

Explicit native targets such as SEntFiN entities are accepted by `predict()`.
The separate `predict_company()` method checks the existing five-company registry:

```python
from finsight.sentiment.companies import load_registry

registry = load_registry(Path("configs/companies.json"))
checked = predictor.predict_company(
    "Apple Inc. reported higher earnings.",
    "Apple",
    registry,
    company_context=True,  # Caller asserts that the name denotes a company.
)
print(checked.resolution.status, checked.prediction)
```

Ambiguous or unsupported targets receive `prediction=None`, not neutral sentiment.
Bare Apple/Amazon/Alphabet require explicit company context. Google-to-Alphabet
rollup requires `allow_parent_rollup=True`. Identity resolution is output metadata:
we do not silently substitute a ticker/canonical name in the model input. This
wrapper neither extracts mentions from headlines nor resolves pronouns. Sparse
five-company supervision still prevents a reliable five-company accuracy claim.

## CLI

From a source checkout with the saved artifact available:

```bash
uv run --locked python -m finsight.sentiment.predict --show-config
uv run --locked python -m finsight.sentiment.predict \
  --text "Acme profits rose while Beta reported a loss." \
  --target Acme --target Beta --device cuda
uv run --locked python -m finsight.sentiment.predict \
  --text "Apple Inc. reported higher earnings." --target Apple \
  --company --company-context --device cuda
```

`--text` supplies the headline. Each `--target` adds a paired request for that same
headline. Output is a JSON list of results. `--data-dir` locates private artifacts;
`--root` locates source configuration/reports, and `--config` selects the pinned
inference config file. `--show-config` verifies public metadata without loading
weights or reading datasets. No predictions are automatically written to disk.

On the current GPU computer, use the updated primary checkout and its existing
Python environment. The old isolated execution checkout is archived:

```powershell
$env:PYTHONPATH = 'C:\Hope\Project\financial-news-intelligence\src'
& 'C:\Hope\Project\financial-news-intelligence\.venv\Scripts\python.exe' `
  -m finsight.sentiment.predict `
  --data-dir 'C:\Hope\Project\financial-news-intelligence\data' `
  --text 'Acme profits rose while Beta reported a loss.' --target Acme --target Beta
```

## Code map

- `sentiment/deployment.py`: validation-only checkpoint selection and metadata checks.
- `sentiment/models/target_predictor.py`: load once, pair encoding, no-gradient
  prediction, label mapping and optional company guard.
- `sentiment/models/transformer.py`: exact local checkpoint loading; runtime-device
  overrides preserve original training provenance.
- `sentiment/predict.py`: argument parsing and JSON output.

A [registry mention baseline](company-mentions.md) now adds automatic targets via
`predict_headline()` and `--detect-companies`. General NER, a new company-labeled
test collection, HTTP serving and sentiment aggregation remain separate tasks. Existing frozen data and evaluation
receipts/results are preserved.

## Verification

All 603 tests pass on Mac and Windows. Isolated-checkout Ruff lint/format passes;
the Mac working tree's unrelated empty `notebooks/experiment.ipynb` blocks full
Ruff parsing and was left unchanged. Checks excluding only that file pass.
The real CUDA checkpoint reproduced all 1,410 original validation predictions and
probabilities exactly, and repeated calls reused the loaded model. The actual CLI
also passed an invented-input smoke test. See the
[verification report](../../reports/phase-04-target-sentiment/target-prediction.md).
