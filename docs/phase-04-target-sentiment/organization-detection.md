# Pretrained organization candidates and target sentiment

Use a pretrained Named Entity Recognition (NER) model to propose names beyond the
five-company registry. This is an experimental path alongside the existing
registry detector; `predict --detect-companies` retains its previous behavior.
[Original detection comparison](../../reports/phase-04-target-sentiment/experiments/organization-detection.md).
The [current resolution stage](company-resolution.md) now routes default sentiment
only to reviewed-registry company matches; unknown names remain visible as unresolved.

## What the two models do

1. `dslim/bert-base-NER` labels tokens as organizations, people, locations or
   miscellaneous entities. The default word-aware `first` aggregation gives each word one consistent
   label before BIO grouping; retain only `ORG` spans. Slice the original headline
   using offsets rather than the pipeline's reconstructed subword text.
2. An ORG is a candidate, not a verified company or ticker. Banks, regulators,
   exchanges and mistaken person-name fragments can appear. The existing registry
   remains responsible for its explicitly supported company identities and
   ambiguity/rollup rules. NER does not override those policies.
3. Resolve complete candidate surfaces through the registry and conservative
   institution-title rules. With `--with-sentiment`, group only verified company
   identities and use the first original surface as target. Other/unresolved spans
   remain in `resolutions`; explicit `--score-all-candidates` retains the former
   raw-ORG diagnostic, with repeated identical surfaces grouped in that mode.
4. The saved RoBERTa predictor receives that original target first and the complete
   original headline second through its existing tokenizer pair encoding. No
   masking, target-name substitution, manual separator insertion or retraining is
   introduced. Results retain all spans and resolution reasons. Verified company
   sentiments identify the registry-match scope; raw diagnostic sentiments keep
   `company_identity_verified: false`.

The NER model is BERT-base-cased fine-tuned by its publisher on English CoNLL-2003;
we do not fine-tune either model here. Its ORG token scores are not calibrated
company probabilities. No confidence threshold is selected using SEntFiN.

`configs/organization-ner-plan.json` remains the immutable original acquisition
and simple-baseline protocol. The v2 loader/CLI selects `first` by default; its
explicit aggregation argument and emitted strategy identify current inference.
The historical audit and comparison explicitly request their declared grouping,
so the original plan is not rewritten to hide the baseline change.

NER can split unfamiliar names into fragments or attach nearby words. Word-aware grouping reduces within-word fragments but can still miss or
misclassify names. Review candidates before treating their sentiment as
company-level output; this path remains experimental and separate from the
default registry-based company detector.

## Setup and acquisition

From the repository root on the Mac, install the existing optional dependencies:

```bash
uv sync --locked --extra transformer
```

The [reviewed local-use decision](../datasets.md#pretrained-organization-detector-review-2026-10-20)
and `configs/organization-ner-plan.json` pin the model revision, seven file hashes
and sizes. Download only these files into ignored storage, using the already
available Hugging Face Hub dependency:

```bash
uv run --locked python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from finsight.sentiment.artifacts import verify_files

plan = json.loads(Path('configs/organization-ner-plan.json').read_text())
directory = Path('data/interim/organization-ner-model') / plan['revision']
snapshot_download(
    repo_id=plan['model_id'],
    revision=plan['revision'],
    allow_patterns=list(plan['files']),
    local_dir=directory,
)
verify_files(directory, plan['files'])
PY
```

Acquisition is explicit; the detector itself loads only verified local files, uses
safetensors and disables remote code. The weight file is approximately 433 MB.
No datasets, credentials or hosted inference service are needed for invented text.
Do not commit downloaded files.

## Detection on the Mac

```bash
uv run --locked python -m finsight.sentiment.organizations \
  --text 'Tesla reported higher profits while Microsoft reported losses.'
```

The verified smoke detects Tesla and Microsoft with original character offsets.
This illustrates a name outside the registry; it does not establish general NER
accuracy. Detection uses CPU by default and needs no saved FinSight sentiment
weights. Requests exceeding the complete 512-token budget fail explicitly rather
than silently losing mentions through truncation.

## Optional sentiment on the GPU PC

The full updated CLI is verified on the GPU PC in an isolated source snapshot.
The PC now has the pinned NER files in the shared ignored model directory. The
primary checkout's Python files were not updated. To run the verified snapshot
in PowerShell:

```powershell
cd C:\Hope\Project\financial-news-intelligence
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
$source = Join-Path (Get-Location) 'data\interim\organization-release\0defc077d350215bd60175d647b962e00e8b0ce35e5fbc182dc73c06ee565107\source'
$env:PYTHONPATH = Join-Path $source 'src'
& $python -m finsight.sentiment.organizations --text 'Microsoft reported higher profits while Tesla reported losses.' --data-dir 'C:\Hope\Project\financial-news-intelligence\data' --with-sentiment --device cuda
Remove-Item Env:PYTHONPATH
```

The full CLI's detected targets and sentiment results reproduce direct saved-model
predictions exactly; the empty-organization CLI also verifies. See the
[current resolution receipt](../../reports/phase-04-target-sentiment/verification/organization-resolution-smoke.json).
An updated primary checkout can use the same module with its existing artifact
data directory once these source changes are installed there.

`--data-dir` locates another existing private artifact directory; `--model-dir`
locates the reviewed NER files. The Mac has the NER weights, but does not have the
saved RoBERTa deployment weights. An empty verified-company list skips sentiment
loading entirely, even if NER proposes unresolved names. Models are reused in memory through the Python detector and
predictor instances when processing multiple requests.

## Reproduce the development comparison

```bash
uv run --locked python -m finsight.sentiment.experiments.organization_detection
```

The runner verifies the existing locked development files and sends only the
1,053 validation headlines to NER. It records model/source/plan/input hashes,
architecture, runtime, candidate counts and positive-only native-target coverage.
The twelve-company allowlist was fixed before inference and is separate from the
five-company registry. No thresholds, model weights, registry rules or deployment
selection are adjusted from this run.

Private source text, target checks and all detections stay under
`data/interim/organization-detection/<provenance-sha256>/`. Deterministic review
membership selects up to ten headlines per declared category; categories can
overlap. Assistant notes are exploratory findings, not independent gold labels.
Only aggregate JSON, review findings and an invented-input smoke are public.

Native sentiment targets include currencies, sectors and indices and are not
exhaustive mention annotations. Exact/overlap coverage is therefore descriptive;
missing non-company targets need not be NER mistakes, and unmatched ORGs need not
be false positives. Exhaustive company precision/recall requires a separate
mention-labeled collection. No frozen sentiment records are read.

## Code

- `sentiment/organizations.py`: verified local model loading, batched ORG spans,
  optional candidate-to-sentiment bridge and CLI.
- `sentiment/experiments/organization_detection.py`: validation comparison and
  private review packet.
- `sentiment/mentions.py`: unchanged registry baseline.
- `tests/unit/test_organizations.py` and `test_organization_audit.py`: synthetic
  behavior checks without downloading models or reading private data.

## Mention-annotated development sample

The separate [50-headline annotation packet and review workflow](data/organization-benchmark.md)
is now prepared, with [provisional exact-span results](../../reports/phase-04-target-sentiment/evaluation/organization-benchmark.md)
for simple versus word-aware grouping. The user has now accepted the reference
and authorized adjudication; see the [reviewed comparison and first-grouping
selection](../../reports/phase-04-target-sentiment/evaluation/organization-benchmark-reviewed.md).
The organization CLI and Python loader default to `first`; pass
`--aggregation-strategy simple` to reproduce the original grouping. CLI responses
record the strategy and detector version. Registry-based `--detect-companies`
continues to use its existing identity policy.
