# FinSight

FinSight is a financial news sentiment intelligence portfolio project for Data
Scientist, ML Engineer, and AI Engineer internships. Its primary research focus
is comparing traditional NLP methods, FinBERT, and company-specific sentiment.
A secondary study will test whether sentiment improves stock prediction over
market-only baselines, using time-respecting evaluation.

**Status: Phases 1 and 2 complete within the documented local research scope.**
Reviewed PhraseBank, SEntFiN, and original-format FiQA files are available in ignored local storage.
VADER, TF-IDF Logistic Regression, and Linear SVM have validation comparisons,
bounded tuning, saved local models, and qualitative error review. See the
[Phase 2 comparison](reports/experiments/phase2-comparison.md). The frozen final
test remains unscored. Phase 3 has begun with an unchanged pretrained FinBERT
validation run on the Windows GPU; see [results](reports/experiments/finbert-validation.md)
and [run instructions](docs/finbert.md). The first SEntFiN FinBERT adaptation
selects epoch 3 with validation macro F1 0.8762; see the
[fine-tuning report](reports/experiments/finbert-finetuning.md) and
[training/loading guide](docs/finbert-finetuning.md). Four RoBERTa/DeBERTa
experiments now compare unchanged financial checkpoints with SEntFiN-trained
base models; see the [transformer comparison](reports/experiments/transformer-comparison.md)
and [reproduction guide](docs/transformer-comparison.md). RoBERTa Base has the
highest observed validation macro F1 (0.8849) in the initial single-seed setup.
Financial checkpoints have also been adapted on SEntFiN, and both original base
models trained separately on PhraseBank; see the
[follow-up results](reports/experiments/transformer-followup.md) and
[training guide](docs/transformer-followup.md). Phase 3 remains in progress.
The [all-model cohort report](reports/experiments/all-models-by-cohort.md) brings
together traditional and transformer variants, class metrics, architecture/size,
transfer results and missing evaluation coverage.
A [controlled learning-rate/seed study](reports/experiments/transformer-tuning.md)
now reports RoBERTa Base macro F1 0.8760 ± 0.0082 and FinBERT 0.8748 ± 0.0023
across three seeds at their selected rates; no superiority claim or test scoring.
A [calibration/error review](reports/experiments/transformer-analysis.md) now
analyzes all six selected-rate runs, with reliability plots, recurring errors and
39 private qualitative reviews; both models remain overconfident on validation.
See the [reproduction guide](docs/transformer-analysis.md).
News ingestion, stock prediction, API, and dashboard remain unimplemented.
See [ROADMAP.md](ROADMAP.md)
for the phase sequence and acceptance criteria.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first.
Python 3.11+ is supported; `.python-version` selects Python 3.12 for local work.
Run these commands from the repository root:

```sh
uv python install 3.12
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
```

`uv sync` creates `.venv`, installs FinSight in editable mode, and installs the
development dependencies from `uv.lock`. No environment activation is required.
Commit `uv.lock`; use `uv lock` deliberately when changing dependencies, then
rerun the checks. To apply formatting, run `uv run --locked ruff format .`.

For the optional PyTorch/Transformers environment on the RTX 4070 SUPER research
computer, see [Windows GPU setup and data integrity checks](docs/windows-gpu-setup.md).
The runtime is optional; baseline installation and CI do not download it.

## Configuration

Defaults work without a `.env` file. For local overrides:

```sh
cp .env.example .env
uv run --locked --env-file .env python -c 'from finsight.config import Settings; print(Settings.from_env())'
```

`FINSIGHT_DATA_DIR` defaults to `data` and is relative to the current working
directory unless absolute. `FINSIGHT_LOG_LEVEL` defaults to `INFO`; allowed values
are `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL` (case-insensitive).
`Settings.from_env()` reads the environment only when called; it does not load
`.env` automatically, create directories, or configure logging. An explicit
mapping can be supplied for tests. No API credentials are needed yet.

Keep secrets in the ignored `.env` file or deployment environment. Add only
redacted variable examples when a later phase introduces a provider.

## Structure

```text
.github/workflows/ci.yml   # Formatting, lint, tests, and package build
configs/                  # Asset identities and provisional split policy
src/finsight/
  config.py               # Typed environment settings
  data/                   # Reviewed acquisition, loaders, and integrity checks
  sentiment/              # Future sentiment research
  features/               # Future aggregation and features
  prediction/             # Future downstream prediction study
  pipelines/              # Future workflow orchestration
  api/                    # Future delivery adapters
tests/unit/              # Configuration and dataset behavior tests
tests/integration/       # Synthetic offline acquisition/loading workflow
data/{raw,interim,processed}/  # Ignored local research data
 models/                  # Ignored model artifacts
 notebooks/               # Reproducible development-only Phase 1 EDA
 scripts/                 # Future command entry points
 reports/{figures,experiments}/ # Research outputs
 frontend/                # Reserved for Phase 8
 infrastructure/          # Reserved for Phase 9
 docs/                    # Architecture and data-use review checklist
```

Data preparation and traditional sentiment baselines are implemented; other folders reserve
later roadmap boundaries. Details and
tradeoffs are in [docs/architecture.md](docs/architecture.md). Review the
[dataset/model/API checklist](docs/datasets.md) before acquiring any research
assets. PhraseBank, SEntFiN, and the selected FiQA copy are reviewed for local noncommercial research; public
source-text redistribution and model-weight license scope remain unresolved.
The [evaluation dataset shortlist](docs/evaluation-dataset-shortlist.md) records
candidate findings. The subsequent [evaluation contract](docs/evaluation-contract.md)
defines the frozen SEntFiN holdout and its independence/coverage limitations.

## Verification and CI

GitHub Actions runs the same locked checks on Python 3.11 and 3.12 for pushes
and pull requests, and builds the package with `uv build`. CI uses
[setup-uv](https://github.com/astral-sh/setup-uv) to install uv and select Python.
After pushing to GitHub, confirm the workflow results before claiming CI passed.
Local checks alone cannot verify a hosted Actions run.

Work one roadmap task at a time, add tests for reusable behavior, and update
checkboxes only after verification. Keep the final test set separate from
training and tuning; record dataset provenance and model versions; compare
against baselines; prevent look-ahead bias. Negative research results are valid.

## Local research datasets

```sh
uv run --locked python -m finsight.data acquire
uv run --locked python -m finsight.data validate
uv run --locked python -m finsight.data analyze-labels
uv run --locked python -m finsight.data analyze-lengths
uv run --locked python -m finsight.data review-annotations
uv run --locked python -m finsight.data prepare-splits
uv run --locked python -m finsight.data freeze-benchmark
uv run --locked --group eda python -m finsight.data eda
```

The first command acquires reviewed pinned files, verifies checksums, and reuses a
valid cache. The second validates schemas and writes an ignored aggregate integrity
report. See [loading instructions and findings](docs/dataset-loading.md),
[PhraseBank terms](docs/phrasebank-asset-review.md), and
[SEntFiN terms](docs/sentfin-asset-review.md), and
[FiQA release and terms](docs/fiqa-asset-review.md).
FiQA training annotations retain continuous scores; its original test inputs
have no gold scores and cannot be used for supervised evaluation. No dataset contents are committed;
CI uses synthetic examples and requires no dataset downloads. The curated
SEntFiN final holdout is frozen and stays withheld from Phase 2 development.

The [label analysis report](reports/label-analysis.md) compares source-class
frequencies and FiQA training-score distributions. Its machine-readable output
is ignored local data; its checked-in Markdown contains aggregate statistics only.
No FiQA class thresholds are assigned by label analysis.

The [text-length report](reports/length-analysis.md) summarizes characters and
whitespace-separated words, with label/source-type comparisons. These counts
are descriptive and do not determine model token limits or filtering rules.

The [duplicate/annotation report](reports/annotation-review.md) records exact and
lexical near-duplicate candidates. [Qualitative findings](docs/annotation-review-findings.md)
document conflicts, target-coverage differences, and FiQA snippet issues. Detailed
source examples remain in ignored local storage; review itself changes no rows.

The [split policy](docs/split-policy.md) applies reversible filtering and groups
known exact/near duplicates before seeded provisional partitioning. The
[split preparation report](reports/split-preparation.md) records retained counts
and class distributions. Raw data stays intact; detailed processed records and
lineage are ignored. Test candidates remain separate from training and tuning,
and the subsequent benchmark release freezes the curated SEntFiN membership.

The [frozen benchmark report](reports/frozen-benchmark.md) describes 1,029 headlines
and 1,360 native target labels. The [freeze receipt](configs/benchmark-freeze.json)
pins exact local artifacts without publishing dataset text. Two malformed-target
groups remain excluded. Use the verified development loader documented in the
[evaluation contract](docs/evaluation-contract.md); changing the official split
after freezing fails. This is held out from FinSight development with no known
accessible-corpus overlap. Complete FinBERT pretraining exposure remains unknown,
and this corpus does not establish performance on all five MVP companies.

## Phase 1 dataset cards and EDA

Read the [PhraseBank](docs/dataset-cards/phrasebank.md),
[SEntFiN](docs/dataset-cards/sentfin.md), and [FiQA copy](docs/dataset-cards/fiqa.md)
cards for schemas, provenance, rights, roles, partitions, and limitations.
The [Phase 1 findings](docs/phase1-findings.md) record completion evidence;
the [EDA report](reports/phase1-eda.md) contains five reviewed aggregate figures.
The figures use only the 14,963 retained train/validation texts.

Install optional plotting/notebook tools with `uv sync --locked --group eda`.
Regenerate the report with `uv run --locked --group eda python -m finsight.data eda`.
Open `notebooks/01_dataset_eda.ipynb` in a Jupyter-compatible editor using the
project `.venv/bin/python` kernel. CLI plotting does not need a notebook server.
Use `uv run --locked --group eda pytest` to include the synthetic plotting check.
Source datasets and notebook outputs stay uncommitted; the five aggregate PNGs,
summary JSON, and Markdown report are safe research outputs. Matplotlib and
ipykernel belong only to the optional EDA group; VADER is a runtime dependency.
Phase 2 includes VADER, TF-IDF + Logistic Regression, and TF-IDF + Linear SVM.

## VADER baseline

```sh
uv sync --locked
uv run --locked python -m finsight.sentiment validate-vader
```

VADER uses the unchanged packaged lexicon and rules, with compound thresholds
<= -0.05 negative, >= 0.05 positive, and neutral between. It needs no training
or inference-time network access. The validation check uses PhraseBank and
single-target SEntFiN development cohorts with a training-majority control;
the frozen benchmark is never scored. Read [how VADER works](docs/vader.md) and
the [validation results](reports/experiments/vader-validation.md). Aggregate JSON
records configurations, scoring-resource hashes, split identity, code provenance,
and timing; per-record predictions stay ignored. VADER's proportions are not
calibrated class probabilities, and its predictions are not company-specific.

## TF-IDF + Logistic Regression

```sh
uv sync --locked
uv run --locked python -m finsight.sentiment validate-tfidf-logreg
```

The pipeline learns word/phrase weights and sentiment coefficients from training
only, then scores validation with separate PhraseBank and single-target SEntFiN
models. Read [how it works and setup](docs/tfidf-logreg.md) and the
[comparison report](reports/experiments/tfidf-logreg-validation.md). Validation
macro F1 is 0.6696 / 0.8165, compared with VADER's 0.4895 / 0.4787 on the same
rows. Fitted models and per-record outputs remain ignored local artifacts.
The frozen test benchmark remains withheld, and this initial configuration has
not been tuned. Probabilities have not been assessed for calibration.

For the subsequent PhraseBank-only class-weight experiment:

```sh
uv run --locked python -m finsight.sentiment validate-tfidf-logreg \
  --cohort phrasebank \
  --config configs/tfidf-logreg-balanced.json \
  --report-path reports/experiments/tfidf-logreg-phrasebank-balanced.md
```

[The comparison and input-flow guide](docs/phrasebank-class-weights.md) explains
how command arguments select settings and how prepared texts/labels enter the
pipeline. Balanced weighting improves validation macro F1 from 0.6696 to 0.7378;
the unweighted preset remains the default, and final-test data stays withheld.

## TF-IDF + Linear SVM

```sh
uv run --locked python -m finsight.sentiment validate-tfidf-svm
uv run --locked python -m finsight.sentiment validate-tfidf-svm \
  --config configs/tfidf-svm-balanced.json \
  --report-path reports/experiments/tfidf-svm-balanced-validation.md
```

These commands train four separate models: unweighted/balanced for PhraseBank and
single-target SEntFiN. Validation macro F1 is 0.7344 / 0.7444 on PhraseBank and
0.8072 / 0.8075 on SEntFiN. Read [how SVM works and setup](docs/tfidf-svm.md),
the [unweighted report](reports/experiments/tfidf-svm-validation.md), and the
[balanced report](reports/experiments/tfidf-svm-balanced-validation.md).
Decision scores are signed margins, not probabilities. TF-IDF fits on training
only; fitted artifacts stay ignored and the frozen final test remains withheld.

## Phase 2 tuning and error review

```sh
uv run --locked python -m finsight.sentiment.tuning
uv run --locked python -m finsight.sentiment.error_analysis
```

The first command compares 24 separately fitted pipelines using the declared
C/class-weight grid; it preserves earlier public baseline reports. The second
verifies saved validation predictions and prepares a deterministic local error
packet without fitting or scoring. Review notes stay local; qualitative judgments
require inspecting the texts. Read the [tuning guide](docs/tfidf-tuning.md),
[qualitative error review](reports/experiments/phase2-error-analysis.md), and
[consolidated Phase 2 report](reports/experiments/phase2-comparison.md).
