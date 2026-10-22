# FinSight

**Financial news sentiment intelligence, from sentence classification to
company-specific analysis.**

FinSight is a Data Scientist, ML Engineer, and AI Engineer portfolio project.
It compares traditional NLP baselines with FinBERT, RoBERTa, and DeBERTa, then
examines how sentiment changes for different companies in the same headline.
A planned secondary study will test whether sentiment improves stock prediction
over market-only baselines using chronological evaluation.

[Project status](#project-status) · [Results](#research-results) ·
[Setup](#setup) · [Usage](#usage) · [Structure](#repository-structure) ·
[Documentation](#documentation)

## Project status

The current implementation covers dataset curation, sentiment experiments,
frozen sentence/target evaluations, and local company-sentiment inference.
News ingestion, daily aggregation, stock prediction, API, and dashboard work
remain ahead in the [roadmap](ROADMAP.md).

| Phase | Verified scope and remaining work | Final report |
|---|---|---|
| 0 — Foundation | Python package, locked environment, tests and CI workflow; hosted CI status must be checked separately | [Report](reports/phase-00-foundation/final-report.md) |
| 1 — Data | Dataset review, quality audits, grouped splits, benchmark freeze and EDA complete within local research scope | [Report](reports/phase-01-data/final-report.md) |
| 2 — Baselines | VADER, TF-IDF Logistic Regression/Linear SVM, bounded tuning and validation error review complete | [Report](reports/phase-02-baselines/final-report.md) |
| 3 — Sentence transformers | Locked single-target evaluation complete; MLflow and length/batch ablations deferred | [Report](reports/phase-03-transformers/final-report.md) |
| 4 — Target sentiment | Target conditioning, organization/issuer resolution and GPU inference verified; independent company-test and per-company evaluation criteria remain open | [Report](reports/phase-04-target-sentiment/final-report.md) |

**How to read this table:** phases are roadmap work areas. Verified scope describes
completed tasks, not unconditional phase closure. Each final report separates
results, decisions, limitations and remaining acceptance criteria.

## Research results

### Sentence-level sentiment

The locked Phase 3 evaluation uses **766 single-target SEntFiN headlines**.
Mean macro F1 across three seeds is **89.32% for DeBERTa Base**, **88.68% for
RoBERTa Base**, and **88.28% for fine-tuned FinBERT**. Supplementary soft voting
reaches **90.28%**, but its paired advantage over DeBERTa is inconclusive because
the 95% interval includes zero. Validation-fitted temperature scaling improves
held-out negative log likelihood for all three transformers.

[Results and interpretation](reports/phase-03-transformers/final-report.md) ·
[Full evaluation and provenance](reports/experiments/phase3-final.md)

### Target-specific sentiment

The locked Phase 4 evaluation uses **262 multi-target headlines with 592 native
target annotations**. It compares matched headline-only and target-aware models:

| Model | Headline-only macro F1 (%) | Target-aware macro F1 (%) | Gain (pp) |
|---|---:|---:|---:|
| TF-IDF Logistic Regression | 68.47 | 76.15 | +7.68 |
| FinBERT | 73.19 | 86.84 | +13.65 |
| RoBERTa Base | 72.82 | 88.34 | +15.52 |
| DeBERTa Base | 74.53 | 88.75 | +14.23 |

**How to read this table:** macro F1 equally averages negative, neutral and positive
class F1 on the same target annotations. Transformer values average seeds 42–44;
TF-IDF is one fixed pipeline. Gain is the paired difference in percentage points,
computed before rounding. These are target-level results, separate from the
sentence cohort above. Native SEntFiN targets include entities other than companies.

RoBERTa's conditioning gain has a paired 95% interval of **[12.16, 19.10] pp**.
The comparison does not establish a clear RoBERTa/DeBERTa winner. The local
predictor uses the validation-selected **RoBERTa Base seed 43, epoch 2** checkpoint;
the three-seed benchmark average is not that checkpoint's individual score.
The target-format study supports retaining original target/headline pairs.

[Results and limitations](reports/phase-04-target-sentiment/final-report.md) ·
[Full evaluation](reports/experiments/phase4-target-final.md) ·
[Target-format study](reports/phase-04-target-sentiment/experiments/target-format-comparison.md)

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
from the repository root. Python 3.11+ is supported; `.python-version` selects 3.12.

```sh
git clone https://github.com/1JamesRoberts/financial-news-intelligence.git
cd financial-news-intelligence
uv python install 3.12
uv sync --locked
```

This creates `.venv` with the locked package and development dependencies.
`uv run` uses that environment without activation. Dataset contents and model
weights are excluded from Git; cloning does not install research artifacts.
The default environment supports traditional baselines and offline tests.

Optional environments:

```sh
# Plotting and notebook kernel
uv sync --locked --group eda

# PyTorch / Transformers for local model work
uv sync --locked --extra transformer
```

For the existing RTX 4070 SUPER environment, use the
[Windows GPU setup guide](docs/guides/windows-gpu-setup.md). The configured
Windows launchers use its existing Python environment and do not require `uv`.

### Configuration

Defaults work without `.env`. Copy `.env.example` to the ignored `.env` file
for local overrides, and pass `--env-file .env` to `uv run` when needed.

- `FINSIGHT_DATA_DIR`: research storage, default `data`; relative paths resolve
  from the current working directory.
- `FINSIGHT_LOG_LEVEL`: default `INFO`; also accepts `DEBUG`, `WARNING`, `ERROR`
  and `CRITICAL`.

Keep credentials and contact details out of committed files.
See [configuration and architecture](docs/architecture.md).

## Usage

### Inspect the selected sentiment configuration

```sh
uv run --locked python -m finsight.sentiment.predict --show-config
```

This verifies public selection metadata without loading weights or scoring data.

### Predict sentiment for explicit targets

On a machine with the saved RoBERTa artifact and transformer dependencies installed:

```sh
uv run --locked --extra transformer python -m finsight.sentiment.predict --text "Acme profits rose while Beta reported a loss." --target Acme --target Beta --device cpu
```

Use `--device cuda` on the configured GPU machine. Each target receives a separate
prediction using **target first, headline second**, with tokenizer-managed special
tokens. Explicit targets do not require company detection or registry membership.
Returned probabilities are uncalibrated. Missing weights fail explicitly; the
command does not download the trained artifact.

[Python API and CLI guide](docs/phase-04-target-sentiment/prediction.md) ·
[Current GPU launchers](docs/phase-04-target-sentiment/issuer-alias-gpu.md)

### Detect companies and route sentiment

The automatic path uses pretrained BERT organization extraction, conservative
company-type checks, and optional US-listed issuer/alias linking before target
sentiment. Organization detection, company classification and issuer identity
are separate decisions; an unresolved identity is not neutral sentiment.
Parent-company rollup requires explicit permission through `--allow-parent-rollup`.

Use the [company detection/resolution guide](docs/phase-04-target-sentiment/company-resolution.md)
and [reviewed-alias GPU commands](docs/phase-04-target-sentiment/issuer-alias-gpu.md)
for required local assets, policy flags and end-to-end examples.

### Reproduce development experiments

Review the [data rights ledger](docs/datasets.md) and
[dataset loading guide](docs/phase-01-data/acquisition/dataset-loading.md)
before acquiring assets. With the reviewed data prepared locally:

```sh
uv run --locked python -m finsight.sentiment validate-vader
uv run --locked python -m finsight.sentiment validate-tfidf-logreg
uv run --locked python -m finsight.sentiment validate-tfidf-svm
```

These commands use development cohorts. Follow the phase guides for class weights,
bounded tuning, EDA, transformer training and saved-output verification.
Completed frozen-test evaluations are documented separately; use saved outputs
to reproduce metrics rather than rerunning test inference during development.

## Repository structure

```text
configs/                    # Pinned assets, experiment plans and inference settings
src/finsight/
  config.py                 # Typed environment settings
  data/                     # Acquisition, validation, curation and split integrity
  sentiment/
    models/                 # Model loading and prediction
    training/               # Shared training loops and configurations
    experiments/            # Reproducible study runners
    evaluation/             # Metrics, calibration and evaluation contracts
    artifacts.py            # Checksums and model manifest verification
  features/                 # Reserved for aggregation/features
  prediction/               # Reserved for stock prediction
  pipelines/                # Reserved for orchestration
  api/                      # Reserved for delivery adapters
tests/                      # Unit and synthetic integration tests
scripts/                    # Research helpers and execution scripts
notebooks/                  # EDA and training walkthroughs
reports/phase-*/             # Phase indexes, final summaries and task evidence
reports/experiments/         # Shared experiment aggregates and run reports
reports/figures/             # Aggregate research figures
docs/                       # Phase guides, architecture and rights ledger
data/{raw,interim,processed}/ # Ignored datasets, caches and private artifacts
models/                     # Ignored local model artifacts
.github/workflows/ci.yml     # Locked lint, formatting, tests and package build
frontend/                   # Reserved for the dashboard
infrastructure/             # Deployment scaffolding
```

Training and inference remain separate. Research runners reuse shared model,
training, evaluation and artifact modules. See the
[sentiment code guide](docs/guides/sentiment-code-guide.md) for an end-to-end walkthrough.

## Verification and CI

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
```

GitHub Actions runs locked checks and a package build on Ubuntu with Python
3.11/3.12 and Windows with Python 3.12. CI also installs the optional EDA group
and uses synthetic data; it requires no dataset or model downloads.
Check hosted workflow results separately from local verification.

The latest local verification has **813 passing tests** and passing full Ruff
lint and formatting checks. Notebook execution outputs are cleared before commit
to keep local dataset contents out of Git. Hosted CI must still be verified separately.

## Data and evaluation boundaries

- PhraseBank, SEntFiN and the selected FiQA release have documented roles and
  local research-use reviews. Source text and weights are not redistributed here;
  consult [dataset/model terms](docs/datasets.md) before acquiring or sharing assets.
- FinBERT was trained on Financial PhraseBank, so its PhraseBank results cannot
  establish independent generalization. Complete pretraining exposure is unknown.
- Training and tuning exclude frozen tests. The evaluated Phase 3 and Phase 4
  SEntFiN cohorts are consumed; PhraseBank tests remain reserved. Phase 4 v2
  excludes a reviewed overlap group while preserving the original freeze.
- SEntFiN company coverage is uneven and does not establish performance for the
  five planned MVP stocks. Mention/linking reviews are small development studies,
  separate from the frozen sentiment benchmark.
- Stock prediction and any claim of trading value remain future research.

## Documentation

- [Final reports for Phases 0–4](reports/README.md): outcomes, decisions and open work.
- [Guides by phase](docs/README.md): data, models, training and evaluation methods.
- [Roadmap](ROADMAP.md): task sequence and acceptance criteria.
- [Architecture](docs/architecture.md): module boundaries and design choices.
- [Dataset/model rights ledger](docs/datasets.md): provenance and usage decisions.
- [Simple transformer training notebook](notebooks/02_simple_transformer_training.ipynb): loading, tokenization, training, saving and prediction.
