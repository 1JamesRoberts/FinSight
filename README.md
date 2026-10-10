# FinSight

FinSight is a financial news sentiment intelligence portfolio project for Data
Scientist, ML Engineer, and AI Engineer internships. Its primary research focus
is comparing traditional NLP methods, FinBERT, and company-specific sentiment.
A secondary study will test whether sentiment improves stock prediction over
market-only baselines, using time-respecting evaluation.

**Status: Phase 0 foundation and Phase 1 local dataset loading implemented.**
Reviewed PhraseBank, SEntFiN, and original-format FiQA files are available in ignored local storage.
No sentiment models, news-provider ingestion, stock prediction, API, or dashboard
are implemented. See [ROADMAP.md](ROADMAP.md)
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
 notebooks/               # Future exploration
 scripts/                 # Future command entry points
 reports/{figures,experiments}/ # Research outputs
 frontend/                # Reserved for Phase 8
 infrastructure/          # Reserved for Phase 9
 docs/                    # Architecture and data-use review checklist
```

The module folders are empty boundaries, not implemented services. Details and
tradeoffs are in [docs/architecture.md](docs/architecture.md). Review the
[dataset/model/API checklist](docs/datasets.md) before acquiring any research
assets. PhraseBank, SEntFiN, and the selected FiQA copy are reviewed for local noncommercial research; public
source-text redistribution and model-weight license scope remain unresolved.
The [evaluation dataset shortlist](docs/evaluation-dataset-shortlist.md) records
candidate findings and the provisional recommendation; no final test is frozen.

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
```

The first command acquires reviewed pinned files, verifies checksums, and reuses a
valid cache. The second validates schemas and writes an ignored aggregate integrity
report. See [loading instructions and findings](docs/dataset-loading.md),
[PhraseBank terms](docs/phrasebank-asset-review.md), and
[SEntFiN terms](docs/sentfin-asset-review.md), and
[FiQA release and terms](docs/fiqa-asset-review.md).
FiQA training annotations retain continuous scores; its original test inputs
have no gold scores and cannot be used for supervised evaluation. No dataset contents are committed;
CI uses synthetic examples and requires no dataset downloads. No final test set
has been frozen and no models have been trained.

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
but the independent final benchmark still requires curation and freezing.
