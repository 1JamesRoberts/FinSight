# FinSight— Coding Agent Instructions

## Project Overview

FinSight is a financial news intelligence platform focused on
financial sentiment analysis, company-specific sentiment,
sentiment aggregation, and downstream stock prediction.

## Development Principles

1. Follow ROADMAP.md.
2. Implement one task at a time.
3. Prefer simple, maintainable solutions.
4. Use Python type hints.
5. Write unit tests for reusable functionality.
6. Use Ruff for linting and formatting.
7. Use pytest for testing.
8. Avoid introducing unnecessary dependencies.
9. Never hardcode API keys or credentials.
10. Do not commit datasets or model weights without
    verifying redistribution rights.

## Machine Learning Principles

1. Never use test data for training or tuning.
2. Make experiments reproducible.
3. Use fixed random seeds where applicable.
4. Log experimental configurations and metrics.
5. Always compare against baseline models.
6. Avoid look-ahead bias in financial predictions.
7. Separate training and inference pipelines.
8. Document dataset limitations.

## Workflow

Before implementation:
- Read ROADMAP.md.
- Identify the active phase.
- Inspect relevant existing code.
- Propose a short implementation plan.

During implementation:
- Make incremental changes.
- Follow the existing architecture.
- Add tests.
- Avoid unrelated refactoring.

After implementation:
- Run relevant tests and linting.
- Summarize the changes.
- Identify remaining issues.
- Report which roadmap tasks are complete.
- Do not mark tasks complete unless verified.

## Important

Do not implement future roadmap phases unless explicitly requested.
Ask before making major architectural changes.

## Incremental verification and research boundaries

- Work only on the requested phase or bounded task; add behavior tests alongside
  reusable functionality, preferably before implementation.
- Run `uv run --locked ruff check .`, `uv run --locked ruff format --check .`,
  and `uv run --locked pytest` before marking tasks complete.
- Verify dataset, model, and API terms before acquisition, training, storage,
  display, or redistribution; record decisions in `docs/datasets.md`.
- Freeze the final sentiment test set; audit FinBERT/Financial PhraseBank overlap
  and qualify any claims of independent evaluation.
- Use only information available at the prediction cutoff and chronological
  splits for financial prediction. Log dataset/model versions and code revisions.
