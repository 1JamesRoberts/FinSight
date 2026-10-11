# FinSight — Project Roadmap

> **Financial News Sentiment Intelligence & Stock Prediction**  
> Primary focus: **financial NLP and company-specific sentiment analysis**.  
> Secondary focus: determine whether sentiment adds predictive value for next-day stock returns.  
> Audience: Data Scientist / ML Engineer / AI Engineer internship portfolio.

## 1. Project goals

Build a reproducible, end-to-end platform that:

1. Collects and processes appropriately licensed financial news.
2. Benchmarks traditional NLP baselines against financial-domain transformers.
3. Identifies **company-specific sentiment**, including in multi-company articles.
4. Aggregates article-level sentiment into timestamped company indicators.
5. Tests whether sentiment features improve next-day stock direction prediction compared with market-only baselines.
6. Serves news, sentiment, metrics, and predictions through a monitored application.

### Research questions

- **RQ1:** How do lexicon and TF-IDF models compare with a financial-domain transformer on an *independent* financial sentiment test set?
- **RQ2:** Does appropriate fine-tuning improve classification, calibration, or robustness?
- **RQ3:** Does target-aware sentiment outperform article-wide sentiment for multi-company text?
- **RQ4:** Which sentiment aggregation methods are stable and informative?
- **RQ5:** Does adding news sentiment improve stock-direction prediction under time-respecting evaluation?

**Success is not defined by beating the stock market.** Negative or inconclusive findings are valid if the experiment is sound.

## 2. Scope and guardrails

### MVP

- Five companies: **AAPL, MSFT, NVDA, AMZN, GOOGL**.
- English news, daily aggregates, daily US equity prices.
- Three sentiment classes: positive / neutral / negative.
- VADER, TF-IDF + Logistic Regression, TF-IDF + Linear SVM, FinBERT.
- Company-targeted sentiment as the distinguishing NLP feature.
- Market-only versus sentiment-augmented LightGBM as the stock-model comparison.
- Scheduled ingestion, PostgreSQL, FastAPI, a compact Next.js dashboard.

### Later / optional

- GRU, more tickers, intraday data, advanced event extraction, streaming, automatic retraining.
- **Do not start** these until the MVP evaluation is complete.

### Non-negotiables

- Verify data/model licenses and provider ToS **before** downloading, persisting, training on, or republishing content.
- Never commit secrets, restricted articles, large datasets, or model weights without permission.
- Pretrained `ProsusAI/finbert` was trained on Financial PhraseBank: **do not present its evaluation on overlapping PhraseBank samples as independent generalization evidence**. Create an independent benchmark or explicitly label the overlap/limitations.
- Freeze a final sentiment test set before model comparisons; do not tune on it.
- For price prediction, use only data *available at the stated prediction cutoff*, including article availability/ingestion time when possible. Use exchange-aware trading calendars and chronological splits.
- Persist model version, dataset version, config, code revision, and evaluation metrics.
- Keep model outputs clearly labeled as experimental predictions, not financial advice.

## 3. Proposed technology stack

| Layer | Tools | Notes |
|---|---|---|
| Python project | Python 3.11+, `uv` | Dependency locking |
| Quality | Ruff, pytest, mypy (optional) | CI gates |
| Data science | pandas, NumPy, scikit-learn | Reproducible pipelines |
| NLP | VADER, Hugging Face Transformers, PyTorch | FinBERT and target-aware model |
| Prediction | LightGBM | GRU optional |
| Experiments | MLflow | Parameters, metrics, artifacts |
| Storage | PostgreSQL, Alembic | Source and inference metadata |
| API | FastAPI, Pydantic | Explicit typed schemas |
| UI | Next.js | Research-first dashboard |
| Jobs | APScheduler or Prefect | Choose one initially |
| Deployment | Docker Compose | Fits Proxmox homelab |
| Operations | GitHub Actions, Prometheus, Grafana | Add progressively |

Avoid Kafka, Kubernetes, and microservices for the MVP unless required by a measured bottleneck.

## 4. Suggested repository layout

```text
finsight/
├── README.md
├── ROADMAP.md
├── AGENTS.md
├── pyproject.toml
├── uv.lock
├── .env.example
├── configs/
│   ├── data.yaml
│   ├── sentiment.yaml
│   └── prediction.yaml
├── data/                  # ignored, except README/placeholders
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
├── src/finsight/
│   ├── data/
│   ├── sentiment/
│   ├── features/
│   ├── prediction/
│   ├── pipelines/
│   └── api/
├── frontend/
├── tests/
│   ├── unit/
│   └── integration/
├── scripts/
├── reports/
│   ├── figures/
│   └── experiments/
├── infrastructure/
│   ├── docker/
│   └── monitoring/
└── docs/
    ├── architecture.md
    ├── datasets.md
    ├── experiments.md
    └── limitations.md
```

## 5. Timeline and milestones

**Indicative effort:** 10–14 weeks at 10–15 hours/week; adjust by progress, not by deadline pressure.

| Phase | Focus | Time | Milestone |
|---|---|---|---|
| 0 | Project initialization | 2–3 days | Foundation |
| 1 | Data exploration/preparation | 1 week | M1 |
| 2 | Traditional sentiment baselines | 1 week | M1 |
| 3 | FinBERT / fine-tuning | 2 weeks | M1 |
| 4 | Company-specific sentiment | 1–2 weeks | **M1: NLP research** |
| 5 | Sentiment aggregation | 1 week | M2 |
| 6 | Stock prediction study | 2 weeks | **M2: Financial ML study** |
| 7 | Backend + news ingestion | 1–2 weeks | M3 |
| 8 | Dashboard | 1 week | M3 |
| 9 | Deployment + observability | 1 week | M3 |
| 10 | Portfolio polish | 2–3 days | **M3: Finished application** |

---

## Phase 0 — Repository foundation

**Goal:** A reproducible development environment; **no ML implementation yet**.

### Tasks
- [x] Initialize Git, `.gitignore`, `README.md`, `.env.example`.
- [x] Configure Python environment and dependencies with `uv`.
- [x] Configure Ruff and pytest; add GitHub Actions lint/test workflow.
- [x] Create package structure under `src/finsight/`.
- [x] Add configuration patterns with no hardcoded secrets.
- [x] Record architectural choices in `docs/architecture.md`.
- [x] Add dataset/model/API terms checklist to `docs/datasets.md`.
- [x] Add `AGENTS.md` with incremental, test-first Codex instructions.

**Deliverables:** Installable skeleton, passing sample tests, setup instructions.

**Acceptance criteria:** Clean checkout can install, lint, and test; CI passes; secret files are ignored.

**Local verification (2026-10-10):** Locked installation in the working tree and a
fresh copy passed on Python 3.12.2. Ruff lint and formatting passed; 11 unit tests
passed; source distribution and wheel builds passed. Installed package imports
were verified outside the repository, and secret/data/model ignore patterns were
checked. All eight implementation tasks above are complete. Hosted GitHub Actions
has not run yet; the CI acceptance criterion remains pending until pushed to GitHub.

**Codex prompt:**
> Read ROADMAP.md. Implement Phase 0 only. Set up a typed Python package using uv, Ruff, pytest, .env.example, minimal CI, and a clear README. Explain architecture decisions; add no modeling or ingestion features. Verify lint and tests.

## Phase 1 — Dataset curation and EDA

**Goal:** Auditable sentiment datasets and leakage-aware splits.

### Tasks
- [x] Compare Financial PhraseBank and FiQA schemas, labels, target entities, licensing, and source provenance.
- [x] Document redistribution/training terms and pin exact selected dataset revisions (local noncommercial scope; public redistribution not cleared).
- [x] Implement repeatable ingestion and schema validation (PhraseBank, SEntFiN, and original-format FiQA participant copy).
- [x] Analyze labels, lengths, duplicates, ambiguous examples, and available annotation-agreement metadata (descriptive review; unresolved cases retained).
- [x] Normalize label mappings; preserve original labels/metadata (PhraseBank and SEntFiN).
- [x] Define train/validation/test partition strategy and hash-based duplicate checks (parent split; final benchmark frozen separately).
- [x] Select/curate an **independent** final evaluation set for pretrained FinBERT comparisons (held-out SEntFiN; no known accessible-corpus overlap, checkpoint pretraining exposure unknown).
- [x] Write dataset cards and EDA figures (three artifact-specific cards; development-only notebook, aggregate JSON, and five reviewed PNGs).

**Comparison verification (2026-10-10):** Public cards, the PhraseBank loader,
research papers, original FiQA challenge, and published license terms were reviewed.
See `docs/dataset-comparison.md`. Comparison is complete; acquisition approval,
exact revision pinning, empirical inspection, and independent-test selection remain
pending. FiQA mirror/original terms differ and require resolution.

**Evaluation shortlist (2026-10-10):** SEntFiN, SemEval financial headlines,
Twitter Financial News Sentiment, and FNSPID were reviewed using published
sources. See `docs/evaluation-dataset-shortlist.md`. SEntFiN is the provisional
preferred candidate; rights, exact revisions, duplicate/model-overlap checks,
and final-test freezing remain pending. No additional checkbox is complete.

**SEntFiN release review (2026-10-10):** Immutable dataset/model repository
revisions and metadata/file identities are recorded in `configs/asset-revisions.json`.
See `docs/sentfin-asset-review.md` for the local-research scope assessment and
unresolved public-text rights, model-weight license, and overlap evidence. No
CSV files or weights were acquired. The broader terms/revisions task and final
independent-test task remain unchecked.

**Dataset loading verification (2026-10-10):** PhraseBank asset terms/revision
review and local acquisition of PhraseBank/SEntFiN are complete. Typed source
loaders, checksummed acquisition, and schema validation pass on real pinned
artifacts; synthetic tests run offline. See `docs/dataset-loading.md` for counts,
source discrepancies, duplicates, and the limited exact-overlap check. The terms
checkbox was left open at that stage because FiQA was deferred. Full EDA,
split design, and independent final-test selection remain incomplete.

**FiQA loading verification (2026-10-10):** The original noncommercial-use
terms support local research. An accessible participant-hosted original-format
copy was pinned, acquired, and validated; see `docs/fiqa-asset-review.md`. Its
1,113 training texts contain 1,178 scored annotations, and 192 test inputs have
no gold scores. Scores, targets, aspects, snippets, groups, and source splits are
preserved. One normalized original train/test text overlap is documented. The
selected dataset terms/revisions task is complete in the stated local scope;
the gated mirror’s MIT scope and organizer-byte equivalence remain unresolved.
No final test was selected, no thresholds assigned, and no models implemented.

**Label analysis verification (2026-10-10):** Reproducible class counts,
percentages, and imbalance ratios are complete for all PhraseBank agreement
variants and SEntFiN target annotations. FiQA original training scores have
continuous descriptive statistics and numeric sign counts; no class thresholds
are assigned and unscored test inputs are excluded. See
`reports/label-analysis.md`. The broader EDA checkbox remains open: lengths,
duplicate/ambiguity investigation, and annotation-quality analysis are incomplete.
No new partitions, final-test selection, or models were implemented.

**Text-length verification (2026-10-10):** Original character and whitespace-word
length summaries, inclusive percentiles, and fixed descriptive length flags are
complete for each PhraseBank agreement subset/label, SEntFiN overall/per-target-
label headline groups, and FiQA training texts by source type. See
`reports/length-analysis.md`. SEntFiN label groups count a headline once per
distinct label; FiQA counts source texts rather than annotations. The supplied
SEntFiN word counts differ from our rule for 88 headlines and remain preserved.
No records were filtered and no model tokenization, thresholds, or splits were
implemented. The broader EDA checkbox remains open for duplicate/ambiguity and
annotation-quality review.

**Duplicate/annotation review verification (2026-10-10):** Exact grouping,
context-aware gold-conflict checks, and conservative word-trigram near-duplicate
screening are complete. See `reports/annotation-review.md` and
`docs/annotation-review-findings.md` for qualitative inspection and limitations.
PhraseBank has eight exact groups (two conflicting); SEntFiN has 67 (seven
same-target conflicts). FiQA has one original train/test overlap. The 25 near
pairs include one FiQA–SEntFiN overlap. FiQA has 49 nonmatching snippet values
across 45 training annotations; raw values and scores are preserved. Agreement
subset metadata was reviewed; individual rater statistics are unavailable. The
descriptive analysis task is complete, while adjudication, partition design,
final-test selection, dataset cards, and figures remain pending.

**Partition preparation verification (2026-10-10):** The versioned filtering policy
and deterministic seed-42 group assignment are implemented and verified. See
`docs/split-policy.md` and `reports/split-preparation.md`. Processed views retain
4,830 PhraseBank sentences, 10,682 SEntFiN headlines, and 1,112 scored FiQA texts;
27 source records are quarantined, 61 redundant copies preserve lineage, and all
192 original unscored FiQA inputs remain outside supervised partitions. All 48
flagged FiQA annotations are excluded from snippet supervision and test candidates.
Known exact/group/near-edge overlap and lineage checks pass. Repeated real-data
runs produce identical content-addressed artifacts; synthetic tests verify policy
changes preserve prior artifacts and corrupted caches fail. This completes the
partition strategy task only. Independent benchmark curation/freezing, ambiguity
adjudication, dataset cards, and EDA figures remain incomplete; no models exist.

**Benchmark freeze verification (2026-10-10):** `sentfin-independent-v1` is curated
and frozen before models: 1,029 headlines, 1,360 target annotations, 1,028 duplicate
groups. Two malformed-target headline groups are excluded without repairing gold.
See `docs/evaluation-contract.md`, `configs/benchmark-freeze.json`, and
`reports/frozen-benchmark.md`. Fixed cohorts separate single-target controls,
multi-target examples, and mixed-target sentiment. Repeat runs reproduce the same
receipt/artifacts; corruption and resplitting are rejected. The verified development
loader returns only train/validation records and excludes frozen/curation groups.
This completes the external-test curation task with qualified independence:
accessible-corpus checks pass, but complete FinBERT pretraining exposure remains
unknown. Five-company US-market coverage is insufficient; no such performance
claim is supported. Model weight terms remain unresolved. Dataset cards/figures
and model implementation are still pending. Local tests and Ruff pass; hosted CI
has not been run for this change.

**Dataset-card/EDA verification (2026-10-10):** Three cards in
`docs/dataset-cards/` document source identity, terms, schemas, labels, annotation
quality, intended roles, partitions, and limitations. The development-only notebook
`notebooks/01_dataset_eda.ipynb` calls the tested EDA command; all three code cells
executed in a fresh project kernel. Five exported figures were visually inspected.
`reports/eda-summary.json` pins the source/freeze/development identities and exact
aggregate values; `reports/phase1-eda.md` and `docs/phase1-findings.md` explain the
14,963 retained development texts and distinct counting units. Repeated local
generation produces identical figures/aggregate files and preserves the frozen
receipt. Optional Matplotlib/ipykernel dependencies are locked in the EDA group;
ingestion and domain logic still use the standard library. All 136 tests, Ruff,
and the package build pass locally; hosted Actions remain unverified.

**Phase 1 status: complete within the documented local research scope.** All eight
tasks and the code/notebook/cards/split-manifest deliverables are verified.
Known duplicate boundaries and immutable holdout checks pass. Unresolved rights,
pretraining exposure, annotation quality, and five-company coverage limitations
remain documented rather than treated as resolved. No model or prediction work
has started; Phase 2 is the next separately requested phase.

**Deliverables:** Data-preparation code, EDA notebook, dataset cards, split manifest.

**Acceptance criteria:** Runs from raw authorized inputs reproducibly; partitions have no known duplicate overlap; independent-test limitations documented.

## Phase 2 — Traditional NLP baselines

**Goal:** Meaningful non-transformer benchmarks.

### Tasks
- [x] Implement VADER with explicitly documented 3-class thresholds.
- [x] Train TF-IDF + Logistic Regression pipeline (fixed initial configuration; train-only fit, validation-only check).
- [ ] Train TF-IDF + Linear SVM pipeline.
- [ ] Fit all learned preprocessing **only on training data**.
- [ ] Tune using validation/CV on training data, never final test.
- [ ] Compute accuracy, macro F1, per-class precision/recall, confusion matrix, latency.
- [ ] Perform qualitative error analysis; persist configs and model artifacts.
- [ ] Add unit tests for preprocessing and label handling.

**Deliverables:** Model comparison report, baseline inference interface.

**VADER verification (2026-10-11):** The MIT-licensed standalone package 3.3.2
and bundled lexicon/emoji resources are reviewed, locked, and used unchanged.
Typed inference preserves input cues and applies inclusive negative <= -0.05 /
positive >= 0.05 compound thresholds. See `docs/vader.md` and `configs/vader.json`.
The validation-only check covers 518 PhraseBank sentences and 781 single-target
SEntFiN headlines, with training-majority controls and no threshold search.
Macro F1 is 0.4895 / 0.4787 respectively; these are development scores, not final
evaluation. Aggregate results/configuration/resource digests/code provenance are
in `reports/experiments/vader-validation.md` and JSON; local per-record predictions
remain ignored. The frozen benchmark and parent split are unchanged. Tests cover
scoring boundaries/rules, invalid inputs, deterministic inference, metric axes,
cohort exclusion, training-majority selection, and output privacy. All 169 tests,
Ruff, and package build pass locally; hosted CI is unverified. Only the VADER
implementation checkbox is complete: other baseline models, tuning, broader
comparisons, preprocessing tests, and qualitative error analysis remain pending.

**Acceptance criteria:** Every metric traceable to split, configuration, and model version.

**Logistic Regression verification (2026-10-11):** Reviewed BSD-3-Clause
scikit-learn 1.7.2 is locked. Separate word unigram/bigram TF-IDF pipelines fit
vocabulary, IDF, and classifier on 3,785 PhraseBank / 6,299 single-target SEntFiN
training texts only; 518 / 781 validation texts are scored with persisted models.
One fixed configuration is recorded in `configs/tfidf-logreg.json`; no tuning,
FiQA classification, mixed-target sentence gold, or final-test evaluation occurs.
Validation accuracy is 0.7490 / 0.8182 and macro F1 is 0.6696 / 0.8165, versus
VADER's 0.4895 / 0.4787 macro F1 on the same rows. See `docs/tfidf-logreg.md`
and `reports/experiments/tfidf-logreg-validation.md` plus aggregate JSON.
Versioned fitted pipelines and per-record predictions stay ignored; artifact
hashes, split/freeze/config/lock/code identities, parameters, and timings are
logged. Repeated predictions/metrics match, and earlier artifacts remain valid.
All 191 tests (including optional plotting), required Ruff checks, and sdist/wheel
builds pass locally; hosted CI remains unverified. Train-only vocabulary/IDF,
preprocessing/labels, convergence handling, local round trips, holdout rejection,
and output privacy have synthetic tests. Only the Logistic Regression task is
newly complete; the remaining broader Phase 2 tasks await SVM, development-only
tuning, full comparisons, and qualitative error analysis.

## Phase 3 — Financial transformer experiments

**Goal:** Reproducible transformer benchmarking and meaningful fine-tuning.

### Tasks
- [ ] Build batched inference for `ProsusAI/finbert` with class-probability output.
- [ ] Record evaluation overlap caveat with Financial PhraseBank.
- [ ] Decide whether to fine-tune a base financial-language model or adapt an existing sentiment checkpoint; document difference.
- [ ] Run controlled training experiments (learning rate, length, batch size); early stopping on validation only.
- [ ] Track dataset version, seeds, model checkpoint, metrics, latency in MLflow.
- [ ] Evaluate on independent frozen test set once final configuration is selected.
- [ ] Analyze calibration, negation, financial jargon, ambiguity, and class-specific failures.

**Deliverables:** Inference module, training pipeline, comparison report, versioned best model.

**Acceptance criteria:** Reproducible runs and credible, clearly qualified external evaluation.

## Phase 4 — Company-targeted sentiment

**Goal:** Distinguish sentiments about different companies in the same article.

### Tasks
- [ ] Build ticker/name/alias mapping for initial five companies.
- [ ] Implement entity identification and explicit ambiguity handling.
- [ ] Define target-level labels and create/validate an entity-labeled test collection.
- [ ] Compare article-wide FinBERT against a target-aware approach (e.g. target + text pair).
- [ ] Evaluate macro F1 **per company target** and on multi-company examples.
- [ ] Track errors from aliases, comparisons, neutral mentions, and conflicting cues.
- [ ] Expose `predict(text, target_company)` returning probabilities and model version.

**Deliverables:** Target-sentiment service and independent target-level evaluation.

**Acceptance criteria:** One article can yield distinct, evaluated sentiments for multiple companies; unsupported entities fail gracefully.

### Milestone M1 — NLP research ready

- [ ] Baselines and transformer compared fairly.
- [ ] Company-specific model evaluated on target-labeled examples.
- [ ] Results, failure cases, and limitations documented.

## Phase 5 — Sentiment aggregation and time-series analytics

**Goal:** Convert news predictions into reliable, time-indexed company signals.

### Tasks
- [ ] Standardize signed score: `P(positive) - P(negative)`.
- [ ] Implement rolling counts, averages, positive/negative ratios, momentum, dispersion.
- [ ] Explore relevance weighting or exponential time decay as *experiments*.
- [ ] Deduplicate syndicated stories; preserve source/provenance.
- [ ] Define missing-news behavior and minimum sample-size indicators.
- [ ] Store window start/end, news availability cutoff, company, aggregation version.
- [ ] Visualize news volume and sentiment changes over time.

**Deliverables:** Versioned feature pipeline and exploratory visualizations.

**Acceptance criteria:** Any aggregate can be reproduced from a known set of articles and cutoff time.

## Phase 6 — Downstream market prediction experiment

**Goal:** Test the **incremental predictive value** of sentiment, not trading profitability.

### Experimental contract

- **Decision time:** immediately after regular US trading-session close.
- **Target:** whether next trading-day adjusted close-to-close return is positive.
- **Feature cutoff:** no market or news observation that was unavailable at decision time.
- **Validation:** chronological train/validation/test with rolling or expanding walk-forward analysis.
- **Stock scope:** start with five selected tickers; document survivorship and selection bias.

### Tasks
- [ ] Ingest documented and licensed daily price data and exchange calendar.
- [ ] Align `published_at`, `ingested_at`, market sessions and target windows correctly.
- [ ] Engineer past returns, volume, moving averages, volatility and sentiment features.
- [ ] Implement a naive benchmark, market-only Logistic Regression and LightGBM.
- [ ] Compare LightGBM market-only with market+VADER, market+FinBERT, market+target-aware sentiment.
- [ ] Report balanced accuracy, ROC-AUC, macro F1, log loss, calibration, class balance, and uncertainty.
- [ ] Compare across companies and time periods; investigate instability and ablations.
- [ ] Optionally test GRU **only after** establishing tabular baselines.
- [ ] State explicitly when adding sentiment does not help.

**Deliverables:** Forecasting notebooks/modules, walk-forward report, model artifacts.

**Acceptance criteria:** An independently reproducible comparison using time-valid data; no leakage or exaggerated financial claims.

### Milestone M2 — Financial ML study ready

- [ ] Sentiment feature series generated and audited.
- [ ] Market-only vs sentiment-enhanced results documented.
- [ ] Negative results and uncertainty reported transparently.

## Phase 7 — Scheduled ingestion, storage, and API

**Goal:** Expose the NLP platform through production-like interfaces.

### Tasks
- [ ] Choose a permitted news provider and document rate limits and use rights.
- [ ] Create a provider adapter supporting incremental retrieval, retry/backoff, and idempotency.
- [ ] Add PostgreSQL tables and Alembic migrations for companies, article metadata, provider IDs, entity links, sentiment predictions, aggregates, prices, and forecasts.
- [ ] Persist `published_at`, `ingested_at`, provider update time (if available), inference time, model version.
- [ ] Avoid storing/re-exposing full article text unless licensed.
- [ ] Run scheduled ingestion and inference; log failures and retries.
- [ ] Add FastAPI endpoints:
  - [ ] `GET /api/v1/stocks`
  - [ ] `GET /api/v1/stocks/{ticker}/news`
  - [ ] `GET /api/v1/stocks/{ticker}/sentiment`
  - [ ] `GET /api/v1/stocks/{ticker}/sentiment/history`
  - [ ] `GET /api/v1/stocks/{ticker}/prediction`
  - [ ] `GET /api/v1/models/metrics`
- [ ] Add typed response schemas, pagination, health endpoint, unit/integration tests.

**Deliverables:** Database migrations, scheduled worker, documented HTTP API.

**Acceptance criteria:** Ingestion reruns do not duplicate articles; API serves versioned predictions from the database; failures are visible and recoverable.

## Phase 8 — Dashboard

**Goal:** Make sentiment analysis the visible center of FinSight.

### Tasks
- [ ] Company selector and current sentiment overview.
- [ ] News feed with per-company sentiment, timestamps, and source links.
- [ ] Sentiment trend and article-volume charts.
- [ ] Multi-company headline examples and target-specific output.
- [ ] Research/model comparison section with metrics and confusion matrices.
- [ ] Secondary prediction panel with cutoff, horizon, confidence, historical performance, and disclaimers.
- [ ] Responsive design and loading/empty/error states.

**Deliverables:** Next.js frontend connected to FastAPI.

**Acceptance criteria:** A viewer can inspect why a company has its sentiment score and see how model quality was assessed.

## Phase 9 — Deployment and monitoring

**Goal:** Operate FinSight reliably on Docker Compose / Proxmox.

### Tasks
- [ ] Containerize worker, API, frontend, database (or use existing managed DB).
- [ ] Add health checks, structured logs, resource limits, and secrets management.
- [ ] Build reproducible CI for lint, tests, and images.
- [ ] Publish deployment instructions without disclosing network secrets.
- [ ] Configure Prometheus/Grafana for ingestion success, freshness, failures, latency, and service uptime.
- [ ] Monitor sentiment distribution and data quality; separate drift **alerts** from automatic retraining.
- [ ] Establish backups/recovery test and basic API security.

**Deliverables:** Running deployment, operator README, monitoring views.

**Acceptance criteria:** Services recover after restart, health checks pass, a failed ingestion is diagnosable, and backups can be restored.

## Phase 10 — Portfolio and documentation

**Goal:** Turn project work into interview-ready evidence.

### Tasks
- [ ] Write an outcome-focused README with one-paragraph problem statement and architecture diagram.
- [ ] Publish reproducible (licensed) examples and redacted configuration.
- [ ] Include sentiment benchmarking results, error analysis, and company-target evaluation.
- [ ] Include downstream stock experiment with baseline, split design, and limitations.
- [ ] Add screenshots, a short demo, API documentation, and deployment steps.
- [ ] Explain engineering tradeoffs, failure cases, and next steps.
- [ ] Draft concise resume bullets using **actual measured results**.

**Deliverables:** Portfolio-ready GitHub repository and demonstration.

**Acceptance criteria:** Another developer can run the demo from documented steps and verify key claims from published experiments.

### Milestone M3 — Finished portfolio application

- [ ] Demonstrable sentiment intelligence application.
- [ ] Independently evaluated sentiment model and target-aware results.
- [ ] Rigorous, clearly qualified stock-prediction ablation.
- [ ] Tests, CI, deployment, monitoring, documentation.

---

## 6. Definition of done (every task)

A task is complete only when:

1. The requested implementation exists and respects current scope.
2. Unit/integration tests appropriate to the change have been added and pass.
3. Formatting/linting pass; public interfaces have types and documentation.
4. Dataset/model assumptions, reproducibility settings, and licenses are recorded when relevant.
5. Existing functionality is not silently broken.
6. The task checkbox is updated **only after verification**.
7. The change summary lists tests run and known limitations.

## 7. Codex workflow

1. Put this file at repository root as `ROADMAP.md`.
2. Maintain `AGENTS.md` with coding standards, no-leakage rules, and instruction to work **one phase or bounded task at a time**.
3. For each task, ask Codex to read both files, inspect existing code, state a short plan, implement only the named scope, add tests, and report verification.
4. Review dataset decisions, model comparisons, data-leakage risks, and research conclusions yourself.
5. Commit at the end of each verified task; keep concise experiment logs in `reports/experiments/`.

### Reusable Codex prompt

> Read `ROADMAP.md` and `AGENTS.md`, then inspect the current repository. Work **only on Phase [N], task [X]**. Before editing, summarize the current state and propose a short plan. Implement a maintainable, typed solution with appropriate tests; avoid unnecessary dependencies and changes outside scope. Run relevant tests and lint. Report files changed, commands/results, any unresolved risks, and which roadmap checkbox can now be marked complete. Do not start the next task automatically.

### First Codex prompt

> Read `ROADMAP.md` and implement **Phase 0 only**. Create a clean, installable Python package with `uv`, Ruff, pytest, GitHub Actions, environment-variable examples, basic documentation, and an `AGENTS.md` covering coding discipline and research guardrails. Do not add data ingestion or model training. Run tests and lint and summarize all design choices.

## 8. Decision log (fill as you go)

| Date | Decision | Alternatives | Reason | Impact |
|---|---|---|---|---|
| TBD | News provider and license | TBD | TBD | Phase 1/7 |
| 2026-10-10 | Freeze curated SEntFiN test groups as `sentfin-independent-v1` | PhraseBank (known supervision overlap); FiQA (continuous scores/provenance uncertainty) | Native human target labels; no known accessible-corpus overlap | Phase 1/3; unknown pretraining exposure and limited US-company coverage |
| TBD | Target-sentiment data annotation strategy | TBD | TBD | Phase 4 |
| TBD | Sentiment aggregation definition | TBD | TBD | Phase 5 |
| TBD | Market data source and cutoff handling | TBD | TBD | Phase 6 |
| TBD | Scheduling framework | TBD | TBD | Phase 7 |

## 9. References to evaluate

- [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) — verify revision and license.
- [FiQA sentiment dataset](https://huggingface.co/datasets/TheFinAI/fiqa-sentiment-classification) — verify task schema and license.
- [ProsusAI FinBERT](https://huggingface.co/ProsusAI/finbert) — review training data overlap.
- [Alpha Vantage documentation](https://www.alphavantage.co/documentation/) — provider capabilities and limits may change.
- [Benzinga APIs](https://www.benzinga.com/apis/) — verify data-use rights.
- [SEC EDGAR](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data) — filings as optional event signals.

**Priority order:** NLP validity → target-aware sentiment → aggregation quality → downstream market comparison → product/deployment polish.
