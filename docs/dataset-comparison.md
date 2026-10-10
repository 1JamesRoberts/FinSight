# Financial PhraseBank vs FiQA

Reviewed: 2026-10-10. Scope: Phase 1, candidate comparison only.
This is a documentation review of public sources, not an empirical dataset audit.
No data files were downloaded, gated terms accepted, or processing implemented.
Exact artifact revisions, checksums, class distributions, duplicate rates, and
company coverage remain unverified. Neither candidate is approved for acquisition.

## Published structure

| Property | Financial PhraseBank | FiQA Task 1 / TheFinAI mirror |
| --- | --- | --- |
| Unit | Financial-news sentence | Headline or post with a sentiment target/aspect |
| Published fields | `sentence`, `label` | `_id`, `sentence`, `target`, `aspect`, `score`, `type` |
| Labels | Negative, neutral, positive | Continuous sentiment score; original task range is -1 to +1 |
| Company supervision | No separate target field in this loader | Explicit target field; identity/ticker normalization still requires inspection |
| Published size | Card lists 4,846 at >=50% agreement; 4,217 at >=66%; 3,453 at >=75%; 2,264 unanimous | Mirror lists train 822, valid 117, test 234 (1,173 total) |
| Splits | No designed train/validation/test partition | Mirror provides three subsets; do not assume these equal original challenge splits |

Structure and counts are published claims from the
[PhraseBank card](https://huggingface.co/datasets/takala/financial_phrasebank) and
[FiQA mirror card](https://huggingface.co/datasets/TheFinAI/en-fiqa-absa).
The PhraseBank card also describes 4,840 sentences elsewhere: verify the actual
artifact count rather than silently choosing one number.

The [PhraseBank loader](https://huggingface.co/datasets/takala/financial_phrasebank/blob/main/financial_phrasebank.py)
encodes `ClassLabel` as 0=negative, 1=neutral, 2=positive, despite the card showing
string labels. It emits a single `train` container; this is not a held-out split.
It reads ISO-8859-1 text and separates sentence/label at the final `@`.
These are loader details, not permission to execute its download code.

The [original FiQA challenge](https://sites.google.com/view/fiqa/home) specifies
aspect-target sentiment regression, evaluated with MSE and R-squared. Its example
allows multiple target annotations for one sentence. A three-class conversion
would therefore be a FinSight experimental choice, not an original gold label.

## Provenance and annotation

The [PhraseBank research paper](https://arxiv.org/html/1307.5336v2) describes English
news about OMX Helsinki companies, sourced through LexisNexis. Sentences were
annotated from an investor perspective using their explicit content. Sixteen
financially knowledgeable annotators participated, with five to eight judgments
per sentence. Positive-vs-neutral boundaries were a source of disagreement.

The FiQA challenge supplies headlines and microblog messages. Keep its Task 1
sentiment corpus separate from Task 2 question answering; Task 2 source descriptions
are not evidence of Task 1 provenance. The
[FLUE paper, section 3.1.1 and Table 1](https://arxiv.org/pdf/2211.00083)
confirms the sentiment-regression task and the 822/117/234 benchmark sizes.
Detailed publisher coverage, original annotation procedures, flattening of
multiple targets, and preservation of snippet annotations need further tracing.

## Access and terms findings

PhraseBank's card declares CC BY-NC-SA 3.0 and directs commercial-use inquiries
to its authors. The loader's Apache-2.0 header covers its code and must not be
substituted for the dataset license. The
[Creative Commons deed](https://creativecommons.org/licenses/by-nc-sa/3.0/)
requires attribution, noncommercial use, and share-alike for distributed
adaptations, subject to the license's scope. It also warns that other rights may
apply. Training-use interpretation, underlying article rights, and public demo
rights still require an asset-specific review; no conclusion about model-weight
licensing is made here.

FiQA's mirror advertises MIT, requires a logged-in access form, and asks users to
cite FinBen. The original challenge states that training and testing data are
available only for noncommercial use. The FLUE table describes FiQA as public;
that does not settle redistribution rights. Resolve the provenance and scope of
these differing statements before acquisition, training, or publication. Do not
assume a mirror can grant broader rights over source material.

The roadmap's FiQA URL currently redirects to `TheFinAI/en-fiqa-absa`; the card's
example still names `ChanceFocus/fiqa-sentiment-classification`. Record this identity
mismatch before choosing and pinning an artifact. PhraseBank's current viewer
reports an unsupported legacy dataset script. Later ingestion should inspect an
authorized, pinned artifact directly rather than copying outdated loader examples.

## Recommended roles (provisional)

- **PhraseBank:** start traditional three-class baseline development here if terms
  are cleared. Its label format matches FinSight directly. An agreement subset
  can support a robustness comparison; treat subsets as potentially overlapping,
  not independent datasets, and verify membership before creating splits.
- **FiQA:** retain as a candidate for target-aware sentiment and external
  evaluation. Preserve targets, aspects, scores, and source type. Define any
  neutral band before final evaluation; select thresholds using training/validation
  only and document information lost through discretization.
- **Independent benchmark:** neither candidate is accepted as the frozen final
  test yet. [ProsusAI's FinBERT card](https://huggingface.co/ProsusAI/finbert)
  confirms PhraseBank fine-tuning. A new random PhraseBank split does not remove
  that contamination. FiQA independence still needs a checkpoint-specific training
  data audit and cross-corpus duplicate checks. The
  [original FinBERT paper](https://arxiv.org/html/1908.10063) includes FiQA
  regression experiments; this does not identify the published classifier's
  training samples. See the
  [evaluation shortlist](evaluation-dataset-shortlist.md) for alternative candidates.

These are recommendations inferred from task fit, not measured performance.
Do not pool the corpora immediately: sentence-wide classes and target/aspect scores
represent different supervision. Report headline and post performance separately
if FiQA is used. Group all annotations of the same source text into one partition
to prevent target rows leaking across splits. Preserve original split assignments
for audit even if a documented replacement is later necessary.

Neither published schema establishes a timestamped news/price panel or sufficient
coverage of AAPL, MSFT, NVDA, AMZN, and GOOGL. Company coverage, full-article
performance, and multi-company evaluation remain future empirical questions.
Neither corpus should be assumed suitable for stock-prediction backtesting.

## Next bounded task

Complete the asset terms records in `datasets.md`, resolve FiQA's licensing/access
questions, and pin exact authorized revisions. Then implement loading and validation
as a separate task. Leave the final test untouched during EDA and tuning.
