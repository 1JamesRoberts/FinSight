# Independent sentiment evaluation: dataset shortlist

Reviewed: 2026-10-10. Scope: Phase 1, evaluation candidate review.
Recommendation is provisional; no final test set is selected or frozen.
The follow-up [release review](sentfin-asset-review.md) now pins SEntFiN and
FinBERT revisions and records a scoped local-research usage assessment. Terms
findings below describe the shortlist stage; follow-up decisions take precedence.
No datasets or weights were downloaded and no access agreement was submitted.
Published statistics below are not measurements of pinned local artifacts.

## Decision

Prioritize **SEntFiN 1.0** for the next asset-review task. It offers the closest
supervision match for three-class, company-targeted sentiment. Keep SemEval-2017
Task 5 headlines as a backup, and Twitter Financial News Sentiment as an optional
social-media robustness benchmark. Exclude FNSPID's generated sentiment from the
human gold-standard shortlist.

This ranking is an inference about research fit, not a performance claim or an
acquisition approval. A supplementary independently annotated US-company set
would address coverage gaps if the shortlisted corpora cannot do so.

## Candidate comparison

| Candidate | Published scope and labels | Target support | Provisional role | Main unresolved issue |
| --- | --- | --- | --- | --- |
| SEntFiN 1.0 | 10,753 human-annotated headlines; positive/negative/neutral | 14,404 entity annotations; 2,847 multi-entity headlines | First choice for external and target-aware evaluation | Indian-market transfer; dataset/article rights and exact-checkpoint overlap audit |
| SemEval-2017 Task 5, subtask 2 | 1,647 news headlines/statements; continuous -1 to +1 scores | Company-specific scores | Backup target-aware news benchmark | Release agreement/access, class conversion, duplicate overlap |
| zeroshot/twitter-financial-news-sentiment | Roughly 12,000 finance tweets; bearish/bullish/neutral | Text-level label; no separate target-label field in viewer | Optional robustness benchmark | Annotation details, inconsistent counts, social-media domain and rights |
| FNSPID | Large news/price corpus; paper describes ChatGPT sentiment scoring | Stock-associated records, not established human target gold labels | Excluded as human evaluation ground truth | Generated labels and uncertain released sentiment coverage |

Sources and qualifications follow; counts are specific to the cited publications
or cards and may differ from files eventually acquired.

## SEntFiN: strongest task fit

The [authors' paper, section 3](https://arxiv.org/html/2305.12257) describes Economic
Times headlines from 2011-2015. Three business/management students annotated
entities, with consensus and curator resolution. Table 1 reports 5,074 positive,
3,814 negative, and 5,516 neutral entity labels. The paper reports 1,233 headlines
with conflicting entity sentiments. Entity coverage includes financial entities
beyond companies; headline evaluation will not establish full-article quality.
The narrative inconsistently says more than two entities while the table groups
multiple entities; verify multiplicity from the artifact later.

The [release repository](https://github.com/pyRis/SEntFiN) lists `SEntFiN.csv` and an
entity database. Its [MIT license](https://github.com/pyRis/SEntFiN/blob/main/LICENSE)
is attributed to the paper's authors, but uses software/documentation wording.
Confirm its scope for the CSV, retain attribution/license notices, and record how
underlying publisher material may be stored or displayed before acquisition.
The paper's CC BY license is not automatically a dataset license.

The corpus is a promising external test candidate, but publication after FinBERT's
2019 paper does not prove no overlap: its underlying headlines are older.
Exact coverage of FinSight's five US companies is unverified. Treat results as
transfer to this corpus, not evidence for every company in the dashboard.

## SemEval: useful but requires label adaptation

The [task paper, sections 2.2-2.4](https://aclanthology.org/S17-2089.pdf) traces news
to RSS sources including Reuters, Bloomberg, and Forbes, collected in August to
November 2015. Three financial experts annotated, with a fourth consolidating.
The schema describes title text, company, and continuous sentiment score;
headline annotator rank correlation is reported as 0.54. Verify actual field
names and split sizes from the authorized artifact later.

The [organizers' access page](https://alt.qcri.org/semeval2017/task5/index.php?id=data-and-tools)
requires a release agreement before download and an emailed form for gold labels.
The linked headline agreement could not be retrieved during this review; its
training, storage, and redistribution permissions remain unresolved. No form was
submitted or organizer contacted. An accessible third-party copy would not remove
these obligations.

Continuous scores do not supply an agreed three-class neutral boundary. Define
and justify any conversion on development data before freezing evaluation. Audit
shared text with FiQA and training corpora rather than assuming distinct names
mean distinct examples.

## Twitter: secondary domain only

The [publisher's card](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)
reports Twitter API collection and MIT licensing. The viewer exposes `text` and
`label`; the stated mapping is 0=bearish, 1=bullish, 2=neutral. Cashtags may occur
in text but do not establish separate gold labels for each company.

Counts conflict: the description says 11,932, total rows show 11,931, and the card's
split table sums to 12,424. Exact counts require pinned-file verification. Annotator
identity, procedure, and agreement are not explained in the reviewed card; do not
assume human consensus. Resolve label provenance and Twitter/source rights before
use. Bearish/bullish is not automatically equivalent to investor-impact sentiment
in news. Do not combine this test with news results into one headline metric.

## FNSPID: exclude generated labels from the gold benchmark

The [FNSPID paper, section 3](https://arxiv.org/html/2402.06698) describes ChatGPT
scores from 1 to 5 on summaries, including 402,546 scored news items. Its ethics
section also says sentiment scores were excluded from final analysis, so released
coverage needs clarification. The [original repository](https://github.com/Zdong104/FNSPID_Financial_News_Dataset)
contains a noncommercial-use restriction in its disclaimer; rights over collected
news need a separate review.

Model-generated labels can support other explicitly qualified experiments, but
are not independent human judgments. FNSPID is not selected for NLP accuracy
claims. No price-data acquisition or prediction work is authorized by this review.

## FinBERT independence: what the sources establish

The [ProsusAI model card](https://huggingface.co/ProsusAI/finbert) confirms
PhraseBank fine-tuning. The [2019 paper, section 4.2.3](https://arxiv.org/html/1908.10063)
reports FiQA regression experiments using cross-validation. That history does
not establish that the published three-class checkpoint trained on FiQA, but
prevents treating FiQA independence as settled without checkpoint provenance.

Neither source provides a complete sample-level training manifest for the exact
checkpoint we will use. This review found no documented inclusion of the three
new candidates in that original training description; absence from documentation
is not proof of independence. Later checkpoints fine-tuned on SEntFiN, SemEval,
or Twitter require their own held-out audits and cannot inherit this assessment.
Separate supervised-label overlap from possible exposure to raw text during
language-model pretraining; report uncertainty in both.

## Acceptance gates before freezing a test

1. Complete rights records for the chosen artifact, pin its immutable revision,
   and record checksums after authorized acquisition. Specify local research and
   public display/redistribution permissions separately.
2. Pin the exact model revision; record training evidence and unresolved exposure.
   Check exact/near duplicates against our development sets and documented model
   training corpora where available. Record limits where those corpora are unavailable.
3. Validate entity types, class mapping, annotation quality, and company coverage.
   Have a curator prepare counts/strata without using final-test labels to tune
   models or thresholds.
4. Keep all targets from a headline and duplicate/syndicated stories together.
   Reserve development material separately; keep final-test material out of
   training, tuning, threshold selection, and iterative model error analysis.
5. Freeze a versioned manifest before comparisons. Report headline-level grouping,
   per-target metrics, and separate multi-entity/conflicting-sentiment results.
   Use a single-company cohort for sentence-wide controls; broadcasting an
   article-wide score to targets is a named baseline, not invented article labels.

If SEntFiN passes these gates, reserve an external test subset before developing
any target-aware model on the remaining material. If rights, coverage, or overlap
cannot be resolved, plan a permitted US-company headline collection with blinded
human annotation, second-review/adjudication, and predeclared sampling. That is
a fallback plan, not a dataset created or approved in this task.

## Task status

Shortlist and recommendation: complete. Acquisition, revision pinning, empirical
checks, and final independent-test selection: pending. The roadmap's final-test
checkbox remains unchecked.
