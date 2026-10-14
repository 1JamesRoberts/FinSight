# FinSight — All-model comparison by cohort

Updated 2026-10-12. This report consolidates recorded Phase 2 and Phase 3
development results; it performs no new training, inference or test evaluation.

In the initial fixed-configuration comparison, SEntFiN-trained RoBERTa Base had
the highest observed SEntFiN validation macro F1
(0.8849). PhraseBank-trained RoBERTa Base has the highest observed PhraseBank
macro F1 among models trained in this project (0.8868), narrowly ahead of
PhraseBank-trained DeBERTa Base (0.8857). These are descriptive validation
leaders, not proven winners on an independent test.

## Evaluation contract

| Cohort | Training texts per fitted model | Validation texts | Validation negative / neutral / positive |
| --- | ---: | ---: | --- |
| PhraseBank ≥50% agreement master view | 3,785 | 518 | 69 / 297 / 152 |
| SEntFiN single-target headlines | 6,299 | 781 | 247 / 252 / 282 |

PhraseBank nested agreement subsets are not separate cohorts. Each corpus uses
its existing group-based frozen development split. No pooling, sequential
PhraseBank→SEntFiN training, validation fitting or test scoring is included.
FiQA and multi-target SEntFiN have no comparable sentence-classification runs;
FiQA has continuous scores, and multi-target sentiment needs a target-aware
evaluation protocol. Both remain outside these tables.

Accuracy measures the fraction correct. Macro F1 averages negative, neutral and
positive F1 equally, which is useful with PhraseBank’s neutral-heavy labels.
Balanced weights derive only from training labels. Unweighted means equal
sample weights. Selected LR/SVM rows maximize validation macro F1 over six
C/weight choices; duplicate configurations remain visible to distinguish the
initial experiment from the selected result.

## PhraseBank validation

| Model / variant | Project training | Selection or evaluation role | Accuracy | Macro F1 |
| --- | --- | --- | ---: | ---: |
| [Training majority](phase2-comparison.md) | PhraseBank only | Fixed control | 0.5734 | 0.2429 |
| [VADER](phase2-comparison.md) | None (lexicon/rules) | Fixed control | 0.5560 | 0.4895 |
| [TF-IDF + Logistic Regression — unweighted, C=1](tfidf-tuning.md) | PhraseBank only | Fixed configuration | 0.7490 | 0.6696 |
| [TF-IDF + Logistic Regression — balanced, C=1](tfidf-tuning.md) | PhraseBank only | Fixed configuration | 0.7645 | 0.7378 |
| [TF-IDF + Logistic Regression — selected, C=10, balanced](phase2-comparison.md) | PhraseBank only | Best of 6 on same validation | 0.7780 | 0.7446 |
| [TF-IDF + Linear SVM — unweighted, C=1](tfidf-tuning.md) | PhraseBank only | Fixed configuration | 0.7799 | 0.7344 |
| [TF-IDF + Linear SVM — balanced, C=1](tfidf-tuning.md) | PhraseBank only | Fixed configuration | 0.7819 | 0.7444 |
| [TF-IDF + Linear SVM — selected, C=1, balanced](phase2-comparison.md) | PhraseBank only | Best of 6 on same validation | 0.7819 | 0.7444 |
| [FinBERT — unchanged](finbert-validation.md) | None; upstream sentiment supervision | No project selection; PB exposure diagnostic | 0.8822 | 0.8823 |
| [RoBERTa Base — PhraseBank training](roberta_phrasebank.md) | PhraseBank only | PhraseBank epoch 5 | 0.8938 | 0.8868 |
| [DeBERTa-v3 Base — PhraseBank training](deberta_phrasebank.md) | PhraseBank only | PhraseBank epoch 4 | 0.8900 | 0.8857 |
| [Financial RoBERTa Large — unchanged](financial_roberta_phrasebank.md) | None; upstream sentiment supervision | No selection; supervised provenance incomplete | 0.7973 | 0.8142 |
| [Financial DeBERTa-v3 Small — unchanged](financial_deberta_phrasebank.md) | None; upstream sentiment supervision | No selection; apparent PB supervision diagnostic | 0.9826 | 0.9851 |

### Class F1 for the same variants

| Model / variant | Negative F1 | Neutral F1 | Positive F1 |
| --- | ---: | ---: | ---: |
| Training majority | 0.0000 | 0.7288 | 0.0000 |
| VADER | 0.3077 | 0.6157 | 0.5450 |
| TF-IDF + Logistic Regression — unweighted, C=1 | 0.5941 | 0.8264 | 0.5882 |
| TF-IDF + Logistic Regression — balanced, C=1 | 0.7206 | 0.8171 | 0.6758 |
| TF-IDF + Logistic Regression — selected, C=10, balanced | 0.6977 | 0.8287 | 0.7075 |
| TF-IDF + Linear SVM — unweighted, C=1 | 0.6783 | 0.8393 | 0.6857 |
| TF-IDF + Linear SVM — balanced, C=1 | 0.6992 | 0.8371 | 0.6969 |
| TF-IDF + Linear SVM — selected, C=1, balanced | 0.6992 | 0.8371 | 0.6969 |
| FinBERT — unchanged | 0.8993 | 0.8948 | 0.8528 |
| RoBERTa Base — PhraseBank training | 0.8857 | 0.9128 | 0.8617 |
| DeBERTa-v3 Base — PhraseBank training | 0.8923 | 0.9075 | 0.8571 |
| Financial RoBERTa Large — unchanged | 0.8811 | 0.8008 | 0.7606 |
| Financial DeBERTa-v3 Small — unchanged | 1.0000 | 0.9848 | 0.9705 |

## SEntFiN single-target validation

| Model / variant | Project training | Selection or evaluation role | Accuracy | Macro F1 |
| --- | --- | --- | ---: | ---: |
| [Training majority](phase2-comparison.md) | SEntFiN only | Fixed control | 0.3611 | 0.1769 |
| [VADER](phase2-comparison.md) | None (lexicon/rules) | Fixed control | 0.4738 | 0.4787 |
| [TF-IDF + Logistic Regression — unweighted, C=1](tfidf-tuning.md) | SEntFiN only | Fixed configuration | 0.8182 | 0.8165 |
| [TF-IDF + Logistic Regression — balanced, C=1](tfidf-tuning.md) | SEntFiN only | Fixed configuration | 0.8195 | 0.8181 |
| [TF-IDF + Logistic Regression — selected, C=1, balanced](phase2-comparison.md) | SEntFiN only | Best of 6 on same validation | 0.8195 | 0.8181 |
| [TF-IDF + Linear SVM — unweighted, C=1](tfidf-tuning.md) | SEntFiN only | Fixed configuration | 0.8092 | 0.8072 |
| [TF-IDF + Linear SVM — balanced, C=1](tfidf-tuning.md) | SEntFiN only | Fixed configuration | 0.8092 | 0.8075 |
| [TF-IDF + Linear SVM — selected, C=0.1, balanced](phase2-comparison.md) | SEntFiN only | Best of 6 on same validation | 0.8169 | 0.8160 |
| [FinBERT — unchanged](finbert-validation.md) | None; upstream sentiment supervision | No project selection | 0.7478 | 0.7492 |
| [FinBERT — SEntFiN adaptation](finbert-finetuning.md) | SEntFiN only | SEntFiN epoch 3 | 0.8771 | 0.8762 |
| [Financial RoBERTa Large — unchanged](financial_roberta.md) | None; upstream sentiment supervision | No project selection | 0.7439 | 0.7333 |
| [Financial DeBERTa-v3 Small — unchanged](financial_deberta.md) | None; upstream sentiment supervision | No project selection | 0.7388 | 0.7414 |
| [RoBERTa Base — SEntFiN training](roberta_base.md) | SEntFiN only | SEntFiN epoch 4 | 0.8860 | 0.8849 |
| [DeBERTa-v3 Base — SEntFiN training](deberta_base.md) | SEntFiN only | SEntFiN epoch 5 | 0.8745 | 0.8738 |
| [Financial RoBERTa Large — SEntFiN adaptation](financial_roberta_adapted.md) | SEntFiN only | SEntFiN epoch 5 | 0.8707 | 0.8696 |
| [Financial DeBERTa-v3 Small — SEntFiN adaptation](financial_deberta_adapted.md) | SEntFiN only | SEntFiN epoch 5 | 0.8745 | 0.8741 |
| [RoBERTa Base — PhraseBank training — transfer](roberta_phrasebank.md) | PhraseBank only | PhraseBank epoch 5; transfer after selection | 0.7836 | 0.7845 |
| [DeBERTa-v3 Base — PhraseBank training — transfer](deberta_phrasebank.md) | PhraseBank only | PhraseBank epoch 4; transfer after selection | 0.7798 | 0.7816 |

### Class F1 for the same variants

| Model / variant | Negative F1 | Neutral F1 | Positive F1 |
| --- | ---: | ---: | ---: |
| Training majority | 0.0000 | 0.0000 | 0.5306 |
| VADER | 0.5276 | 0.4340 | 0.4745 |
| TF-IDF + Logistic Regression — unweighted, C=1 | 0.8077 | 0.7907 | 0.8512 |
| TF-IDF + Logistic Regression — balanced, C=1 | 0.8075 | 0.7892 | 0.8577 |
| TF-IDF + Logistic Regression — selected, C=1, balanced | 0.8075 | 0.7892 | 0.8577 |
| TF-IDF + Linear SVM — unweighted, C=1 | 0.7992 | 0.7771 | 0.8454 |
| TF-IDF + Linear SVM — balanced, C=1 | 0.8000 | 0.7787 | 0.8438 |
| TF-IDF + Linear SVM — selected, C=0.1, balanced | 0.8109 | 0.7901 | 0.8470 |
| FinBERT — unchanged | 0.7743 | 0.6922 | 0.7811 |
| FinBERT — SEntFiN adaptation | 0.8815 | 0.8378 | 0.9094 |
| Financial RoBERTa Large — unchanged | 0.8317 | 0.5915 | 0.7765 |
| Financial DeBERTa-v3 Small — unchanged | 0.8060 | 0.6629 | 0.7554 |
| RoBERTa Base — SEntFiN training | 0.8939 | 0.8554 | 0.9053 |
| DeBERTa-v3 Base — SEntFiN training | 0.8984 | 0.8340 | 0.8889 |
| Financial RoBERTa Large — SEntFiN adaptation | 0.8830 | 0.8330 | 0.8927 |
| Financial DeBERTa-v3 Small — SEntFiN adaptation | 0.8921 | 0.8396 | 0.8904 |
| RoBERTa Base — PhraseBank training — transfer | 0.8302 | 0.7097 | 0.8136 |
| DeBERTa-v3 Base — PhraseBank training — transfer | 0.8176 | 0.7130 | 0.8143 |

## Training-corpus comparison and missing coverage

| Model artifact | PhraseBank validation macro F1 | SEntFiN validation macro F1 |
| --- | ---: | ---: |
| FinBERT unchanged | 0.8823 (known supervision exposure) | 0.7492 |
| RoBERTa Base trained on PhraseBank | 0.8868 (epoch selection) | 0.7845 (transfer only) |
| DeBERTa-v3 Base trained on PhraseBank | 0.8857 (epoch selection) | 0.7816 (transfer only) |
| RoBERTa Base trained on SEntFiN | Not evaluated | 0.8849 (epoch selection) |
| DeBERTa-v3 Base trained on SEntFiN | Not evaluated | 0.8738 (epoch selection) |
| FinBERT adapted on SEntFiN | Not evaluated | 0.8762 (epoch selection) |
| Financial RoBERTa Large, unchanged | 0.8142 (provenance incomplete) | 0.7333 |
| Financial RoBERTa Large, SEntFiN adapted | Not evaluated | 0.8696 |
| Financial DeBERTa-v3 Small, unchanged | 0.9851 (apparent supervision exposure) | 0.7414 |
| Financial DeBERTa-v3 Small, SEntFiN adapted | Not evaluated | 0.8741 |

LR/SVM each have distinct PhraseBank and SEntFiN fitted artifacts; their two
cohort scores do not measure transfer by one shared model. “Not evaluated” is
an explicit coverage gap, not a zero score. No missing score is inferred.

The PhraseBank-only bases select epochs on PhraseBank before SEntFiN transfer.
The SEntFiN-trained bases select epochs on SEntFiN itself. Their roughly 0.10
and 0.09 macro F1 gaps favor SEntFiN training on these development headlines,
but different selection sets and training sizes prevent isolating corpus effects.

On SEntFiN validation, RoBERTa Base’s 0.8849 macro F1 is 0.0668 above selected
TF-IDF Logistic Regression (0.8181). On PhraseBank, RoBERTa Base’s 0.8868 is
0.1421 above selected LR (0.7446). These compare development-selected models;
the small transformer gaps do not establish statistical superiority.

Financial RoBERTa adaptation improves SEntFiN macro F1 by 0.1363; Financial
DeBERTa by 0.1326; FinBERT by 0.1271. All retain their existing heads and train
all encoder parameters. Financial RoBERTa is Large and financial DeBERTa is
Small, while the new-head models are Base; differences cannot be attributed
solely to financial pretraining.

Financial DeBERTa is nearly perfect on PhraseBank (509/518 correct) but its source
description indicates supervision on that corpus, with incomplete split provenance.
This does not demonstrate superiority on unseen data. Exact overlap cannot be
confirmed from the available upstream information. Financial RoBERTa scores
413/518 correct; its incomplete provenance also prevents an independence claim.
Both rows use original sources, with no project training or checkpoint selection.

## Controlled learning-rate and seed follow-up

The same SEntFiN cohort now has ten fresh runs of RoBERTa Base and FinBERT.
Rates 1e-5/2e-5/3e-5 were compared at seed 42; then the selected rates were
repeated at seeds 43/44. Original tables above retain historical results.

| Model | Learning rate | Seed | Role | Accuracy | Macro F1 |
| --- | ---: | ---: | --- | ---: | ---: |
| [FinBERT](transformer-tuning-runs/finbert/lr-1e-5-seed-42/finbert.md) | 1e-05 | 42 | Rate selection | 0.8681 | 0.8663 |
| [FinBERT](transformer-tuning-runs/finbert/lr-2e-5-seed-42/finbert.md) | 2e-05 | 42 | Rate selection | 0.8771 | 0.8762 |
| [FinBERT](transformer-tuning-runs/finbert/lr-3e-5-seed-42/finbert.md) | 3e-05 | 42 | Rate selection | 0.8771 | 0.8766 |
| [RoBERTa Base](transformer-tuning-runs/roberta_base/lr-1e-5-seed-42/roberta_base.md) | 1e-05 | 42 | Rate selection | 0.8822 | 0.8810 |
| [RoBERTa Base](transformer-tuning-runs/roberta_base/lr-2e-5-seed-42/roberta_base.md) | 2e-05 | 42 | Rate selection | 0.8860 | 0.8849 |
| [RoBERTa Base](transformer-tuning-runs/roberta_base/lr-3e-5-seed-42/roberta_base.md) | 3e-05 | 42 | Rate selection | 0.8758 | 0.8749 |
| [FinBERT](transformer-tuning-runs/finbert/lr-3e-5-seed-43/finbert.md) | 3e-05 | 43 | Fixed-rate repeat | 0.8771 | 0.8756 |
| [FinBERT](transformer-tuning-runs/finbert/lr-3e-5-seed-44/finbert.md) | 3e-05 | 44 | Fixed-rate repeat | 0.8732 | 0.8722 |
| [RoBERTa Base](transformer-tuning-runs/roberta_base/lr-2e-5-seed-43/roberta_base.md) | 2e-05 | 43 | Fixed-rate repeat | 0.8707 | 0.8686 |
| [RoBERTa Base](transformer-tuning-runs/roberta_base/lr-2e-5-seed-44/roberta_base.md) | 2e-05 | 44 | Fixed-rate repeat | 0.8758 | 0.8746 |

| Model | Selected rate | Accuracy mean ± sample SD | Macro F1 mean ± sample SD |
| --- | ---: | ---: | ---: |
| FinBERT | 3e-05 | 0.8758 ± 0.0022 | 0.8748 ± 0.0023 |
| RoBERTa Base | 2e-05 | 0.8775 ± 0.0078 | 0.8760 ± 0.0082 |

Mean macro F1 differs by only 0.0013; per-seed ranking reverses at seed 43.
RoBERTa has larger observed variation in this small study. The three-seed mean
includes seed 42, which selected the rate, and reuses development validation.
It is not an ensemble score or independent test result. No best seed is chosen.
See the [full tuning report](transformer-tuning.md) for class means, grid rules,
architecture, timing, checksummed artifacts and limitations.

## Architecture and model size

| Model family | Architecture | Parameters | FP32 parameter MiB |
| --- | --- | ---: | ---: |
| FinBERT | 12 layers; hidden 768; 12 attention heads; 3 classes | 109,484,547 | 417.65 |
| Financial RoBERTa Large | 24 layers; hidden 1024; 16 attention heads; 3 classes | 355,362,819 | 1355.60 |
| Financial DeBERTa-v3 Small | 6 layers; hidden 768; 12 attention heads; 3 classes | 141,897,219 | 541.29 |
| RoBERTa Base | 12 layers; hidden 768; 12 attention heads; 3 classes | 124,647,939 | 475.49 |
| DeBERTa-v3 Base | 12 layers; hidden 768; 12 attention heads; 3 classes | 184,424,451 | 703.52 |

Adaptation does not change parameter count. Base models initialize new sentiment
heads; upstream masked-language-model heads are replaced. Parameter memory
excludes optimizer state, activations and tokenizer/configuration files.

| Traditional model / cohort | Architecture | Stored classifier parameters | Additional IDF values | Selected pipeline KiB |
| --- | --- | ---: | ---: | ---: |
| Logistic Regression / phrasebank | 12,351 unigram/bigram TF-IDF features → 3 classes | 37,056 | 12,351 | 768.14 |
| Linear SVM / phrasebank | 12,351 unigram/bigram TF-IDF features → 3 classes | 37,056 | 12,351 | 768.00 |
| Logistic Regression / sentfin_single_target | 10,475 unigram/bigram TF-IDF features → 3 classes | 31,428 | 10,475 | 641.68 |
| Linear SVM / sentfin_single_target | 10,475 unigram/bigram TF-IDF features → 3 classes | 31,428 | 10,475 | 641.54 |

VADER uses a lexicon and rules without a fitted corpus classifier. Majority
stores a training-selected label. LR uses multinomial logistic classification;
Linear SVM uses one-versus-rest squared hinge classifiers. Pipeline sizes include
vocabulary/serialization overhead and are not RAM measurements. IDF state is
separate from classifier parameter counts. Transformer softmax and LR outputs
remain uncalibrated; SVM decision scores and VADER proportions are not confidence
probabilities.

## Full traditional-model grid

All 24 fitted corpus/configuration combinations are included below. Selected
rows above are drawn from this grid; they are not additional independent runs.

| Cohort | Classifier | C | Class weight | Accuracy | Macro F1 |
| --- | --- | ---: | --- | ---: | ---: |
| phrasebank | logistic_regression | 0.1 | unweighted | 0.6274 | 0.3848 |
| phrasebank | logistic_regression | 0.1 | balanced | 0.7220 | 0.6720 |
| phrasebank | logistic_regression | 1 | unweighted | 0.7490 | 0.6696 |
| phrasebank | logistic_regression | 1 | balanced | 0.7645 | 0.7378 |
| phrasebank | logistic_regression | 10 | unweighted | 0.7722 | 0.7183 |
| phrasebank | logistic_regression | 10 | balanced | 0.7780 | 0.7446 |
| phrasebank | linear_svm | 0.1 | unweighted | 0.7297 | 0.6285 |
| phrasebank | linear_svm | 0.1 | balanced | 0.7683 | 0.7205 |
| phrasebank | linear_svm | 1 | unweighted | 0.7799 | 0.7344 |
| phrasebank | linear_svm | 1 | balanced | 0.7819 | 0.7444 |
| phrasebank | linear_svm | 10 | unweighted | 0.7606 | 0.7176 |
| phrasebank | linear_svm | 10 | balanced | 0.7568 | 0.7146 |
| sentfin_single_target | logistic_regression | 0.1 | unweighted | 0.7439 | 0.7384 |
| sentfin_single_target | logistic_regression | 0.1 | balanced | 0.7657 | 0.7646 |
| sentfin_single_target | logistic_regression | 1 | unweighted | 0.8182 | 0.8165 |
| sentfin_single_target | logistic_regression | 1 | balanced | 0.8195 | 0.8181 |
| sentfin_single_target | logistic_regression | 10 | unweighted | 0.8156 | 0.8141 |
| sentfin_single_target | logistic_regression | 10 | balanced | 0.8169 | 0.8154 |
| sentfin_single_target | linear_svm | 0.1 | unweighted | 0.8156 | 0.8145 |
| sentfin_single_target | linear_svm | 0.1 | balanced | 0.8169 | 0.8160 |
| sentfin_single_target | linear_svm | 1 | unweighted | 0.8092 | 0.8072 |
| sentfin_single_target | linear_svm | 1 | balanced | 0.8092 | 0.8075 |
| sentfin_single_target | linear_svm | 10 | unweighted | 0.7913 | 0.7902 |
| sentfin_single_target | linear_svm | 10 | balanced | 0.7913 | 0.7902 |

## Limitations and reproducibility

FinBERT used PhraseBank during upstream sentiment supervision, so its PhraseBank
result is diagnostic. Financial DeBERTa’s source description also indicates
PhraseBank supervision. Financial RoBERTa lacks complete supervised sample
provenance. Raw-text pretraining exposure is unknown. Neither cohort can be
certified unseen by every upstream model.

All transformer training uses one fixed seed/configuration with validation epoch
selection. Traditional models use a bounded validation grid. Repeated selection
and inspection make these scores optimistic for final generalization; no
confidence intervals, multiple-seed superiority or calibrated probabilities are
claimed. Larger scores on one cohort do not establish that it is a better dataset.

The report reuses recorded measurements. It does not compare CPU baseline
latency (batch one, Mac) directly with GPU transformer latency (batch 32, Windows).
Per-run reports linked above retain timing, environment, truncation, confusion
matrices, configurations, source/code hashes and saved artifact identities.
Weights, vocabulary, source text and private prediction records remain ignored.

During report generation, confusion-matrix totals, supports, accuracy and macro
F1 were checked for every displayed metric, including all grid candidates.
Candidate and transformer validation-membership digests match the two cohort
digests below; parent split/frozen benchmark identities agree. This is aggregate
verification of existing results, not a new per-record inference audit.

Parent split: `a9a67c231903bc2e3e71afead1d6daaf6b7201211fc25a05efd2bb2ee2d5e898`.
Frozen benchmark: `b7cc249017ddd87f0b2c52a3abf3ff5b96f6920832b411f18eac547fec0b0f1c`; remains unscored.

| Cohort | Validation membership SHA-256 |
| --- | --- |
| phrasebank | `515be8ccf4c59545e553d53aeea2451614c615c5b4534fc6aec19309ee3a4d97` |
| sentfin_single_target | `df68a4fa8844a432385a0f23487f3e582513ca9b7e04e76c8c17b0f1584130da` |

### Input report identities

These SHA-256 values pin the exact aggregate inputs used to create this report.

| Input JSON | SHA-256 |
| --- | --- |
| [tfidf-tuning.json](tfidf-tuning.json) | `329457f5dc8b951883897dfcf0b0a3ca88357093730d86315c9c5183cefdc5e1` |
| [phase2-comparison.json](phase2-comparison.json) | `32b3c353839c8aa903b82310eada60200c14eee34925d39d882a90a1f2af4d2d` |
| [finbert-validation.json](finbert-validation.json) | `ee63070bdb16c74bef562cddd8a1f30c92ba757489d8203477c6fb3d45b43e2c` |
| [finbert-finetuning.json](finbert-finetuning.json) | `560203a92f04621eb80ee98c14183ff9896d20594cbd292aef242a68996d9d9e` |
| [financial_roberta.json](financial_roberta.json) | `02bb45821689a958c7fc4ea035fc8a1dc73dcdf1b6db88fae6ed00cc9cdbfa29` |
| [financial_deberta.json](financial_deberta.json) | `f2c8cac19238ca68ed55a07d26bafc70e9e635d9fbe65220c9be298c68c69c27` |
| [roberta_base.json](roberta_base.json) | `6ebe322cdca4ce6094f28509dd7482fc11e5deec0c3d097902f3689caacae660` |
| [deberta_base.json](deberta_base.json) | `a3b9f03d1d019e2830b0cebf02b399a2d8fa5993f2a1507540034fd8056fc66a` |
| [financial_roberta_adapted.json](financial_roberta_adapted.json) | `5739c37f8d6b07fec1127ae0cf12cf47a077df4efde7fcab1bb1ad952e2ce607` |
| [financial_deberta_adapted.json](financial_deberta_adapted.json) | `85d7e5e0020da9529d20e65df1e8dde37c3475d42f68b98ec2354d75ece85df9` |
| [roberta_phrasebank.json](roberta_phrasebank.json) | `e71a9a56f66fbf81d0c5f57d848472b9f2aaea475a256b8b98bf89e622d67763` |
| [deberta_phrasebank.json](deberta_phrasebank.json) | `15bfcd783eda29d2e9cd4934ae2fadbda7225218ae0d65ef010b64b09f34a564` |
| [financial_roberta_phrasebank.json](financial_roberta_phrasebank.json) | `79cbf3b2c770ab61493016807e47fd4c957c820d4b57198b3a0f6e2a342dd380` |
| [financial_deberta_phrasebank.json](financial_deberta_phrasebank.json) | `a609d031440430f665b6cc69060fbe594949ce91b3a4ecc034bd9d52697c432b` |
| [transformer-tuning.json](transformer-tuning.json) | `206e36ffd82fa287e8e2d0ce49b6780600047125e1ff4dd9b8a56c9804919d21` |

New PhraseBank inference checks (2026-10-12): exact ordered prediction IDs/gold
labels, probability normalization, recomputed metrics, original checkpoint hashes
and recorded code/configuration lineage pass on Mac/Windows. Training samples
are zero; no derived artifacts or transfer step. The frozen benchmark is unchanged.
Ruff and all 282 tests pass on both machines.

This completes the consolidated cohort report. Phase 3 hyperparameter
comparisons, MLflow, calibration/error analysis and final evaluation remain open.
No roadmap phase is closed by this report.

## DeBERTa Base rate/seed extension

SEntFiN-only validation extension; five fresh runs, selected rate at seed 42, then seeds 43/44. Historical single-configuration and earlier two-model study rows remain intact. See [DeBERTa study](deberta-tuning.md) and [calibration/error review](deberta-analysis.md).

| Rate | Seed | Selected epoch | Accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| 1e-05 | 42 | 2 | 0.8835 | 0.8826 |
| 2e-05 | 42 | 5 | 0.8745 | 0.8738 |
| 3e-05 | 42 | 4 | 0.8835 | 0.8824 |
| 1e-05 | 43 | 4 | 0.8771 | 0.8760 |
| 1e-05 | 44 | 5 | 0.8848 | 0.8844 |

Selected LR 1e-05; accuracy 0.8818 ± 0.0041, macro F1 0.8810 ± 0.0044 (sample SD, three separate seeds). Seed 42 participates in selection; no superiority claim, ensemble or test score.

## Locked final single-target SEntFiN evaluation (2026-10-13)

The [final report](phase3-final.md) and [exact aggregate JSON](phase3-final.json) now score all 766 locked single-target test headlines. Mean macro F1: soft voting 0.9028, hard voting 0.9003, DeBERTa Base 0.8932, RoBERTa Base 0.8868 and fine-tuned FinBERT 0.8828. Their supplementary/primary roles, all three seeds, temperatures and baselines were fixed before inference. Paired ensemble-minus-DeBERTa intervals span zero; the observed ensemble lead is inconclusive. Calibration reduces held-out NLL for all three models.

The tables and JSON above retain their historical development-only scope; they are not replaced with test scores. Final-test coverage applies only to the locked models and single-target SEntFiN cohort. Financial variants and PhraseBank-trained transfer variants remain development diagnostics; reserved multi-target/PhraseBank tests are not scored. Pretraining exposure is unknown and no five-company US-domain generalization is established.

## Exploratory checkpoint added after final evaluation

[ProPrak01 DeBERTa-v3 Base, unchanged](proprak-deberta-validation.md) scored
0.7900 accuracy / 0.7915 macro F1 on the same 781 SEntFiN validation headlines.
It reports PhraseBank/Twitter/FiQA upstream training; no project training or
calibration fitting was performed. This post-final validation comparison is
separate from the locked final results. It has no PhraseBank/FiQA independence
claim and no new final-test score.

[ProPrak01 financial Base, SEntFiN-adapted](proprak-deberta-adaptation.md)
subsequently averages 0.8818 accuracy / 0.8815 macro F1 over seeds 42/43/44
on that validation cohort (macro-F1 sample SD 0.0062), versus 0.8818/0.8810
for the existing original Base three-seed comparison. Selected models are saved
and verified; regression head frozen, raw probabilities only. This optional
post-final experiment adds no frozen-test result or superiority claim.
