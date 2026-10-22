# Organization detection: registry versus pretrained NER

Reviewed 2026-10-20. [Method and commands](../../../docs/phase-04-target-sentiment/organization-detection.md) · [Pinned plan](../../../configs/organization-ner-plan.json) · [Aggregate JSON](organization-detection.json) · [Verification](../verification/organization-detection-verification.json) · [Invented-input sentiment smoke](../verification/organization-sentiment-smoke.json).

A pretrained organization detector expands coverage beyond the five-company registry, including Tesla in the invented-input check. However, this baseline has fragmented names, overextended boundaries and organization-type mistakes. Keep it as an explicit experimental candidate path; retain the existing default registry behavior and saved RoBERTa sentiment checkpoint. ORG extraction does not establish a company identity or ticker.

## Configuration and architecture

Use `dslim/bert-base-NER` revision `d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc`, a published BERT-base-cased token classifier trained on English CoNLL-2003. The publisher's [pinned card](https://huggingface.co/dslim/bert-base-NER/blob/d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc/README.md) declares MIT terms and notes domain and subword limitations. The [local-use review](../../../docs/datasets.md#pretrained-organization-detector-review-2026-10-20) precedes acquisition. Only seven checksummed model/card/tokenizer files were acquired; no CoNLL/Reuters training text was downloaded or redistributed.

The loaded model has **107,726,601 parameters**, 12 layers, hidden size 768, 12 attention heads and nine BIO token classes (O; beginning/inside tags for ORG, PER, LOC and MISC). FP32 parameters occupy 430,906,404 bytes; the safetensors file is 433,292,294 bytes. It is separate from the saved three-class RoBERTa sentiment model. No new dependencies or training are introduced.

The fixed `simple` aggregation groups compatible BIO predictions, preserving original character offsets. Retain all ORG groups, with no tuned score threshold. Use batch size 16 and a complete-input 512-token limit; all validation headlines fit, with a maximum of 34 encoded tokens. CPU inference on all 1,053 headlines took 6.41 seconds, excluding loading and review; this single local timing is not a production latency benchmark. Torch is 2.11.0 and Transformers is pinned to 4.57.6 in `uv.lock`. Inference disables gradients/dropout and seeds Torch with 42.

## Validation coverage

| Measure | Count |
| --- | ---: |
| Unique validation headlines | 1,053 |
| Native financial-target annotations | 1,410 |
| Headlines with registry-resolved mentions | 2 |
| Registry-resolved spans | 2 |
| Headlines with registry-ambiguous mentions | 1 |
| Registry-ambiguous spans | 1 |
| Headlines with NER ORG proposals | 749 |
| NER ORG proposal spans | 1,062 |
| Headlines with multiple distinct ORG surfaces | 217 |
| NER-positive headlines with no registry candidate | 747 |

**Table attributes:** Headlines counts each native record once; spans counts each proposed occurrence, including repeated names and fragments. Native annotations counts target–headline pairs, not exhaustive named entities. Resolved and ambiguous registry headline counts can overlap. Multiple distinct ORG surfaces means more than one case-insensitively distinct predicted surface in a headline; fragmented names can inflate this count, so it is not a count of verified multi-company headlines. No registry candidate means neither a resolved nor an ambiguous registry span; those 747 headlines demonstrate broader proposal coverage, not verified detection accuracy. The NER and registry methods overlap and these rows must not be added together.

The registry is deliberately limited to five US companies. This corpus mostly concerns other financial entities, so larger NER counts are expected from broader scope and do not alone establish that NER is more accurate.

## Annotated-target match diagnostic

The twelve-company allowlist was fixed before inference and includes Infosys, TCS, ONGC, Sun Pharma, SBI, BPCL, HUL, Tata Steel, Tech Mahindra, ICICI Bank, Wipro and HDFC Bank. The all-native view additionally includes indices, commodities, currencies, countries and sectors.

| Scope | Native annotations | Literal target annotations | Nonliteral target annotations |
| --- | ---: | ---: | ---: |
| All native financial targets | 1,410 | 1,402 | 8 |
| Fixed twelve-company allowlist | 70 | 70 | 0 |

**Table attributes:** Native annotations is the number of target–headline pairs in the scope. Literal annotations have at least one case-insensitive, word-boundary literal occurrence of the full native target in the headline. Nonliteral annotations have none under that rule; aliases are not inferred. Literal and nonliteral counts partition each scope. The company scope is a subset of the all-native scope, so scopes overlap. This run uses the complete prepared validation view; the earlier format-training diagnostic used a smaller shared span-eligible view, hence 70 company annotations here versus 67 in the name-substitution check.

| Detector | All native: exact / 1,402 | All native: overlap / 1,402 | Company allowlist: exact / 70 | Company allowlist: overlap / 70 |
| --- | ---: | ---: | ---: | ---: |
| Registry resolved | 2 | 2 | 0 | 0 |
| Registry resolved plus ambiguous | 3 | 3 | 0 | 0 |
| Pretrained NER ORG | 661 | 865 | 44 | 62 |

**Table attributes:** Each cell counts literal target annotations with at least one matching detector span; repeated mentions of the same target do not increase its annotation count. Exact requires the detector's start/end to equal one full native-target occurrence. Overlap requires any character intersection and includes exact matches. Registry resolved plus ambiguous is a coverage view, not permission to score ambiguous mentions. Denominators exclude nonliteral annotations. On the company subset, 44/70 (62.86%) have an exact match and 62/70 (88.57%) have any overlap; the additional 18 are overlap-only, while eight have none. These are positive-only annotation coverage values, not exhaustive company precision/recall. The five-company registry has no members in this twelve-company allowlist, so its zeroes reflect scope rather than a failed attempt at those names.

A partial fragment can count as overlap even if it is not a usable target name. Review confirms this is material: some unfamiliar company names or acronyms split into pieces, and one proposed span includes a trading instruction alongside the company name. Consequently the 62/70 overlap result should not be treated as 62 correctly extracted companies.

## Relation of NER proposals to native annotations

| Relation of a proposed ORG span | Spans |
| --- | ---: |
| Exact match to a literal native target | 661 |
| Overlap with a literal target, without any exact match | 271 |
| No overlap with a literal native target | 130 |

**Table attributes:** This table counts predicted spans, whereas the previous table counts target annotations. A span may intersect multiple target annotations; an annotation may intersect several fragmented spans. The three relation categories are disjoint and sum to 1,062 proposals. No overlap means unannotated under this limited literal comparison, not false positive. Conversely, an exact native-target match does not prove that the target is a company: native targets can be other financial entities.

## Qualitative review

The fixed sampling rule selects up to ten headlines per category, sorting by a hash of native record identity. Category membership and full cases are private. The assistant inspected all selected cases; this is exploratory review rather than independent human adjudication. No model, aggregation rule, threshold, registry context or rollup policy was changed after reviewing the outputs.

| Review category | Selected headlines |
| --- | ---: |
| ORG without a literal annotation overlap | 10 |
| Literal native target without an overlapping ORG | 10 |
| Multiple distinct ORG surfaces | 10 |
| Registry ambiguity | 1 |
| Distinct reviewed headlines across categories | 24 |

**Table attributes:** Selected headlines counts up to ten sampled records in each declared category; the ambiguity category has only one eligible validation record. Categories overlap, so their 31 memberships represent 24 distinct headlines. Sampling is descriptive and enriched for differences, not a random prevalence estimate or a precision/recall benchmark.

Findings from the reviewed cases:

- **Real company coverage misses:** a company in a recommendation headline was omitted even though another company was extracted. Some website-style targets were also missed.
- **Boundary and fragmentation failures:** a full company name, an acronym and another company name were split into shorter ORG spans. This can inflate multi-organization counts and create inappropriate sentiment targets.
- **Non-company targets:** many absent native targets were commodities, currencies, sectors or stock indices. Their absence is not automatically a company-detector error.
- **ORG does not mean company:** central banks, a regulator, an exchange and a court can be legitimate organizations without being operating companies.
- **Extra proposals are mixed:** an unannotated bank alias appears useful, while person names or person-name suffixes were also incorrectly proposed as ORG. Missing native annotations alone cannot distinguish these cases.
- **Ambiguity remains explicit:** NER identifies Microsoft and Amazon in one validation headline, while the registry retains its existing context requirement for Amazon. NER evidence does not silently alter registry identity policy.

## Candidate-to-sentiment verification

On the invented headline “Tesla reported higher profits while Microsoft reported losses.”, the actual NER model returns the complete Tesla and Microsoft spans. These detected surfaces were sent through the new bridge to the existing saved CUDA RoBERTa checkpoint `2f742f69854f097b5b1351e4a1de126b54c28fe2c900edb3c2e41526cd32e855`. It returned positive for Tesla and negative for Microsoft. Each result exactly matches calling the existing predictor directly with the same target. This is a functional smoke check, not real-data accuracy evidence.

The bridge uses original target–headline pair encoding and retains model version and raw, uncalibrated probabilities. Repeated identical surfaces share one sentiment call; distinct aliases remain separate. An empty ORG list skips sentiment model loading/forward inference. Outputs explicitly mark company identity as unverified. The normal registry-based automatic CLI and deployed checkpoint remain unchanged. No sentiment predictions were generated on SEntFiN in this detection comparison.

The new CLI's detection path was run on the Mac. Its bridge was separately verified against the GPU PC's existing sentiment artifact using the actual locally detected spans; no NER weight copy or primary-source update on the PC was necessary for that bridge smoke. See setup instructions before attempting the full new CLI there.

## Verification and next decision

The local private packet contains all 1,053 validation records and all 1,410 native annotations; hashes, membership and aggregate coverage/relation counts were checked after execution. Plan/model/source identities and all acquired file sizes/hashes verify. The public verification receipt also pins the private assistant review. Source content remains under ignored data, while aggregate results and invented-input outputs are public. No frozen records were read, neither model was trained and no deployment choice changed.

Twenty-one new synthetic behavior checks cover span validity, complete-length guards, positive-only coverage, reserved-partition rejection, candidate grouping and the no-candidate sentiment path. All **721 pytest tests pass**. Required full Ruff commands were run and encounter only the unchanged scratch notebook's existing nine lint issues and formatting; excluding `notebooks/experiment.ipynb`, lint and formatting pass. `git diff --check` passes.

This completes the bounded pretrained NER comparison and experimental sentiment bridge. It does not close the company-labeled evaluation collection or establish company extraction accuracy. Before enabling broader automatic detection, improve word-level boundary handling and evaluate a separately reviewed collection with company/non-company types and full mention spans. Keep that extraction evaluation separate from sentiment accuracy, and keep ticker linking explicit.
