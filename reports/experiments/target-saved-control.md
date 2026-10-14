# Saved DeBERTa headline control: target-level validation

Three existing seeds (42/43/44), no new training or temperature scaling.
Each model scores 1,053 headlines once, then joins predictions to 1,410 native targets.

| Cohort | Targets | Headlines | Accuracy % | Macro F1 % | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all_targets | 1410 | 1053 | 80.52 ± 0.15 | 80.58 ± 0.21 | 0.9043 | 0.3516 | 0.1582 |
| single_target | 781 | 781 | 88.18 ± 0.41 | 88.10 ± 0.44 | 0.4995 | 0.2082 | 0.0858 |
| multi_target | 629 | 272 | 71.01 ± 0.18 | 71.22 ± 0.09 | 1.4071 | 0.5296 | 0.2500 |
| mixed_label_headlines | 283 | 122 | 51.24 ± 0.35 | 50.15 ± 1.20 | 2.4892 | 0.9060 | 0.4416 |
| seen_native_target | 1147 | 891 | 80.27 ± 0.18 | 80.30 ± 0.26 | 0.9204 | 0.3553 | 0.1607 |
| unseen_native_target | 263 | 243 | 81.62 ± 0.44 | 81.74 ± 0.40 | 0.8345 | 0.3356 | 0.1539 |

Accuracy/F1 are mean ± sample standard deviation across seeds, not confidence intervals.
All metrics weight target annotations equally; cohorts overlap. Three-class macro F1 includes all classes.
Mixed-label headlines necessarily receive the same predicted label for conflicting native targets.
Single-target predictions reproduce every earlier label and metric; raw probability differences are checked at 1e-5 tolerance.

| Seed | All-target F1 % | Multi-target F1 % | Mixed-label F1 % |
| --- | ---: | ---: | ---: |
| 42 | 80.63 | 71.13 | 51.42 |
| 43 | 80.34 | 71.29 | 49.02 |
| 44 | 80.76 | 71.25 | 50.00 |

Model: DebertaV2ForSequenceClassification, 12 layers, hidden size 768, 12 attention heads, 184,424,451 parameters. FP32, eager attention.
Previously trained on 6,299 single-target headlines; no target conditioning. Full profiles, per-class metrics, confusion matrices and calibration bins are in the companion JSON.

These are reused development validation results, not independent test evidence. No target-aware improvement claim is established.
The primary future comparison remains freshly trained matched headline-only versus target-pair models.
Reserved multi-target test data remain unscored; the original Phase 3 final aggregate is unchanged.
Native text, target names and row predictions remain in ignored local storage.

Plan SHA-256: `d8c87e545ec2546cace4a41f3f5840b1ad077d04cdcefec81447973524185a81`.
Private view: `be33e779e278b2c786c48ee3205cbad4f9e6bee98839d4c3b6509009a51a856c`.
Execution revision: `8a94c8ee8f99037c9218682374b361950510050c`.
