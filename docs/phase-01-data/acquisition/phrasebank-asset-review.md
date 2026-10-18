# Financial PhraseBank asset review

Reviewed: 2026-10-10. Scope: local, noncommercial FinSight research.

The [dataset card](https://huggingface.co/datasets/takala/financial_phrasebank)
declares CC BY-NC-SA 3.0 and identifies the Malo et al. financial-news corpus.
The [license](https://creativecommons.org/licenses/by-nc-sa/3.0/legalcode) permits
sharing/adaptation within its noncommercial, attribution, and share-alike terms.
Project decision: acquiring the released archive, validating it locally, and later
conducting noncommercial research are supported under that grant. Retain source,
authors, title, license link, and modification notices. Distributed adaptations
must comply with the applicable share-alike terms. This does not establish
separate rights to republish underlying news or license resulting model weights.
Commercial use and public source-text demos require a separate review.

Pinned repository revision: `8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964`.
Archive: `data/FinancialPhraseBank-v1.0.zip` (681,890 bytes).
Hugging Face's pinned tree metadata supplies archive SHA-256
`0e1a06c4900fdae46091d031068601e3773ba067c7cecb5b0da1dcba5ce989a6`.
The download must match it before loading. The dataset license applies to data;
the Apache header in the historical loader does not replace it. The loader is
not executed. Original text files will be read directly from the archive using
its documented ISO-8859-1 encoding and final `@` label delimiter.

Use the >=50% agreement variant as the primary local corpus for integrity checks;
load other variants separately to validate their counts. Do not concatenate
agreement variants because they overlap. No split is assigned by this review.
PhraseBank is a baseline development candidate and cannot be presumed an
independent test for the pretrained FinBERT checkpoint.

Store source archive, local license evidence, and acquisition receipt under
ignored `data/raw/financial_phrasebank/`. No dataset content is committed.
No retention deadline is specified in the reviewed dataset card/license.
FiQA is deferred because its release terms remain unresolved; it is not acquired.
