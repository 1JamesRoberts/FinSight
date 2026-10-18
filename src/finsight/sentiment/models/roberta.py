"""RoBERTa-specific head rules used by shared checked loading and prediction.

The pretrained language head may be replaced only during base-model training.
Financial sentiment checkpoints must load their complete existing head exactly.
"""

BASE_HEAD_KEYS = frozenset(
    {
        "classifier.dense.weight",
        "classifier.dense.bias",
        "classifier.out_proj.weight",
        "classifier.out_proj.bias",
    }
)
LANGUAGE_HEAD_PREFIXES = ("lm_head.",)
