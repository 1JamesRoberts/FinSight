"""Private native target/text examples and target-preserving tokenizer inputs."""

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from finsight.data.audit import text_identity

LABELS = ("negative", "neutral", "positive")


@dataclass(frozen=True)
class TargetText:
    """Inference receives text and identity only; no gold labels or partition flags."""

    record_id: str
    headline: str
    target: str


@dataclass(frozen=True)
class TargetExample:
    example_id: str
    record_id: str
    group_id: str
    partition: str
    annotation_index: int
    target: str
    headline: str
    label: str
    multi_target: bool
    mixed_labels: bool
    development_only: bool
    review_reasons: tuple[str, ...]


def build_examples(records: list[dict]) -> tuple[TargetExample, ...]:
    """Expand native annotations only after validating the original partitions."""
    if any(r["partition"] not in ("train", "validation") for r in records):
        raise ValueError("target preparation accepts development records only")
    rows = [r for r in records if r["corpus"] == "sentfin"]
    if len({r["record_id"] for r in rows}) != len(rows):
        raise ValueError("source IDs must be unique")
    groups: dict[str, set[str]] = defaultdict(set)
    texts: dict[str, set[str]] = defaultdict(set)
    examples = []
    for row in sorted(rows, key=lambda r: r["record_id"]):
        native = row["native_record"]
        if not isinstance(native["title"], str) or not native["title"].strip():
            raise ValueError("headline must be nonempty text")
        annotations = native["sentiments"]
        if not annotations or any(
            not isinstance(a["target"], str)
            or not a["target"].strip()
            or a["label"] not in LABELS
            for a in annotations
        ):
            raise ValueError("invalid target annotation")
        if len({a["target"] for a in annotations}) != len(annotations):
            raise ValueError("duplicate native target within a headline")
        if row["development_only"] and row["partition"] != "train":
            raise ValueError("development-only rows must remain in training")
        groups[row["group_id"]].add(row["partition"])
        texts[text_identity(native["title"])].add(row["partition"])
        for index, annotation in enumerate(annotations):
            identity = json.dumps([row["record_id"], index, annotation["target"]])
            examples.append(
                TargetExample(
                    hashlib.sha256(identity.encode()).hexdigest(),
                    row["record_id"],
                    row["group_id"],
                    row["partition"],
                    index,
                    annotation["target"],
                    native["title"],
                    annotation["label"],
                    len(annotations) > 1,
                    len({a["label"] for a in annotations}) > 1,
                    row["development_only"],
                    tuple(row["review_reasons"]),
                )
            )
    if any(len(partitions) > 1 for partitions in groups.values()):
        raise ValueError("duplicate group crossed partitions")
    if any(len(partitions) > 1 for partitions in texts.values()):
        raise ValueError("normalized headline crossed partitions")
    return tuple(examples)


def summarize_examples(examples: tuple[TargetExample, ...]) -> dict:
    """Count targets separately from headlines; native seen/unseen is not company ID."""
    training_targets = {e.target for e in examples if e.partition == "train"}
    result = {}
    for partition in ("train", "validation"):
        selected = [e for e in examples if e.partition == partition]
        result[partition] = {
            "examples": len(selected),
            "headlines": len({e.record_id for e in selected}),
            "duplicate_groups": len({e.group_id for e in selected}),
            "class_counts": {
                label: sum(e.label == label for e in selected) for label in LABELS
            },
            "multi_target_examples": sum(e.multi_target for e in selected),
            "mixed_label_examples": sum(e.mixed_labels for e in selected),
            "development_only_headlines": len(
                {e.record_id for e in selected if e.development_only}
            ),
            "development_only_examples": sum(e.development_only for e in selected),
            "review_reason_headlines": dict(
                Counter(
                    reason
                    for reason, _ in {
                        (reason, e.record_id)
                        for e in selected
                        for reason in e.review_reasons
                    }
                )
            ),
            "seen_native_target_examples": sum(
                e.target in training_targets for e in selected
            ),
            "unseen_native_target_examples": sum(
                e.target not in training_targets for e in selected
            ),
        }
    return result


def examples_bytes(examples: tuple[TargetExample, ...], partition: str) -> bytes:
    """Serialize actual native text and labels privately, with stable membership IDs."""
    return "".join(
        json.dumps(asdict(e), sort_keys=True) + "\n"
        for e in examples
        if e.partition == partition
    ).encode()


def encode_batch(
    tokenizer: Any,
    examples: tuple[TargetExample, ...] | list[TargetExample] | list[TargetText],
    mode: str,
    *,
    max_length: int,
) -> Any:
    """Build model inputs without labels; fail rather than truncate the target."""
    if mode not in ("matched_headline_only", "target_pair"):
        raise ValueError("unsupported input mode")
    if not examples:
        raise ValueError("encoding requires nonempty examples")
    if type(max_length) is not int or not 4 <= max_length <= 512:
        raise ValueError("invalid maximum sequence length")
    if tokenizer.truncation_side != "right" or tokenizer.padding_side != "right":
        raise ValueError("protocol requires right truncation and padding")
    headlines = [e.headline for e in examples]
    if mode == "matched_headline_only":
        return tokenizer(
            headlines,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )
    targets = [e.target for e in examples]
    target_ids = tokenizer(targets, add_special_tokens=False, truncation=False)[
        "input_ids"
    ]
    headline_ids = tokenizer(headlines, add_special_tokens=False, truncation=False)[
        "input_ids"
    ]
    special = tokenizer.num_special_tokens_to_add(pair=True)
    for target, headline in zip(target_ids, headline_ids, strict=True):
        if not target or not headline:
            raise ValueError("target and headline must encode to nonempty tokens")
        if len(target) + special >= max_length:
            raise ValueError("full target leaves no headline token; review overflow")
    encoded = tokenizer(
        targets,
        headlines,
        padding=True,
        truncation="only_second",
        max_length=max_length,
        return_tensors=None,
    )
    for i, (target, headline) in enumerate(zip(target_ids, headline_ids, strict=True)):
        expected = tokenizer.build_inputs_with_special_tokens(
            target, headline[: max_length - len(target) - special]
        )
        actual = [
            token
            for token, active in zip(
                encoded["input_ids"][i], encoded["attention_mask"][i], strict=True
            )
            if active
        ]
        if actual != expected:
            raise ValueError(
                "pair encoding did not preserve target and headline budget"
            )
    return encoded


def audit_tokenization(
    tokenizer: Any,
    examples: tuple[TargetExample, ...],
    *,
    max_length: int,
) -> dict:
    """Inspect both input formats without loading weights or running inference."""
    result = {}
    for partition in ("train", "validation"):
        selected = [e for e in examples if e.partition == partition]
        formats = {}
        for mode in ("matched_headline_only", "target_pair"):
            lengths = []
            truncated = unknown_targets = 0
            conflicts: dict[tuple[int, ...], set[str]] = defaultdict(set)
            for start in range(0, len(selected), 32):
                batch = selected[start : start + 32]
                encoded = encode_batch(tokenizer, batch, mode, max_length=max_length)
                hids = tokenizer(
                    [e.headline for e in batch],
                    add_special_tokens=False,
                    truncation=False,
                )["input_ids"]
                tids = tokenizer(
                    [e.target for e in batch],
                    add_special_tokens=False,
                    truncation=False,
                )["input_ids"]
                special = tokenizer.num_special_tokens_to_add(
                    pair=mode == "target_pair"
                )
                for i, example in enumerate(batch):
                    ids = tuple(
                        t
                        for t, active in zip(
                            encoded["input_ids"][i],
                            encoded["attention_mask"][i],
                            strict=True,
                        )
                        if active
                    )
                    lengths.append(len(ids))
                    full_length = (
                        len(hids[i])
                        + special
                        + (len(tids[i]) if mode == "target_pair" else 0)
                    )
                    truncated += full_length > max_length
                    unknown_targets += tokenizer.unk_token_id in tids[i]
                    conflicts[ids].add(example.label)
            formats[mode] = {
                "examples": len(selected),
                "maximum_encoded_length": max(lengths, default=0),
                "headline_truncated_examples": truncated,
                "target_unknown_token_examples": unknown_targets,
                "identical_inputs_with_conflicting_labels": sum(
                    len(v) > 1 for v in conflicts.values()
                ),
            }
        result[partition] = formats
    return result
