"""Verified development inputs and guarded frozen target membership expansion."""

import json
from collections import Counter
from pathlib import Path

from finsight.data.audit import text_identity
from finsight.data.review import text_hash
from finsight.sentiment.artifacts import file_sha256, sha, verify_files
from finsight.sentiment.evaluation.metrics import LABELS
from finsight.sentiment.evaluation.protocol import safe_path


def development_inputs(
    root: Path, data_dir: Path
) -> tuple[list[dict], list[dict], dict]:
    """Read locked native development views without opening frozen files."""
    lock = json.loads((root / "configs/target-sentiment-lock.json").read_bytes())
    directory = data_dir / "interim/target-pairs" / lock["data_manifest_sha256"]
    if file_sha256(directory / "manifest.json") != lock["data_manifest_sha256"]:
        raise ValueError("native development manifest differs")
    manifest = json.loads((directory / "manifest.json").read_bytes())
    verify_files(directory, manifest["files"])
    partitions = []
    for partition in ("train", "validation"):
        rows = [
            json.loads(line)
            for line in (directory / f"{partition}.jsonl").read_bytes().splitlines()
        ]
        if not rows or any(r["partition"] != partition for r in rows):
            raise ValueError("native development partition differs")
        if len({r["example_id"] for r in rows}) != len(rows):
            raise ValueError("duplicate native example identity")
        partitions.append(rows)
    training, validation = partitions
    if {r["group_id"] for r in training} & {r["group_id"] for r in validation}:
        raise ValueError("native development group overlap")
    return (
        training,
        validation,
        {"manifest_sha256": lock["data_manifest_sha256"], "files": manifest["files"]},
    )


def evaluation_rows(examples: list[dict]) -> list[dict]:
    """Use unique annotation IDs for metric alignment; keep headline IDs separately."""
    return [
        {
            "record_id": r["example_id"],
            "headline_id": r["record_id"],
            "group_id": r["group_id"],
            "headline": r["headline"],
            "target": r["target"],
            "gold": r["label"],
            "multi_target": r["multi_target"],
            "mixed_labels": r["mixed_labels"],
        }
        for r in examples
    ]


def expand_frozen(
    rows: list[dict],
    development: list[dict],
    expected: dict,
    *,
    exclusions: list[dict] | None = None,
) -> list[dict]:
    """Audit all groups before emitting any reserved multi-target model inputs."""
    if len({r["record_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate frozen headline identity")
    consumed = [r for r in rows if len(r["native_record"]["sentiments"]) == 1]
    selected = [r for r in rows if len(r["native_record"]["sentiments"]) > 1]
    # An explicit v2 registry is the only permitted exclusion. Default behavior
    # retains the v1 failure gate. Match both members and labels before filtering;
    # any unexpected overlap or missing/changed registry member still aborts.
    excluded = set()
    for exclusion in exclusions or []:
        group = exclusion["group_id"]
        reserved = [r for r in selected if r["group_id"] == group]
        prior = [r for r in consumed if r["group_id"] == group]
        if (
            group in excluded
            or len(reserved) != 1
            or len(prior) != 1
            or any(r["group_id"] == group for r in development)
            or text_hash(reserved[0]["native_record"]["title"])
            != exclusion["reserved_text_sha256"]
            or text_hash(prior[0]["native_record"]["title"])
            != exclusion["consumed_text_sha256"]
            or dict(
                Counter(a["label"] for a in reserved[0]["native_record"]["sentiments"])
            )
            != exclusion["reserved_class_counts"]
            or exclusion["reason"]
            != "near_duplicate_group_previously_evaluated_in_phase3"
        ):
            raise ValueError("previously consumed group exclusion differs")
        excluded.add(group)
    selected = [r for r in selected if r["group_id"] not in excluded]
    groups = {r["group_id"] for r in selected}
    forbidden_groups = {r["group_id"] for r in development + consumed}
    forbidden_text = {text_identity(r["headline"]) for r in development} | {
        text_identity(r["native_record"]["title"]) for r in consumed
    }
    if groups & forbidden_groups or any(
        text_identity(r["native_record"]["title"]) in forbidden_text for r in selected
    ):
        raise ValueError(
            "reserved multi-target overlaps development or consumed single-target data"
        )
    examples = []
    for row in rows:
        if (
            row["corpus"] != "sentfin"
            or row["partition"] != "test_candidate"
            or row["development_only"]
            or row["review_reasons"]
        ):
            raise ValueError("invalid frozen eligibility")
        native = row["native_record"]
        annotations = native["sentiments"]
        if (
            not annotations
            or not isinstance(native["title"], str)
            or not native["title"].strip()
        ):
            raise ValueError("invalid frozen headline")
        if len({a["target"] for a in annotations}) != len(annotations) or any(
            a["label"] not in LABELS
            or not isinstance(a["target"], str)
            or not a["target"].strip()
            for a in annotations
        ):
            raise ValueError("invalid frozen annotations")
        if len(annotations) < 2 or row["group_id"] in excluded:
            continue
        for index, annotation in enumerate(annotations):
            identity = sha(
                json.dumps([row["record_id"], index, annotation["target"]]).encode()
            )
            examples.append(
                {
                    "record_id": identity,
                    "headline_id": row["record_id"],
                    "group_id": row["group_id"],
                    "headline": native["title"],
                    "target": annotation["target"],
                    "gold": annotation["label"],
                    "multi_target": True,
                    "mixed_labels": len({a["label"] for a in annotations}) > 1,
                }
            )
    for name, selected in (
        ("multi_target", examples),
        ("mixed_label_headlines", [r for r in examples if r["mixed_labels"]]),
    ):
        counts = {
            "headlines": len({r["headline_id"] for r in selected}),
            "target_annotations": len(selected),
            "class_counts": dict(Counter(r["gold"] for r in selected)),
        }
        if counts != expected[name]:
            raise ValueError("frozen cohort membership/class support differs")
    return examples


def frozen_inputs(
    root: Path, data_dir: Path, plan: dict, development: list[dict]
) -> list[dict]:
    """Explicit final execution only: verify bytes, then decode and audit membership."""
    relative = next(p for p in plan["frozen_files"] if p.endswith("test-records.jsonl"))
    path = safe_path(data_dir, relative)
    if file_sha256(path) != plan["frozen_files"][relative]:
        raise ValueError("frozen records changed before decoding")
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    return expand_frozen(
        rows,
        development,
        plan["cohorts"]["expected"],
        exclusions=plan["cohorts"].get("excluded_previously_consumed_groups"),
    )
