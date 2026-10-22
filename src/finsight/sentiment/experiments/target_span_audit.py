"""Audit shared literal-span eligibility before training target-format variants."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from finsight.data.target_pairs import TargetExample, summarize_examples
from finsight.data.target_preparation import json_bytes
from finsight.data.target_spans import (
    SPAN_POLICY,
    VARIANTS,
    align_targets,
    transform_input,
)
from finsight.sentiment.artifacts import file_sha256, project_root, sha
from finsight.sentiment.evaluation.target_data import development_inputs

REPORT = Path("reports/phase-04-target-sentiment/evaluation/target-span-audit.json")


def audit_partition(
    rows: list[dict], partition: str
) -> tuple[dict, list[str], list[dict]]:
    """Keep a whole headline only when every native target can be transformed.

    Selection uses text alignment only, never gold polarity or a model score.
    All variants share the same retained membership; original splits stay fixed.
    """
    if partition not in ("train", "validation") or not rows:
        raise ValueError("nonempty development partition required")
    if any(r["partition"] != partition for r in rows):
        raise ValueError("unexpected partition")
    if len({r["example_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate annotation identity")
    headlines: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        headlines[row["record_id"]].append(row)
    counters: Counter = Counter()
    reasons: Counter = Counter()
    selected, review = [], []
    hashes = {variant: hashlib.sha256() for variant in VARIANTS}
    for identifier, annotations in sorted(headlines.items()):
        text = annotations[0]["headline"]
        if len({r["headline"] for r in annotations}) != 1:
            raise ValueError("headline identity has inconsistent text")
        aligned = align_targets(text, [r["target"] for r in annotations])
        complete = all(a.eligible for a in aligned)
        counters["headlines"] += 1
        counters["complete_eligible_headlines"] += complete
        counters["annotations"] += len(annotations)
        counters["individually_eligible_annotations"] += sum(
            a.eligible for a in aligned
        )
        counters["literal_aligned_annotations"] += sum(bool(a.spans) for a in aligned)
        counters["repeated_target_annotations"] += sum(
            len(a.spans) > 1 for a in aligned
        )
        counters["case_insensitive_only_annotations"] += sum(
            bool(a.spans)
            and not any(text[s.start : s.end] == a.target for s in a.spans)
            for a in aligned
        )
        reasons.update(reason for a in aligned for reason in a.reasons)
        if complete:
            for row, alignment in zip(annotations, aligned, strict=True):
                selected.append(row["example_id"])
                # Dry-run every variant on every eligible annotation and record
                # content hashes without publishing text or fitting a tokenizer.
                for variant in VARIANTS:
                    target, headline = transform_input(text, alignment, variant)
                    hashes[variant].update(
                        json_bytes(
                            {
                                "example_id": row["example_id"],
                                "target_sequence": target,
                                "headline_sequence": headline,
                            }
                        )
                    )
        if not complete or any(len(a.spans) > 1 for a in aligned):
            review.append(
                {
                    "record_id": identifier,
                    "partition": partition,
                    "headline": text,
                    "headline_eligible": complete,
                    "annotations": [
                        {
                            "example_id": row["example_id"],
                            "label": row["label"],
                            **asdict(alignment),
                        }
                        for row, alignment in zip(annotations, aligned, strict=True)
                    ],
                }
            )
    counters["shared_eligible_annotations"] = len(selected)
    counters["excluded_headlines"] = (
        len(headlines) - counters["complete_eligible_headlines"]
    )
    counters["excluded_annotations"] = len(rows) - len(selected)
    return (
        {
            "counts": dict(counters),
            "annotation_exclusion_reasons": dict(reasons),
            "eligible_membership_sha256": sha(json_bytes(selected)),
            "variant_text_sha256": {k: v.hexdigest() for k, v in hashes.items()},
        },
        selected,
        review,
    )


def run(root: Path, data_dir: Path) -> dict:
    """Use locked development files only; write no source records under reports."""
    training, validation, inputs = development_inputs(root, data_dir)
    summary = {
        "schema_version": 1,
        "experiment": "target-format-span-audit-v1",
        "status": "character_alignment_audited_not_trained",
        "policy": SPAN_POLICY,
        "variants": list(VARIANTS),
        "selection_unit": "complete_native_target_headline",
        "inputs": inputs,
        "frozen_records_read": False,
        "training_performed": False,
        "tokenizer_eligibility_verified": False,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_sha256": {
            name: file_sha256(root / "src/finsight" / name)
            for name in (
                "data/target_spans.py",
                "sentiment/experiments/target_span_audit.py",
                "sentiment/evaluation/target_data.py",
            )
        },
        "partitions": {},
    }
    membership = {}
    review = []
    examples = []
    retained = []
    for partition, rows in (("train", training), ("validation", validation)):
        counts, identifiers, cases = audit_partition(rows, partition)
        summary["partitions"][partition] = counts
        membership[partition] = identifiers
        review.extend(cases)
        selected = set(identifiers)
        for row in rows:
            example = TargetExample(
                **{**row, "review_reasons": tuple(row["review_reasons"])}
            )
            examples.append(example)
            if example.example_id in selected:
                retained.append(example)
    summary["original_cohorts"] = summarize_examples(tuple(examples))
    summary["eligible_cohorts"] = summarize_examples(tuple(retained))
    bodies = {
        "membership.json": json_bytes(membership),
        "review.jsonl": b"".join(
            (json.dumps(row, sort_keys=True) + "\n").encode() for row in review
        ),
    }
    manifest = {
        "policy": SPAN_POLICY,
        "inputs": inputs,
        "source_sha256": summary["source_sha256"],
        "files": {
            name: {"sha256": sha(body), "size_bytes": len(body)}
            for name, body in bodies.items()
        },
    }
    manifest_body = json_bytes(manifest)
    identity = sha(manifest_body)
    directory = data_dir / "interim/target-span-audit" / identity
    directory.mkdir(parents=True, exist_ok=True)
    for name, body in {**bodies, "manifest.json": manifest_body}.items():
        path = directory / name
        if path.exists() and path.read_bytes() != body:
            raise ValueError("existing private audit differs")
        path.write_bytes(body)
    summary["private_manifest_sha256"] = identity
    (root / REPORT).parent.mkdir(parents=True, exist_ok=True)
    (root / REPORT).write_bytes(json_bytes(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    result = run(root, args.data_dir or root / "data")
    print(json.dumps(result["partitions"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
