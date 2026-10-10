"""Prepare reversible filtering decisions and provisional, group-safe partitions."""

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from finsight.data.acquisition import artifact_path
from finsight.data.audit import text_identity
from finsight.data.fiqa import load_fiqa
from finsight.data.loaders import load_phrasebank, load_sentfin
from finsight.data.review import (
    fiqa_annotation_context,
    review_datasets,
    snippet_mismatches,
    text_hash,
)

PARTITIONS = ("train", "validation", "test_candidate")


@dataclass(frozen=True)
class SourceExample:
    corpus: str
    source_id: str
    text: str
    annotations: tuple[tuple[str, str | None], ...]
    native_record: dict[str, Any]
    snippet_issues: tuple[int, ...] = ()
    unscored: bool = False

    @property
    def reference(self) -> str:
        return f"{self.corpus}/{self.source_id}"


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def duplicate_components(
    hashes: set[str], edges: list[tuple[str, str]]
) -> dict[str, str]:
    """Join exact text identities and near edges transitively with stable group IDs."""
    parent = {key: key for key in hashes}

    def root(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for left, right in sorted(edges):
        if left not in parent or right not in parent:
            raise ValueError("near-duplicate edge references unknown text")
        a, b = root(left), root(right)
        if a != b:
            parent[max(a, b)] = min(a, b)
    groups: dict[str, list[str]] = defaultdict(list)
    for key in sorted(hashes):
        groups[root(key)].append(key)
    return {key: _digest(members) for members in groups.values() for key in members}


def assign_partition(
    group_id: str, policy: dict[str, Any], *, development_only: bool
) -> str:
    """Use a seeded SHA-256 bucket; approximate ratios without seed searching."""
    weights = policy["partition_weights"]
    names = PARTITIONS[:2] if development_only else PARTITIONS
    total = sum(weights[name] for name in names)
    bucket = int.from_bytes(
        hashlib.sha256(f"{policy['seed']}:{group_id}".encode()).digest()[:8], "big"
    )
    boundary = 0
    for name in names:
        boundary += weights[name]
        if bucket * total < boundary * 2**64:
            return name
    raise AssertionError("partition bucket was not assigned")


def validate_policy(policy: dict[str, Any], review: dict[str, Any]) -> None:
    """Only supported decisions are accepted; policy changes cannot be ignored."""
    expected = {
        "schema_version": 1,
        "phrasebank_agreement": "50",
        "near_duplicates": review["summary"]["near_method"],
        "quarantine": [
            "exact_gold_conflict",
            "near_gold_difference",
            "original_unscored_holdout_overlap",
        ],
        "development_only": [
            "cross_corpus_overlap",
            "target_coverage_difference",
            "fiqa_snippet_review",
        ],
        "deduplicate": (
            "same corpus, normalized text, and equivalent annotation contexts/values"
        ),
        "fiqa_snippet_supervision": (
            "exclude annotations with missing or nonmatching snippets"
        ),
        "final_test_frozen": False,
    }
    if set(policy) != set(expected) | {"seed", "partition_weights"}:
        raise ValueError("unexpected split policy fields")
    if any(policy[key] != value for key, value in expected.items()):
        raise ValueError("unsupported filtering or review policy")
    if type(policy["seed"]) is not int:
        raise ValueError("split seed must be an integer")
    weights = policy["partition_weights"]
    if not isinstance(weights, dict) or set(weights) != {
        "train",
        "validation",
        "test_candidate",
    }:
        raise ValueError("partition weights must name all three partitions")
    if (
        any(type(value) is not int or value <= 0 for value in weights.values())
        or sum(weights.values()) != 100
    ):
        raise ValueError("positive integer partition weights must sum to 100")


def _load_sources(manifest: dict[str, Any], data_dir: Path) -> list[SourceExample]:
    assets = manifest["assets"]
    phrasebank = assets["financial_phrasebank"]
    examples = [
        SourceExample(
            "phrasebank",
            row.source_id,
            row.sentence,
            (("sentence", row.label.value),),
            asdict(row),
        )
        for row in load_phrasebank(
            artifact_path(data_dir, phrasebank["acquisition_files"][0]),
            revision=phrasebank["revision"],
            agreement="50",
        )
    ]
    examples.extend(
        SourceExample(
            "sentfin",
            row.source_id,
            row.title,
            tuple((text_identity(a.target), a.label.value) for a in row.sentiments),
            asdict(row),
        )
        for row in load_sentfin(
            data_dir / "raw/sentfin/SEntFiN.csv",
            revision=assets["sentfin_1_0"]["revision"],
        )
    )
    if "fiqa_task1_original" in manifest["local_dataset_assets"]:
        fiqa = assets["fiqa_task1_original"]
        for kind in ("headline", "post"):
            for split in ("train", "test"):
                for row in load_fiqa(
                    data_dir / f"raw/fiqa/task1_{kind}_ABSA_{split}.json",
                    revision=fiqa["revision"],
                    source_type=kind,
                    source_split=split,
                ):
                    examples.append(
                        SourceExample(
                            "fiqa",
                            f"{kind}/{split}/{row.source_id}",
                            row.sentence,
                            tuple(
                                (
                                    fiqa_annotation_context(a),
                                    str(a.score or 0.0)
                                    if a.score is not None
                                    else None,
                                )
                                for a in row.annotations
                            ),
                            asdict(row),
                            tuple(
                                i
                                for i, a in enumerate(row.annotations)
                                if any(value is None for value in a.snippets)
                                or snippet_mismatches(row.sentence, a.snippets)
                            ),
                            split == "test",
                        )
                    )
    return examples


def make_split_plan(
    examples: list[SourceExample], review: dict[str, Any], policy: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Keep every source decision, representative lineage, and target group intact."""
    validate_policy(policy, review)
    ordered = sorted(examples, key=lambda row: row.reference)
    if len({row.reference for row in ordered}) != len(ordered):
        raise ValueError("source references must be unique")
    hashes = {text_hash(row.text) for row in ordered}
    edges = [
        (pair["left_sha256"], pair["right_sha256"]) for pair in review["near_pairs"]
    ]
    components = duplicate_components(hashes, edges)
    reasons: dict[str, set[str]] = defaultdict(set)
    for group in review["exact_groups"]:
        group_id = components[group["text_sha256"]]
        if group["conflicting_contexts"]:
            reasons[group_id].add("exact_gold_conflict")
        elif group["annotation_sets_differ"] and group["corpus"] == "sentfin":
            reasons[group_id].add("target_coverage_difference")
    for pair in review["near_pairs"]:
        if pair["shared_context_gold_differs"]:
            reasons[components[pair["left_sha256"]]].add("near_gold_difference")
    corpora: dict[str, set[str]] = defaultdict(set)
    members: dict[str, list[SourceExample]] = defaultdict(list)
    for row in ordered:
        group_id = components[text_hash(row.text)]
        corpora[group_id].add(row.corpus)
        members[group_id].append(row)
        if row.snippet_issues:
            reasons[group_id].add("fiqa_snippet_review")
    for group_id, rows in members.items():
        if len(corpora[group_id]) > 1:
            reasons[group_id].add("cross_corpus_overlap")
        if any(row.unscored for row in rows) and any(not row.unscored for row in rows):
            reasons[group_id].add("original_unscored_holdout_overlap")
    representatives: dict[tuple[str, str, str], str] = {}
    source_decisions = []
    prepared = []
    for row in ordered:
        digest = text_hash(row.text)
        group_id = components[digest]
        flags = sorted(reasons[group_id])
        quarantine = bool(set(flags) & set(policy["quarantine"]))
        dev_only = bool(set(flags) & set(policy["development_only"]))
        partition = (
            None
            if row.unscored or quarantine
            else assign_partition(
                group_id,
                policy,
                development_only=dev_only,
            )
        )
        annotation_signature = _json(
            sorted(set(row.annotations), key=lambda a: (a[0], a[1] or ""))
        )
        key = (row.corpus, digest, annotation_signature)
        representative = None
        if row.unscored:
            status = "unscored"
        elif quarantine:
            status = "quarantined"
        elif key in representatives:
            status = "redundant_exact_duplicate"
            representative = representatives[key]
        else:
            status = "retained"
            representative = row.reference
            representatives[key] = representative
            prepared.append(
                {
                    "record_id": representative,
                    "corpus": row.corpus,
                    "group_id": group_id,
                    "text_sha256": digest,
                    "partition": partition,
                    "development_only": dev_only,
                    "review_reasons": flags,
                    "native_record": row.native_record,
                    "annotation_eligibility": [
                        {
                            "annotation_index": i,
                            "target_score_supervision": True,
                            "snippet_supervision": i not in row.snippet_issues,
                            "review_required": i in row.snippet_issues,
                        }
                        for i in range(len(row.annotations))
                    ]
                    if row.corpus == "fiqa"
                    else None,
                    "source_references": [],
                }
            )
        source_decisions.append(
            {
                "source_reference": row.reference,
                "corpus": row.corpus,
                "text_sha256": digest,
                "group_id": group_id,
                "partition": partition,
                "status": status,
                "representative_id": representative,
                "reasons": flags
                + (["original_test_without_gold"] if row.unscored else []),
                "flagged_snippet_annotation_indices": list(row.snippet_issues),
            }
        )
    lineage: dict[str, list[str]] = defaultdict(list)
    for decision in source_decisions:
        if decision["representative_id"] is not None:
            lineage[decision["representative_id"]].append(decision["source_reference"])
    for row in prepared:
        row["source_references"] = lineage[row["record_id"]]
    manifest = {
        "algorithm_version": 1,
        "asset_manifest_sha256": review["summary"]["manifest_sha256"],
        "revisions": review["summary"]["revisions"],
        "policy": policy,
        "policy_sha256": _digest(policy),
        "sources": source_decisions,
        "known_near_edges": [list(edge) for edge in sorted(edges)],
        "final_test_frozen": False,
        "test_role": (
            "provisional candidate; not independent FinBERT generalization evidence"
        ),
    }
    validate_split_plan(manifest, prepared)
    return manifest, prepared


def validate_split_plan(
    manifest: dict[str, Any], prepared: list[dict[str, Any]]
) -> None:
    """Fail on known duplicate overlap or incorrect representative lineage."""
    partitions: dict[str, set[str]] = defaultdict(set)
    hashes: dict[str, set[str]] = defaultdict(set)
    kept = {row["record_id"]: row for row in prepared}
    if len(kept) != len(prepared):
        raise ValueError("duplicate processed record IDs")
    decisions = {row["source_reference"]: row for row in manifest["sources"]}
    if len(decisions) != len(manifest["sources"]):
        raise ValueError("duplicate source references")
    for row in prepared:
        source = decisions.get(row["record_id"])
        if source is None or source["status"] != "retained":
            raise ValueError("excluded or unknown source entered processed records")
        if any(
            source[key] != row[key]
            for key in ("corpus", "group_id", "text_sha256", "partition")
        ):
            raise ValueError("processed record differs from source decision")
        if row["partition"] not in PARTITIONS:
            raise ValueError("invalid supervised partition")
        partitions[row["group_id"]].add(row["partition"])
        hashes[row["text_sha256"]].add(row["partition"])
        if row["development_only"] and row["partition"] == "test_candidate":
            raise ValueError("development-only group entered test candidate")
    if any(len(values) > 1 for values in (*partitions.values(), *hashes.values())):
        raise ValueError("known duplicate overlap across partitions")
    for left, right in manifest["known_near_edges"]:
        if len(hashes[left] | hashes[right]) > 1:
            raise ValueError("near duplicates cross partition boundaries")
    expected_lineage: dict[str, list[str]] = defaultdict(list)
    for decision in manifest["sources"]:
        representative = decision["representative_id"]
        if representative is not None:
            expected_lineage[representative].append(decision["source_reference"])
            if (
                representative not in kept
                or decision["source_reference"]
                not in kept[representative]["source_references"]
            ):
                raise ValueError("missing representative lineage")
            if decision["partition"] != kept[representative]["partition"]:
                raise ValueError("source and representative partition differ")
        elif decision["partition"] is not None:
            raise ValueError("excluded source assigned a supervised partition")
    for representative, row in kept.items():
        expected = sorted(expected_lineage[representative])
        if sorted(row["source_references"]) != expected:
            raise ValueError("incorrect representative lineage")


def prepare_splits(
    manifest_path: Path, policy_path: Path, data_dir: Path
) -> dict[str, Any]:
    """Recompute source review and save content-addressed local split artifacts."""
    review = review_datasets(manifest_path, data_dir)
    manifest = json.loads(manifest_path.read_text())
    policy = json.loads(policy_path.read_text())
    plan, prepared = make_split_plan(_load_sources(manifest, data_dir), review, policy)
    body = "".join(_json(row) + "\n" for row in prepared)
    plan["prepared_records_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    plan_body = _json(plan) + "\n"
    fingerprint = hashlib.sha256(plan_body.encode()).hexdigest()
    directory = data_dir / "processed/splits" / fingerprint
    directory.mkdir(parents=True, exist_ok=True)
    files = (
        ("split-manifest.json", plan_body),
        ("prepared-records.jsonl", body),
    )
    for name, content in files:
        path = directory / name
        if path.exists() and path.read_text() != content:
            raise ValueError(
                "existing immutable split artifact differs; inspect corruption"
            )
    for name, content in files:
        path = directory / name
        if not path.exists():
            path.write_text(content, encoding="utf-8")
    counts = {}
    for corpus in sorted({source["corpus"] for source in plan["sources"]}):
        sources = [source for source in plan["sources"] if source["corpus"] == corpus]
        rows = [row for row in prepared if row["corpus"] == corpus]
        counts[corpus] = {
            "sources": len(sources),
            "source_statuses": dict(
                sorted(Counter(s["status"] for s in sources).items())
            ),
            "prepared_records": len(rows),
            "partitions": {
                name: sum(row["partition"] == name for row in rows)
                for name in PARTITIONS
            },
            "class_counts_by_partition": {
                name: dict(
                    sorted(
                        Counter(
                            a["label"]
                            for row in rows
                            if row["partition"] == name
                            for a in (
                                row["native_record"].get("sentiments")
                                or [{"label": row["native_record"].get("label")}]
                            )
                            if a["label"] is not None
                        ).items()
                    )
                )
                for name in PARTITIONS
            },
        }
    return {
        "split_manifest_sha256": fingerprint,
        "directory": str(directory),
        "policy": policy,
        "corpora": counts,
        "quarantine_reasons": dict(
            sorted(
                Counter(
                    reason
                    for source in plan["sources"]
                    if source["status"] == "quarantined"
                    for reason in source["reasons"]
                    if reason in policy["quarantine"]
                ).items()
            )
        ),
        "fiqa_snippet_annotations_excluded_from_snippet_supervision": sum(
            eligibility["review_required"]
            for row in prepared
            for eligibility in row["annotation_eligibility"] or []
        ),
        "known_duplicate_overlap_checks_passed": True,
        "final_test_frozen": False,
    }


def render_split_report(summary: dict[str, Any]) -> str:
    """Publish aggregates only, keeping source IDs and processed text ignored."""
    weights = summary["policy"]["partition_weights"]
    ratios = "/".join(str(weights[name]) for name in PARTITIONS)
    lines = [
        "# FinSight provisional split preparation",
        "",
        f"Policy: approximate {ratios} group partitions, "
        f"seed {summary['policy']['seed']}; SHA-256 bucket",
        "assignment without stratification or seed searching. Development-only",
        "groups renormalize the train/validation weights. Counts can",
        "differ from the requested ratios because connected groups stay intact.",
        "",
        "| Corpus | Raw sources | Retained | Redundant | Quarantined | Unscored "
        "| Train | Validation | Test candidate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for corpus, counts in summary["corpora"].items():
        statuses = counts["source_statuses"]
        cells = [
            counts["sources"],
            counts["prepared_records"],
            statuses.get("redundant_exact_duplicate", 0),
            statuses.get("quarantined", 0),
            statuses.get("unscored", 0),
            *(counts["partitions"][name] for name in PARTITIONS),
        ]
        lines.append(
            f"| {corpus} | " + " | ".join(str(value) for value in cells) + " |"
        )
    lines += [
        "",
        "Exact equivalent copies retain one processed representative and all source",
        "references. Near pairs form transitive groups but are not automatically",
        "merged. Unresolved exact/near gold differences quarantine whole groups.",
        "Scored groups touching original unscored holdouts are also quarantined.",
        "Cross-corpus overlap, target-coverage differences, and FiQA snippet flags",
        "restrict a group to development. Mixed labels across different targets",
        "remain intact. FiQA continuous scores are never converted to classes.",
        "",
        "Flagged FiQA annotations excluded from snippet supervision: "
        f"{summary['fiqa_snippet_annotations_excluded_from_snippet_supervision']}.",
        "",
        "## Class counts after preparation",
        "",
        "PhraseBank counts sentences; SEntFiN counts target annotations. FiQA",
        "retains continuous scores, so its class-count dictionaries are empty.",
        "",
        "| Corpus | Partition | Negative | Neutral | Positive |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for corpus, counts in summary["corpora"].items():
        if corpus == "fiqa":
            continue
        for partition, labels in counts["class_counts_by_partition"].items():
            lines.append(
                f"| {corpus} | {partition} | "
                + " | ".join(
                    str(labels.get(label, 0))
                    for label in ("negative", "neutral", "positive")
                )
                + " |"
            )
    lines += [
        "",
        "## Reproduction and limitations",
        "",
        "Run `uv run --locked python -m finsight.data prepare-splits`.",
        "Source files are verified before processing and remain unchanged.",
        "Processed records and the full lineage manifest are ignored local data.",
        "Existing content-addressed artifacts must match exactly; new policies",
        "produce new directories rather than silently replacing prior splits.",
        "",
        f"Split manifest SHA-256: `{summary['split_manifest_sha256']}`.",
        "",
        "Known exact text, connected-group, and near-edge partition overlap",
        "checks pass. This covers the current lexical screen, not all semantic",
        "overlap or inaccessible model-training corpora. PhraseBank test",
        "candidates remain unsuitable as independent pretrained FinBERT evidence.",
        "SEntFiN candidates still need independent-benchmark curation and review.",
        "No final test is frozen and no model training or tuning is performed.",
    ]
    return "\n".join(lines) + "\n"
