"""Freeze a qualified external holdout before any model development."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from finsight.data.audit import text_identity
from finsight.data.splits import prepare_splits, validate_split_plan


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def curate_benchmark(
    plan: dict[str, Any], records: list[dict[str, Any]], policy: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the preassigned holdout, preserving targets and native gold labels."""
    if policy["schema_version"] != 1 or policy["corpus"] != "sentfin":
        raise ValueError("unsupported benchmark policy")
    if policy["selection"] != (
        "all eligible sentfin test_candidate records from the pinned provisional split"
    ):
        raise ValueError("unsupported benchmark selection")
    if policy["labels"] != ["negative", "neutral", "positive"]:
        raise ValueError("unsupported benchmark labels")
    if (
        policy["primary_metric"] != "macro_f1_per_target"
        or policy["sentence_control_cohort"] != "single_target"
        or policy["uncertainty_unit"] != "duplicate_group"
        or policy["target_cohorts"]
        != ["all", "single_target", "multi_target", "mixed_target_sentiment"]
    ):
        raise ValueError("unsupported evaluation protocol")
    validate_split_plan(plan, records)
    decisions = {row["source_reference"]: row for row in plan["sources"]}
    exclusions = policy["curation_exclusions"]
    if any(reference not in decisions for reference in exclusions):
        raise ValueError("curation exclusion references unknown source")
    excluded_groups = {decisions[reference]["group_id"] for reference in exclusions}
    selected = sorted(
        (
            row
            for row in records
            if row["corpus"] == "sentfin"
            and row["partition"] == "test_candidate"
            and row["group_id"] not in excluded_groups
        ),
        key=lambda row: row["record_id"],
    )
    if not selected:
        raise ValueError("benchmark selection is empty")
    cohorts: dict[str, list[dict[str, Any]]] = {
        name: [] for name in policy["target_cohorts"]
    }
    labels: Counter[str] = Counter()
    targets: set[str] = set()
    nonliteral_targets = 0
    for row in selected:
        if row["development_only"] or row["review_reasons"]:
            raise ValueError("flagged record entered benchmark")
        annotations = row["native_record"]["sentiments"]
        if not annotations or any(
            a["label"] not in policy["labels"] for a in annotations
        ):
            raise ValueError("benchmark has missing or invalid gold labels")
        cohorts["all"].append(row)
        cohorts["single_target" if len(annotations) == 1 else "multi_target"].append(
            row
        )
        if len({a["label"] for a in annotations}) > 1:
            cohorts["mixed_target_sentiment"].append(row)
        labels.update(a["label"] for a in annotations)
        targets.update(text_identity(a["target"]) for a in annotations)
        title = text_identity(row["native_record"]["title"])
        nonliteral_targets += sum(
            text_identity(a["target"]) not in title for a in annotations
        )
    if set(labels) != set(policy["labels"]):
        raise ValueError("benchmark must contain all three gold classes")
    summary = {
        "curation_excluded_groups": len(excluded_groups),
        "curation_excluded_candidate_records": sum(
            row["corpus"] == "sentfin"
            and row["partition"] == "test_candidate"
            and row["group_id"] in excluded_groups
            for row in records
        ),
        "headlines": len(selected),
        "duplicate_groups": len({row["group_id"] for row in selected}),
        "target_annotations": sum(labels.values()),
        "target_class_counts": dict(sorted(labels.items())),
        "unique_normalized_targets": len(targets),
        "nonliteral_target_annotations": nonliteral_targets,
        "cohorts": {
            name: {
                "headlines": len(rows),
                "target_annotations": sum(
                    len(r["native_record"]["sentiments"]) for r in rows
                ),
                "class_counts": dict(
                    sorted(
                        Counter(
                            a["label"]
                            for r in rows
                            for a in r["native_record"]["sentiments"]
                        ).items()
                    )
                ),
            }
            for name, rows in cohorts.items()
        },
        "mvp_exact_target_name_annotations": {
            ticker: sum(
                text_identity(a["target"]) in {text_identity(n) for n in names}
                for row in selected
                for a in row["native_record"]["sentiments"]
            )
            for ticker, names in policy["coverage_names"].items()
        },
    }
    return selected, summary


def freeze_benchmark(
    manifest_path: Path,
    split_policy_path: Path,
    policy_path: Path,
    data_dir: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Reproduce the holdout and fail rather than replace an existing freeze."""
    splits = prepare_splits(
        manifest_path, split_policy_path, data_dir, freeze_path=receipt_path
    )
    directory = Path(splits["directory"])
    plan_body = (directory / "split-manifest.json").read_text(encoding="utf-8")
    if _sha(plan_body) != splits["split_manifest_sha256"]:
        raise ValueError("split manifest checksum mismatch")
    plan = json.loads(plan_body)
    records_body = (directory / "prepared-records.jsonl").read_text(encoding="utf-8")
    if _sha(records_body) != plan["prepared_records_sha256"]:
        raise ValueError("processed records checksum mismatch")
    records = [json.loads(line) for line in records_body.splitlines()]
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    selected, summary = curate_benchmark(plan, records, policy)
    assets = json.loads(manifest_path.read_text(encoding="utf-8"))
    body = "".join(_json(row) + "\n" for row in selected)
    frozen = {
        "schema_version": 1,
        "benchmark_id": policy["benchmark_id"],
        "final_test_frozen": True,
        "parent_split_manifest_sha256": splits["split_manifest_sha256"],
        "asset_manifest_sha256": plan["asset_manifest_sha256"],
        "policy": policy,
        "summary": summary,
        "model_reference": {
            key: assets["assets"]["prosusai_finbert"][key]
            for key in ("repository", "revision")
        },
        "records_sha256": _sha(body),
        "reserved_groups": sorted({row["group_id"] for row in selected}),
        "curation_excluded_groups": sorted(
            {
                row["group_id"]
                for row in plan["sources"]
                if row["source_reference"] in policy["curation_exclusions"]
            }
        ),
    }
    frozen_body = _json(frozen) + "\n"
    fingerprint = _sha(frozen_body)
    receipt = {
        "schema_version": 1,
        "benchmark_id": policy["benchmark_id"],
        "final_test_frozen": True,
        "parent_split_manifest_sha256": splits["split_manifest_sha256"],
        "benchmark_manifest_sha256": fingerprint,
        "records_sha256": _sha(body),
        "policy_sha256": _sha(_json(policy)),
        "summary": summary,
    }
    destination = data_dir / "processed/benchmarks" / fingerprint
    outputs = [
        (destination / "benchmark-manifest.json", frozen_body),
        (destination / "test-records.jsonl", body),
        (receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n"),
    ]
    for path, content in outputs:
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError(
                "frozen benchmark differs; never overwrite a frozen release"
            )
    for path, content in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(content, encoding="utf-8")
    return receipt


def load_development_records(
    data_dir: Path, receipt_path: Path
) -> list[dict[str, Any]]:
    """Load only verified train/validation records from the freeze's parent split."""
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    frozen_dir = (
        data_dir / "processed/benchmarks" / receipt["benchmark_manifest_sha256"]
    )
    body = (frozen_dir / "benchmark-manifest.json").read_text(encoding="utf-8")
    if _sha(body) != receipt["benchmark_manifest_sha256"]:
        raise ValueError("frozen manifest checksum mismatch")
    frozen = json.loads(body)
    if (
        frozen["parent_split_manifest_sha256"]
        != receipt["parent_split_manifest_sha256"]
    ):
        raise ValueError("freeze receipt and parent split differ")
    test_body = (frozen_dir / "test-records.jsonl").read_text(encoding="utf-8")
    if (
        _sha(test_body) != frozen["records_sha256"]
        or _sha(test_body) != receipt["records_sha256"]
    ):
        raise ValueError("frozen records checksum mismatch")
    split_dir = data_dir / "processed/splits" / receipt["parent_split_manifest_sha256"]
    plan_body = (split_dir / "split-manifest.json").read_text(encoding="utf-8")
    if _sha(plan_body) != receipt["parent_split_manifest_sha256"]:
        raise ValueError("parent split checksum mismatch")
    plan = json.loads(plan_body)
    records_body = (split_dir / "prepared-records.jsonl").read_text(encoding="utf-8")
    if _sha(records_body) != plan["prepared_records_sha256"]:
        raise ValueError("development records checksum mismatch")
    records = [json.loads(line) for line in records_body.splitlines()]
    validate_split_plan(plan, records)
    development = [
        row for row in records if row["partition"] in {"train", "validation"}
    ]
    reserved = set(frozen["reserved_groups"]) | set(frozen["curation_excluded_groups"])
    if any(row["group_id"] in reserved for row in development):
        raise ValueError("frozen group entered development data")
    return development


def render_benchmark_report(receipt: dict[str, Any]) -> str:
    """Render aggregate release evidence without publishing source texts or IDs."""
    summary = receipt["summary"]
    lines = [
        "# Frozen SEntFiN evaluation benchmark",
        "",
        f"Release: `{receipt['benchmark_id']}`. Final holdout frozen before models.",
        "",
        f"Headlines: {summary['headlines']}; target annotations: "
        f"{summary['target_annotations']}; duplicate groups: "
        f"{summary['duplicate_groups']}.",
        f"Curation excluded {summary['curation_excluded_candidate_records']} "
        "candidate records with malformed target annotations; "
        "their groups stay withheld.",
        "",
        "| Cohort | Headlines | Targets | Negative | Neutral | Positive |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, counts in summary["cohorts"].items():
        labels = counts["class_counts"]
        lines.append(
            f"| {name} | {counts['headlines']} | {counts['target_annotations']} | "
            + " | ".join(
                str(labels.get(label, 0))
                for label in ("negative", "neutral", "positive")
            )
            + " |"
        )
    lines += [
        "",
        f"Unique normalized targets: {summary['unique_normalized_targets']}.",
        f"Nonliteral target names: {summary['nonliteral_target_annotations']} "
        "annotations "
        "(retained; canonical names need not appear literally in headlines).",
        "",
        "MVP exact target-name matches (descriptive; no entity resolution):",
    ]
    lines += [
        f"- {ticker}: {count}"
        for ticker, count in summary["mvp_exact_target_name_annotations"].items()
    ]
    lines += [
        "",
        "This is held out from FinSight development. Known accessible-corpus",
        "exact/near overlap is excluded. Full checkpoint training exposure is unknown.",
        "Financial entities are not all verified companies; this is not a five-company",
        "US-market benchmark. See `docs/evaluation-contract.md` for the protocol.",
        "",
        f"Benchmark manifest SHA-256: `{receipt['benchmark_manifest_sha256']}`.",
        f"Records SHA-256: `{receipt['records_sha256']}`.",
        "",
        "Raw and frozen records remain ignored local research data. The checked-in",
        "receipt pins membership and labels through content digests, without text.",
        "No model training, inference, or evaluation metrics have been implemented.",
    ]
    return "\n".join(lines) + "\n"
