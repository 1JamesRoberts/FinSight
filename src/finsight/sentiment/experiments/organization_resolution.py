"""Verified-cache evaluation of conservative identity resolution; no model fitting."""

import json
from collections import Counter
from dataclasses import asdict

from finsight.data.organization_annotations import validate_annotations
from finsight.data.target_preparation import json_bytes
from finsight.sentiment.artifacts import file_sha256, project_root, sha
from finsight.sentiment.companies import load_registry
from finsight.sentiment.evaluation.organization_metrics import exact_span_metrics
from finsight.sentiment.experiments.organization_benchmark import sample_rows
from finsight.sentiment.organization_resolution import (
    RESOLVER_VERSION,
    resolve_organizations,
)
from finsight.sentiment.organizations import OrganizationMention


def evaluate_resolutions(
    references: list[dict],
    predictions: dict[str, list[dict]],
    registry,
) -> tuple[dict, list[dict], list[dict]]:
    """Resolve before reference joining; keep all unknown names in the denominator."""
    expected = {r["record_id"] for r in references}
    if len(expected) != len(references) or set(predictions) != expected:
        raise ValueError("resolution evaluation membership differs")
    cells = {
        status: Counter({"company": 0, "other_organization": 0, "unmatched_span": 0})
        for status in ("verified_company", "other_organization", "unresolved")
    }
    accepted_company, accepted_other, raw = {}, {}, {}
    output, errors = [], []
    kinds = Counter(
        {
            "overextended_span": 0,
            "merged_names": 0,
            "spurious_span": 0,
            "reference_lost_to_boundary": 0,
            "complete_omission": 0,
        }
    )
    for row in references:
        identifier, text = row["record_id"], row["headline"]
        mentions = tuple(
            OrganizationMention(m["text"], m["start"], m["end"], None)
            for m in predictions[identifier]
        )
        resolved = resolve_organizations(text, mentions, registry)
        gold = {(e["start"], e["end"]): e for e in row["entities"]}
        raw[identifier] = [(m.start, m.end) for m in mentions]
        accepted_company[identifier] = []
        accepted_other[identifier] = []
        for r in resolved:
            key = (r.mention.start, r.mention.end)
            entity = gold.get(key)
            relation = entity["type"] if entity else "unmatched_span"
            cells[r.status][relation] += 1
            if r.status == "verified_company":
                accepted_company[identifier].append(key)
            if r.status == "other_organization":
                accepted_other[identifier].append(key)
            output.append(
                {"record_id": identifier, **asdict(r), "reference_relation": relation}
            )
            if not entity:
                overlap = [e for (a, b), e in gold.items() if a < key[1] and key[0] < b]
                category = (
                    "merged_names"
                    if len(overlap) > 1
                    else "overextended_span"
                    if overlap
                    else "spurious_span"
                )
                kinds[category] += 1
                errors.append(
                    {
                        "record_id": identifier,
                        "headline": text,
                        "error": "extra_prediction",
                        "category": category,
                        "mention": asdict(r.mention),
                        "overlapping_reference": overlap,
                    }
                )
        for key, e in gold.items():
            if key in raw[identifier]:
                continue
            overlap = any(a < key[1] and key[0] < b for a, b in raw[identifier])
            category = "reference_lost_to_boundary" if overlap else "complete_omission"
            kinds[category] += 1
            errors.append(
                {
                    "record_id": identifier,
                    "headline": text,
                    "error": "missed_reference",
                    "category": category,
                    "entity": e,
                }
            )
    company = {
        r["record_id"]: [
            (e["start"], e["end"]) for e in r["entities"] if e["type"] == "company"
        ]
        for r in references
    }
    other = {
        r["record_id"]: [
            (e["start"], e["end"])
            for e in r["entities"]
            if e["type"] == "other_organization"
        ]
        for r in references
    }
    total = sum(sum(c.values()) for c in cells.values())
    answered = sum(
        sum(c.values()) for status, c in cells.items() if status != "unresolved"
    )
    correct = (
        cells["verified_company"]["company"]
        + cells["other_organization"]["other_organization"]
    )
    return (
        {
            "headlines": len(references),
            "proposed_spans": total,
            "decisions_by_reference": {k: dict(v) for k, v in cells.items()},
            "decision_coverage": answered / total if total else None,
            "correct_typed_exact_spans": correct,
            "selective_typed_exact_precision": correct / answered if answered else None,
            "raw_org_as_company": exact_span_metrics(company, raw),
            "verified_company_extraction": exact_span_metrics(
                company, accepted_company
            ),
            "other_organization_extraction": exact_span_metrics(other, accepted_other),
            "extraction_error_categories": dict(kinds),
        },
        output,
        errors,
    )


def main() -> None:
    root = project_root()
    result_path = (
        root
        / "reports/phase-04-target-sentiment/evaluation"
        / "organization-benchmark-reviewed.json"
    )
    benchmark = json.loads(result_path.read_bytes())
    verification = json.loads(
        (
            root
            / "reports/phase-04-target-sentiment/verification"
            / "organization-benchmark-reviewed-verification.json"
        ).read_bytes()
    )
    if file_sha256(result_path) != verification["result_sha256"]:
        raise ValueError("reviewed benchmark checksum differs")
    packet = (
        root
        / "data/interim/organization-annotations"
        / benchmark["sample_manifest_sha256"]
    )
    annotations = packet / "annotations.reviewed.jsonl"
    if file_sha256(annotations) != benchmark["annotation_sha256"]:
        raise ValueError("reference checksum differs")
    rows = [json.loads(x) for x in annotations.read_bytes().splitlines()]
    validate_annotations(sample_rows(packet, root), rows)
    runs = [
        d
        for d in (packet / "evaluations").iterdir()
        if file_sha256(d / "result.json") == verification["result_sha256"]
    ]
    if len(runs) != 1:
        raise ValueError("exact benchmark execution cache required")
    cache = runs[0] / "predictions.jsonl"
    if file_sha256(cache) != benchmark["predictions_sha256"]:
        raise ValueError("prediction cache differs")
    predictions = {
        r["record_id"]: r["mentions"]
        for r in map(json.loads, cache.read_bytes().splitlines())
        if r["method"] == "ner_first"
    }
    registry = load_registry(root / "configs/companies.json")
    summary, decisions, errors = evaluate_resolutions(rows, predictions, registry)
    provenance = {
        "reference_sha256": benchmark["annotation_sha256"],
        "prediction_cache_sha256": benchmark["predictions_sha256"],
        "benchmark_sha256": verification["result_sha256"],
        "registry_sha256": registry.sha256,
        "source_sha256": {
            n: file_sha256(root / "src/finsight/sentiment" / n)
            for n in (
                "organization_resolution.py",
                "companies.py",
                "experiments/organization_resolution.py",
            )
        },
    }
    private = (
        root / "data/interim/organization-resolution" / sha(json_bytes(provenance))
    )
    private.mkdir(parents=True, exist_ok=True)
    for name, rs in (("decisions.jsonl", decisions), ("error-review.jsonl", errors)):
        (private / name).write_bytes(
            b"".join((json.dumps(r, sort_keys=True) + "\n").encode() for r in rs)
        )
    result = {
        "experiment": RESOLVER_VERSION,
        **provenance,
        **summary,
        "private_packet_id": private.name,
        "decisions_sha256": file_sha256(private / "decisions.jsonl"),
        "error_review_sha256": file_sha256(private / "error-review.jsonl"),
        "policy": {
            "company_context": False,
            "allow_parent_rollup": False,
            "unknown_names": "unresolved",
            "gold_in_resolver": False,
        },
        "reference_quality": "assistant_drafted_user_accepted_development",
        "model_inference_performed": False,
        "training_performed": False,
        "frozen_records_read": False,
    }
    body = json_bytes(result)
    (private / "result.json").write_bytes(body)
    output = (
        root
        / "reports/phase-04-target-sentiment/experiments/organization-resolution.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(body)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
