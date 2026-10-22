"""Validation-only comparison of registry mentions and pretrained ORG proposals."""

import argparse
import json
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from finsight.data.target_preparation import json_bytes
from finsight.data.target_spans import literal_spans
from finsight.sentiment.artifacts import file_sha256, project_root, sha
from finsight.sentiment.companies import CompanyRegistry, load_registry
from finsight.sentiment.evaluation.target_data import development_inputs
from finsight.sentiment.mentions import detect_companies
from finsight.sentiment.organizations import (
    PLAN,
    REVISION,
    OrganizationDetector,
    OrganizationMention,
)


def validation_headlines(rows: list[dict]) -> dict[str, dict]:
    """Group annotations without opening or accepting any test/training records."""
    headlines: dict[str, dict] = {}
    for row in rows:
        if row["partition"] != "validation":
            raise ValueError("only validation records are supported")
        item = headlines.setdefault(
            row["record_id"], {"headline": row["headline"], "targets": []}
        )
        if item["headline"] != row["headline"]:
            raise ValueError("inconsistent headline identity")
        if row["target"] in item["targets"]:
            raise ValueError("duplicate target annotation")
        item["targets"].append(row["target"])
    return dict(sorted(headlines.items()))


def _relation(spans: list, mentions: list | tuple) -> tuple[bool, bool]:
    exact = any((s.start, s.end) == (m.start, m.end) for s in spans for m in mentions)
    overlap = any(s.start < m.end and m.start < s.end for s in spans for m in mentions)
    return exact, overlap


def compare_detections(
    rows: list[dict],
    organizations: dict[str, tuple[OrganizationMention, ...]],
    registry: CompanyRegistry,
    company_allowlist: set[str],
) -> tuple[dict, list[dict]]:
    """Count annotated-target coverage without inventing exhaustive mention gold.

    A target counts once per headline, even if it appears repeatedly. Exact and
    overlap coverage are inclusive: exact matches are also overlaps. The separate
    proposal relation table uses disjoint exact / overlap-only / unmatched bins.
    """
    headlines = validation_headlines(rows)
    if set(organizations) != set(headlines):
        raise ValueError("NER headline membership differs")
    scopes = {
        name: {
            "annotations": 0,
            "literal_annotations": 0,
            "nonliteral_annotations": 0,
            **{
                d: {"exact": 0, "overlap": 0}
                for d in ("registry_resolved", "registry_all_candidates", "ner_org")
            },
        }
        for name in ("all_native_financial_targets", "fixed_company_allowlist")
    }
    totals: Counter = Counter()
    relation: Counter = Counter(
        {"exact": 0, "overlap_only": 0, "no_literal_annotation_overlap": 0}
    )
    cases = []
    for identifier, item in headlines.items():
        text = item["headline"]
        orgs = organizations[identifier]
        if any(text[m.start : m.end] != m.text for m in orgs):
            raise ValueError("NER span differs from source text")
        baseline = detect_companies(text, registry)
        candidates = {
            "registry_resolved": baseline.resolved,
            "registry_all_candidates": (*baseline.resolved, *baseline.ambiguous),
            "ner_org": orgs,
        }
        target_checks, all_spans = [], []
        for target in item["targets"]:
            spans = literal_spans(text, target)
            all_spans.extend(spans)
            checks = {
                d: dict(zip(("exact", "overlap"), _relation(spans, ms), strict=True))
                for d, ms in candidates.items()
            }
            target_checks.append({"target": target, "literal": bool(spans), **checks})
            for scope, counts in scopes.items():
                if (
                    scope == "fixed_company_allowlist"
                    and target not in company_allowlist
                ):
                    continue
                counts["annotations"] += 1
                counts["literal_annotations"] += bool(spans)
                counts["nonliteral_annotations"] += not spans
                for d, check in checks.items():
                    for measure, matched in check.items():
                        counts[d][measure] += matched
        org_relations = []
        for org in orgs:
            exact, overlap = _relation(all_spans, [org])
            category = (
                "exact"
                if exact
                else "overlap_only"
                if overlap
                else "no_literal_annotation_overlap"
            )
            relation[category] += 1
            org_relations.append({**asdict(org), "annotation_relation": category})
        totals.update(
            {
                "headlines": 1,
                "native_annotations": len(item["targets"]),
                "ner_org_spans": len(orgs),
                "ner_org_headlines": bool(orgs),
                "multiple_org_headlines": len({m.text.casefold() for m in orgs}) > 1,
                "registry_resolved_spans": len(baseline.resolved),
                "registry_resolved_headlines": bool(baseline.resolved),
                "registry_ambiguous_spans": len(baseline.ambiguous),
                "registry_ambiguous_headlines": bool(baseline.ambiguous),
                "ner_org_without_registry_candidate_headlines": bool(orgs)
                and not (baseline.resolved or baseline.ambiguous),
            }
        )
        categories = []
        if any(
            o["annotation_relation"] == "no_literal_annotation_overlap"
            for o in org_relations
        ):
            categories.append("unmatched_org")
        if any(c["literal"] and not c["ner_org"]["overlap"] for c in target_checks):
            categories.append("missed_literal_target")
        if len({m.text.casefold() for m in orgs}) > 1:
            categories.append("multiple_orgs")
        if baseline.ambiguous:
            categories.append("registry_ambiguous")
        cases.append(
            {
                "record_id": identifier,
                "partition": "validation",
                **item,
                "organizations": org_relations,
                "target_checks": target_checks,
                "registry": asdict(baseline),
                "review_categories": categories,
            }
        )
    return {
        "totals": dict(totals),
        "coverage": scopes,
        "ner_span_annotation_relation": dict(relation),
    }, cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    args = parser.parse_args()
    root = args.root.resolve()
    data = args.data_dir or root / "data"
    plan = json.loads((root / PLAN).read_bytes())
    # Existing helper verifies both development partitions; only validation is
    # sent to NER. It never reads the reserved records.
    _, rows, inputs = development_inputs(root, data)
    headlines = validation_headlines(rows)
    registry = load_registry(root / "configs/companies.json")
    model_dir = args.model_dir or data / "interim/organization-ner-model" / REVISION
    # Keep this historical baseline reproducible when the interactive default changes.
    detector = OrganizationDetector.load(
        model_dir, root=root, device=args.device, aggregation_strategy="simple"
    )
    import torch

    torch.manual_seed(42)
    torch.set_num_threads(4)
    texts = [x["headline"] for x in headlines.values()]
    lengths = detector.tokenizer(texts, truncation=False)["input_ids"]
    started = perf_counter()
    detected = detector.detect_many(texts, batch_size=plan["batch_size"])
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = perf_counter() - started
    counts, cases = compare_detections(
        rows,
        dict(zip(headlines, detected, strict=True)),
        registry,
        set(plan["company_target_allowlist"]),
    )
    model = detector.pipeline.model
    c = model.config
    source_files = (
        "sentiment/organizations.py",
        "sentiment/mentions.py",
        "sentiment/companies.py",
        "sentiment/evaluation/target_data.py",
        "sentiment/experiments/organization_detection.py",
        "data/target_spans.py",
    )
    provenance = {
        "plan_sha256": file_sha256(root / PLAN),
        "inputs": inputs,
        "registry_sha256": registry.sha256,
        "source_sha256": {
            name: file_sha256(root / "src/finsight" / name) for name in source_files
        },
    }
    private = data / "interim/organization-detection" / sha(json_bytes(provenance))
    private.mkdir(parents=True, exist_ok=True)
    (private / "cases.jsonl").write_bytes(
        b"".join((json.dumps(case, sort_keys=True) + "\n").encode() for case in cases)
    )
    # Fixed deterministic per-category review. Categories may share a headline;
    # this is exploratory inspection, never independent human gold annotation.
    review_ids: dict[str, list[str]] = {}
    for category in plan["review_categories"]:
        eligible = [c for c in cases if category in c["review_categories"]]
        eligible.sort(key=lambda c: sha(c["record_id"].encode()))
        review_ids[category] = [
            c["record_id"] for c in eligible[: plan["review_per_category"]]
        ]
    (private / "review-membership.json").write_bytes(json_bytes(review_ids))
    summary = {
        "experiment": plan["experiment"],
        **provenance,
        **counts,
        "model_id": plan["model_id"],
        "revision": plan["revision"],
        "model_files": plan["files"],
        "configuration": {
            k: plan[k]
            for k in (
                "aggregation_strategy",
                "score_threshold",
                "partition",
                "batch_size",
                "max_length",
                "registry_policy",
                "company_target_allowlist",
            )
        },
        "model": {
            "class": type(model).__name__,
            "layers": c.num_hidden_layers,
            "hidden_size": c.hidden_size,
            "attention_heads": c.num_attention_heads,
            "token_classes": c.num_labels,
            "parameters": sum(p.numel() for p in model.parameters()),
            "parameter_bytes": sum(
                p.numel() * p.element_size() for p in model.parameters()
            ),
            "device": args.device,
            "inference_seconds": elapsed,
            "maximum_encoded_tokens": max(map(len, lengths)),
            "torch_version": torch.__version__,
        },
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "private_packet_sha256": file_sha256(private / "cases.jsonl"),
        "private_directory_id": private.name,
        "review_membership_sha256": file_sha256(private / "review-membership.json"),
        "review_counts": {k: len(v) for k, v in review_ids.items()},
        "review_unique_headlines": len({i for ids in review_ids.values() for i in ids}),
        "frozen_records_read": False,
        "training_performed": False,
        "sentiment_inference_on_dataset": False,
        "precision_recall_computed": False,
    }
    (private / "summary.json").write_bytes(json_bytes(summary))
    output = (
        root
        / "reports/phase-04-target-sentiment/experiments/organization-detection.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(json_bytes(summary))
    print(json.dumps({"private": str(private), "totals": summary["totals"]}, indent=2))


if __name__ == "__main__":
    main()
