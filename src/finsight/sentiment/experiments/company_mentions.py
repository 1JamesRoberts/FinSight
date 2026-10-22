"""Development-only coverage audit; native targets are not exhaustive NER gold."""

import argparse
import json
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from finsight.sentiment.artifacts import file_sha256, project_root
from finsight.sentiment.companies import CompanyRegistry, load_registry, resolve_company
from finsight.sentiment.evaluation.target_data import development_inputs
from finsight.sentiment.mentions import DETECTOR_VERSION, detect_companies


def audit_partition(
    rows: list[dict], registry: CompanyRegistry, partition: str
) -> tuple[dict, list[dict]]:
    """Count unique headlines, retaining only relevant cases in a private packet."""
    if partition not in {"train", "validation"}:
        raise ValueError("only development partitions are supported")
    headlines: dict[str, dict] = {}
    for row in rows:
        if row["partition"] != partition:
            raise ValueError("unexpected partition")
        item = headlines.setdefault(
            row["record_id"], {"headline": row["headline"], "annotations": []}
        )
        if item["headline"] != row["headline"]:
            raise ValueError("headline identity has inconsistent text")
        item["annotations"].append({"target": row["target"], "label": row["label"]})

    totals: Counter = Counter()
    companies = {c.company_id: Counter() for c in registry.companies}
    reasons: Counter = Counter()
    packet = []
    for identifier, item in sorted(headlines.items()):
        detection = detect_companies(item["headline"], registry)
        resolved = {m.resolution.company.company_id for m in detection.resolved}
        ambiguous = {
            c.company_id for m in detection.ambiguous for c in m.resolution.candidates
        }
        # Target annotations explicitly denote financial entities. This is a
        # comparison view only: do not pass that context into headline detection.
        native = set()
        for annotation in item["annotations"]:
            identity = resolve_company(
                annotation["target"], registry, company_context=True
            )
            if identity.company is not None:
                native.add(identity.company.company_id)
        totals.update(
            {
                "headlines": 1,
                "target_annotations": len(item["annotations"]),
                "resolved_headlines": bool(resolved),
                "ambiguous_headlines": bool(ambiguous),
                "candidate_headlines": bool(resolved | ambiguous),
                "multi_resolved_company_headlines": len(resolved) > 1,
                "resolved_spans": len(detection.resolved),
                "ambiguous_spans": len(detection.ambiguous),
            }
        )
        reasons.update(m.resolution.reason for m in detection.ambiguous)
        for company, counts in companies.items():
            counts.update(
                {
                    "resolved_headlines": company in resolved,
                    "ambiguous_headlines": company in ambiguous,
                    "native_target_headlines": company in native,
                    "native_target_resolved": company in native & resolved,
                    "native_target_ambiguous_only": company
                    in (native & ambiguous) - resolved,
                    "native_target_no_candidate": company
                    in native - resolved - ambiguous,
                    "resolved_without_native_target": company in resolved - native,
                }
            )
        if resolved or ambiguous or native:
            packet.append(
                {
                    "record_id": identifier,
                    "partition": partition,
                    **item,
                    "detection": asdict(detection),
                    "native_registry_companies": sorted(native),
                }
            )
    return {
        "totals": dict(totals),
        "companies": {k: dict(v) for k, v in companies.items()},
        "ambiguous_span_reasons": dict(reasons),
    }, packet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    data = args.data_dir or root / "data"
    training, validation, inputs = development_inputs(root, data)
    registry = load_registry(root / "configs/companies.json")
    summary = {
        "detector_version": DETECTOR_VERSION,
        "registry_sha256": registry.sha256,
        "inputs": inputs,
        "policy": {"company_context": False, "allow_parent_rollup": False},
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_sha256": {
            name: file_sha256(root / "src/finsight/sentiment" / name)
            for name in (
                "mentions.py",
                "companies.py",
                "experiments/company_mentions.py",
                "evaluation/target_data.py",
            )
        },
        "frozen_records_read": False,
        "sentiment_inference": False,
        "partitions": {},
    }
    packet = []
    for partition, rows in (("train", training), ("validation", validation)):
        counts, cases = audit_partition(rows, registry, partition)
        summary["partitions"][partition] = counts
        packet.extend(cases)
    # Full source text/annotations stay under ignored data; publish counts only.
    private = data / "interim/company-mention-review"
    private.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in packet)
    (private / "cases.jsonl").write_text(body, encoding="utf-8")
    summary["private_packet_sha256"] = file_sha256(private / "cases.jsonl")
    summary["private_packet_headlines"] = len(packet)
    output = (
        root
        / "reports/phase-04-target-sentiment/experiments/company-mention-coverage.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
