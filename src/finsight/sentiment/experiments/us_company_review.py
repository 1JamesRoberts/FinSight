"""Run a pinned US-company review packet; accuracy requires approved references."""

import argparse
import json
from collections import Counter
from pathlib import Path

from finsight.data.target_preparation import json_bytes
from finsight.data.us_company_review import validate_review
from finsight.data.us_listed_companies import load_directory
from finsight.sentiment.artifacts import file_sha256, project_root, sha
from finsight.sentiment.companies import load_registry
from finsight.sentiment.evaluation.company_linking import linking_metrics
from finsight.sentiment.organization_resolution import resolve_organizations
from finsight.sentiment.organizations import REVISION, OrganizationDetector


def read_packet(packet: Path, root: Path) -> list[dict]:
    manifest = json.loads((packet / "manifest.json").read_bytes())
    if (
        file_sha256(packet / "manifest.json") != packet.name
        or file_sha256(packet / "sample.jsonl") != manifest["sample_sha256"]
        or file_sha256(root / "configs/us-company-review-plan.json")
        != manifest["plan_sha256"]
        or file_sha256(root / "src/finsight/data/us_company_review.py")
        != manifest["selection_source_sha256"]
    ):
        raise ValueError("review packet, plan or sampling implementation differs")
    return [
        json.loads(line) for line in (packet / "sample.jsonl").read_bytes().splitlines()
    ]


def run(
    root: Path,
    data: Path,
    packet: Path,
    *,
    annotations: Path | None = None,
    device: str = "cpu",
) -> dict:
    rows = read_packet(packet, root)
    plan = json.loads((root / "configs/us-company-review-plan.json").read_bytes())
    reviewed = None
    if annotations is not None:
        reviewed = [json.loads(line) for line in annotations.read_bytes().splitlines()]
        # Reject unapproved drafts before any detector inference or metrics.
        validate_review(rows, reviewed)
    directory = load_directory(
        data / "interim/us-listed-directory" / plan["directory_sha256"]
    )
    registry = load_registry(root / "configs/companies.json")
    detector = OrganizationDetector.load(
        data / "interim/organization-ner-model" / REVISION,
        root=root,
        device=device,
        aggregation_strategy="first",
    )
    mentions = detector.detect_many([row["headline"] for row in rows])
    outputs = {"registry_only": {}, "registry_plus_sec": {}}
    counts = {}
    for method in outputs:
        reasons: Counter = Counter()
        for row, proposed in zip(rows, mentions, strict=True):
            resolved = resolve_organizations(
                row["headline"],
                proposed,
                registry,
                company_context=plan["company_context"],
                allow_parent_rollup=plan["allow_parent_rollup"],
                us_directory=directory if method == "registry_plus_sec" else None,
            )
            predictions = []
            for item in resolved:
                cik = item.sec_cik
                # Evaluation identity crosswalk only: legacy matches retain their
                # original acceptance policy. Never turn an abstention into a link.
                if item.status == "verified_company" and cik is None:
                    known = next(
                        c for c in registry.companies if c.company_id == item.company_id
                    )
                    candidates = set().union(
                        *(directory.tickers.get(t, set()) for t in known.tickers)
                    )
                    if len(candidates) == 1:
                        cik = next(iter(candidates))
                predictions.append(
                    {
                        "text": item.mention.text,
                        "start": item.mention.start,
                        "end": item.mention.end,
                        "status": item.status,
                        "reason": item.reason,
                        "cik": cik,
                    }
                )
                reasons[item.reason] += 1
            outputs[method][row["record_id"]] = predictions
        counts[method] = {
            "proposed_spans": sum(len(v) for v in outputs[method].values()),
            "linked_spans": sum(
                p["cik"] is not None for v in outputs[method].values() for p in v
            ),
            "decision_reasons": dict(sorted(reasons.items())),
        }
    prediction_body = json_bytes(outputs)
    evidence = {
        "sample_manifest_sha256": packet.name,
        "plan_sha256": file_sha256(root / "configs/us-company-review-plan.json"),
        "directory_sha256": directory.snapshot_sha256,
        "registry_sha256": registry.sha256,
        "ner_revision": REVISION,
        "device": device,
        "prediction_sha256": sha(prediction_body),
        "annotation_sha256": file_sha256(annotations) if annotations else None,
        "runner_sha256": file_sha256(Path(__file__)),
        "source_sha256": {
            str(p.relative_to(root)): file_sha256(p)
            for p in [
                root / "src/finsight/sentiment/organizations.py",
                root / "src/finsight/sentiment/organization_resolution.py",
                root / "src/finsight/data/us_listed_companies.py",
                root / "src/finsight/sentiment/evaluation/company_linking.py",
            ]
        },
        "uv_lock_sha256": file_sha256(root / "uv.lock"),
    }
    report = {
        "experiment": plan["experiment"],
        "headlines": len(rows),
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "reference_status": "human_reviewed" if reviewed else "pending_human_review",
        "counts_are_not_accuracy": True,
        "methods": counts,
        "provenance": evidence,
        "training": False,
        "frozen_records_read": False,
    }
    if reviewed is not None:
        report["metrics"] = {
            method: linking_metrics(rows, reviewed, predictions)
            for method, predictions in outputs.items()
        }
    run_dir = packet / "runs" / sha(json_bytes(evidence))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions.json").write_bytes(prediction_body)
    (run_dir / "report.json").write_bytes(json_bytes(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=project_root() / "data")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                project_root(),
                args.data_dir,
                args.packet,
                annotations=args.annotations,
                device=args.device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
