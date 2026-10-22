"""Frozen-code baseline versus reviewed aliases on a disjoint development packet."""

import argparse
import json
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from finsight.data.target_preparation import json_bytes
from finsight.data.us_company_review import validate_review
from finsight.data.us_listed_companies import load_directory, name_key
from finsight.sentiment.artifacts import file_sha256, project_root, sha
from finsight.sentiment.companies import load_registry
from finsight.sentiment.evaluation.company_linking import linking_metrics
from finsight.sentiment.evaluation.organization_metrics import exact_span_metrics
from finsight.sentiment.issuer_aliases import load_aliases
from finsight.sentiment.organization_resolution import resolve_organizations
from finsight.sentiment.organizations import REVISION, OrganizationDetector

PLAN = Path("configs/issuer-alias-review-plan.json")

# Run the archived production implementation in a fresh process, avoiding module
# cache mixing. It receives identical current NER spans, not new model inference.
BASELINE_SCRIPT = r"""
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from finsight.sentiment.companies import load_registry
from finsight.data.us_listed_companies import load_directory
from finsight.sentiment.organization_resolution import resolve_organizations
from finsight.sentiment.organizations import OrganizationMention
payload=json.load(sys.stdin)
registry=load_registry(Path(sys.argv[2])/'configs/companies.json')
directory=load_directory(Path(sys.argv[3]))
output={}
for row in payload:
    mentions=tuple(OrganizationMention(**m) for m in row['mentions'])
    resolved=resolve_organizations(row['headline'],mentions,registry,company_context=True,us_directory=directory)
    predictions=[]
    for r in resolved:
        cik=r.sec_cik
        if r.company_id and cik is None:
            company=next(c for c in registry.companies if c.company_id==r.company_id)
            candidates=set().union(*(
                directory.tickers.get(t,set()) for t in company.tickers
            ))
            if len(candidates)==1:cik=next(iter(candidates))
        predictions.append({'text':r.mention.text,'start':r.mention.start,'end':r.mention.end,'cik':cik,'company_type':r.company_type,'reason':r.reason})
    output[row['record_id']]=predictions
print(json.dumps(output))
"""


def verify_release(path: Path) -> None:
    """Reject changed baseline code rather than silently rerunning new logic."""
    manifest = path / "source-manifest.json"
    if file_sha256(manifest) != path.name:
        raise ValueError("baseline manifest checksum differs")
    for name, expected in json.loads(manifest.read_bytes()).items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or file_sha256(path / relative) != expected
        ):
            raise ValueError("baseline source checksum differs")


def read_packet(root: Path, packet: Path) -> tuple[dict, list[dict]]:
    plan = json.loads((root / PLAN).read_bytes())
    manifest = json.loads((packet / "manifest.json").read_bytes())
    if (
        file_sha256(packet / "manifest.json") != packet.name
        or file_sha256(packet / "sample.jsonl") != manifest["sample_sha256"]
        or file_sha256(root / PLAN) != manifest["plan_sha256"]
        or file_sha256(root / "src/finsight/data/issuer_alias_review.py")
        != manifest["sampling_source_sha256"]
        or file_sha256(root / "src/finsight/data/us_company_review.py")
        != manifest["base_sampler_sha256"]
        or file_sha256(root / plan["aliases_path"]) != plan["aliases_sha256"]
    ):
        raise ValueError("packet, sampler, aliases or plan differs")
    previous = root / plan["previous_packet"]
    if file_sha256(previous / "sample.jsonl") != manifest["previous_sample_sha256"]:
        raise ValueError("previous sample differs")
    old = [
        json.loads(line)
        for line in (previous / "sample.jsonl").read_bytes().splitlines()
    ]
    rows = [
        json.loads(line) for line in (packet / "sample.jsonl").read_bytes().splitlines()
    ]
    if (
        len(rows) != plan["sample_size"]
        or {r["record_id"] for r in rows} & {r["record_id"] for r in old}
        or {name_key(r["headline"]) for r in rows}
        & {name_key(r["headline"]) for r in old}
    ):
        raise ValueError("fresh membership differs or overlaps old sample")
    return plan, rows


def run(
    root: Path,
    data: Path,
    packet: Path,
    *,
    annotations: Path | None = None,
    device: str = "cpu",
) -> dict:
    plan, rows = read_packet(root, packet)
    reference = None
    if annotations:
        reference = [json.loads(line) for line in annotations.read_bytes().splitlines()]
        validate_review(rows, reference)
    baseline = data / "interim/organization-release" / plan["baseline_release"]
    verify_release(baseline)
    directory_path = (
        data / "interim/us-listed-directory" / plan["directory_sha256"]
    ).resolve()
    directory = load_directory(directory_path)
    catalog = load_aliases(root / plan["aliases_path"], directory)
    registry = load_registry(root / "configs/companies.json")
    detector = OrganizationDetector.load(
        data / "interim/organization-ner-model" / REVISION,
        root=root,
        device=device,
        aggregation_strategy="first",
    )
    proposed = detector.detect_many([row["headline"] for row in rows])
    payload = [
        {**row, "mentions": [asdict(m) for m in mentions]}
        for row, mentions in zip(rows, proposed, strict=True)
    ]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            BASELINE_SCRIPT,
            str((baseline / "source/src").resolve()),
            str((baseline / "source").resolve()),
            str(directory_path),
        ],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
    )
    predictions = {"archived_v3": json.loads(result.stdout)}
    for method in plan["methods"][1:]:
        output = {}
        for row, mentions in zip(rows, proposed, strict=True):
            resolved = resolve_organizations(
                row["headline"],
                mentions,
                registry,
                company_context=True,
                us_directory=directory,
                aliases=catalog if method != "suffix_only" else None,
                allow_parent_rollup=method == "reviewed_aliases_parent_rollup",
            )
            values = []
            for r in resolved:
                cik = r.sec_cik
                if r.company_id and cik is None:
                    company = next(
                        c for c in registry.companies if c.company_id == r.company_id
                    )
                    candidates = set().union(
                        *(directory.tickers.get(t, set()) for t in company.tickers)
                    )
                    if len(candidates) == 1:
                        cik = next(iter(candidates))
                values.append(
                    {
                        "text": r.mention.text,
                        "start": r.mention.start,
                        "end": r.mention.end,
                        "cik": cik,
                        "company_type": r.company_type,
                        "reason": r.reason,
                    }
                )
            output[row["record_id"]] = values
        predictions[method] = output
    counts = {
        method: {
            "proposed_spans": sum(len(v) for v in output.values()),
            "company_classified_spans": sum(
                p["company_type"] == "company"
                for values in output.values()
                for p in values
            ),
            "linked_spans": sum(
                p["cik"] is not None for values in output.values() for p in values
            ),
            "unlinked_company_spans": sum(
                p["company_type"] == "company" and p["cik"] is None
                for values in output.values()
                for p in values
            ),
            "reasons": dict(
                Counter(p["reason"] for values in output.values() for p in values)
            ),
        }
        for method, output in predictions.items()
    }
    provenance = {
        "sample_manifest_sha256": packet.name,
        "plan_sha256": file_sha256(root / PLAN),
        "aliases_sha256": catalog.sha256,
        "baseline_release": baseline.name,
        "directory_sha256": directory.snapshot_sha256,
        "registry_sha256": registry.sha256,
        "runner_sha256": file_sha256(Path(__file__)),
        "source_sha256": {
            name: file_sha256(root / name)
            for name in [
                "src/finsight/sentiment/issuer_aliases.py",
                "src/finsight/data/us_listed_companies.py",
                "src/finsight/sentiment/organization_resolution.py",
                "src/finsight/sentiment/organizations.py",
            ]
        },
        "ner_revision": REVISION,
        "device": device,
        "uv_lock_sha256": file_sha256(root / "uv.lock"),
        "prediction_sha256": sha(json_bytes(predictions)),
        "reference_sha256": file_sha256(annotations) if annotations else None,
    }
    report = {
        "experiment": plan["experiment"],
        "headlines": len(rows),
        "reference_status": "user_reviewed" if reference else "pending_new_user_review",
        "methods": counts,
        "counts_are_not_accuracy": True,
        "provenance": provenance,
        "training": False,
        "frozen_records_read": False,
        "overlap_with_previous_ids_or_normalized_titles": 0,
    }
    if reference:
        # Explicit parent roll-up changes identity policy; its counts are a
        # diagnostic, not accuracy against a no-parent-roll-up reference.
        report["metrics"] = {
            method: linking_metrics(rows, reference, output)
            for method, output in predictions.items()
            if method != "reviewed_aliases_parent_rollup"
        }
        gold = {
            row["record_id"]: [
                (e["start"], e["end"])
                for e in row["entities"]
                if e["type"] == "company"
            ]
            for row in reference
        }
        report["classified_company_metrics"] = {
            method: exact_span_metrics(
                gold,
                {
                    identifier: [
                        (p["start"], p["end"])
                        for p in values
                        if p["company_type"] == "company"
                    ]
                    for identifier, values in output.items()
                },
            )
            for method, output in predictions.items()
            if method != "reviewed_aliases_parent_rollup"
        }
    target = packet / "runs" / sha(json_bytes(provenance))
    target.mkdir(parents=True, exist_ok=True)
    (target / "predictions.json").write_bytes(json_bytes(predictions))
    (target / "report.json").write_bytes(json_bytes(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=project_root() / "data")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
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
