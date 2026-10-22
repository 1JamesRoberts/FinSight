"""Small mention-annotated development packet; drafts never become human gold."""

import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

from finsight.data.organization_annotations import (
    select_headlines,
    validate_annotations,
)
from finsight.data.target_preparation import json_bytes
from finsight.sentiment.artifacts import file_sha256, project_root, sha
from finsight.sentiment.companies import load_registry
from finsight.sentiment.evaluation.organization_metrics import exact_span_metrics
from finsight.sentiment.evaluation.target_data import development_inputs
from finsight.sentiment.mentions import detect_companies
from finsight.sentiment.organizations import REVISION, OrganizationDetector

PLAN = Path("configs/organization-annotation-plan.json")


def jsonl(rows: list[dict]) -> bytes:
    return b"".join((json.dumps(r, sort_keys=True) + "\n").encode() for r in rows)


def prepare(root: Path, data: Path) -> Path:
    plan = json.loads((root / PLAN).read_bytes())
    _, validation, inputs = development_inputs(root, data)
    rows = select_headlines(validation, size=plan["sample_size"], seed=plan["seed"])
    body = jsonl(rows)
    manifest = {
        "experiment": plan["experiment"],
        "plan_sha256": file_sha256(root / PLAN),
        "inputs": inputs,
        "sample_sha256": sha(body),
        "headlines": len(rows),
        "selection_source_sha256": file_sha256(
            root / "src/finsight/data/organization_annotations.py"
        ),
        "selection_uses_predictions": False,
        "frozen_records_read": False,
        "previous_full_validation_inference": True,
    }
    manifest_body = json_bytes(manifest)
    directory = data / "interim/organization-annotations" / sha(manifest_body)
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in (("manifest.json", manifest_body), ("sample.jsonl", body)):
        path = directory / name
        if path.exists() and path.read_bytes() != content:
            raise ValueError("immutable sample differs")
        path.write_bytes(content)
    return directory


def sample_rows(packet: Path, root: Path) -> list[dict]:
    manifest = json.loads((packet / "manifest.json").read_bytes())
    if (
        file_sha256(packet / "manifest.json") != packet.name
        or file_sha256(packet / "sample.jsonl") != manifest["sample_sha256"]
        or file_sha256(root / PLAN) != manifest["plan_sha256"]
    ):
        raise ValueError("sample or plan checksum differs")
    return [
        json.loads(line) for line in (packet / "sample.jsonl").read_bytes().splitlines()
    ]


def review_markdown(rows: list[dict]) -> str:
    """Source-only review sheet: no detector outputs or native sentiment labels."""
    lines = [
        "# Company and organization mention review",
        "",
        "All annotations below are assistant drafts, awaiting human review.",
        "Mark every named company and other organization, including repeated mentions.",
        "Offsets are zero-based start-inclusive/end-exclusive. Empty lists mean the",
        "assistant found no organization; check these too. Do not infer missing names.",
        "",
        "Edit annotations.provisional.jsonl and save as annotations.reviewed.jsonl.",
        'For each checked row set review.status="approved".',
        "Set review.reviewer to your name after checking that row.",
        "Resolve uncertain flags and fix types/spans before approving each row.",
        "Approval is a reviewer assertion; the tool cannot verify reviewer identity.",
        "",
    ]
    for i, row in enumerate(rows, 1):
        lines += [f"## {i}. {row['record_id']}", "", row["headline"], ""]
        if not row["entities"]:
            lines += ["Draft: no organization mentions.", ""]
        for e in row["entities"]:
            flag = " — uncertain, needs adjudication" if e["uncertain"] else ""
            lines.append(
                f"- `{e['text']}` [{e['start']}, {e['end']}) — {e['type']}{flag}"
            )
        if row["notes"]:
            lines += ["", f"Draft note: {row['notes']}"]
        lines += ["", f"Review status: {row['review']['status']}", ""]
    return "\n".join(lines)


def evaluate(
    root: Path,
    data: Path,
    packet: Path,
    annotation_path: Path,
    *,
    provisional: bool = False,
    device: str = "cpu",
) -> dict:
    sources = sample_rows(packet, root)
    rows = [json.loads(line) for line in annotation_path.read_bytes().splitlines()]
    validate_annotations(sources, rows, provisional=provisional)
    annotations = {r["record_id"]: r for r in rows}
    # Freeze the exact reference bytes before constructing/scoring either pipeline.
    reference_sha = file_sha256(annotation_path)
    reference_path = packet / f"reference-{reference_sha}.jsonl"
    reference_path.write_bytes(annotation_path.read_bytes())
    status = "assistant_provisional" if provisional else "human_reviewed"
    locked_reference = {
        "sample_manifest_sha256": packet.name,
        "annotation_sha256": reference_sha,
        "reference_status": status,
        "human_review_required_for_reference_claims": True,
        "runner_sha256": file_sha256(Path(__file__)),
        "policy_sha256": file_sha256(
            root
            / "docs/phase-04-target-sentiment/data/organization-annotation-policy.md"
        ),
        "uv_lock_sha256": file_sha256(root / "uv.lock"),
    }
    run = packet / "evaluations" / sha(json_bytes(locked_reference))
    run.mkdir(parents=True, exist_ok=True)
    (run / "reference-lock.json").write_bytes(json_bytes(locked_reference))
    registry = load_registry(root / "configs/companies.json")
    detector = OrganizationDetector.load(
        data / "interim/organization-ner-model" / REVISION,
        root=root,
        device=device,
        aggregation_strategy="simple",
    )
    import torch
    from transformers import TokenClassificationPipeline

    torch.manual_seed(42)
    torch.set_num_threads(4)
    texts = [s["headline"] for s in sources]
    proposals = {}
    timings = {}
    for strategy in ("simple", "first"):
        candidate = (
            detector
            if strategy == "simple"
            else OrganizationDetector(
                detector.tokenizer,
                TokenClassificationPipeline(
                    model=detector.pipeline.model,
                    tokenizer=detector.tokenizer,
                    aggregation_strategy="first",
                    device=device,
                ),
                max_length=512,
            )
        )
        start = perf_counter()
        proposals[f"ner_{strategy}"] = candidate.detect_many(texts)
        if device == "cuda":
            torch.cuda.synchronize()
        timings[strategy] = perf_counter() - start
    baseline = [detect_companies(text, registry) for text in texts]
    proposals["registry_resolved"] = [b.resolved for b in baseline]
    proposals["registry_all_candidates"] = [
        (*b.resolved, *b.ambiguous) for b in baseline
    ]
    metrics, private = {}, []
    for method, detections in proposals.items():
        prediction = {
            s["record_id"]: [(m.start, m.end) for m in mentions]
            for s, mentions in zip(sources, detections, strict=True)
        }
        metrics[method] = {}
        for scope in ("all_organizations", "all_proposals_as_company_candidates"):
            reference = {
                s["record_id"]: [
                    (e["start"], e["end"])
                    for e in annotations[s["record_id"]]["entities"]
                    if scope == "all_organizations" or e["type"] == "company"
                ]
                for s in sources
            }
            metrics[method][scope] = exact_span_metrics(reference, prediction)
        for source, mentions in zip(sources, detections, strict=True):
            private.append(
                {
                    "record_id": source["record_id"],
                    "method": method,
                    "mentions": [
                        dict(text=m.text, start=m.start, end=m.end) for m in mentions
                    ],
                }
            )
    (run / "predictions.jsonl").write_bytes(jsonl(private))
    support = Counter(e["type"] for row in rows for e in row["entities"])
    result = {
        "experiment": "organization-mentions-dev-v1",
        **locked_reference,
        "headlines": len(sources),
        "reference_entity_counts": dict(support),
        "empty_reference_headlines": sum(not r["entities"] for r in rows),
        "uncertain_reference_entities": sum(
            e["uncertain"] for r in rows for e in r["entities"]
        ),
        "approved_reference_headlines": sum(
            r["review"]["status"] == "approved" for r in rows
        ),
        "metrics": metrics,
        "inference_seconds": timings,
        "device": device,
        "model_revision": REVISION,
        "ner_model_plan_sha256": file_sha256(
            root / "configs/organization-ner-plan.json"
        ),
        "plan_sha256": file_sha256(root / PLAN),
        "registry_sha256": registry.sha256,
        "predictions_sha256": file_sha256(run / "predictions.jsonl"),
        "source_sha256": {
            name: file_sha256(root / "src/finsight" / name)
            for name in (
                "data/organization_annotations.py",
                "sentiment/organizations.py",
                "sentiment/evaluation/organization_metrics.py",
                "sentiment/experiments/organization_benchmark.py",
            )
        },
        "training_performed": False,
        "frozen_records_read": False,
        "deployment_change": False,
    }
    (run / "result.json").write_bytes(json_bytes(result))
    suffix = "provisional" if provisional else "reviewed"
    output = (
        root
        / "reports/phase-04-target-sentiment/evaluation"
        / f"organization-benchmark-{suffix}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(json_bytes(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check", "evaluate"))
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--provisional", action="store_true")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    args = parser.parse_args()
    root, data = args.root, args.data_dir or args.root / "data"
    if args.action == "prepare":
        print(prepare(root, data))
        return
    if not args.packet or not args.annotations:
        parser.error("--packet and --annotations are required")
    if args.action == "check":
        sources = sample_rows(args.packet, root)
        rows = [json.loads(line) for line in args.annotations.read_bytes().splitlines()]
        validate_annotations(sources, rows, provisional=args.provisional)
        (args.packet / "review.md").write_text(review_markdown(rows), encoding="utf-8")
        print("Annotations validated; review.md written")
        return
    result = evaluate(
        root,
        data,
        args.packet,
        args.annotations,
        provisional=args.provisional,
        device=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
