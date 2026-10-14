"""Development-only target coverage; literal matches are diagnostic candidates."""

import argparse
import hashlib
import json
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from finsight.data.benchmark import load_development_records
from finsight.sentiment.companies import (
    Company,
    CompanyRegistry,
    load_registry,
    resolve_company,
)

LABELS = ("negative", "neutral", "positive")
VIEWS = {"strict": False, "with_parent_rollup": True}
PLAN = Path("configs/company-coverage.json")
REGISTRY = Path("configs/companies.json")
RECEIPT = Path("configs/benchmark-freeze.json")
OUTPUT = Path("reports/company-coverage")


def _normalized(text: str) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", text).casefold().replace(",", " ").split()
    )


def literal_match(text: str, target: str) -> bool:
    """Screen whole-token literals without semantic recognition or span alignment."""
    needle = _normalized(target).rstrip(".")
    if not needle:
        raise ValueError("empty target annotation")
    return (
        re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", _normalized(text))
        is not None
    )


def _company_literal(text: str, company: Company, rollup: bool) -> bool:
    names = [company.name, *company.tickers]
    names += [a.name for a in company.aliases if a.kind == "name" or rollup]
    return any(literal_match(text, name) for name in names)


def _counts(
    rows: list[dict], company: Company, registry: CompanyRegistry, rollup: bool
) -> dict:
    matched = []
    literal_candidates = unannotated = conflicts = repeated = 0
    matches = []
    for row in rows:
        native = row["native_record"]
        annotations = [
            a
            for a in native["sentiments"]
            if (
                resolve_company(
                    a["target"],
                    registry,
                    company_context=True,
                    allow_parent_rollup=rollup,
                ).company
                == company
            )
        ]
        literal = _company_literal(native["title"], company, rollup)
        literal_candidates += literal
        unannotated += literal and not annotations
        if annotations:
            matched.append(row)
            repeated += len(annotations) > 1
            conflicts += len({a["label"] for a in annotations}) > 1
            matches.extend((a, native["title"]) for a in annotations)
    literal_targets = sum(literal_match(title, a["target"]) for a, title in matches)
    alias_only = sum(
        not literal_match(title, a["target"])
        and _company_literal(title, company, rollup)
        for a, title in matches
    )
    return {
        "annotated_headlines": len(matched),
        "target_annotations": len(matches),
        "duplicate_groups": len({r["group_id"] for r in matched}),
        "class_counts": {
            label: sum(a["label"] == label for a, _ in matches) for label in LABELS
        },
        "single_target_headlines": sum(
            len(r["native_record"]["sentiments"]) == 1 for r in matched
        ),
        "multi_target_headlines": sum(
            len(r["native_record"]["sentiments"]) > 1 for r in matched
        ),
        "mixed_label_headlines": sum(
            len({a["label"] for a in r["native_record"]["sentiments"]}) > 1
            for r in matched
        ),
        "literal_target_annotations": literal_targets,
        "registered_alias_only_annotations": alias_only,
        "no_registered_alias_annotations": len(matches) - literal_targets - alias_only,
        "same_company_multi_annotation_headlines": repeated,
        "same_company_conflict_headlines": conflicts,
        "literal_candidate_headlines": literal_candidates,
        "unannotated_literal_candidate_headlines": unannotated,
    }


def summarize_coverage(
    records: list[dict], registry: CompanyRegistry
) -> tuple[dict, list]:
    """Count source annotations without relabeling, filtering or expanding training."""
    if any(r["partition"] not in ("train", "validation") for r in records):
        raise ValueError("coverage accepts development records only")
    rows = [r for r in records if r["corpus"] == "sentfin"]
    if len({r["record_id"] for r in rows}) != len(rows):
        raise ValueError("record IDs must be unique")
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups[row["group_id"]].add(row["partition"])
        annotations = row["native_record"]["sentiments"]
        if not annotations or any(
            not isinstance(a["target"], str)
            or not a["target"].strip()
            or a["label"] not in LABELS
            for a in annotations
        ):
            raise ValueError("invalid target annotation")
    if any(len(parts) > 1 for parts in groups.values()):
        raise ValueError("duplicate group crossed development partitions")
    summary: dict[str, Any] = {"partitions": {}}
    review = []
    for partition in ("train", "validation"):
        selected = [r for r in rows if r["partition"] == partition]
        annotations = [a for r in selected for a in r["native_record"]["sentiments"]]
        resolutions = Counter()
        nonliteral = unsupported_cues = 0
        for row in selected:
            native = row["native_record"]
            for annotation in native["sentiments"]:
                target = annotation["target"]
                resolution = resolve_company(target, registry, company_context=True)
                resolutions[resolution.status] += 1
                reasons = []
                if resolution.status == "ambiguous":
                    reasons.append(resolution.reason)
                if not literal_match(native["title"], target):
                    nonliteral += 1
                    reasons.append("target_not_literal")
                if resolution.status == "unsupported" and any(
                    _company_literal(target, c, True) for c in registry.companies
                ):
                    unsupported_cues += 1
                    reasons.append("unsupported_with_registry_name_cue")
                for reason in reasons:
                    review.append(
                        {
                            "record_id": row["record_id"],
                            "partition": partition,
                            "target": target,
                            "label": annotation["label"],
                            "reason": reason,
                            "candidate_company_ids": [
                                c.company_id for c in resolution.candidates
                            ],
                        }
                    )
        summary["partitions"][partition] = {
            "overall": {
                "headlines": len(selected),
                "target_annotations": len(annotations),
                "duplicate_groups": len({r["group_id"] for r in selected}),
                "unique_target_strings": len({a["target"] for a in annotations}),
                "class_counts": dict(Counter(a["label"] for a in annotations)),
                "single_target_headlines": sum(
                    len(r["native_record"]["sentiments"]) == 1 for r in selected
                ),
                "multi_target_headlines": sum(
                    len(r["native_record"]["sentiments"]) > 1 for r in selected
                ),
                "mixed_label_headlines": sum(
                    len({a["label"] for a in r["native_record"]["sentiments"]}) > 1
                    for r in selected
                ),
                "nonliteral_target_annotations": nonliteral,
            },
            "target_resolution": {
                s: resolutions[s] for s in ("resolved", "ambiguous", "unsupported")
            },
            "unsupported_with_registry_name_cue_annotations": unsupported_cues,
            "views": {
                view: {
                    c.primary_ticker: _counts(selected, c, registry, rollup)
                    for c in registry.companies
                }
                for view, rollup in VIEWS.items()
            },
        }
    return summary, review


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_report(result: dict) -> str:
    lines = [
        "# Five-company SEntFiN development coverage",
        "",
        "Provided annotations only; no test analysis, label changes or training.",
        "Literal candidates are diagnostic name matches, not verified entity labels.",
        "",
        "## Complete development cohort",
        "",
        "| Split | Headlines | Target annotations | Single-target | Multi-target | "
        "Mixed-label |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for part, summary in result["summary"]["partitions"].items():
        c = summary["overall"]
        lines.append(
            f"| {part} | {c['headlines']} | {c['target_annotations']} | "
            f"{c['single_target_headlines']} | {c['multi_target_headlines']} | "
            f"{c['mixed_label_headlines']} |"
        )
    for view in VIEWS:
        lines += [
            "",
            f"## {view}",
            "",
            "Registry matches treat the annotated targets as financial entities.",
            "The separate parent-rollup view does not rewrite source targets.",
            "",
            "| Split | Ticker | Labeled headlines | Targets | Negative | Neutral | "
            "Positive | Single-target | Multi-target | Mixed-label |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for part, summary in result["summary"]["partitions"].items():
            for ticker, c in summary["views"][view].items():
                labels = c["class_counts"]
                lines.append(
                    f"| {part} | {ticker} | {c['annotated_headlines']} | "
                    f"{c['target_annotations']} | "
                    + " | ".join(str(labels[label]) for label in LABELS)
                    + f" | {c['single_target_headlines']} | "
                    f"{c['multi_target_headlines']} | {c['mixed_label_headlines']} |"
                )
        lines += [
            "",
            "| Split | Ticker | Literal target | Alias only | No registered alias | "
            "Candidate headlines | Unannotated candidates | Conflicts |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for part, summary in result["summary"]["partitions"].items():
            for ticker, c in summary["views"][view].items():
                lines.append(
                    f"| {part} | {ticker} | "
                    + " | ".join(
                        str(c[k])
                        for k in (
                            "literal_target_annotations",
                            "registered_alias_only_annotations",
                            "no_registered_alias_annotations",
                            "literal_candidate_headlines",
                            "unannotated_literal_candidate_headlines",
                            "same_company_conflict_headlines",
                        )
                    )
                    + " |"
                )
    lines += [
        "",
        "## Resolution and annotation diagnostics",
        "",
        "| Split | Resolved annotations | Parent grouping pending | Outside registry | "
        "Unknown targets with name cue | Nonliteral targets (whole corpus) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for part, c in result["summary"]["partitions"].items():
        r = c["target_resolution"]
        lines.append(
            f"| {part} | {r['resolved']} | {r['ambiguous']} | {r['unsupported']} | "
            f"{c['unsupported_with_registry_name_cue_annotations']} | "
            f"{c['overall']['nonliteral_target_annotations']} |"
        )
    lines += [
        "",
        "Count headlines once per company, and annotations as provided.",
        "Company rows overlap; their sum is not a unique headline count.",
        "Mixed-label means the original headline has different target sentiments.",
        "Nonliteral targets can be abbreviations or implicit references, not errors.",
        "Outside-registry targets generally belong to other entities, not errors.",
        "Literal candidates do not create sentiment labels. No rows are dropped.",
        "",
        f"Registry `{result['registry_version']}`.",
        f"Registry SHA-256 `{result['registry_sha256']}`.",
        f"Plan SHA-256 `{result['plan_sha256']}`; code `{result['code_revision']}`.",
        f"Private review flags SHA-256 `{result['review_sha256']}`.",
        "See [method and interpretation](../docs/company-coverage.md). "
        "Full provenance is in the sibling JSON.",
        "",
    ]
    return "\n".join(lines)


def run(data_dir: Path) -> dict:
    """Verify lineage and publish aggregates; keep annotation-level flags private."""
    if OUTPUT.with_suffix(".json").exists() or OUTPUT.with_suffix(".md").exists():
        raise FileExistsError(
            "coverage outputs exist; inspect or archive before rerunning"
        )
    plan = json.loads(PLAN.read_bytes())
    registry = load_registry(REGISTRY)
    if (
        plan["schema_version"] != 1
        or plan["partitions"] != ["train", "validation"]
        or plan["registry_sha256"] != registry.sha256
        or plan["freeze_receipt_sha256"] != _sha(RECEIPT)
        or plan["views"] != VIEWS
        or plan["company_context"] is not True
        or plan["labels"] != list(LABELS)
    ):
        raise ValueError("coverage plan identities or scope differ")
    records = load_development_records(data_dir, RECEIPT)
    summary, review = summarize_coverage(records, registry)
    review_body = (
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in review)
    ).encode()
    receipt = json.loads(RECEIPT.read_bytes())
    result = {
        "schema_version": 1,
        "analysis": plan["analysis"],
        "plan": plan,
        "plan_sha256": _sha(PLAN),
        "registry_version": registry.version,
        "registry_sha256": registry.sha256,
        "summary": summary,
        "review_sha256": hashlib.sha256(review_body).hexdigest(),
        "review_flags": len(review),
        "final_test_analyzed": False,
        "parent_split_manifest_sha256": receipt["parent_split_manifest_sha256"],
        "benchmark_manifest_sha256": receipt["benchmark_manifest_sha256"],
        "freeze_receipt_sha256": _sha(RECEIPT),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "implementation_sha256": {
            p: _sha(Path(p))
            for p in (
                "src/finsight/data/company_coverage.py",
                "src/finsight/sentiment/companies.py",
            )
        },
        "lockfile_sha256": _sha(Path("uv.lock")),
    }
    load_development_records(data_dir, RECEIPT)
    body = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    identifier = hashlib.sha256(body).hexdigest()
    private = data_dir / "interim/company-coverage" / identifier
    private.mkdir(parents=True, exist_ok=False)
    (private / "aggregate.json").write_bytes(body)
    (private / "review.jsonl").write_bytes(review_body)
    OUTPUT.with_suffix(".json").write_bytes(body)
    OUTPUT.with_suffix(".md").write_bytes(render_report(result).encode())
    print(json.dumps({"analysis_id": identifier, "summary": summary}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    run(parser.parse_args().data_dir)


if __name__ == "__main__":
    main()
