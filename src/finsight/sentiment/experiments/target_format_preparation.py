"""Prepare and lock the nine-run RoBERTa target-format study without training."""

import argparse
import json
import platform
import subprocess
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

from finsight.data.target_pairs import TargetExample, summarize_examples
from finsight.data.target_preparation import json_bytes
from finsight.data.target_spans import VARIANTS, align_targets, transform_input
from finsight.data.target_tokenization import audit_pairs
from finsight.sentiment.artifacts import file_sha256, project_root, sha, verify_files
from finsight.sentiment.evaluation.target_data import development_inputs
from finsight.sentiment.experiments.target_span_audit import audit_partition
from finsight.sentiment.training.config import FineTuneConfig

PLAN = Path("configs/target-format-plan.json")
LOCK = Path("configs/target-format-lock.json")
REPORT = Path(
    "reports/phase-04-target-sentiment/evaluation/target-tokenizer-audit.json"
)
SOURCES = (
    "data/target_spans.py",
    "data/target_tokenization.py",
    "data/target_pairs.py",
    "sentiment/experiments/target_span_audit.py",
    "sentiment/experiments/target_format_preparation.py",
    "sentiment/evaluation/target_data.py",
)


def plan_reference(root: Path, path: str) -> Path:
    """Locate the moved audit without rewriting the frozen plan or its checksum."""
    original = root / path
    if (
        path == "reports/phase-04-target-sentiment/target-span-audit.json"
        and not original.exists()
    ):
        return original.parent / "evaluation" / original.name
    return original


def load_plan(root: Path) -> tuple[dict, dict]:
    """Resolve only the pinned, already-reviewed backbone and training settings."""
    plan = json.loads((root / PLAN).read_bytes())
    for path, digest in (
        (plan["span_audit_path"], plan["span_audit_sha256"]),
        (plan["backbone_plan_path"], plan["backbone_plan_sha256"]),
        (plan["training_config_path"], plan["training_config_sha256"]),
    ):
        if file_sha256(plan_reference(root, path)) != digest:
            raise ValueError("target-format plan reference differs")
    entry = json.loads((root / plan["backbone_plan_path"]).read_bytes())["models"][
        "roberta_base"
    ]
    config = FineTuneConfig.load(root / plan["training_config_path"])
    if (
        plan["schema_version"] != 1
        or plan["model_key"] != "roberta_base"
        or plan["asset"] != entry["asset"]
        or list(VARIANTS) != plan["variants"]
        or plan["seeds"] != [42, 43, 44]
        or plan["trials"]
        != [{"variant": v, "seed": s} for s in (42, 43, 44) for v in VARIANTS]
        or config.max_length != plan["max_length"]
        or plan["max_length"] != 128
        or plan["frozen_test_access"]
        or plan["train_validation_combined"]
        or plan["hyperparameter_search"]
    ):
        raise ValueError("target-format protocol differs")
    # All model weights remain untouched; only tokenizer/config files are read.
    files = {
        name: item
        for name, item in entry["asset"]["files"].items()
        if name != "model.safetensors"
    }
    return plan, files


def load_tokenizers(directory: Path, files: dict) -> tuple[Any, Any]:
    verify_files(directory, files)
    from transformers import AutoTokenizer

    tokenizers = tuple(
        AutoTokenizer.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False, use_fast=fast
        )
        for fast in (False, True)
    )
    vocab = json.loads((directory / "config.json").read_bytes())["vocab_size"]
    for tokenizer in tokenizers:
        if (
            len(tokenizer) != vocab
            or tokenizer.padding_side != "right"
            or tokenizer.truncation_side != "right"
            or tokenizer.num_special_tokens_to_add(pair=True) != 4
        ):
            raise ValueError("tokenizer vocabulary or pair policy differs")
    return tokenizers


def audit_partition_views(
    rows: list[dict], partition: str, slow: Any, fast: Any, max_length: int
) -> tuple[dict, list[dict], list[dict]]:
    """Audit all character-eligible annotations; exclude complete headlines jointly."""
    _, identifiers, _ = audit_partition(rows, partition)
    eligible = set(identifiers)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["example_id"] in eligible:
            grouped[row["record_id"]].append(row)
    prepared = []
    for _, annotations in sorted(grouped.items()):
        headline = annotations[0]["headline"]
        aligned = align_targets(headline, [r["target"] for r in annotations])
        for row, alignment in zip(annotations, aligned, strict=True):
            views = {}
            for variant in VARIANTS:
                target, text = transform_input(headline, alignment, variant)
                views[variant] = {"target": target, "headline": text}
            prepared.append({"example": row, "inputs": views})
    if [r["example"]["example_id"] for r in prepared] != identifiers:
        raise ValueError("character-eligible row order differs")
    if not prepared:
        raise ValueError("no character-eligible examples")
    statuses = {}
    for variant in VARIANTS:
        pairs = [
            (r["inputs"][variant]["target"], r["inputs"][variant]["headline"])
            for r in prepared
        ]
        statuses[variant] = audit_pairs(slow, fast, pairs, max_length=max_length)
    excluded_headlines = {
        row["example"]["record_id"]
        for i, row in enumerate(prepared)
        if any(not statuses[v][i]["eligible"] for v in VARIANTS)
    }
    retained = [
        r for r in prepared if r["example"]["record_id"] not in excluded_headlines
    ]
    if not retained:
        raise ValueError("no shared token-eligible examples")
    retained_indices = [
        i
        for i, r in enumerate(prepared)
        if r["example"]["record_id"] not in excluded_headlines
    ]
    variants = {}
    for variant in VARIANTS:
        checks = statuses[variant]
        reasons = Counter(reason for row in checks for reason in row["reasons"])
        variants[variant] = {
            "audited_annotations": len(checks),
            "full_pair_tokens_max_before_exclusions": max(
                r["full_pair_tokens"] for r in checks
            ),
            "full_pair_tokens_max_retained": max(
                checks[i]["full_pair_tokens"] for i in retained_indices
            ),
            "target_tokens_max": max(r["target_tokens"] for r in checks),
            "headline_tokens_max": max(r["headline_tokens"] for r in checks),
            "individually_token_eligible_annotations": sum(
                r["eligible"] for r in checks
            ),
            "annotation_failure_reasons": dict(reasons),
            "slow_fast_equivalent": True,
            "all_retained_text_and_markers_preserved": True,
        }
    exclusions = [
        {**row, "token_audit": {v: statuses[v][i] for v in VARIANTS}}
        for i, row in enumerate(prepared)
        if row["example"]["record_id"] in excluded_headlines
    ]
    summary = {
        "character_eligible_headlines": len(grouped),
        "character_eligible_annotations": len(prepared),
        "retained_headlines": len({r["example"]["record_id"] for r in retained}),
        "retained_annotations": len(retained),
        "additional_excluded_headlines": len(excluded_headlines),
        "additional_excluded_annotations": len(prepared) - len(retained),
        "membership_sha256": sha(
            json_bytes([r["example"]["example_id"] for r in retained])
        ),
        "variants": variants,
    }
    return summary, retained, exclusions


def prepare(root: Path, data_dir: Path, tokenizer_dir: Path) -> dict:
    if (root / LOCK).exists() or (root / REPORT).exists():
        raise FileExistsError(
            "target-format preparation already published; use --check"
        )
    plan, files = load_plan(root)
    span = json.loads(plan_reference(root, plan["span_audit_path"]).read_bytes())
    for name, digest in span["source_sha256"].items():
        if file_sha256(root / "src/finsight" / name) != digest:
            raise ValueError("character alignment implementation differs")
    training, validation, inputs = development_inputs(root, data_dir)
    if inputs != span["inputs"]:
        raise ValueError("character audit input files differ")
    slow, fast = load_tokenizers(tokenizer_dir, files)
    summary = {
        "schema_version": 1,
        "experiment": plan["experiment"],
        "status": "tokenizer_verified_training_locked_not_started",
        "plan_sha256": file_sha256(root / PLAN),
        "source_sha256": {
            name: file_sha256(root / "src/finsight" / name) for name in SOURCES
        },
        "inputs": inputs,
        "tokenizer": {
            "model_id": plan["asset"]["model_id"],
            "revision": plan["asset"]["revision"],
            "files": files,
            "classes": [type(slow).__name__, type(fast).__name__],
            "vocabulary_size": len(slow),
            "vocabulary_tokens_added": 0,
            "paired_special_tokens": 4,
        },
        "environment": {
            "python": platform.python_version(),
            "transformers": version("transformers"),
            "tokenizers": version("tokenizers"),
        },
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "frozen_records_read": False,
        "model_weights_loaded": False,
        "training_performed": False,
        "partitions": {},
    }
    bodies = {}
    all_retained = []
    exclusions = []
    for partition, rows in (("train", training), ("validation", validation)):
        character, _, _ = audit_partition(rows, partition)
        if character != span["partitions"][partition]:
            raise ValueError("character audit no longer reproduces")
        counts, retained, rejected = audit_partition_views(
            rows, partition, slow, fast, plan["max_length"]
        )
        summary["partitions"][partition] = counts
        all_retained.extend(
            TargetExample(
                **{
                    **r["example"],
                    "review_reasons": tuple(r["example"]["review_reasons"]),
                }
            )
            for r in retained
        )
        exclusions.extend(rejected)
        bodies[f"{partition}.jsonl"] = b"".join(
            (json.dumps(r, sort_keys=True) + "\n").encode() for r in retained
        )
    summary["retained_cohorts"] = summarize_examples(tuple(all_retained))
    bodies["tokenizer-exclusions.jsonl"] = b"".join(
        (json.dumps(r, sort_keys=True) + "\n").encode() for r in exclusions
    )
    manifest = {
        "experiment": plan["experiment"],
        "plan_sha256": summary["plan_sha256"],
        "inputs": inputs,
        "source_sha256": summary["source_sha256"],
        "tokenizer": summary["tokenizer"],
        "files": {
            name: {"sha256": sha(body), "size_bytes": len(body)}
            for name, body in bodies.items()
        },
    }
    manifest_body = json_bytes(manifest)
    identity = sha(manifest_body)
    directory = data_dir / "interim/target-format" / identity
    directory.mkdir(parents=True, exist_ok=True)
    for name, body in {**bodies, "manifest.json": manifest_body}.items():
        path = directory / name
        if path.exists() and path.read_bytes() != body:
            raise ValueError("existing private target-format files differ")
        path.write_bytes(body)
    summary["data_manifest_sha256"] = identity
    (root / REPORT).parent.mkdir(parents=True, exist_ok=True)
    (root / REPORT).write_bytes(json_bytes(summary))
    lock = {
        "experiment": plan["experiment"],
        "plan_sha256": summary["plan_sha256"],
        "report_sha256": file_sha256(root / REPORT),
        "data_manifest_sha256": identity,
        "uv_lock_sha256": file_sha256(root / "uv.lock"),
        "frozen_test_access": False,
        "training_started": False,
    }
    (root / LOCK).write_bytes(json_bytes(lock))
    return summary


def check_preparation(root: Path, data_dir: Path, tokenizer_dir: Path) -> dict:
    """Check the locked sources, tokenizer, original inputs and prepared bytes."""
    _, files = load_plan(root)
    lock = json.loads((root / LOCK).read_bytes())
    if (
        file_sha256(root / PLAN) != lock["plan_sha256"]
        or file_sha256(root / REPORT) != lock["report_sha256"]
        or file_sha256(root / "uv.lock") != lock["uv_lock_sha256"]
        or lock["frozen_test_access"]
        or lock["training_started"]
    ):
        raise ValueError("preparation lock differs")
    report = json.loads((root / REPORT).read_bytes())
    for name, digest in report["source_sha256"].items():
        if file_sha256(root / "src/finsight" / name) != digest:
            raise ValueError("locked preparation implementation differs")
    _, _, inputs = development_inputs(root, data_dir)
    directory = data_dir / "interim/target-format" / lock["data_manifest_sha256"]
    if file_sha256(directory / "manifest.json") != lock["data_manifest_sha256"]:
        raise ValueError("target-format manifest differs")
    manifest = json.loads((directory / "manifest.json").read_bytes())
    if (
        inputs != report["inputs"]
        or inputs != manifest["inputs"]
        or manifest["plan_sha256"] != lock["plan_sha256"]
        or report["data_manifest_sha256"] != lock["data_manifest_sha256"]
        or manifest["source_sha256"] != report["source_sha256"]
        or manifest["tokenizer"] != report["tokenizer"]
        or files != report["tokenizer"]["files"]
    ):
        raise ValueError("prepared data lineage differs")
    verify_files(directory, manifest["files"])
    verify_files(tokenizer_dir, files)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    action = check_preparation if args.check else prepare
    result = action(root, args.data_dir or root / "data", args.tokenizer_dir)
    print(json.dumps(result["partitions"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
