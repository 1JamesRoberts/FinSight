"""Prepare private target rows and lock input/baseline readiness without training."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from finsight.data.benchmark import load_development_records
from finsight.data.target_pairs import (
    audit_tokenization,
    build_examples,
    examples_bytes,
    summarize_examples,
)
from finsight.sentiment.artifacts import file_sha256, verify_files

PLAN = Path("configs/target-sentiment-plan.json")
ASSETS = Path("configs/transformer-assets.json")
RECEIPT = Path("configs/benchmark-freeze.json")
LOCK = Path("configs/target-sentiment-lock.json")
OUTPUT = Path("reports/target-input-preparation")
TOKENIZER_FILES = ("README.md", "config.json", "spm.model", "tokenizer_config.json")


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def load_plan() -> tuple[dict, dict]:
    plan = json.loads(PLAN.read_bytes())
    if (
        plan["schema_version"] != 1
        or plan["partitions"] != ["train", "validation"]
        or plan["labels"] != ["negative", "neutral", "positive"]
        or plan["seeds"] != [42, 43, 44]
        or plan["arms"]
        != ["saved_headline_control", "matched_headline_only", "target_pair"]
        or plan["max_length"] != 128
        or plan["truncation"] != "only_second"
        or plan["final_test_evaluated"] is not False
        or file_sha256(RECEIPT) != plan["freeze_receipt_sha256"]
        or file_sha256(ASSETS) != plan["asset_manifest_sha256"]
        or file_sha256(Path(plan["training_config"])) != plan["training_config_sha256"]
        or file_sha256(Path("reports/experiments/deberta-tuning.json"))
        != plan["baseline_study_sha256"]
    ):
        raise ValueError("target experiment plan or referenced identities differ")
    asset = json.loads(ASSETS.read_bytes())["models"][plan["model_key"]]
    receipt = json.loads(RECEIPT.read_bytes())
    if plan["parent_split_manifest_sha256"] != receipt["parent_split_manifest_sha256"]:
        raise ValueError("parent split identity differs")
    if (
        asset["model_id"] != plan["model_id"]
        or asset["revision"] != plan["model_revision"]
    ):
        raise ValueError("base checkpoint differs")
    for control in plan["saved_headline_controls"]:
        if file_sha256(Path(control["report_path"])) != control["report_sha256"]:
            raise ValueError("saved control report differs")
    return plan, asset


def prepare(data_dir: Path, tokenizer_dir: Path) -> dict:
    """Use cached reviewed tokenizer files only; no hub calls or model weights."""
    if (
        LOCK.exists()
        or OUTPUT.with_suffix(".json").exists()
        or OUTPUT.with_suffix(".md").exists()
    ):
        raise FileExistsError("preparation already published; inspect saved outputs")
    plan, asset = load_plan()
    from transformers import AutoTokenizer

    files = {name: asset["files"][name] for name in TOKENIZER_FILES}
    verify_files(tokenizer_dir, files)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=False,
    )
    records = load_development_records(data_dir, RECEIPT)
    examples = build_examples(records)
    summary = summarize_examples(examples)
    for part in plan["partitions"]:
        if (
            summary[part]["examples"] != plan["expected_examples"][part]
            or summary[part]["headlines"] != plan["expected_headlines"][part]
        ):
            raise ValueError("prepared membership/counts differ from declared view")
    token_audit = audit_tokenization(tokenizer, examples, max_length=plan["max_length"])
    controls = []
    for control in plan["saved_headline_controls"]:
        path = data_dir / control["artifact_relative_path"] / "artifact-manifest.json"
        if file_sha256(path) != control["artifact_manifest_sha256"]:
            raise ValueError("saved headline-control artifact manifest differs")
        controls.append({"seed": control["seed"], "manifest_verified": True})
    load_development_records(data_dir, RECEIPT)
    bodies = {
        f"{part}.jsonl": examples_bytes(examples, part) for part in plan["partitions"]
    }
    manifest = {
        "schema_version": 1,
        "dataset_id": "sentfin-native-targets-v1",
        "plan_sha256": file_sha256(PLAN),
        "summary": summary,
        "parent_split_manifest_sha256": plan["parent_split_manifest_sha256"],
        "freeze_receipt_sha256": plan["freeze_receipt_sha256"],
        "files": {
            name: {"sha256": hashlib.sha256(body).hexdigest(), "size_bytes": len(body)}
            for name, body in bodies.items()
        },
    }
    manifest_body = json_bytes(manifest)
    identity = hashlib.sha256(manifest_body).hexdigest()
    root = data_dir / "interim/target-pairs" / identity
    root.mkdir(parents=True, exist_ok=False)
    for name, body in bodies.items():
        (root / name).write_bytes(body)
    (root / "manifest.json").write_bytes(manifest_body)
    result = {
        "schema_version": 1,
        "experiment": plan["experiment"],
        "status": "prepared_not_trained",
        "plan_sha256": file_sha256(PLAN),
        "data_manifest_sha256": identity,
        "data_manifest": manifest,
        "tokenizer": {
            "model_id": asset["model_id"],
            "revision": asset["revision"],
            "class": type(tokenizer).__name__,
            "files": files,
            "pair_special_tokens": tokenizer.num_special_tokens_to_add(pair=True),
        },
        "tokenization": token_audit,
        "saved_controls": controls,
        "models_loaded": 0,
        "models_trained": 0,
        "final_test_evaluated": False,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "implementation_sha256": {
            name: file_sha256(Path(name))
            for name in (
                "src/finsight/data/target_pairs.py",
                "src/finsight/data/target_preparation.py",
            )
        },
        "lockfile_sha256": file_sha256(Path("uv.lock")),
    }
    body = json_bytes(result)
    (root / "preparation.json").write_bytes(body)
    OUTPUT.with_suffix(".json").write_bytes(body)
    OUTPUT.with_suffix(".md").write_bytes(render_report(result).encode())
    LOCK.write_bytes(
        json_bytes(
            {
                "schema_version": 1,
                "experiment": plan["experiment"],
                "status": "prepared_not_trained",
                "plan_sha256": file_sha256(PLAN),
                "data_manifest_sha256": identity,
                "preparation_report_sha256": hashlib.sha256(body).hexdigest(),
                "final_test_evaluated": False,
            }
        )
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def render_report(result: dict) -> str:
    lines = [
        "# Target-aware input preparation",
        "",
        "Status: **prepared, not trained**. Native target–headline pairs and matched",
        "headline-only rows share source IDs, partitions, groups and labels.",
        "",
        "| Split | Headlines | Examples | Negative | Neutral | Positive | "
        "Multi-target examples | Mixed-label examples |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for part, counts in result["data_manifest"]["summary"].items():
        labels = counts["class_counts"]
        lines.append(
            f"| {part} | {counts['headlines']} | {counts['examples']} | "
            + " | ".join(str(labels[c]) for c in ("negative", "neutral", "positive"))
            + f" | {counts['multi_target_examples']} | "
            f"{counts['mixed_label_examples']} |"
        )
    lines += [
        "",
        "| Split | Input | Max tokens | Truncated headlines | Targets with UNK | "
        "Identical inputs with conflicting labels |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for part, modes in result["tokenization"].items():
        for mode, counts in modes.items():
            lines.append(
                f"| {part} | {mode} | {counts['maximum_encoded_length']} | "
                f"{counts['headline_truncated_examples']} | "
                f"{counts['target_unknown_token_examples']} | "
                f"{counts['identical_inputs_with_conflicting_labels']} |"
            )
    lines += [
        "",
        "The full first-sequence target is verified in every encoded pair.",
        "Only the headline may be right-truncated; target overflow fails preparation.",
        "Headline-only label conflicts are expected for mixed-sentiment headlines.",
        "The label-conflict count is distinct encoded input patterns, not row count.",
        "UNK and nonliteral targets remain in the view; no labels are changed.",
        "",
        f"Plan SHA-256: `{result['plan_sha256']}`.",
        f"Private data manifest: `{result['data_manifest_sha256']}`.",
        f"Execution revision: `{result['code_revision']}`.",
        "Native text/labels remain private under `data/interim/target-pairs/`.",
        "No training, inference, calibration fitting or test analysis occurred.",
        "See [protocol and use](../docs/target-sentiment-protocol.md).",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.data_dir, args.tokenizer_dir)


if __name__ == "__main__":
    main()
