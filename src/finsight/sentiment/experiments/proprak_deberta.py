"""Unchanged custom multitask DeBERTa on development validation only."""

import argparse
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from finsight.data.benchmark import load_development_records
from finsight.sentiment.artifacts import file_sha256, verify_files
from finsight.sentiment.evaluation.cohorts import sentence_and_label, sentence_cohorts
from finsight.sentiment.evaluation.metrics import classification_metrics
from finsight.sentiment.experiments.sentence_comparison import predictions_bytes
from finsight.sentiment.models.deberta import (
    build_model,
    load_tensor_state,
    load_tokenizer,
)
from finsight.sentiment.models.deberta import masked_mean as masked_mean
from finsight.sentiment.models.transformer import runtime, score_model
from finsight.sentiment.training.config import FineTuneConfig
from finsight.sentiment.training.sentence import membership_digest

ASSET = Path("configs/proprak-deberta-asset.json")
RECEIPT = Path("configs/benchmark-freeze.json")


def validation_rows(records: list[dict]) -> list[dict]:
    """Reject reserved records before selecting single-target SEntFiN validation."""
    rows = sentence_cohorts(records)["sentfin_single_target"]
    selected = [row for row in rows if row["partition"] == "validation"]
    if not selected:
        raise ValueError("SEntFiN validation must be nonempty")
    return selected


def run(data_dir: Path, device: str) -> dict:
    """One fixed inference run; private predictions and aggregate public report."""
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig

    public = Path("reports/experiments/proprak-deberta-validation.json")
    if public.exists():
        raise FileExistsError("validation report exists; inspect it before rerunning")
    asset = json.loads(ASSET.read_bytes())
    config = FineTuneConfig(device=device)
    records = load_development_records(data_dir, RECEIPT)
    rows = validation_rows(records)
    membership = membership_digest(rows)
    controls = {}
    for name in ("financial_deberta", "financial_deberta_adapted", "deberta_base"):
        path = Path("reports/experiments") / f"{name}.json"
        previous = json.loads(path.read_bytes())
        if (
            previous["validation_membership_sha256"] != membership
            or previous["final_test_evaluated"] is not False
        ):
            raise ValueError("historical validation membership or scope differs")
        controls[name] = {
            "metrics": previous["metrics"],
            "report_sha256": file_sha256(path),
            "scope": "Historical one-seed development run; not final-test results",
        }
    torch = runtime(config)
    directory = Path(
        snapshot_download(
            repo_id=asset["model_id"],
            revision=asset["revision"],
            allow_patterns=list(asset["files"]),
            cache_dir=str(data_dir / "interim/transformer-checkpoint-cache"),
        )
    )
    verify_files(directory, asset["files"])
    source_config = AutoConfig.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False
    )
    source_config._attn_implementation = "eager"
    model = build_model(source_config)
    load_tensor_state(model, directory / "pytorch_model.bin")
    tokenizer = load_tokenizer(directory)
    model.to(device=device, dtype=torch.float32)
    texts, gold = zip(*(sentence_and_label(row) for row in rows), strict=True)
    predictions, profile = score_model(torch, tokenizer, model, list(texts), config)
    private = predictions_bytes(rows, predictions)
    receipt = json.loads(RECEIPT.read_bytes())
    result = {
        "schema_version": 1,
        "experiment": "proprak-deberta-validation",
        "scope": "Post-final exploratory SEntFiN validation; unchanged classification "
        "head; no training, calibration fitting or final-test inference. "
        "Upstream SEntFiN exposure unknown; PhraseBank/FiQA supervision reported.",
        "asset": asset,
        "asset_manifest_sha256": file_sha256(ASSET),
        "configuration": asdict(config.inference_config()),
        "training_samples": 0,
        "validation_samples": len(rows),
        "validation_membership_sha256": membership,
        "final_test_evaluated": False,
        "frozen_test_records_sha256": receipt["records_sha256"],
        "benchmark_manifest_sha256": receipt["benchmark_manifest_sha256"],
        "parent_split_manifest_sha256": receipt["parent_split_manifest_sha256"],
        "metrics": classification_metrics(
            gold=list(gold), predictions=[p.label.value for p in predictions]
        ),
        "model": profile,
        "pooling": "attention-mask mean over non-padding tokens, including specials",
        "heads": "768->3 classification, 768->1 regression; dropout 0.1; "
        "regression head loaded strictly but not evaluated",
        "model_file_bytes": asset["files"]["pytorch_model.bin"]["size_bytes"],
        "predictions_sha256": hashlib.sha256(private).hexdigest(),
        "historical_comparisons": controls,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "implementation_sha256": file_sha256(Path(__file__)),
        "lockfile_sha256": file_sha256(Path("uv.lock")),
        "environment": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("torch", "transformers", "tokenizers")},
        },
    }
    # Validate the frozen/data manifests again after inference without scoring tests.
    load_development_records(data_dir, RECEIPT)
    body = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    identifier = hashlib.sha256(body).hexdigest()
    destination = data_dir / "interim/proprak-deberta-validation" / identifier
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "run.json").write_bytes(body)
    (destination / "predictions.jsonl").write_bytes(private)
    with public.open("xb") as stream:
        stream.write(body)
    print(
        json.dumps(
            {"run_id": identifier, "metrics": result["metrics"], "model": profile},
            indent=2,
        ),
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    run(args.data_dir, args.device)


if __name__ == "__main__":
    main()
