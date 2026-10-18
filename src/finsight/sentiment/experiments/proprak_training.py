"""Fixed three-seed adaptation of the reviewed multitask financial checkpoint."""

import argparse
import gc
import hashlib
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from finsight.data.benchmark import load_development_records
from finsight.sentiment.artifacts import (
    file_sha256,
    json_body,
    save_model,
    source_manifest,
    verify_files,
)
from finsight.sentiment.evaluation.calibration import calibration_metrics
from finsight.sentiment.evaluation.cohorts import sentence_and_label
from finsight.sentiment.evaluation.metrics import LABELS, classification_metrics
from finsight.sentiment.experiments.proprak_deberta import ASSET, RECEIPT
from finsight.sentiment.experiments.sentence_comparison import predictions_bytes
from finsight.sentiment.models.deberta import (
    build_model,
    load_tensor_state,
    load_tokenizer,
)
from finsight.sentiment.models.transformer import runtime, score_model
from finsight.sentiment.training.config import FineTuneConfig
from finsight.sentiment.training.sentence import (
    fit_checkpoint,
    membership_digest,
    sentfin_partitions,
)

PLAN = Path("configs/proprak-deberta-training-plan.json")
REPORTS = Path("reports/experiments/proprak-deberta-adapted")
FROZEN = ("reg_head.weight", "reg_head.bias")


def load_plan(path: Path = PLAN) -> tuple[dict, FineTuneConfig]:
    """Require the fixed matched protocol and source/reference identities."""
    plan = json.loads(path.read_bytes())
    if (
        plan["schema_version"] != 1
        or plan["seeds"] != [42, 43, 44]
        or plan["frozen_parameters"] != list(FROZEN)
        or plan["cohort"] != "sentfin_single_target"
        or plan["final_test_evaluated"] is not False
        or file_sha256(ASSET) != plan["asset_manifest_sha256"]
        or file_sha256(RECEIPT) != plan["freeze_receipt_sha256"]
        or file_sha256(Path(plan["configuration_template"]))
        != plan["configuration_template_sha256"]
    ):
        raise ValueError("training plan or source identities differ")
    for name, expected in plan["reference_reports"].items():
        if file_sha256(Path(name)) != expected:
            raise ValueError("reference report differs")
    config = FineTuneConfig.load(Path(plan["configuration_template"]))
    if config != FineTuneConfig(learning_rate=1e-5):
        raise ValueError("matched training budget differs")
    return plan, config


def source_model(asset: dict, config: FineTuneConfig, data_dir: Path) -> tuple:
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig

    torch = runtime(config)
    directory = Path(
        snapshot_download(
            repo_id=asset["model_id"],
            revision=asset["revision"],
            cache_dir=str(data_dir / "interim/transformer-checkpoint-cache"),
            allow_patterns=list(asset["files"]),
        )
    )
    verify_files(directory, asset["files"])
    c = AutoConfig.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False
    )
    c._attn_implementation = "eager"
    model = build_model(c)
    load_tensor_state(model, directory / "pytorch_model.bin")
    return torch, load_tokenizer(directory), model, directory


def reload_model(
    directory: Path,
    identifier: str,
    asset: dict,
    config: FineTuneConfig,
    training_membership: str,
) -> tuple[Any, Any]:
    """Verify complete artifact ancestry before rebuilding the custom architecture."""
    from transformers import AutoConfig, PreTrainedTokenizerFast

    manifest_path = directory / "artifact-manifest.json"
    if file_sha256(manifest_path) != identifier:
        raise ValueError("artifact manifest differs")
    manifest = json.loads(manifest_path.read_bytes())
    if (
        manifest["source"] != asset
        or manifest["configuration"] != asdict(config)
        or manifest["training_membership_sha256"] != training_membership
    ):
        raise ValueError("artifact ancestry differs")
    verify_files(directory, manifest["files"])
    c = AutoConfig.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False
    )
    c._attn_implementation = "eager"
    model = build_model(c)
    load_tensor_state(model, directory / "model.safetensors")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False
    )
    return tokenizer, model


def run_seed(seed: int, data_dir: Path) -> dict:
    """Always initialize from source, preserve frozen regression head, reload best."""
    plan, template = load_plan()
    if seed not in plan["seeds"]:
        raise ValueError("seed not in declared plan")
    public = REPORTS / f"seed-{seed}.json"
    if public.exists():
        raise FileExistsError("seed report already exists; inspect saved outputs")
    config = replace(template, seed=seed)
    asset = json.loads(ASSET.read_bytes())
    training, validation = sentfin_partitions(
        load_development_records(data_dir, RECEIPT)
    )
    train_digest = membership_digest(training)
    validation_digest = membership_digest(validation)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], text=True
    ).splitlines()
    allowed = {f"?? {REPORTS.as_posix()}/seed-{s}.json" for s in plan["seeds"]}
    if set(status) - allowed:
        raise ValueError("training requires unchanged code; only seed outputs allowed")
    dirty = bool(status)
    source_regression = {}

    def loader(*_: Any) -> tuple:
        loaded = source_model(asset, config, data_dir)
        source_regression.update(
            {
                name: p.detach().clone()
                for name, p in loaded[2].named_parameters()
                if name in FROZEN
            }
        )
        return loaded

    def saver(
        model: Any,
        tokenizer: Any,
        _root: Path,
        cfg: FineTuneConfig,
        epoch: int,
        membership: str,
    ) -> tuple:
        return save_model(model, tokenizer, data_dir, asset, cfg, epoch, membership)

    print(
        f"Starting ProPrak01 adaptation seed {seed}: {len(training)} train, "
        f"{len(validation)} validation",
        flush=True,
    )
    fit = fit_checkpoint(
        config,
        training,
        validation,
        data_dir,
        checkpoint_loader=loader,
        artifact_saver=saver,
        checkpoint_files=tuple(asset["files"]),
        run_namespace=f"proprak-seed-{seed}",
        index_labels=LABELS,
        frozen_parameters=FROZEN,
    )
    directory = fit.pop("artifact_directory")
    torch = runtime(config)
    tokenizer, model = reload_model(
        directory, fit["artifact_manifest_sha256"], asset, config, train_digest
    )
    if not all(
        torch.equal(source_regression[n], dict(model.named_parameters())[n])
        for n in FROZEN
    ):
        raise ValueError("frozen regression weights changed")
    model.to(device=config.device, dtype=torch.float32)
    texts, gold = zip(*(sentence_and_label(row) for row in validation), strict=True)
    predictions, profile = score_model(torch, tokenizer, model, list(texts), config)
    metrics = classification_metrics(list(gold), [p.label.value for p in predictions])
    if metrics != fit["history"][fit["best_epoch"] - 1]["validation"]:
        raise ValueError("saved checkpoint failed to reproduce selected-epoch metrics")
    private = predictions_bytes(validation, predictions)
    packet = [json.loads(line) for line in private.splitlines()]
    receipt = json.loads(RECEIPT.read_bytes())
    result = {
        "schema_version": 1,
        "experiment": "proprak-deberta-adaptation",
        "scope": "Post-final exploratory development only; no test inference; "
        "raw probabilities; no temperature fitting; no best-seed selection",
        "seed": seed,
        "configuration": asdict(config),
        "asset": asset,
        "plan_sha256": file_sha256(PLAN),
        "plan": plan,
        "training_samples": len(training),
        "validation_samples": len(validation),
        "training_membership_sha256": train_digest,
        "validation_membership_sha256": validation_digest,
        "final_test_evaluated": False,
        "benchmark_manifest_sha256": receipt["benchmark_manifest_sha256"],
        "frozen_records_sha256": receipt["records_sha256"],
        "metrics": metrics,
        "calibration": calibration_metrics(packet),
        "fit": fit,
        "model": profile,
        "artifact_relative_path": directory.relative_to(data_dir).as_posix(),
        "artifact_file_bytes": sum(p.stat().st_size for p in directory.iterdir()),
        "frozen_parameters": list(FROZEN),
        "regression_weights_unchanged": True,
        "predictions_sha256": hashlib.sha256(private).hexdigest(),
        "code_revision": revision,
        "working_tree_dirty": dirty,
        "working_tree_status": status,
        "implementation_sha256": source_manifest(),
        "lockfile_sha256": file_sha256(Path("uv.lock")),
    }
    load_development_records(data_dir, RECEIPT)
    body = json_body(result)
    identifier = hashlib.sha256(body).hexdigest()
    destination = data_dir / "interim/proprak-deberta-adaptation" / identifier
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "run.json").write_bytes(body)
    (destination / "predictions.jsonl").write_bytes(private)
    public.parent.mkdir(parents=True, exist_ok=True)
    with public.open("xb") as stream:
        stream.write(body)
    print(
        json.dumps(
            {
                "seed": seed,
                "run_id": identifier,
                "best_epoch": fit["best_epoch"],
                "metrics": metrics,
            }
        ),
        flush=True,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    run_seed(args.seed, args.data_dir)


if __name__ == "__main__":
    main()
