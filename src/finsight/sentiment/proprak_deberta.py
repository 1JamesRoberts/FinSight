"""Unchanged custom multitask DeBERTa on development validation only."""

import argparse
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from finsight.data.benchmark import load_development_records
from finsight.sentiment.finbert import file_sha256
from finsight.sentiment.finbert_training import FineTuneConfig, membership_digest
from finsight.sentiment.tfidf_validation import sentence_and_label
from finsight.sentiment.transformer_comparison import (
    predictions_bytes,
    runtime,
    score_model,
    verify_files,
)
from finsight.sentiment.validation import (
    LABELS,
    classification_metrics,
    sentence_cohorts,
)

ASSET = Path("configs/proprak-deberta-asset.json")
RECEIPT = Path("configs/benchmark-freeze.json")


def masked_mean(hidden: Any, attention_mask: Any) -> Any:
    """Match the upstream mean over every non-padding token, including specials."""
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def build_model(config: Any) -> Any:
    """Reconstruct the published architecture without executing remote code."""
    import torch
    from transformers import AutoModel

    class MultiTaskDeberta(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = AutoModel.from_config(config, trust_remote_code=False)
            self.dropout = torch.nn.Dropout(0.1)
            self.cls_head = torch.nn.Linear(config.hidden_size, 3)
            self.reg_head = torch.nn.Linear(config.hidden_size, 1)
            self.config = config
            self.config.id2label = dict(enumerate(LABELS))

        def save_pretrained(self, directory: Path, *, safe_serialization: bool) -> None:
            from safetensors.torch import save_file

            if not safe_serialization:
                raise ValueError("derived weights require safe serialization")
            directory.mkdir(parents=True, exist_ok=True)
            save_file(self.state_dict(), str(directory / "model.safetensors"))
            self.config.save_pretrained(directory)

        def forward(
            self,
            input_ids: Any,
            attention_mask: Any,
            labels: Any = None,
            **kwargs: Any,
        ) -> Any:
            output = self.backbone(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )
            pooled = self.dropout(masked_mean(output.last_hidden_state, attention_mask))
            logits = self.cls_head(pooled)
            return SimpleNamespace(
                logits=logits,
                regression=self.reg_head(pooled).squeeze(-1),
                loss=torch.nn.functional.cross_entropy(logits, labels)
                if labels is not None
                else None,
            )

    return MultiTaskDeberta()


def load_tensor_state(model: Any, path: Path) -> None:
    """Require all backbone and both head tensors; never allow random fallback."""
    import torch

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError("checkpoint must be a tensor state dictionary")
    model.load_state_dict(state, strict=True)


def load_tokenizer(directory: Path) -> Any:
    """Read the complete serialized pipeline despite v5-only wrapper metadata."""
    from transformers import PreTrainedTokenizerFast

    metadata = json.loads((directory / "tokenizer_config.json").read_bytes())
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(directory / "tokenizer.json"),
        model_input_names=["input_ids", "attention_mask"],
        **{
            name: metadata[name]
            for name in (
                "bos_token",
                "eos_token",
                "cls_token",
                "sep_token",
                "pad_token",
                "unk_token",
                "mask_token",
            )
        },
    )
    # Assigning token roles must not change the serialized vocabulary or pipeline.
    from tokenizers import Tokenizer

    original = Tokenizer.from_file(str(directory / "tokenizer.json"))
    if tokenizer.get_vocab() != original.get_vocab():
        raise ValueError("tokenizer role assignment changed the vocabulary")
    # The saved backend has batch padding/truncation enabled; compare under the
    # same explicit settings as the wrapper's unpadded, untruncated probe.
    original.no_padding()
    original.no_truncation()
    probe = "Company earnings rose 10%, while debt fell."
    if tokenizer(probe)["input_ids"] != original.encode(probe).ids:
        raise ValueError("serialized tokenizer encoding differs")
    return tokenizer


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
