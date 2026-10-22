"""Synthetic native-company-name sensitivity; no counterfactual gold or fitting."""

import argparse
import gc
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev

from finsight.data.target_pairs import TargetText
from finsight.data.target_preparation import json_bytes
from finsight.data.target_spans import (
    VARIANTS,
    align_targets,
    literal_spans,
    transform_input,
)
from finsight.data.target_tokenization import audit_pairs
from finsight.sentiment.artifacts import file_sha256, project_root, sha
from finsight.sentiment.evaluation.metrics import LABELS
from finsight.sentiment.experiments.target_format_preparation import (
    load_plan,
    load_tokenizers,
)
from finsight.sentiment.experiments.target_format_training import locked_inputs
from finsight.sentiment.models.target_inference import score_inputs
from finsight.sentiment.models.transformer import load_saved
from finsight.sentiment.training.config import FineTuneConfig

PLAN = Path("configs/target-name-stress-plan.json")
DEPENDENCIES = (
    "data/target_spans.py",
    "data/target_pairs.py",
    "data/target_tokenization.py",
    "sentiment/models/target_inference.py",
    "sentiment/models/transformer.py",
    "sentiment/experiments/target_format_preparation.py",
    "sentiment/experiments/target_format_training.py",
)


def renamed_views(
    headline: str, target: str, other_targets: list[str], name: str
) -> dict:
    """Change only selected literal spans, then apply each trained input format."""
    if not name.strip() or name.casefold() == target.casefold():
        raise ValueError("distinct nonempty substitute required")
    if literal_spans(headline, name) or any(
        literal_spans(name, other) or literal_spans(other, name)
        for other in other_targets
    ):
        raise ValueError("substitute collides with original context or native targets")
    alignment = align_targets(headline, [target, *other_targets])[0]
    if not alignment.eligible:
        raise ValueError("eligible native alignment required")
    changed = headline
    for span in reversed(alignment.spans):
        changed = changed[: span.start] + name + changed[span.end :]
    replacement = align_targets(changed, [name, *other_targets])[0]
    views = {
        v: dict(
            zip(
                ("target", "headline"),
                transform_input(changed, replacement, v),
                strict=True,
            )
        )
        for v in VARIANTS
    }
    original_masked = transform_input(headline, alignment, "target_replacement")
    if (
        views["target_replacement"]["target"],
        views["target_replacement"]["headline"],
    ) != original_masked:
        raise ValueError("substitution changed non-target context")
    return views


def sensitivity(original: list[dict], renamed: list[dict]) -> dict:
    """Measure output changes without receiving counterfactual sentiment labels."""
    if not original or len(original) != len(renamed):
        raise ValueError("aligned nonempty prediction lists required")
    flips, tv, maximum = 0, [], []
    for a, b in zip(original, renamed, strict=True):
        for row in (a, b):
            p = row["probabilities"]
            if (
                set(p) != set(LABELS)
                or any(not math.isfinite(x) or not 0 <= x <= 1 for x in p.values())
                or not math.isclose(sum(p.values()), 1, abs_tol=1e-6)
            ):
                raise ValueError("invalid named probabilities")
            if row["prediction"] != max(LABELS, key=p.__getitem__):
                raise ValueError("prediction/probability mapping differs")
        changes = [
            abs(a["probabilities"][label] - b["probabilities"][label])
            for label in LABELS
        ]
        flips += a["prediction"] != b["prediction"]
        tv.append(sum(changes) / 2)
        maximum.append(max(changes))
    return {
        "comparisons": len(original),
        "label_flips": flips,
        "label_flip_rate": flips / len(original),
        "mean_total_variation": mean(tv),
        "maximum_absolute_class_probability_change": max(maximum),
    }


def prepare(root: Path, data: Path, tokenizer_dir: Path, plan_path: Path) -> Path:
    plan = json.loads(plan_path.read_bytes())
    if (
        plan["partition"] != "validation"
        or plan["frozen_test_access"]
        or plan["training_performed"]
    ):
        raise ValueError("development inference-only protocol required")
    report_path = root / "reports/experiments/target-format.json"
    if (
        file_sha256(report_path) != plan["source_report_sha256"]
        or file_sha256(root / "configs/target-format-lock.json")
        != plan["preparation_lock_sha256"]
    ):
        raise ValueError("original experiment lineage differs")
    _, examples, _ = locked_inputs(root, data, tokenizer_dir)
    grouped = defaultdict(list)
    for e in examples:
        if e.partition == "validation":
            grouped[e.record_id].append(e)
    cases = []
    for _, annotations in sorted(grouped.items()):
        for e in annotations:
            if e.target not in plan["native_target_allowlist"]:
                continue
            others = [a.target for a in annotations if a.example_id != e.example_id]
            alignment = align_targets(e.headline, [e.target, *others])[0]
            original = {
                v: dict(
                    zip(
                        ("target", "headline"),
                        transform_input(e.headline, alignment, v),
                        strict=True,
                    )
                )
                for v in VARIANTS
            }
            substitutions = [
                renamed_views(e.headline, e.target, others, name)
                for name in plan["substitute_names"]
            ]
            cases.append(
                {
                    "example": asdict(e),
                    "original": original,
                    "substitutions": substitutions,
                }
            )
    if not cases:
        raise ValueError("empty native company-target subset")
    _, files = load_plan(root)
    slow, fast = load_tokenizers(tokenizer_dir, files)
    maximum = {}
    for variant in VARIANTS:
        inputs = [
            view[variant]
            for case in cases
            for view in [case["original"], *case["substitutions"]]
        ]
        checks = audit_pairs(
            slow,
            fast,
            [(v["target"], v["headline"]) for v in inputs],
            max_length=plan["max_length"],
        )
        if any(not c["eligible"] for c in checks):
            raise ValueError(
                "stress input failed token eligibility; review before scoring"
            )
        maximum[variant] = max(c["full_pair_tokens"] for c in checks)
    body = b"".join((json.dumps(c, sort_keys=True) + "\n").encode() for c in cases)
    manifest = {
        "plan_sha256": file_sha256(plan_path),
        "runner_sha256": file_sha256(Path(__file__)),
        "dependency_sha256": {
            name: file_sha256(root / "src/finsight" / name) for name in DEPENDENCIES
        },
        "cases_sha256": sha(body),
        "cases": len(cases),
        "headlines": len({c["example"]["record_id"] for c in cases}),
        "native_target_counts": dict(Counter(c["example"]["target"] for c in cases)),
        "paired_comparisons_per_checkpoint": len(cases) * len(plan["substitute_names"]),
        "maximum_encoded_tokens": maximum,
    }
    identity = sha(json_bytes(manifest))
    directory = data / "interim/target-name-stress" / identity
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in {
        "cases.jsonl": body,
        "manifest.json": json_bytes(manifest),
        "plan.json": plan_path.read_bytes(),
    }.items():
        path = directory / name
        if path.exists() and path.read_bytes() != content:
            raise ValueError("existing probe preparation differs")
        path.write_bytes(content)
    print(json.dumps(manifest, indent=2))
    return directory


def run(root: Path, data: Path, packet: Path) -> dict:
    plan = json.loads((packet / "plan.json").read_bytes())
    manifest = json.loads((packet / "manifest.json").read_bytes())
    if (
        sha(json_bytes(manifest)) != packet.name
        or file_sha256(packet / "plan.json") != manifest["plan_sha256"]
        or file_sha256(packet / "cases.jsonl") != manifest["cases_sha256"]
        or file_sha256(Path(__file__)) != manifest["runner_sha256"]
    ):
        raise ValueError("probe packet or implementation differs")
    for name, digest in manifest["dependency_sha256"].items():
        if file_sha256(root / "src/finsight" / name) != digest:
            raise ValueError("probe dependency differs")
    source = root / "reports/experiments/target-format.json"
    if file_sha256(source) != plan["source_report_sha256"]:
        raise ValueError("saved study differs")
    study = json.loads(source.read_bytes())
    base_plan, _ = load_plan(root)
    cases = [
        json.loads(line) for line in (packet / "cases.jsonl").read_bytes().splitlines()
    ]
    trials = []
    private = []
    for trial in study["trials"]:
        variant = trial["input_variant"]
        seed = trial["seed"]
        config = FineTuneConfig(**trial["configuration"])
        torch, tokenizer, model, artifact = load_saved(
            data / trial["artifact_relative_path"],
            trial["artifact_manifest_sha256"],
            base_plan["asset"],
            config,
        )
        if artifact["context"] != trial["context"]:
            raise ValueError("saved checkpoint context differs")
        inputs = []
        for case in cases:
            for index, views in enumerate([case["original"], *case["substitutions"]]):
                v = views[variant]
                inputs.append(
                    TargetText(
                        case["example"]["example_id"] + "/" + str(index),
                        v["headline"],
                        v["target"],
                    )
                )
        predictions, profile = score_inputs(
            torch, tokenizer, model, inputs, "target_pair", config
        )
        # Verify original-input replay against checksummed saved validation output.
        cache = (
            data
            / "interim/target-deberta"
            / sha(json_bytes(trial))
            / "predictions.jsonl"
        )
        if file_sha256(cache) != trial["predictions_sha256"]:
            raise ValueError("native prediction cache differs")
        saved = {
            r["example_id"]: r
            for r in (json.loads(line) for line in cache.read_bytes().splitlines())
        }
        originals = []
        changed = []
        replay_max = 0.0
        width = 1 + len(plan["substitute_names"])
        for i, case in enumerate(cases):
            base = predictions[i * width]
            reference = saved[case["example"]["example_id"]]
            replay_max = max(
                replay_max,
                max(
                    abs(
                        base["probabilities"][label] - reference["probabilities"][label]
                    )
                    for label in LABELS
                ),
            )
            if base["prediction"] != reference["prediction"]:
                raise ValueError("original prediction replay class differs")
            for j in range(1, width):
                originals.append(base)
                changed.append(predictions[i * width + j])
                private.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "example_id": case["example"]["example_id"],
                        "substitute_name": plan["substitute_names"][j - 1],
                        "original": base,
                        "renamed": predictions[i * width + j],
                    }
                )
        if replay_max > plan["probability_tolerance"]:
            raise ValueError("original probability replay differs")
        metrics = sensitivity(originals, changed)
        if variant == "target_replacement" and (
            metrics["label_flips"]
            or metrics["maximum_absolute_class_probability_change"]
            > plan["probability_tolerance"]
        ):
            raise ValueError("identical replacement-input control failed")
        trials.append(
            {
                "variant": variant,
                "seed": seed,
                "artifact_manifest_sha256": trial["artifact_manifest_sha256"],
                "original_replay_max_probability_delta": replay_max,
                "metrics": metrics,
                "model": profile,
            }
        )
        print(f"{variant} seed {seed}: {metrics}", flush=True)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    result = {
        "experiment": plan["experiment"],
        "plan_sha256": manifest["plan_sha256"],
        "preparation_manifest_sha256": packet.name,
        "preparation": manifest,
        "trials": trials,
        "aggregate": {},
        "training_performed": False,
        "frozen_test_evaluated": False,
        "counterfactual_accuracy_computed": False,
    }
    for variant in VARIANTS:
        cells = [t["metrics"] for t in trials if t["variant"] == variant]
        result["aggregate"][variant] = {
            k: {
                "mean": mean(c[k] for c in cells),
                "sample_std": stdev(c[k] for c in cells),
            }
            for k in ("label_flip_rate", "mean_total_variation")
        }
        result["aggregate"][variant]["maximum_absolute_class_probability_change"] = max(
            c["maximum_absolute_class_probability_change"] for c in cells
        )
    private_body = b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode() for row in private
    )
    (packet / "predictions-private.jsonl").write_bytes(private_body)
    result["predictions_sha256"] = sha(private_body)
    (packet / "results.json").write_bytes(json_bytes(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    if args.run:
        run(args.root.resolve(), args.data_dir.resolve(), args.run.resolve())
    else:
        if not args.tokenizer_dir:
            parser.error("--tokenizer-dir required for preparation")
        print(
            prepare(
                args.root.resolve(),
                args.data_dir.resolve(),
                args.tokenizer_dir,
                args.plan,
            )
        )


if __name__ == "__main__":
    main()
