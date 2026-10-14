"""Fixed, matched TF-IDF Logistic Regression comparison on target development data."""

import argparse
import hashlib
import json
import platform
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from threadpoolctl import threadpool_limits

from finsight.data.target_preparation import LOCK, PLAN, json_bytes
from finsight.sentiment.finbert import file_sha256
from finsight.sentiment.target_control import FINAL, evaluate_cohorts, verified_inputs
from finsight.sentiment.target_tfidf import TargetTfidf
from finsight.sentiment.tfidf import TfidfConfig, dependency_versions

CONFIG = Path("configs/target-tfidf.json")
REPORT = Path("reports/experiments/target-tfidf-logreg")
CONTROL = Path("reports/experiments/target-saved-control.json")
COHORTS = (
    "all_targets",
    "single_target",
    "multi_target",
    "mixed_label_headlines",
    "seen_native_target",
    "unseen_native_target",
)


def sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def run(data_dir: Path, config_path: Path = CONFIG) -> dict:
    """Train both arms on exactly the same annotations; never select using test."""
    if any(REPORT.with_suffix(s).exists() for s in (".json", ".md")):
        raise FileExistsError("target TF-IDF results already published")
    settings = json.loads(config_path.read_bytes())
    if (
        settings["schema_version"] != 1
        or settings["arms"] != ["headline_only", "headline_target"]
        or settings["threads"] != 1
        or settings["final_test_evaluated"] is not False
        or settings["data_plan_sha256"] != file_sha256(PLAN)
        or settings["data_manifest_sha256"]
        != json.loads(LOCK.read_bytes())["data_manifest_sha256"]
        or settings["saved_control_report_sha256"] != file_sha256(CONTROL)
        or settings["classifier"]["class_weight"] is not None
    ):
        raise ValueError("target TF-IDF configuration/lineage differs")
    _, _, examples, _ = verified_inputs(data_dir)
    final_sha = file_sha256(FINAL)
    training = tuple(e for e in examples if e.partition == "train")
    validation = tuple(e for e in examples if e.partition == "validation")
    train_targets = {e.target for e in training}
    texts, targets = [e.headline for e in validation], [e.target for e in validation]
    configuration = TfidfConfig(**settings["classifier"])
    models, private, vocabularies = {}, {}, []
    for arm in settings["arms"]:
        model = TargetTfidf(
            arm,
            configuration,
            target_min_df=settings["target_min_df"],
            target_max_features=settings["target_max_features"],
        )
        print(f"Training {arm}: {len(training)} target annotations", flush=True)
        with threadpool_limits(limits=settings["threads"]):
            started = perf_counter()
            model.fit(training)
            training_seconds = perf_counter() - started
            before = model.predict_many(texts, targets)
            # Serialized-byte addressing preserves artifacts from previous runs.
            artifact_root = data_dir / "interim/target-tfidf-models"
            artifact_root.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=artifact_root) as temporary:
                staging = Path(temporary) / "model.pkl"
                digest = model.save(staging)
                artifact = artifact_root / digest / "model.pkl"
                artifact.parent.mkdir(parents=True, exist_ok=True)
                if artifact.exists():
                    if artifact.read_bytes() != staging.read_bytes():
                        raise ValueError("existing fitted artifact differs")
                else:
                    staging.replace(artifact)
            model = TargetTfidf.load(artifact, expected_sha256=digest)
            model.predict_many(["The company reports earnings."], ["Company"])
            started = perf_counter()
            predictions = model.predict_many(texts, targets)
            elapsed = perf_counter() - started
        if predictions != before:
            raise ValueError("saved/reloaded model predictions differ")
        vocabulary = model.headline_vectorizer
        vocabularies.append(
            sha(
                json_bytes(
                    {
                        "vocabulary": {
                            k: int(v) for k, v in vocabulary.vocabulary_.items()
                        },
                        "idf": vocabulary.idf_.tolist(),
                    }
                )
            )
        )
        rows = [
            {
                **asdict(e),
                "gold": e.label,
                "prediction": p.label.value,
                "probabilities": p.probabilities,
            }
            for e, p in zip(validation, predictions, strict=True)
        ]
        cohorts = evaluate_cohorts(rows, train_targets)
        by_headline: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            by_headline[row["record_id"]].add(row["prediction"])
        differing = {
            identity for identity, labels in by_headline.items() if len(labels) > 1
        }
        if arm == "headline_only" and differing:
            raise ValueError("headline-only predictions depend on target")
        body = b"".join((json.dumps(r, sort_keys=True) + "\n").encode() for r in rows)
        private[arm] = body
        models[arm] = {
            "cohorts": cohorts,
            "model": model.metadata(),
            "training_seconds": training_seconds,
            "inference_seconds": elapsed,
            "inference_ms_per_annotation": elapsed * 1000 / len(validation),
            "artifact_relative_path": artifact.relative_to(data_dir).as_posix(),
            "artifact_sha256": digest,
            "artifact_bytes": artifact.stat().st_size,
            "predictions_sha256": sha(body),
            "reload_probabilities_identical": True,
            "headline_vocabulary_and_idf_sha256": vocabularies[-1],
            "feature_diagnostics": {
                "train": model.diagnostics(
                    [e.headline for e in training], [e.target for e in training]
                ),
                "validation": model.diagnostics(texts, targets),
            },
            "headlines_with_different_target_predictions": len(differing),
            "mixed_label_headlines_with_different_predictions": len(
                differing & {e.record_id for e in validation if e.mixed_labels}
            ),
        }
        score = cohorts["all_targets"]["classification"]["macro_f1"]
        print(
            f"{arm}: validation macro F1={score:.6f}",
            flush=True,
        )
    if len(set(vocabularies)) != 1:
        raise ValueError("headline feature fits differ between matched arms")
    verified_inputs(data_dir)
    if file_sha256(FINAL) != final_sha:
        raise ValueError("original final aggregate changed")
    control = json.loads(CONTROL.read_bytes())
    result = {
        "schema_version": 1,
        "experiment": settings["experiment"],
        "configuration": settings,
        "configuration_sha256": file_sha256(config_path),
        "plan_sha256": file_sha256(PLAN),
        "lock_sha256": file_sha256(LOCK),
        "data_manifest_sha256": settings["data_manifest_sha256"],
        "training_examples": len(training),
        "training_headlines": len({e.record_id for e in training}),
        "training_class_counts": dict(
            sorted(Counter(e.label for e in training).items())
        ),
        "training_membership_sha256": sha(json_bytes([e.example_id for e in training])),
        "validation_membership_sha256": sha(
            json_bytes([e.example_id for e in validation])
        ),
        "models": models,
        "matched_headline_features_identical": True,
        "saved_deberta_control": {
            "report_sha256": file_sha256(CONTROL),
            "aggregate": control["aggregate"],
        },
        "original_final_aggregate_sha256": final_sha,
        "hyperparameters_tuned": False,
        "final_test_evaluated": False,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "lockfile_sha256": file_sha256(Path("uv.lock")),
        "implementation_sha256": {
            name: file_sha256(Path(name))
            for name in (
                "src/finsight/sentiment/target_tfidf.py",
                "src/finsight/sentiment/target_tfidf_experiment.py",
                "src/finsight/sentiment/target_control.py",
                "src/finsight/sentiment/tfidf.py",
                "src/finsight/sentiment/validation.py",
                "src/finsight/sentiment/transformer_analysis.py",
                "src/finsight/data/target_pairs.py",
            )
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            **dependency_versions(),
        },
        "scope": "Exploratory native-target validation only. One fixed "
        "deterministic CPU fit per arm, equal annotation weights; "
        "additive target features, no interactions. No temperature "
        "fitting or test inference.",
    }
    body = json_bytes(result)
    root = data_dir / "interim/target-tfidf-logreg" / sha(body)
    root.mkdir(parents=True, exist_ok=False)
    for arm, predictions in private.items():
        (root / f"{arm}.jsonl").write_bytes(predictions)
    (root / "run.json").write_bytes(body)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.with_suffix(".json").write_bytes(body)
    REPORT.with_suffix(".md").write_bytes(render_report(result).encode())
    return result


def render_report(result: dict) -> str:
    settings = result["configuration"]
    classifier = settings["classifier"]
    lines = [
        "# Matched TF-IDF + Logistic Regression target baselines",
        "",
        f"Both arms fit the same {result['training_examples']:,} target annotations "
        f"from {result['training_headlines']:,} training headlines.",
        f"One fixed CPU fit per arm: C={classifier['c']}, "
        "L2 multinomial logistic regression, lbfgs, "
        f"seed {classifier['seed']}, no class weights or search. Validation only.",
        "",
        "| Cohort | Targets | Headline LR F1 % | Headline + target LR "
        "F1 % | Saved DeBERTa F1 % |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for cohort in COHORTS:
        first = result["models"]["headline_only"]["cohorts"][cohort]
        second = result["models"]["headline_target"]["cohorts"][cohort]
        control = result["saved_deberta_control"]["aggregate"][cohort]["macro_f1"][
            "mean"
        ]
        a, b = first["classification"]["macro_f1"], second["classification"]["macro_f1"]
        lines.append(
            f"| {cohort} | {first['examples']} | {a * 100:.2f} | "
            f"{b * 100:.2f} | {control * 100:.2f} |"
        )
    lines += [
        "",
        "The saved DeBERTa column is a three-seed mean, previously trained on 6,299",
        "single-target headlines. Its data/update budget differs; this is context,",
        "not a controlled model-architecture comparison. LR arms share "
        "all training rows.",
        "",
        "| Arm | Features (headline + target) | Coefficients shape | "
        "Learned parameters | Artifact bytes |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for arm in ("headline_only", "headline_target"):
        run = result["models"][arm]
        m = run["model"]
        lines.append(
            f"| {arm} | {m['headline_features']} + {m['target_features']} | "
            f"{m['coefficient_shape']} | {m['parameters']} | {run['artifact_bytes']} |"
        )
    lines += [
        "",
        "Learned parameter counts include classifier coefficients and "
        "intercepts; fitted",
        "IDF values and vocabulary are stored separately in the saved artifact.",
        "Headline vocabulary/IDF hashes match across arms. Each TF-IDF block uses",
        "word unigrams/bigrams, lowercase, sublinear TF, smoothed IDF and L2 norm.",
        f"Headline min_df={classifier['min_df']}, "
        f"max_features={classifier['max_features']:,}; "
        f"target min_df={settings['target_min_df']}, "
        f"max_features={settings['target_max_features']:,}.",
        "Both are fitted only on training annotations, including repeated headlines.",
        "The blocks are concatenated at scale 1 with no joint renormalization.",
        "",
        "| Arm | All-target accuracy % | NLL | Brier | ECE | Zero "
        "target rows (validation) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ("headline_only", "headline_target"):
        run = result["models"][arm]
        c = run["cohorts"]["all_targets"]
        p = c["calibration"]
        zero = run["feature_diagnostics"]["validation"]["zero_target_rows"]
        lines.append(
            f"| {arm} | {c['classification']['accuracy'] * 100:.2f} | "
            f"{p['nll']:.4f} | {p['brier']:.4f} | {p['ece']:.4f} | "
            f"{zero if zero is not None else 'N/A'} |"
        )
    lines += [
        "",
        "## Interpretation and limitations",
        "",
        "The target block adds target-specific class offsets, not target/headline",
        "interactions. It can learn entity sentiment tendencies; it cannot reliably",
        "bind individual sentiment expressions to their entity. Changes in predictions",
        "do not demonstrate contextual aspect understanding. Native target strings",
        "are not verified company identities, and unseen strings may "
        "share word features.",
        "Unseen tokens are ignored; a fully zero target block leaves headline evidence",
        "plus the intercept from the target-trained model, not the "
        "headline-only model.",
        "Default word tokenization drops punctuation and single-character tokens.",
        "",
    ]
    for arm in ("headline_only", "headline_target"):
        run = result["models"][arm]
        lines.append(
            f"{arm}: different predicted labels across targets in "
            f"{run['headlines_with_different_target_predictions']} "
            "validation headlines, "
            f"including {run['mixed_label_headlines_with_different_predictions']} "
            "mixed-label headlines."
        )
    lines += [
        "",
        "Metrics weight target annotations equally. Cohorts overlap. "
        "Probabilities are raw;",
        "ECE uses ten equal-width top-confidence bins. No seed-variation estimate is",
        "appropriate for one deterministic fit. No confidence interval or superiority",
        "claim is made from this reused validation set. Reserved tests "
        "remain untouched.",
        "Full per-class metrics, confusion matrices, calibration bins and timings are",
        "in the companion JSON. Both private models reproduce all "
        "probabilities after reload.",
        "Native inputs, fitted vocabularies/weights and row "
        "predictions stay ignored/private.",
        "",
        "Reproduce from a clean checkout with the existing private "
        "data and absent outputs:",
        "",
        "```sh",
        "uv run --locked python -m finsight.sentiment.target_tfidf_experiment",
        "```",
        "",
        "The runner refuses existing published reports. Inspect saved "
        "results for routine use.",
        f"Data manifest: `{result['data_manifest_sha256']}`.",
        f"Configuration SHA-256: `{result['configuration_sha256']}`.",
        f"Execution revision: `{result['code_revision']}`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    run(args.data_dir, args.config)


if __name__ == "__main__":
    main()
