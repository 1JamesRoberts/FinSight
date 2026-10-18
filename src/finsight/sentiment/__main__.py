"""Run a sentiment baseline against verified development data."""

import argparse
from pathlib import Path

from finsight.config import Settings
from finsight.sentiment.experiments.vader_validation import run_validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "validate-vader",
            "validate-tfidf-logreg",
            "validate-tfidf-svm",
            "validate-finbert",
            "fine-tune-finbert",
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=Settings.from_env().data_dir)
    parser.add_argument(
        "--freeze-receipt", type=Path, default=Path("configs/benchmark-freeze.json")
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--vader-config", type=Path, default=Path("configs/vader.json"))
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=Path("reports/experiments/tfidf-tuning.json"),
        help="Saved Phase 2 tuning report for FinBERT comparisons",
    )
    parser.add_argument(
        "--pretrained-report",
        type=Path,
        default=Path("reports/experiments/finbert-validation.json"),
    )
    parser.add_argument(
        "--cohort",
        choices=("all", "phrasebank", "sentfin_single_target"),
        default="all",
        help="Sentence cohort for TF-IDF or FinBERT (default: both)",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
    )
    args = parser.parse_args()
    if args.command == "validate-vader":
        if args.cohort != "all":
            parser.error("--cohort is supported for validate-tfidf-logreg only")
        config = args.config or Path("configs/vader.json")
        report = args.report_path or Path("reports/experiments/vader-validation.md")
        run = run_validation(args.data_dir, args.freeze_receipt, config, report)
        key = "vader"
    elif args.command == "fine-tune-finbert":
        from finsight.sentiment.training.sentence import run_finetuning

        if args.cohort == "phrasebank":
            parser.error("fine-tune-finbert supports SEntFiN only")
        config = args.config or Path("configs/finbert-finetune.json")
        report = args.report_path or Path("reports/experiments/finbert-finetuning.md")
        run = run_finetuning(
            args.data_dir,
            args.freeze_receipt,
            config,
            report,
            args.baseline_report,
            args.pretrained_report,
        )
        print(f"Saved checkpoint verified on {run['validation_samples']} texts.")
        print(f"Aggregate report saved to {report}")
        return
    elif args.command == "validate-finbert":
        from finsight.sentiment.experiments.finbert_validation import (
            run_finbert_validation,
        )

        config = args.config or Path("configs/finbert.json")
        report = args.report_path or Path("reports/experiments/finbert-validation.md")
        run = run_finbert_validation(
            args.data_dir,
            args.freeze_receipt,
            config,
            report,
            args.baseline_report,
            cohort=args.cohort,
        )
        key = "finbert"
    else:
        from finsight.sentiment.experiments.tfidf_validation import run_tfidf_validation

        svm = args.command == "validate-tfidf-svm"
        prefix = "tfidf-svm" if svm else "tfidf-logreg"
        config = args.config or Path(f"configs/{prefix}.json")
        report = args.report_path or Path(f"reports/experiments/{prefix}-validation.md")
        run = run_tfidf_validation(
            args.data_dir,
            args.freeze_receipt,
            config,
            report,
            args.vader_config,
            cohort=args.cohort,
            model_kind="linear_svm" if svm else "logistic_regression",
        )
        key = "linear_svm" if svm else "logistic_regression"
    count = sum(result[key]["samples"] for result in run["metrics"].values())
    print(f"{args.command} checked on {count} validation texts.")
    print(f"Aggregate report saved to {report}")


if __name__ == "__main__":
    main()
