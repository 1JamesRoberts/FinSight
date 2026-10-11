"""Run a sentiment baseline against verified development data."""

import argparse
from pathlib import Path

from finsight.config import Settings
from finsight.sentiment.validation import run_validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate-vader", "validate-tfidf-logreg"))
    parser.add_argument("--data-dir", type=Path, default=Settings.from_env().data_dir)
    parser.add_argument(
        "--freeze-receipt", type=Path, default=Path("configs/benchmark-freeze.json")
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--vader-config", type=Path, default=Path("configs/vader.json"))
    parser.add_argument(
        "--report-path",
        type=Path,
    )
    args = parser.parse_args()
    if args.command == "validate-vader":
        config = args.config or Path("configs/vader.json")
        report = args.report_path or Path("reports/experiments/vader-validation.md")
        run = run_validation(args.data_dir, args.freeze_receipt, config, report)
        key = "vader"
    else:
        from finsight.sentiment.tfidf_validation import run_tfidf_validation

        config = args.config or Path("configs/tfidf-logreg.json")
        report = args.report_path or Path(
            "reports/experiments/tfidf-logreg-validation.md"
        )
        run = run_tfidf_validation(
            args.data_dir, args.freeze_receipt, config, report, args.vader_config
        )
        key = "logistic_regression"
    count = sum(result[key]["samples"] for result in run["metrics"].values())
    print(f"{args.command} checked on {count} validation texts.")
    print(f"Aggregate report saved to {report}")


if __name__ == "__main__":
    main()
