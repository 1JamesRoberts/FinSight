"""Run one validation-only VADER check against verified development data."""

import argparse
from pathlib import Path

from finsight.config import Settings
from finsight.sentiment.validation import run_validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate-vader",))
    parser.add_argument("--data-dir", type=Path, default=Settings.from_env().data_dir)
    parser.add_argument(
        "--freeze-receipt", type=Path, default=Path("configs/benchmark-freeze.json")
    )
    parser.add_argument("--config", type=Path, default=Path("configs/vader.json"))
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/experiments/vader-validation.md"),
    )
    args = parser.parse_args()
    run = run_validation(
        args.data_dir, args.freeze_receipt, args.config, args.report_path
    )
    count = sum(result["vader"]["samples"] for result in run["metrics"].values())
    print(f"VADER checked on {count} validation texts.")
    print(f"Aggregate report saved to {args.report_path}")


if __name__ == "__main__":
    main()
