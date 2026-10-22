"""Local pretrained ORG proposals, separate from confirmed company identities."""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from finsight.sentiment.artifacts import project_root, verify_files

MODEL_ID = "dslim/bert-base-NER"
REVISION = "d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc"
PLAN = Path("configs/organization-ner-plan.json")
DETECTOR_VERSION = "organization-ner-v2"


@dataclass(frozen=True)
class OrganizationMention:
    text: str
    start: int
    end: int
    score: float | None  # None only when evaluating a cache that omitted scores.


def organization_spans(
    text: str, entities: list[dict]
) -> tuple[OrganizationMention, ...]:
    """Keep aggregated ORG spans; slice original text instead of joining subwords.

    Scores come from pipeline aggregation, not calibrated probabilities that an
    organization is a company. No score threshold is tuned here.
    """
    mentions = []
    for entity in entities:
        if entity["entity_group"] != "ORG":
            continue
        start, end, score = entity["start"], entity["end"], float(entity["score"])
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(text)
            or not text[start:end].strip()
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ValueError("invalid organization span or score")
        mentions.append(OrganizationMention(text[start:end], start, end, score))
    mentions.sort(key=lambda m: (m.start, m.end))
    if any(a.end > b.start for a, b in zip(mentions, mentions[1:], strict=False)):
        raise ValueError("organization spans overlap")
    return tuple(mentions)


class OrganizationDetector:
    def __init__(
        self,
        tokenizer: Any,
        pipeline: Any,
        *,
        max_length: int = 512,
        aggregation_strategy: str = "first",
    ) -> None:
        self.tokenizer = tokenizer
        self.pipeline = pipeline
        self.max_length = max_length
        self.aggregation_strategy = aggregation_strategy

    @classmethod
    def load(
        cls,
        directory: Path,
        *,
        root: Path | None = None,
        device: str = "cpu",
        aggregation_strategy: str = "first",
    ) -> "OrganizationDetector":
        """Verify pinned local bytes before loading; never download implicitly."""
        root = root or project_root()
        if aggregation_strategy not in {"simple", "first"}:
            raise ValueError("unsupported organization aggregation strategy")
        plan = json.loads((root / PLAN).read_bytes())
        if (plan["model_id"], plan["revision"]) != (MODEL_ID, REVISION):
            raise ValueError("unreviewed organization model")
        verify_files(directory, plan["files"])
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            pipeline,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            directory, local_files_only=True, use_fast=True, trust_remote_code=False
        )
        model = AutoModelForTokenClassification.from_pretrained(
            directory,
            local_files_only=True,
            use_safetensors=True,
            trust_remote_code=False,
        )
        if not tokenizer.is_fast or model.config.id2label.get(5) != "B-ORG":
            raise ValueError("NER tokenizer/label schema differs")
        model.eval()
        model.requires_grad_(False)
        classifier = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy=aggregation_strategy,
            device=device,
        )
        return cls(
            tokenizer,
            classifier,
            max_length=plan["max_length"],
            aggregation_strategy=aggregation_strategy,
        )

    def detect_many(
        self, texts: list[str], *, batch_size: int = 16
    ) -> list[tuple[OrganizationMention, ...]]:
        """Check complete lengths first so pipeline truncation cannot hide mentions."""
        if (
            not texts
            or any(not isinstance(t, str) or not t.strip() for t in texts)
            or type(batch_size) is not int
            or batch_size < 1
        ):
            raise ValueError("nonempty texts and positive batch size required")
        encoded = self.tokenizer(texts, add_special_tokens=True, truncation=False)
        if any(len(ids) > self.max_length for ids in encoded["input_ids"]):
            raise ValueError("complete headline exceeds organization model budget")
        results = self.pipeline(texts, batch_size=batch_size)
        return [
            organization_spans(text, entities)
            for text, entities in zip(texts, results, strict=True)
        ]


def organization_sentiments(
    text: str, mentions: tuple[OrganizationMention, ...], predictor: Any
) -> list[dict]:
    """Score candidate surfaces with the existing target-first sentiment model.

    Repeated exact surfaces share one prediction; distinct aliases remain separate.
    No ticker, parent mapping or company status is inferred from the ORG label.
    """
    groups: dict[str, list[OrganizationMention]] = {}
    for mention in mentions:
        if text[mention.start : mention.end] != mention.text:
            raise ValueError("organization surface differs from source text")
        key = " ".join(mention.text.casefold().split())
        groups.setdefault(key, []).append(mention)
    if not groups:
        return []
    targets = [group[0].text for group in groups.values()]
    predictions = predictor.predict_many([text] * len(targets), targets)
    return [
        {
            "target": target,
            "mentions": [asdict(m) for m in group],
            "company_identity_verified": False,
            "sentiment": asdict(prediction),
        }
        for target, group, prediction in zip(
            targets, groups.values(), predictions, strict=True
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--data-dir", type=Path, default=project_root() / "data")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--with-sentiment", action="store_true")
    parser.add_argument(
        "--registry", type=Path, default=project_root() / "configs/companies.json"
    )
    parser.add_argument(
        "--us-directory",
        type=Path,
        help="Local checksum-verified SEC snapshot; no download",
    )
    parser.add_argument(
        "--confirmed-company",
        action="append",
        default=[],
        metavar="NAME",
        help="Exact detected name confirmed as a company; supplies no issuer identity",
    )
    parser.add_argument(
        "--company-aliases",
        type=Path,
        help="Reviewed alias catalog; requires --us-directory",
    )
    parser.add_argument("--company-context", action="store_true")
    parser.add_argument("--allow-parent-rollup", action="store_true")
    parser.add_argument("--score-all-candidates", action="store_true")
    parser.add_argument(
        "--aggregation-strategy", choices=("first", "simple"), default="first"
    )
    args = parser.parse_args()
    if args.company_aliases and not args.us_directory:
        parser.error("--company-aliases requires --us-directory")
    if args.score_all_candidates and not args.with_sentiment:
        parser.error("--score-all-candidates requires --with-sentiment")
    directory = (
        args.model_dir or args.data_dir / "interim/organization-ner-model" / REVISION
    )
    detector = OrganizationDetector.load(
        directory, device=args.device, aggregation_strategy=args.aggregation_strategy
    )
    mentions = detector.detect_many([args.text])[0]
    from finsight.data.us_listed_companies import load_directory
    from finsight.sentiment.companies import load_registry
    from finsight.sentiment.organization_resolution import (
        RESOLVER_VERSION,
        company_mention_sentiments,
        resolve_organizations,
    )

    us_directory = load_directory(args.us_directory) if args.us_directory else None
    from finsight.sentiment.issuer_aliases import load_aliases

    aliases = (
        load_aliases(args.company_aliases, us_directory)
        if args.company_aliases
        else None
    )
    registry = load_registry(args.registry)
    resolutions = resolve_organizations(
        args.text,
        mentions,
        registry,
        company_context=args.company_context,
        allow_parent_rollup=args.allow_parent_rollup,
        us_directory=us_directory,
        confirmed_companies=tuple(args.confirmed_company),
        aliases=aliases,
    )
    result = {
        "detector_version": DETECTOR_VERSION,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "aggregation_strategy": args.aggregation_strategy,
        "candidate_type": "organization_not_verified_company",
        "mentions": [asdict(m) for m in mentions],
        "resolver_version": RESOLVER_VERSION,
        "registry_sha256": registry.sha256,
        "us_directory_sha256": us_directory.snapshot_sha256 if us_directory else None,
        "aliases_sha256": aliases.sha256 if aliases else None,
        "resolution_policy": {
            "company_context": args.company_context,
            "confirmed_companies": args.confirmed_company,
            "allow_parent_rollup": args.allow_parent_rollup,
        },
        "resolutions": [
            {**asdict(r), "identity_status": r.identity_status} for r in resolutions
        ],
    }
    if args.with_sentiment:
        # Company type gates default scoring; issuer linking gates only future
        # grouping/stock mapping. Unknown ORGs still receive no placeholder score.
        field = (
            "candidate_sentiments"
            if args.score_all_candidates
            else "company_sentiments"
        )
        result[field] = []
        result["sentiment_policy"] = (
            "all_candidates_diagnostic"
            if args.score_all_candidates
            else "company_mentions_only"
        )
        eligible = (
            bool(mentions)
            if args.score_all_candidates
            else any(r.company_type == "company" for r in resolutions)
        )
        if eligible:
            from finsight.sentiment.models.target_predictor import (
                TargetSentimentPredictor,
            )

            predictor = TargetSentimentPredictor.load(
                data_dir=args.data_dir, device=args.device
            )
            result[field] = (
                organization_sentiments(args.text, mentions, predictor)
                if args.score_all_candidates
                else company_mention_sentiments(args.text, resolutions, predictor)
            )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
