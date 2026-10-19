"""Conservative company mentions from the reviewed registry; no learned NER."""

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from finsight.sentiment.artifacts import project_root
from finsight.sentiment.companies import (
    CompanyRegistry,
    CompanyResolution,
    load_registry,
    resolve_company,
)

DETECTOR_VERSION = "registry-mentions-v1"
LEFT = r"(?<![\w$.:@])"
RIGHT = r"(?!\w|\.\w)"


@dataclass(frozen=True)
class CompanyMention:
    text: str
    start: int
    end: int
    resolution: CompanyResolution


@dataclass(frozen=True)
class MentionDetection:
    resolved: tuple[CompanyMention, ...]
    ambiguous: tuple[CompanyMention, ...]
    registry_version: str
    registry_sha256: str
    detector_version: str = DETECTOR_VERSION


def _name_pattern(name: str) -> str:
    # Match punctuation/whitespace variants already accepted by whole-target
    # resolution, without modifying the headline or its original character spans.
    tokens = re.split(r"[\s,]+", name.rstrip("."))
    body = r"[\s,]+".join(re.escape(token) for token in tokens)
    return body + (r"\.?" if name.endswith(".") else "")


def detect_companies(
    text: str,
    registry: CompanyRegistry,
    *,
    company_context: bool = False,
    allow_parent_rollup: bool = False,
) -> MentionDetection:
    """Leftmost longest matches; ambiguous words never inherit financial context.

    Bare tickers must be uppercase. Cashtags and NASDAQ-prefixed forms are
    case-insensitive. Exact registry names/aliases are case-insensitive. Unknown
    companies are outside this detector's coverage, not neutral-sentiment examples.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("nonempty headline text required")
    if type(company_context) is not bool or type(allow_parent_rollup) is not bool:
        raise ValueError("mention policy flags must be booleans")
    candidates = []
    for company in registry.companies:
        patterns = [(re.escape(ticker), 0) for ticker in company.tickers]
        patterns += [
            (r"(?:\$|NASDAQ\s*:\s*)" + re.escape(ticker), re.IGNORECASE)
            for ticker in company.tickers
        ]
        patterns += [
            (_name_pattern(name), re.IGNORECASE)
            for name in (company.name, *(a.name for a in company.aliases))
        ]
        for body, flags in patterns:
            for match in re.finditer(LEFT + body + RIGHT, text, flags=flags):
                surface = match.group()
                # Do not detach a bare ticker from an unsupported exchange or
                # colon prefix. Reviewed NASDAQ forms match as a complete span.
                if surface in company.tickers and re.search(
                    r"\b[A-Za-z]+\s*:\s*$", text[: match.start()]
                ):
                    continue
                resolution = resolve_company(
                    surface,
                    registry,
                    company_context=company_context,
                    allow_parent_rollup=allow_parent_rollup,
                )
                if resolution.status != "unsupported":
                    candidates.append(
                        CompanyMention(surface, match.start(), match.end(), resolution)
                    )
    # Prefer "Apple Inc." over its nested "Apple" candidate. Repeated mentions
    # elsewhere remain visible; no document-wide alias/coreference inference.
    mentions = []
    end = -1
    for candidate in sorted(candidates, key=lambda m: (m.start, -(m.end - m.start))):
        if candidate.start >= end:
            mentions.append(candidate)
            end = candidate.end
    return MentionDetection(
        tuple(m for m in mentions if m.resolution.status == "resolved"),
        tuple(m for m in mentions if m.resolution.status == "ambiguous"),
        registry.version,
        registry.sha256,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument(
        "--registry", type=Path, default=project_root() / "configs/companies.json"
    )
    parser.add_argument("--company-context", action="store_true")
    parser.add_argument("--allow-parent-rollup", action="store_true")
    args = parser.parse_args()
    detection = detect_companies(
        args.text,
        load_registry(args.registry),
        company_context=args.company_context,
        allow_parent_rollup=args.allow_parent_rollup,
    )
    print(json.dumps(asdict(detection), indent=2))


if __name__ == "__main__":
    main()
