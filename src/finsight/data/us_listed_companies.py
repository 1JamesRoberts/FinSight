"""Versioned SEC issuer/ticker snapshot; no historical membership or fuzzy linking."""

import argparse
import json
import os
import re
import unicodedata
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from finsight.data.target_preparation import json_bytes
from finsight.sentiment.artifacts import file_sha256, project_root, sha

SOURCE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
EXCHANGES = {"Nasdaq", "NYSE", "CBOE", "TXSE"}
SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "ltd",
    "limited",
    "plc",
    "nv",
    "co",
    "company",
}


def name_key(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.sub(r"[^\w&]+", " ", text).split())


def short_key(value: str) -> str:
    # SEC conformed names can end in jurisdiction annotations, not brand aliases.
    text = re.sub(r"\s*/[A-Za-z]{2,4}/?\s*$", "", value)
    words = name_key(text).split()
    while words and words[-1] in SUFFIXES:
        words.pop()
    return " ".join(words)


@dataclass(frozen=True)
class Listing:
    exchange: str
    ticker: str


@dataclass(frozen=True)
class Issuer:
    cik: str
    names: tuple[str, ...]
    listings: tuple[Listing, ...]
    type_review_required: bool


@dataclass(frozen=True)
class DirectoryMatch:
    status: Literal["matched", "ambiguous", "unresolved"]
    reason: str
    issuers: tuple[Issuer, ...] = ()
    matched_ticker: str | None = None
    company_type_known: bool = False


class USListedDirectory:
    def __init__(
        self, issuers: tuple[Issuer, ...], *, snapshot_sha256: str | None = None
    ):
        self.issuers = issuers
        self.snapshot_sha256 = snapshot_sha256
        self.by_cik = {i.cik: i for i in issuers}
        self.names: dict[str, set[str]] = defaultdict(set)
        self.shorts: dict[str, set[str]] = defaultdict(set)
        self.tickers: dict[str, set[str]] = defaultdict(set)
        for issuer in issuers:
            for name in issuer.names:
                self.names[name_key(name)].add(issuer.cik)
                if key := short_key(name):
                    self.shorts[key].add(issuer.cik)
            for listing in issuer.listings:
                self.tickers[listing.ticker].add(issuer.cik)

    def match(self, value: str, *, company_context: bool = False) -> DirectoryMatch:
        if (
            not isinstance(value, str)
            or not value.strip()
            or type(company_context) is not bool
        ):
            raise ValueError("nonempty name and explicit boolean context required")
        value = value.strip()
        explicit = re.fullmatch(
            r"(?:\$|NASDAQ\s*:\s*|NYSE\s*:\s*|CBOE\s*:\s*|TXSE\s*:\s*)([A-Z0-9.-]+)",
            value,
            re.IGNORECASE,
        )
        ticker = (
            explicit[1].upper()
            if explicit
            else value
            if re.fullmatch(r"[A-Z][A-Z0-9.-]*", value)
            else None
        )
        candidates = set()
        reason = "unknown_us_listed_name"
        if ticker and ticker in self.tickers:
            candidates = set(self.tickers[ticker])
            reason = "exact_ticker"
            if explicit and not value.startswith("$"):
                exchange = {
                    "NASDAQ": "Nasdaq",
                    "NYSE": "NYSE",
                    "CBOE": "CBOE",
                    "TXSE": "TXSE",
                }[value.split(":")[0].strip().upper()]
                candidates = {
                    c
                    for c in candidates
                    if any(
                        listing.ticker == ticker and listing.exchange == exchange
                        for listing in self.by_cik[c].listings
                    )
                }
            if not explicit and not company_context:
                return DirectoryMatch("unresolved", "bare_ticker_requires_context")
        elif candidates := self.names.get(name_key(value), set()):
            reason = "exact_sec_name"
        elif candidates := self.shorts.get(name_key(value), set()):
            reason = "suffix_normalized_name"
            if not company_context:
                return DirectoryMatch("unresolved", "short_name_requires_context")
        issuers = tuple(self.by_cik[c] for c in sorted(candidates))
        if len(issuers) > 1:
            return DirectoryMatch("ambiguous", "multiple_sec_issuers", issuers)
        if not issuers:
            return DirectoryMatch("unresolved", reason)
        if issuers[0].type_review_required:
            return DirectoryMatch("unresolved", "entity_type_requires_review")
        return DirectoryMatch(
            "matched", reason, issuers, ticker if reason == "exact_ticker" else None
        )


def build_directory(
    raw: dict, *, snapshot_sha256: str | None = None
) -> USListedDirectory:
    if raw.get("fields") != ["cik", "name", "ticker", "exchange"] or not isinstance(
        raw.get("data"), list
    ):
        raise ValueError("SEC directory schema differs")
    names: dict[str, set[str]] = defaultdict(set)
    listings: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in raw["data"]:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError("invalid SEC directory row")
        cik, name, ticker, exchange = row
        if (
            type(cik) is not int
            or not 0 < cik < 10**10
            or not isinstance(name, str)
            or not name.strip()
            or not isinstance(ticker, str)
            or not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]*", ticker)
            or (exchange is not None and not isinstance(exchange, str))
        ):
            raise ValueError("invalid SEC issuer identity or listing")
        if exchange not in EXCHANGES:
            continue
        key = f"{cik:010d}"
        names[key].add(name)
        listings[key].add((exchange, ticker))
    issuers = tuple(
        Issuer(
            c,
            tuple(sorted(names[c])),
            tuple(Listing(*listing) for listing in sorted(listings[c])),
            any(
                re.search(r"\b(?:ETF|FUND|TRUST)\b", n, re.IGNORECASE) for n in names[c]
            ),
        )
        for c in sorted(names)
    )
    return USListedDirectory(issuers, snapshot_sha256=snapshot_sha256)


def load_directory(directory: Path) -> USListedDirectory:
    manifest_path = directory / "manifest.json"
    if file_sha256(manifest_path) != directory.name:
        raise ValueError("directory manifest checksum differs")
    manifest = json.loads(manifest_path.read_bytes())
    if (
        manifest["source_url"] != SOURCE_URL
        or manifest.get("allowed_exchanges") != sorted(EXCHANGES)
        or file_sha256(directory / "source.json") != manifest["source_sha256"]
    ):
        raise ValueError("directory source identity differs")
    return build_directory(
        json.loads((directory / "source.json").read_bytes()),
        snapshot_sha256=directory.name,
    )


def acquire(data: Path, user_agent: str) -> Path:
    if (
        not user_agent.strip()
        or "@" not in user_agent
        or "\n" in user_agent
        or "\r" in user_agent
    ):
        raise ValueError(
            "FINSIGHT_SEC_USER_AGENT must identify FinSight and a real contact email"
        )
    request = urllib.request.Request(
        SOURCE_URL, headers={"User-Agent": user_agent, "Accept": "application/json"}
    )
    # One bulk request, no crawling/retry loop or browser impersonation.
    with urllib.request.urlopen(request, timeout=45) as response:
        body = response.read(10_000_001)
    if len(body) > 10_000_000:
        raise ValueError("unexpected directory size")
    return save_snapshot(body, data, retrieved_at=datetime.now(UTC).isoformat())


def save_snapshot(body: bytes, data: Path, *, retrieved_at: str) -> Path:
    """Freeze raw bytes and the extraction policy; never store contact details."""
    directory = build_directory(json.loads(body))
    if not directory.issuers:
        raise ValueError("empty US-listed directory")
    manifest = {
        "schema_version": 1,
        "source_url": SOURCE_URL,
        "source_sha256": sha(body),
        "size_bytes": len(body),
        "retrieved_at_utc": retrieved_at,
        "allowed_exchanges": sorted(EXCHANGES),
        "issuer_count": len(directory.issuers),
        "listing_count": sum(len(i.listings) for i in directory.issuers),
        "type_review_required_issuers": sum(
            i.type_review_required for i in directory.issuers
        ),
        "user_agent_contact_configured": True,
        "historical_membership": False,
    }
    contents = json_bytes(manifest)
    path = data / "interim/us-listed-directory" / sha(contents)
    path.mkdir(parents=True, exist_ok=True)
    (path / "source.json").write_bytes(body)
    (path / "manifest.json").write_bytes(contents)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["acquire", "lookup"])
    parser.add_argument("--data-dir", type=Path, default=project_root() / "data")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--company-context", action="store_true")
    args = parser.parse_args()
    if args.action == "acquire":
        print(acquire(args.data_dir, os.environ.get("FINSIGHT_SEC_USER_AGENT", "")))
        return
    if not args.directory or not args.name:
        parser.error("--directory and --name required")
    from dataclasses import asdict

    print(
        json.dumps(
            asdict(
                load_directory(args.directory).match(
                    args.name, company_context=args.company_context
                )
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
