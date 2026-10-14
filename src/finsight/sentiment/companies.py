"""Versioned identities for explicit company targets, without free-text NER."""

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

AliasKind = Literal["name", "subsidiary"]


@dataclass(frozen=True)
class CompanyAlias:
    name: str
    kind: AliasKind
    requires_context: bool


@dataclass(frozen=True)
class Company:
    company_id: str
    primary_ticker: str
    name: str
    other_tickers: tuple[str, ...]
    aliases: tuple[CompanyAlias, ...]
    sources: tuple[str, ...]

    @property
    def tickers(self) -> tuple[str, ...]:
        return (self.primary_ticker, *self.other_tickers)


@dataclass(frozen=True)
class CompanyRegistry:
    version: str
    reviewed_on: str
    sha256: str
    companies: tuple[Company, ...]


@dataclass(frozen=True)
class CompanyResolution:
    input_text: str
    status: Literal["resolved", "ambiguous", "unsupported"]
    company: Company | None
    candidates: tuple[Company, ...]
    reason: str
    alias_kind: Literal["ticker", "name", "subsidiary"] | None
    matched_ticker: str | None
    registry_version: str
    registry_sha256: str


def _identity(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("target and alias names must be strings")
    text = unicodedata.normalize("NFKC", value).casefold().replace(",", " ")
    return " ".join(text.split()).rstrip(".")


def _entries(company: Company) -> tuple[CompanyAlias, ...]:
    return (CompanyAlias(company.name, "name", False), *company.aliases)


def _keys(value: object, expected: set[str]) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("registry fields differ from the supported schema")
    return value


def load_registry(path: Path) -> CompanyRegistry:
    """Read explicitly; reject collisions and unknown rules before resolving targets."""
    body = path.read_bytes()
    raw = _keys(
        json.loads(body),
        {
            "schema_version",
            "version",
            "reviewed_on",
            "companies",
        },
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or not isinstance(raw["version"], str)
        or not raw["version"].strip()
        or not isinstance(raw["reviewed_on"], str)
        or not isinstance(raw["companies"], list)
        or not raw["companies"]
    ):
        raise ValueError("invalid registry metadata")
    date.fromisoformat(raw["reviewed_on"])
    companies = []
    ids: set[str] = set()
    owners: dict[str, str] = {}
    for entry in raw["companies"]:
        row = _keys(
            entry,
            {
                "company_id",
                "primary_ticker",
                "name",
                "other_tickers",
                "aliases",
                "sources",
            },
        )
        identifier = row["company_id"]
        if (
            not isinstance(identifier, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]*", identifier)
            or identifier in ids
        ):
            raise ValueError("company identities must be unique slugs")
        ids.add(identifier)
        if (
            not isinstance(row["other_tickers"], list)
            or not isinstance(row["aliases"], list)
            or not isinstance(row["sources"], list)
            or not row["sources"]
            or any(
                not isinstance(s, str) or not s.startswith("https://")
                for s in row["sources"]
            )
        ):
            raise ValueError("invalid ticker, alias or source collection")
        tickers = [row["primary_ticker"], *row["other_tickers"]]
        if any(
            not isinstance(t, str) or not re.fullmatch(r"[A-Z]{1,5}", t)
            for t in tickers
        ) or len(set(tickers)) != len(tickers):
            raise ValueError("invalid or duplicate ticker")
        if not _identity(row["name"]):
            raise ValueError("company name must be nonempty")
        aliases = []
        for item in row["aliases"]:
            alias = _keys(item, {"name", "kind", "requires_context"})
            if (
                not _identity(alias["name"])
                or alias["kind"] not in ("name", "subsidiary")
                or type(alias["requires_context"]) is not bool
            ):
                raise ValueError("invalid alias rule")
            aliases.append(CompanyAlias(**alias))
        company = Company(
            identifier,
            row["primary_ticker"],
            row["name"],
            tuple(row["other_tickers"]),
            tuple(aliases),
            tuple(row["sources"]),
        )
        keys = [_identity(t) for t in company.tickers]
        keys += [_identity(a.name) for a in _entries(company)]
        for key in keys:
            if key in owners:
                raise ValueError("duplicate or colliding ticker/name/alias")
            owners[key] = identifier
        companies.append(company)
    return CompanyRegistry(
        raw["version"],
        raw["reviewed_on"],
        hashlib.sha256(body).hexdigest(),
        tuple(companies),
    )


def resolve_company(
    value: str,
    registry: CompanyRegistry,
    *,
    company_context: bool = False,
    allow_parent_rollup: bool = False,
) -> CompanyResolution:
    """Resolve a whole target string; ambiguity never silently becomes a company.

    `company_context` asserts prior evidence that a name denotes a company.
    `allow_parent_rollup` explicitly permits the reviewed subsidiary grouping.
    Neither option extracts entities from headlines or infers a sentiment label.
    """
    key = _identity(value)
    if type(company_context) is not bool or type(allow_parent_rollup) is not bool:
        raise ValueError("resolution policy flags must be booleans")
    ticker = re.fullmatch(r"(?:\$|nasdaq\s*:\s*)?([a-z]{1,5})", key)
    for company in registry.companies:
        if ticker and ticker[1].upper() in company.tickers:
            return CompanyResolution(
                value,
                "resolved",
                company,
                (company,),
                "exact_ticker",
                "ticker",
                ticker[1].upper(),
                registry.version,
                registry.sha256,
            )
    for company in registry.companies:
        for alias in _entries(company):
            if key != _identity(alias.name):
                continue
            reason = (
                "parent_rollup_required"
                if alias.kind == "subsidiary" and not allow_parent_rollup
                else "company_context_required"
                if alias.requires_context and not company_context
                else "exact_alias"
            )
            ambiguous = reason != "exact_alias"
            return CompanyResolution(
                value,
                "ambiguous" if ambiguous else "resolved",
                None if ambiguous else company,
                (company,),
                reason,
                alias.kind,
                None,
                registry.version,
                registry.sha256,
            )
    return CompanyResolution(
        value,
        "unsupported",
        None,
        (),
        "unknown_target",
        None,
        None,
        registry.version,
        registry.sha256,
    )
