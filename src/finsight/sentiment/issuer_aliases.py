"""Small source-reviewed issuer aliases; never fuzzy matching or implicit roll-up."""

import json
from dataclasses import dataclass
from pathlib import Path

from finsight.data.us_listed_companies import (
    DirectoryMatch,
    USListedDirectory,
    name_key,
)
from finsight.sentiment.artifacts import file_sha256


@dataclass(frozen=True)
class IssuerAlias:
    name: str
    cik: str
    kind: str
    requires_context: bool
    source_urls: tuple[str, ...]


@dataclass(frozen=True)
class AliasCatalog:
    sha256: str
    directory_sha256: str
    entries: tuple[IssuerAlias, ...]

    def match(
        self,
        value: str,
        directory: USListedDirectory,
        *,
        company_context: bool = False,
        allow_parent_rollup: bool = False,
    ) -> DirectoryMatch:
        """Union competing identities; never let an alias override a collision."""
        if (
            directory.snapshot_sha256 != self.directory_sha256
            or type(allow_parent_rollup) is not bool
        ):
            raise ValueError("alias directory or parent policy differs")
        base = directory.match(value, company_context=company_context)
        entries = [
            entry for entry in self.entries if name_key(entry.name) == name_key(value)
        ]
        if not entries:
            return base
        if not company_context and any(entry.requires_context for entry in entries):
            return (
                base
                if base.status == "matched"
                else DirectoryMatch("unresolved", "alias_requires_company_context")
            )
        # Include held canonical names too: their abstention cannot authorize an
        # alias to another issuer. Full names, short names and tickers all count.
        candidates = {issuer.cik for issuer in base.issuers}
        candidates.update(directory.names.get(name_key(value), set()))
        if company_context:
            candidates.update(directory.shorts.get(name_key(value), set()))
            candidates.update(directory.tickers.get(value.strip().upper(), set()))
        candidates.update(entry.cik for entry in entries)
        issuers = tuple(directory.by_cik[cik] for cik in sorted(candidates))
        if len(issuers) != 1:
            return DirectoryMatch(
                "ambiguous",
                "alias_identity_collision",
                issuers,
                company_type_known=all(
                    not issuer.type_review_required for issuer in issuers
                ),
            )
        if base.status == "matched":
            return base
        if (
            any(entry.kind == "subsidiary" for entry in entries)
            and not allow_parent_rollup
        ):
            return DirectoryMatch(
                "unresolved",
                "alias_parent_rollup_required",
                issuers,
                company_type_known=True,
            )
        reason = (
            "reviewed_parent_rollup_alias"
            if any(entry.kind == "subsidiary" for entry in entries)
            else "reviewed_issuer_alias"
        )
        return DirectoryMatch("matched", reason, issuers, company_type_known=True)


def load_aliases(path: Path, directory: USListedDirectory) -> AliasCatalog:
    """Validate the reviewed catalog against the pinned current issuer snapshot."""
    raw = json.loads(path.read_bytes())
    if (
        raw.get("schema_version") != 1
        or not directory.snapshot_sha256
        or raw.get("directory_sha256") != directory.snapshot_sha256
        or not isinstance(raw.get("aliases"), list)
        or not raw["aliases"]
    ):
        raise ValueError("alias schema or pinned directory differs")
    entries = []
    seen = set()
    for row in raw["aliases"]:
        name, cik, kind = row.get("name"), row.get("cik"), row.get("kind")
        urls = row.get("source_urls")
        if (
            not isinstance(name, str)
            or not name_key(name)
            or name.startswith("$")
            or ":" in name
            or not isinstance(cik, str)
            or cik not in directory.by_cik
            or directory.by_cik[cik].type_review_required
            or kind not in {"issuer_name", "subsidiary"}
            or type(row.get("requires_context")) is not bool
            or not isinstance(urls, list)
            or not urls
            or any(
                not isinstance(url, str) or not url.startswith("https://")
                for url in urls
            )
        ):
            raise ValueError("invalid or unsupported reviewed alias")
        key = name_key(name), cik
        if key in seen:
            raise ValueError("duplicate reviewed alias")
        seen.add(key)
        entries.append(
            IssuerAlias(name, cik, kind, row["requires_context"], tuple(urls))
        )
    return AliasCatalog(file_sha256(path), directory.snapshot_sha256, tuple(entries))
