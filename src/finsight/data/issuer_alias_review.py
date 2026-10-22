"""Select a disjoint alias-development packet without reading predictions."""

from finsight.data.us_company_review import select_sample
from finsight.data.us_listed_companies import name_key


def select_fresh_sample(
    categories: dict[str, list[dict]],
    previous: list[dict],
    *,
    per_category: int = 4,
    seed: int = 43,
) -> list[dict]:
    """Exclude prior page IDs and normalized titles, then use the existing ranker."""
    ids = {row["record_id"] for row in previous}
    titles = {name_key(row["headline"]) for row in previous}
    remaining = {
        category: [
            row
            for row in rows
            if f"wikinews:{row['pageid']}" not in ids
            and name_key(row["title"]) not in titles
        ]
        for category, rows in categories.items()
    }
    return select_sample(remaining, per_category=per_category, seed=seed)
