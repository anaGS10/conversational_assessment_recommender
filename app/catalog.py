from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.config import settings
from app.metadata import infer_assessment_tags, infer_product_kind
from app.schemas import Recommendation

KEY_TO_TEST_TYPE: dict[str, str] = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Simulations": "S",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}


@dataclass(frozen=True)
class CatalogItem:
    entity_id: str
    name: str
    url: str
    description: str
    keys: tuple[str, ...]
    job_levels: tuple[str, ...]
    languages: tuple[str, ...]
    duration: str
    product_kind: str
    assessment_tags: tuple[str, ...]

    @property
    def test_type(self) -> str:
        codes = [KEY_TO_TEST_TYPE[k] for k in self.keys if k in KEY_TO_TEST_TYPE]
        return ",".join(codes) if codes else "K"

    def to_search_text(self) -> str:
        parts = [
            self.name,
            self.description,
            ", ".join(self.keys),
            ", ".join(self.job_levels),
            ", ".join(self.languages),
            self.duration,
            f"kind:{self.product_kind}",
            f"tags:{', '.join(self.assessment_tags)}",
        ]
        return " | ".join(p for p in parts if p)

    def to_context_block(self) -> str:
        return (
            f"ID: {self.entity_id}\n"
            f"Name: {self.name}\n"
            f"URL: {self.url}\n"
            f"Product kind: {self.product_kind}\n"
            f"Assessment tags: {', '.join(self.assessment_tags)}\n"
            f"Test types: {self.test_type}\n"
            f"Keys: {', '.join(self.keys)}\n"
            f"Job levels: {', '.join(self.job_levels)}\n"
            f"Languages: {', '.join(self.languages) or 'N/A'}\n"
            f"Duration: {self.duration or 'N/A'}\n"
            f"Description: {self.description}"
        )


def _normalize_url(url: str) -> str:
    return url.rstrip("/").lower()


def keys_to_test_type(keys: list[str]) -> str:
    codes = [KEY_TO_TEST_TYPE[k] for k in keys if k in KEY_TO_TEST_TYPE]
    return ",".join(codes) if codes else "K"


def _build_catalog_item(row: dict) -> CatalogItem:
    name = row["name"]
    description = row.get("description") or ""
    keys = tuple(row.get("keys") or [])
    return CatalogItem(
        entity_id=str(row["entity_id"]),
        name=name,
        url=row["link"],
        description=description,
        keys=keys,
        job_levels=tuple(row.get("job_levels") or []),
        languages=tuple(row.get("languages") or []),
        duration=row.get("duration") or "",
        product_kind=infer_product_kind(name, description),
        assessment_tags=infer_assessment_tags(name, description, keys),
    )


def load_catalog(path: Path | None = None) -> list[CatalogItem]:
    catalog_path = path or settings.catalog_path
    with catalog_path.open(encoding="utf-8") as f:
        raw = json.load(f)

    items: list[CatalogItem] = []
    for row in raw:
        if row.get("status") != "ok":
            continue
        items.append(_build_catalog_item(row))
    return items


@lru_cache(maxsize=1)
def get_catalog() -> tuple[CatalogItem, ...]:
    return tuple(load_catalog())


@lru_cache(maxsize=1)
def get_catalog_by_id() -> dict[str, CatalogItem]:
    return {item.entity_id: item for item in get_catalog()}


@lru_cache(maxsize=1)
def get_catalog_by_url() -> dict[str, CatalogItem]:
    return {_normalize_url(item.url): item for item in get_catalog()}


@lru_cache(maxsize=1)
def get_catalog_by_name_lower() -> dict[str, CatalogItem]:
    return {item.name.lower(): item for item in get_catalog()}


@lru_cache(maxsize=1)
def get_catalog_names() -> frozenset[str]:
    return frozenset(item.name for item in get_catalog())


@lru_cache(maxsize=1)
def get_catalog_urls() -> frozenset[str]:
    return frozenset(_normalize_url(item.url) for item in get_catalog())


def find_by_name_fragment(fragment: str, limit: int = 5) -> list[CatalogItem]:
    fragment_lower = fragment.lower().strip()
    if not fragment_lower:
        return []

    by_id = get_catalog_by_id()
    exact = get_catalog_by_name_lower().get(fragment_lower)
    if exact:
        return [exact]

    scored: list[tuple[int, CatalogItem]] = []
    for item in by_id.values():
        name_lower = item.name.lower()
        if fragment_lower in name_lower:
            scored.append((len(name_lower), item))
        elif _token_overlap(fragment_lower, name_lower) >= 0.5:
            scored.append((1000 + len(name_lower), item))

    scored.sort(key=lambda x: x[0])
    return [item for _, item in scored[:limit]]


def resolve_entity_ids(
    entity_ids: list[str],
    names: list[str] | None = None,
    max_items: int | None = None,
    allowed_ids: set[str] | None = None,
) -> list[CatalogItem]:
    by_id = get_catalog_by_id()
    resolved: list[CatalogItem] = []
    seen: set[str] = set()

    for entity_id in entity_ids:
        if allowed_ids is not None and str(entity_id) not in allowed_ids:
            continue
        item = by_id.get(str(entity_id))
        if item and item.entity_id not in seen:
            resolved.append(item)
            seen.add(item.entity_id)

    for name in names or []:
        for match in find_by_name_fragment(name, limit=3):
            if allowed_ids is not None and match.entity_id not in allowed_ids:
                continue
            if match.entity_id not in seen:
                resolved.append(match)
                seen.add(match.entity_id)

    limit = max_items or settings.max_recommendations
    return resolved[:limit]


def validate_recommendations(
    entity_ids: list[str],
    names: list[str] | None = None,
    allowed_ids: set[str] | None = None,
) -> list[CatalogItem]:
    return resolve_entity_ids(
        entity_ids,
        names=names,
        max_items=settings.max_recommendations,
        allowed_ids=allowed_ids,
    )


def items_to_recommendations(items: list[CatalogItem]) -> list[Recommendation]:
    return [
        Recommendation(name=item.name, url=item.url, test_type=item.test_type)
        for item in items[: settings.max_recommendations]
    ]


def format_languages_display(languages: tuple[str, ...], max_show: int = 4) -> str:
    if not languages:
        return "—"
    shown = ", ".join(languages[:max_show])
    remaining = len(languages) - max_show
    if remaining > 0:
        shown += f" _(+{remaining} more)_"
    return shown


def format_keys_display(keys: tuple[str, ...]) -> str:
    return ", ".join(keys) if keys else "—"


def format_duration_display(duration: str) -> str:
    return duration if duration else "—"


def format_recommendation_table(items: list[CatalogItem]) -> str:
    """Markdown table matching the sample conversation format."""
    if not items:
        return ""
    lines = [
        "| # | Name | Test Type | Keys | Duration | Languages | URL |",
        "|---|------|-----------|------|----------|-----------|-----|",
    ]
    for index, item in enumerate(items, start=1):
        name = item.name.replace("|", "\\|")
        url = f"<{item.url}>" if item.url else ""
        lines.append(
            "| {index} | {name} | {test_type} | {keys} | {duration} | {languages} | {url} |".format(
                index=index,
                name=name,
                test_type=item.test_type,
                keys=format_keys_display(item.keys),
                duration=format_duration_display(item.duration),
                languages=format_languages_display(item.languages),
                url=url,
            )
        )
    return "\n".join(lines)


def items_for_recommendations(recommendations: list[Recommendation]) -> list[CatalogItem]:
    """Resolve validated recommendations back to catalog items (preserves order)."""
    by_url = get_catalog_by_url()
    items: list[CatalogItem] = []
    seen: set[str] = set()
    for rec in recommendations:
        item = by_url.get(_normalize_url(rec.url))
        if item and item.entity_id not in seen:
            items.append(item)
            seen.add(item.entity_id)
    return items


def final_validate_recommendations(recommendations: list[Recommendation]) -> list[Recommendation]:
    """Last gate: every name and URL must match a single catalog record exactly."""
    by_url = get_catalog_by_url()
    by_name = get_catalog_by_name_lower()
    valid: list[Recommendation] = []
    seen: set[str] = set()

    for rec in recommendations:
        url_key = _normalize_url(rec.url)
        item = by_url.get(url_key) or by_name.get(rec.name.lower())
        if item is None:
            continue
        if item.name != rec.name:
            continue
        if _normalize_url(item.url) != url_key:
            continue
        if item.entity_id in seen:
            continue
        seen.add(item.entity_id)
        valid.append(Recommendation(name=item.name, url=item.url, test_type=item.test_type))

    return valid[: settings.max_recommendations]


def extract_comparison_terms(text: str) -> list[str]:
    patterns = [
        r"difference between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
        r"(?:is|how is)\s+(.+?)\s+different\s+(?:from|than|to)\s+(.+?)(?:\?|$)",
        r"compare\s+(.+?)\s+(?:vs\.?|versus|and|with)\s+(.+?)(?:\?|$)",
        r"what(?:'s| is) the difference between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
    ]
    text_clean = text.strip()
    for pattern in patterns:
        match = re.search(pattern, text_clean, re.IGNORECASE)
        if match:
            return [match.group(1).strip(), match.group(2).strip()]
    return []


def _token_overlap(a: str, b: str) -> float:
    tokens_a = set(re.findall(r"[a-z0-9]+", a))
    tokens_b = set(re.findall(r"[a-z0-9]+", b))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a)
