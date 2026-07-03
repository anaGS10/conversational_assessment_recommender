"""Conversation memory helpers — recover prior shortlists and apply deterministic refine."""

from __future__ import annotations

import re

from app.catalog import CatalogItem, find_by_name_fragment, items_for_recommendations
from app.config import settings
from app.filters import extract_refine_terms
from app.schemas import Message, Recommendation


def items_from_assistant_message(msg: Message) -> list[CatalogItem]:
    """Resolve catalog items from one assistant turn (structured field, then URLs in content)."""
    if msg.role != "assistant":
        return []

    if msg.recommendations:
        items = items_for_recommendations(msg.recommendations)
        if items:
            return items

    from app.catalog import get_catalog_by_url

    by_url = get_catalog_by_url()
    found: list[CatalogItem] = []
    seen: set[str] = set()
    for url in re.findall(
        r"https://www\.shl\.com/products/product-catalog/view/[^/\s>]+/",
        msg.content,
    ):
        item = by_url.get(url.rstrip("/").lower())
        if item and item.entity_id not in seen:
            found.append(item)
            seen.add(item.entity_id)
    return found


def extract_prior_recommendations(messages: list[Message]) -> list[CatalogItem]:
    """Most recent assistant shortlist (skips clarify turns with no recommendations)."""
    for msg in reversed(messages):
        if msg.role != "assistant":
            continue
        items = items_from_assistant_message(msg)
        if items:
            return items
    return []


def format_shortlist_context(items: list[CatalogItem]) -> str:
    if not items:
        return "(none)"
    lines = []
    for index, item in enumerate(items, start=1):
        lines.append(f"{index}. {item.name} [id={item.entity_id}] {item.url}")
    return "\n".join(lines)


def _item_matches_drop_term(item: CatalogItem, term: str) -> bool:
    name = item.name.lower()
    term = term.lower().strip()
    aliases = {
        "rest": ("rest", "restful"),
        "restful": ("rest", "restful"),
        "rest api": ("rest", "restful"),
        "opq": ("opq", "personality"),
        "opq32r": ("opq", "personality"),
        "personality": ("opq", "personality"),
        "verify": ("verify",),
        "verify g+": ("verify",),
        "docker": ("docker",),
        "aws": ("aws", "amazon web services"),
        "angular": ("angular",),
    }
    needles = aliases.get(term, (term,))
    return any(needle in name for needle in needles)


def _resolve_add_term(term: str, candidates: list[CatalogItem]) -> list[CatalogItem]:
    term = term.strip()
    if not term:
        return []

    found: list[CatalogItem] = []
    seen: set[str] = set()

    for item in find_by_name_fragment(term, limit=3):
        if item.entity_id not in seen:
            found.append(item)
            seen.add(item.entity_id)

    term_lower = term.lower()
    for item in candidates:
        if item.entity_id in seen:
            continue
        if term_lower in item.name.lower():
            found.append(item)
            seen.add(item.entity_id)

    return found


def apply_refine_to_shortlist(
    prior: list[CatalogItem],
    user_text: str,
    candidates: list[CatalogItem],
) -> list[CatalogItem] | None:
    """Deterministically add/drop items on the prior shortlist. Returns None if not a refine edit."""
    if not prior:
        return None

    drop_terms, add_terms = extract_refine_terms(user_text)
    if not drop_terms and not add_terms:
        return None

    updated = list(prior)

    for term in drop_terms:
        updated = [item for item in updated if not _item_matches_drop_term(item, term)]

    seen = {item.entity_id for item in updated}
    for term in add_terms:
        for item in _resolve_add_term(term, candidates):
            if item.entity_id not in seen:
                updated.append(item)
                seen.add(item.entity_id)

    if not updated:
        return prior[: settings.max_recommendations]

    return updated[: settings.max_recommendations]


def assistant_message_from_response(reply: str, recommendations: list[Recommendation]) -> Message:
    """Build an assistant message that preserves shortlist context for the next turn."""
    return Message(
        role="assistant",
        content=reply,
        recommendations=recommendations or None,
    )
