"""Shortlist helpers for conversation continuity (confirm/refine memory)."""

from __future__ import annotations

from app.catalog import CatalogItem
from app.config import settings


def confirm_prior_shortlist(
    prior: list[CatalogItem],
    *,
    hint: str | None,
) -> list[CatalogItem] | None:
    """Return the prior shortlist when the user confirms; otherwise None."""
    if hint == "confirm" and prior:
        return prior[: settings.max_recommendations]
    return None


REPLY_ONLY_PROMPT = """You write brief SHL assessment recommendation intros.

Return JSON only: {{"reply": "2-4 sentence conversational intro"}}

Rules:
- Do NOT include markdown tables, numbered lists, or URLs.
- Mention the role or goal naturally.
- The system appends the product table separately."""


async def generate_reply_intro(
    conversation: str,
    items: list[CatalogItem],
    *,
    chat_completion,
) -> str:
    names = ", ".join(item.name for item in items[:6])
    user_prompt = f"""CONVERSATION:
{conversation}

SELECTED PRODUCTS (for context only):
{names}

Write the intro reply."""
    try:
        result = await chat_completion(REPLY_ONLY_PROMPT, user_prompt, temperature=0.0)
        reply = str(result.get("reply", "")).strip()
        if reply:
            return reply
    except Exception:
        pass
    return f"Based on your needs, here are {len(items)} SHL assessments that may fit."
