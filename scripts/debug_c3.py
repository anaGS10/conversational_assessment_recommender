"""Debug C3 contact centre recommendation path."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent import _build_candidate_pool, _conversation_text, _recommend_path
from app.schemas import Message
from app.taxonomy import fallback_match_keywords, score_item_by_keywords
from app.taxonomy_mapper import map_conversation_to_keywords


async def main() -> None:
    msgs = [
        Message(
            role="user",
            content=(
                "We're screening 500 entry-level contact centre agents. Inbound calls, "
                "customer service focus. What should we use?"
            ),
        ),
        Message(role="assistant", content="Before I shape the stack — what language are the calls in?"),
        Message(role="user", content="English."),
        Message(
            role="assistant",
            content="SVAR has four English variants in the catalog: US, UK, Australian, and Indian accent.",
        ),
        Message(role="user", content="US."),
    ]
    conv = _conversation_text(msgs)
    user = "\n".join(m.content for m in msgs if m.role == "user")
    fb = fallback_match_keywords(user)
    llm = await map_conversation_to_keywords(conv, user_text=user)
    print("Fallback keywords:", fb)
    print("LLM keywords:", llm)

    pool = await _build_candidate_pool(msgs, "recommend")
    print("Pool size:", len(pool))
    print("Top 15 pool:")
    for item in pool[:15]:
        score = score_item_by_keywords(item, llm)
        print(f"  [{score:.0f}] {item.name}")

    resp = await _recommend_path(msgs, "recommend")
    print("Final recs:", [r.name for r in resp.recommendations[:10]])


if __name__ == "__main__":
    asyncio.run(main())
