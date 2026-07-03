"""Debug C2 Rust engineer recommendation path."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent import _build_candidate_pool, _conversation_text
from app.filters import detect_query_signals
from app.schemas import Message
from app.taxonomy import fallback_match_keywords, score_item_by_keywords
from app.taxonomy_mapper import map_conversation_to_keywords


async def main() -> None:
    msgs = [
        Message(
            role="user",
            content=(
                "I'm hiring a senior Rust engineer for high-performance networking "
                "infrastructure. What assessments should I use?"
            ),
        ),
        Message(
            role="assistant",
            content=(
                "SHL's catalog doesn't currently include a Rust-specific knowledge test. "
                "The closest fit for a senior IC is Smart Interview Live Coding."
            ),
        ),
        Message(
            role="user",
            content="Yes, go ahead. Should I also add a cognitive test for this level?",
        ),
    ]
    conv = _conversation_text(msgs)
    user = "\n".join(m.content for m in msgs if m.role == "user")
    fb = fallback_match_keywords(user)
    llm = await map_conversation_to_keywords(conv, user_text=user)
    sig = detect_query_signals(conv)
    print("Fallback keywords:", fb)
    print("LLM keywords:", llm)
    print("wants_development:", sig.wants_development)
    print("wants_knowledge:", sig.wants_knowledge)
    print("wants_ability:", sig.wants_ability)

    pool = await _build_candidate_pool(msgs, "recommend")
    print("Pool size:", len(pool))
    print("Top 15 pool (with keyword scores):")
    for item in pool[:15]:
        score = score_item_by_keywords(item, llm)
        print(f"  [{score:.0f}] {item.name}")


if __name__ == "__main__":
    asyncio.run(main())
