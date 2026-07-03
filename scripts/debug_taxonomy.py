"""Quick debug for taxonomy candidate pool."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent import _build_candidate_pool, _conversation_text
from app.schemas import Message
from app.taxonomy import fallback_match_keywords
from app.taxonomy_mapper import map_conversation_to_keywords


async def debug_c7_t2() -> None:
    msgs = [
        Message(
            role="user",
            content=(
                "We're hiring bilingual healthcare admin staff in South Texas — "
                "they handle patient records and need to be assessed in Spanish. "
                "HIPAA compliance is critical. What assessments work?"
            ),
        ),
        Message(role="assistant", content="clarify about language constraints"),
        Message(
            role="user",
            content=(
                "They're functionally bilingual — English fluent for written work. "
                "Go with the hybrid."
            ),
        ),
    ]
    conv = _conversation_text(msgs)
    llm = await map_conversation_to_keywords(conv)
    fb = fallback_match_keywords(conv)
    print("LLM concepts:", llm)
    print("Fallback concepts:", fb)
    pool = await _build_candidate_pool(msgs, "recommend")
    print("Pool size:", len(pool))
    print("Top 15:", [c.name for c in pool[:15]])
    wants = [
        "HIPAA (Security)",
        "Medical Terminology (New)",
        "Microsoft Word 365 - Essentials (New)",
        "Dependability and Safety Instrument (DSI)",
        "Occupational Personality Questionnaire (OPQ32r)",
    ]
    names = [c.name for c in pool]
    for want in wants:
        hit = any(want.lower() in n.lower() for n in names)
        print(f"  {want}: {'YES' if hit else 'NO'}")


if __name__ == "__main__":
    asyncio.run(debug_c7_t2())
