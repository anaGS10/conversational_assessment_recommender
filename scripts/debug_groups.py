"""Quick end-to-end check of the group-based recommendation pipeline."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent import _build_candidate_pool, handle_chat  # noqa: E402
from app.groups import route_query_to_groups  # noqa: E402
from app.schemas import Message  # noqa: E402

CASES = {
    "C2 Rust engineer": "I'm hiring a senior Rust engineer for high-performance networking infrastructure. What assessments should I use?",
    "C3 contact centre": "We're screening 500 entry-level contact centre agents. Inbound calls, customer service focus, English. What should we use?",
    "C4 graduate finance": "Hiring graduate financial analysts — final-year students, no work experience. We need numerical reasoning and a finance knowledge test.",
}


async def main() -> None:
    for label, text in CASES.items():
        msgs = [Message(role="user", content=text)]
        route = await route_query_to_groups(text)
        pool = await _build_candidate_pool(msgs, "recommend")
        resp = await handle_chat(msgs)
        print(f"\n=== {label} ===")
        print("groups:", route["groups"], "| skills:", route["skills"], "| seniority:", route["seniority"])
        print("pool (top 8):", [i.name for i in pool[:8]])
        print("recs:", [r.name for r in resp.recommendations[:10]])
        print("reply:", resp.reply[:220].replace("\n", " "))


if __name__ == "__main__":
    asyncio.run(main())
