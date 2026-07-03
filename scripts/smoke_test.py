"""Quick local smoke tests (no LLM required for retrieval/schema)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent import handle_chat  # noqa: E402
from app.catalog import (  # noqa: E402
    final_validate_recommendations,
    format_recommendation_table,
    get_catalog,
    items_to_recommendations,
    validate_recommendations,
)
from app.filters import apply_rule_filter, detect_query_signals  # noqa: E402
from app.retrieval import get_retrieval_index  # noqa: E402
from app.schemas import ChatResponse, Message, Recommendation  # noqa: E402


def test_schema() -> None:
    resp = ChatResponse(
        reply="test",
        recommendations=[
            Recommendation(
                name="Occupational Personality Questionnaire OPQ32r",
                url="https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
                test_type="P",
            )
        ],
        end_of_conversation=False,
    )
    data = resp.model_dump()
    assert "reply" in data and "recommendations" in data and "end_of_conversation" in data
    print("schema ok")


def test_catalog_metadata() -> None:
    catalog = get_catalog()
    assert len(catalog) > 300
    opq = [x for x in catalog if "opq32r" in x.name.lower()]
    assert opq and opq[0].product_kind == "instrument"
    assert "personality" in opq[0].assessment_tags

    gsa_dev = [x for x in catalog if "global skills development" in x.name.lower()]
    assert gsa_dev and gsa_dev[0].product_kind == "report"
    assert "development" in gsa_dev[0].assessment_tags
    print("catalog metadata ok")


def test_final_validation() -> None:
    good = validate_recommendations([], names=["OPQ32r"])
    recs = items_to_recommendations(good)
    assert final_validate_recommendations(recs) == recs

    bad = [Recommendation(name="Fake Assessment", url="https://example.com", test_type="K")]
    assert final_validate_recommendations(bad) == []
    print("final validation ok")


def test_development_filter() -> None:
    index = get_retrieval_index()
    index.ensure_loaded()
    raw = index.search("development gaps training plans re-skill talent audit", top_k=25)
    signals = detect_query_signals(
        "we need to re-skill our sales organization and identify development gaps for training plans"
    )
    filtered = apply_rule_filter(raw, signals, max_pool=10)
    names = " ".join(i.name.lower() for i in filtered)
    assert "global skills" in names or "development" in names
    top_name = filtered[0].name.lower()
    assert "opq32r" not in top_name or "development" in top_name or "skills" in top_name
    print("development filter ok:", filtered[0].name)


def test_retrieval() -> None:
    index = get_retrieval_index()
    index.ensure_loaded()
    results = index.search("senior java developer spring sql", top_k=5)
    names = [r.name.lower() for r in results]
    assert any("java" in n or "spring" in n or "sql" in n for n in names)
    print("retrieval ok:", [r.name for r in results[:3]])


async def test_turn_cap() -> None:
    messages: list[Message] = []
    for i in range(8):
        messages.append(Message(role="user", content=f"User turn {i + 1}"))
        messages.append(Message(role="assistant", content=f"Assistant turn {i + 1}"))
    messages.append(Message(role="user", content="User turn 9"))

    resp = await handle_chat(messages)
    assert resp.recommendations == []
    assert resp.end_of_conversation is True
    print("turn cap ok")


async def test_fallback_agent() -> None:
    resp = await handle_chat([Message(role="user", content="We need a solution for senior leadership.")])
    assert resp.recommendations == []
    assert resp.reply
    print("fallback clarify ok")

    resp2 = await handle_chat(
        [Message(role="user", content="Are we legally required under HIPAA to test all staff?")]
    )
    assert resp2.recommendations == []
    print("fallback refuse ok")


def test_recommendation_table_format() -> None:
    opq = [x for x in get_catalog() if "opq32r" in x.name.lower()][0]
    table = format_recommendation_table([opq])
    assert "| # | Name | Test Type | Keys | Duration | Languages | URL |" in table
    assert opq.url in table
    assert f"<{opq.url}>" in table
    print("recommendation table format ok")


def test_conversation_memory() -> None:
    from app.conversation_state import apply_refine_to_shortlist, extract_prior_recommendations
    from app.schemas import Message, Recommendation

    opq = [x for x in get_catalog() if "opq32r" in x.name.lower()][0]
    verify = [x for x in get_catalog() if x.name == "SHL Verify Interactive G+"][0]
    grad = [x for x in get_catalog() if "graduate scenarios" in x.name.lower()][0]

    messages = [
        Message(role="user", content="graduate trainee scheme"),
        Message(
            role="assistant",
            content="shortlist",
            recommendations=[
                Recommendation(name=opq.name, url=opq.url, test_type=opq.test_type),
                Recommendation(name=verify.name, url=verify.url, test_type=verify.test_type),
                Recommendation(name=grad.name, url=grad.url, test_type=grad.test_type),
            ],
        ),
        Message(role="user", content="Drop the OPQ. Final list: Verify G+ and Graduate Scenarios."),
    ]
    prior = extract_prior_recommendations(messages)
    assert len(prior) == 3
    refined = apply_refine_to_shortlist(prior, messages[-1].content, get_catalog()[:50])
    assert refined is not None
    assert all("opq" not in i.name.lower() for i in refined)
    assert any("verify" in i.name.lower() for i in refined)
    print("conversation memory ok")


def main() -> None:
    test_schema()
    test_catalog_metadata()
    test_final_validation()
    test_recommendation_table_format()
    test_conversation_memory()
    test_development_filter()
    test_retrieval()
    asyncio.run(test_turn_cap())
    asyncio.run(test_fallback_agent())
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
