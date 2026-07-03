"""Replay sample conversations against the agent and compare to expected traces."""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent import handle_chat  # noqa: E402
from app.schemas import Message  # noqa: E402

SAMPLE_DIR = ROOT / "sample_conversations" / "GenAI_SampleConversations"


def parse_user_turns(content: str) -> list[str]:
    turns: list[str] = []
    blocks = re.split(r"### Turn \d+", content)[1:]
    for block in blocks:
        match = re.search(r"\*\*User\*\*\s*\n\s*\n(.*?)\n\n\*\*Agent\*\*", block, re.DOTALL)
        if not match:
            continue
        raw = match.group(1).strip()
        lines: list[str] = []
        for line in raw.split("\n"):
            if line.startswith(">"):
                lines.append(line[1:].strip())
            elif lines:
                lines.append(line.strip())
        turns.append("\n".join(lines).strip())
    return turns


def parse_turn_expectations(content: str) -> list[dict]:
    blocks = re.split(r"### Turn \d+", content)[1:]
    expectations: list[dict] = []
    for block in blocks:
        no_recs = "recommendations: null" in block.lower()
        end = "end_of_conversation`: **true**" in block or "end_of_conversation`: true" in block.lower()
        urls = re.findall(r"https://www\.shl\.com/products/product-catalog/view/[^/\s>]+/", block)
        expectations.append({"no_recommendations": no_recs, "end_of_conversation": end, "urls_in_turn": urls})
    return expectations


def parse_final_expected_urls(content: str) -> list[str]:
    expectations = parse_turn_expectations(content)
    for exp in reversed(expectations):
        if exp["urls_in_turn"]:
            return exp["urls_in_turn"]
    return []


def recall_at_k(predicted_urls: list[str], relevant_urls: list[str], k: int = 10) -> float:
    if not relevant_urls:
        return 1.0
    top = predicted_urls[:k]
    hits = sum(1 for url in relevant_urls if url in top)
    return hits / len(relevant_urls)


async def replay_file(path: Path) -> dict:
    content = path.read_text(encoding="utf-8")
    user_turns = parse_user_turns(content)
    expectations = parse_turn_expectations(content)
    expected_final_urls = parse_final_expected_urls(content)

    messages: list[Message] = []
    turn_results: list[dict] = []
    issues: list[str] = []
    last_urls: list[str] = []
    last_end = False

    for i, user_text in enumerate(user_turns):
        messages.append(Message(role="user", content=user_text))
        resp = await handle_chat(messages)
        messages.append(
            Message(
                role="assistant",
                content=resp.reply,
                recommendations=resp.recommendations or None,
            )
        )

        exp = expectations[i] if i < len(expectations) else {}
        rec_count = len(resp.recommendations)
        last_urls = [r.url for r in resp.recommendations]
        last_end = resp.end_of_conversation

        if exp.get("no_recommendations") and rec_count > 0:
            issues.append(f"T{i + 1}: expected empty recommendations, got {rec_count}")
        if not exp.get("no_recommendations") and exp.get("urls_in_turn") and rec_count == 0:
            issues.append(f"T{i + 1}: expected recommendations, got 0")

        turn_results.append(
            {
                "turn": i + 1,
                "user_preview": user_text[:80].replace("\n", " "),
                "rec_count": rec_count,
                "end": resp.end_of_conversation,
                "reply_preview": resp.reply[:120].replace("\n", " "),
                "names": [r.name for r in resp.recommendations[:5]],
            }
        )

    recall = recall_at_k(last_urls, expected_final_urls)

    return {
        "file": path.name,
        "turns": len(user_turns),
        "turn_results": turn_results,
        "issues": issues,
        "expected_final_count": len(expected_final_urls),
        "actual_final_count": len(last_urls),
        "recall_at_10": recall,
        "expected_end": expectations[-1]["end_of_conversation"] if expectations else False,
        "actual_end": last_end,
        "missing_final": [u for u in expected_final_urls if u not in last_urls],
        "extra_final": [u for u in last_urls if u not in expected_final_urls],
    }


async def main() -> None:
    files = sorted(SAMPLE_DIR.glob("C*.md"), key=lambda p: int(re.search(r"C(\d+)", p.name).group(1)))
    print(f"Replaying {len(files)} conversations via {ROOT}\n")
    print("=" * 80)

    total_recall = 0.0
    for path in files:
        result = await replay_file(path)
        total_recall += result["recall_at_10"]

        print(f"\n## {result['file']} ({result['turns']} turns)")
        for tr in result["turn_results"]:
            recs = f"{tr['rec_count']} recs" if tr["rec_count"] else "no recs"
            end = " END" if tr["end"] else ""
            print(f"  T{tr['turn']}: [{recs}{end}] {tr['user_preview'][:60]}...")
            if tr["names"]:
                for n in tr["names"]:
                    print(f"         - {n[:70]}")
        print(f"  Recall@10: {result['recall_at_10']:.0%} ({result['actual_final_count']}/{result['expected_final_count']} expected items matched)")
        if result["missing_final"]:
            print(f"  Missing: {len(result['missing_final'])} expected URL(s)")
            for u in result["missing_final"][:3]:
                print(f"    - {u}")
        if result["issues"]:
            print(f"  Issues: {'; '.join(result['issues'])}")
        if result["expected_end"] != result["actual_end"]:
            print(f"  end_of_conversation: expected {result['expected_end']}, got {result['actual_end']}")

    mean_recall = total_recall / len(files) if files else 0
    print("\n" + "=" * 80)
    print(f"Mean Recall@10 across {len(files)} conversations: {mean_recall:.0%}")


if __name__ == "__main__":
    asyncio.run(main())
