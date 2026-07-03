"""Diagnose LLM connectivity."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import settings
from app.llm import chat_completion, LLMError

async def main():
    print(f"Provider: {settings.llm_provider}")
    print(f"Groq key set: {bool(settings.groq_api_key)}")
    print(f"Gemini key set: {bool(settings.gemini_api_key)}")
    print(f"Groq model: {settings.groq_model}")
    print(f"Gemini model: {settings.gemini_model}")

    try:
        result = await chat_completion(
            "You are a helpful assistant. Reply with JSON only.",
            'Say hello and return {"status": "ok", "model": "test"}'
        )
        print(f"\nLLM SUCCESS: {result}")
    except LLMError as e:
        print(f"\nLLMError: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {type(e).__name__}: {e}")

    # Also test which one the agent actually uses
    from app.agent import handle_chat
    from app.schemas import Message
    resp = await handle_chat([
        Message(role="user", content="Hiring a senior Java developer with Spring and SQL experience. 5+ years, backend focused.")
    ])
    print(f"\nAgent response:")
    print(f"  reply: {resp.reply[:100]}...")
    print(f"  recs: {len(resp.recommendations)}")
    if resp.recommendations:
        for r in resp.recommendations[:3]:
            print(f"    - {r.name}")
    print(f"  end: {resp.end_of_conversation}")

if __name__ == "__main__":
    asyncio.run(main())
