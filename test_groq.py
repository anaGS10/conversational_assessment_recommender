import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import settings
from app.llm import chat_completion

async def main():
    settings.llm_provider = "groq"
    print(f"Testing Groq with model: {settings.groq_model}")
    try:
        result = await chat_completion(
            "You are a helpful assistant. Reply with JSON only.",
            'Return {"status": "ok", "provider": "groq"}'
        )
        print(f"SUCCESS: {result}")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")

asyncio.run(main())
