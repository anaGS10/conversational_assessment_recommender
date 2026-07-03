import asyncio
from app.llm import chat_completion

async def test():
    try:
        result = await chat_completion("Reply with JSON", 'Say hi and return {"test": true}')
        print(f"LLM works: {result}")
    except Exception as e:
        print(f"LLM error: {type(e).__name__}: {e}")

asyncio.run(test())
