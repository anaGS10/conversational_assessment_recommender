from __future__ import annotations

import json
import re
from typing import Any

import httpx

from app.config import settings


class LLMError(Exception):
    pass


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {429, 500, 502, 503, 504}
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


async def chat_completion(
    system: str,
    user: str,
    temperature: float | None = None,
) -> dict[str, Any]:
    temp = settings.llm_temperature if temperature is None else temperature

    primary = settings.llm_provider
    fallback = "groq" if primary == "gemini" else "gemini"
    runners = {"gemini": _gemini_completion, "groq": _groq_completion}

    try:
        return await runners[primary](system, user, temp)
    except Exception as exc:  # noqa: BLE001
        # Fall back to the other provider when the primary is throttled/unavailable
        # and the fallback is actually configured.
        fallback_key = settings.groq_api_key if fallback == "groq" else settings.gemini_api_key
        if _is_transient(exc) and fallback_key:
            return await runners[fallback](system, user, temp)
        raise


async def _groq_completion(system: str, user: str, temperature: float) -> dict[str, Any]:
    if not settings.groq_api_key:
        raise LLMError("GROQ_API_KEY is not set")

    payload = {
        "model": settings.groq_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=settings.llm_request_timeout_seconds) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    content = data["choices"][0]["message"]["content"]
    return _extract_json(content)


async def _gemini_completion(system: str, user: str, temperature: float) -> dict[str, Any]:
    if not settings.gemini_api_key:
        raise LLMError("GEMINI_API_KEY is not set")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_model}:generateContent?key={settings.gemini_api_key}"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
        },
    }

    async with httpx.AsyncClient(timeout=settings.llm_request_timeout_seconds) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

    content = data["candidates"][0]["content"]["parts"][0]["text"]
    return _extract_json(content)
