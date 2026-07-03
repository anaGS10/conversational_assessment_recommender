from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    # Optional: prior assistant shortlist (echo back from last ChatResponse for reliable refine/compare).
    recommendations: list[Recommendation] | None = None


class ChatRequest(BaseModel):
    messages: list[Message]


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
