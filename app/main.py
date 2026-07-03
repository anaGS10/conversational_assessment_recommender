import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.agent import handle_chat
from app.config import settings
from app.groups import load_groups
from app.retrieval import get_retrieval_index
from app.schemas import ChatRequest, ChatResponse, HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    index = get_retrieval_index()
    index.ensure_loaded()
    load_groups()
    yield


app = FastAPI(title="SHL Assessment Recommender", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        return await asyncio.wait_for(
            handle_chat(request.messages),
            timeout=settings.chat_request_timeout_seconds,
        )
    except asyncio.TimeoutError:
        return ChatResponse(
            reply=(
                "Sorry, that request took too long to process. "
                "Please try again or start a new conversation."
            ),
            recommendations=[],
            end_of_conversation=False,
        )
