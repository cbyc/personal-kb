"""REST API entry point for the Personal Knowledge Base."""

import argparse
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.agents.orchestrator import OrchestratorAgent
from src.config import get_settings
from src.memory import ConversationMemory
from src.models import QueryResult
from src.pipeline import build_pipeline
from src.tracing import setup_tracing

logger = logging.getLogger(__name__)

agent: OrchestratorAgent
memory: ConversationMemory


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


class ErrorResponse(BaseModel):
    error: str
    detail: str


class HealthResponse(BaseModel):
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Personal KB - API Server")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Clear all indexed data and reindex from scratch before starting.",
    )
    return parser.parse_args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, memory
    settings = get_settings()
    setup_tracing()

    logger.info("Loading knowledge base...")
    agent = build_pipeline(settings, reindex=getattr(app.state, "reindex", False))
    memory = ConversationMemory(max_turns=settings.conversation_history_length)
    logger.info("Knowledge base ready.")

    yield


app = FastAPI(title="Personal KB API", lifespan=lifespan)


def _configure_cors():
    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )


_configure_cors()


@app.post(
    "/api/v1/query",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def query(request: QueryRequest):
    settings = get_settings()

    if len(request.question) > settings.max_query_length:
        return JSONResponse(
            status_code=400,
            content={
                "error": "bad_request",
                "detail": f"Question exceeds maximum length of {settings.max_query_length} characters.",
            },
        )

    result: QueryResult = await agent.ask_async(
        request.question, message_history=memory.get_history()
    )

    memory.add_turn(request.question, result.answer)
    return QueryResponse(answer=result.answer, sources=result.sources)


@app.get("/api/v1/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")


if __name__ == "__main__":
    args = parse_args()
    app.state.reindex = args.reindex

    settings = get_settings()
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
