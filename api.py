"""Day 5: FastAPI service around the RAG assistant.

Run locally (repo root, PowerShell):
    $env:GEMINI_API_KEY="your-key"
    uvicorn api:app --port 8000
Interactive docs:  http://localhost:8000/docs

Endpoints
    POST /ask      {"question": "...", "mode": "rerank"}  -> cited answer
    GET  /health   liveness + which model and how many chunks are loaded
    GET  /metrics  request counts and p50/p95 latency over the last 500 LLM calls

Settings (environment variables, all optional)
    RATE_LIMIT_PER_MIN  requests per minute per client      (default 10)
    DAILY_LIMIT         total /ask requests per UTC day     (default 300)
    TRUST_PROXY         "1" if behind exactly one reverse proxy (reads X-Forwarded-For)
    LOG_QUESTIONS       "1" to put the question text in logs (off by default for privacy)
    LOG_LEVEL           INFO (default), DEBUG, WARNING ...
"""
import hashlib
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Callable, Literal

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator

from service_utils import Metrics, RateLimiter, get_logger

VERSION = "1.0.0"
log = get_logger()


# ---------- request / response models ----------
class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: Literal["vector", "bm25", "hybrid", "rerank"] | None = None

    @field_validator("question", mode="before")
    @classmethod
    def strip_question(cls, v):
        return v.strip() if isinstance(v, str) else v


class Source(BaseModel):
    n: int
    title: str
    url: str
    updated: str | None = None


class AskResponse(BaseModel):
    request_id: str
    answer: str
    answered: bool
    sources: list[Source]
    best_similarity: float
    retrieve_ms: int
    total_ms: int


# ---------- helpers ----------
def _default_factory():
    """Build the real assistant. Imported lazily so tests do not need torch."""
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not set")
    from generate import RagAssistant

    assistant = RagAssistant()
    assistant.retriever.search("warm up", k=1, mode="rerank")  # load every model before serving
    return assistant


def client_ip(request: Request, trust_proxy: bool) -> str:
    if trust_proxy:
        # With one trusted proxy, the LAST entry is the one the proxy added.
        # Earlier entries can be forged by the client, so they are ignored.
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


# ---------- app ----------
def create_app(
    assistant_factory: Callable[[], object] | None = None,
    *,
    rate_limit_per_min: int | None = None,
    daily_limit: int | None = None,
    trust_proxy: bool | None = None,
    log_questions: bool | None = None,
) -> FastAPI:
    per_min = rate_limit_per_min if rate_limit_per_min is not None else int(os.getenv("RATE_LIMIT_PER_MIN", "10"))
    daily = daily_limit if daily_limit is not None else int(os.getenv("DAILY_LIMIT", "300"))
    trust = trust_proxy if trust_proxy is not None else os.getenv("TRUST_PROXY") == "1"
    log_q = log_questions if log_questions is not None else os.getenv("LOG_QUESTIONS") == "1"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        t0 = time.perf_counter()
        app.state.assistant = (assistant_factory or _default_factory)()
        log.info("startup", extra={"fields": {"version": VERSION, "load_s": round(time.perf_counter() - t0, 1)}})
        yield
        log.info("shutdown")

    app = FastAPI(title="GOV.UK RAG assistant", version=VERSION, lifespan=lifespan)
    app.state.metrics = Metrics()
    app.state.limiter = RateLimiter(per_min, daily)

    @app.get("/")
    def root():
        return {"service": "govuk-rag", "docs": "/docs", "health": "/health"}

    @app.get("/health")
    def health(request: Request):
        assistant = request.app.state.assistant
        chunks = getattr(getattr(assistant, "retriever", None), "chunks", {})
        return {
            "status": "ok",
            "version": VERSION,
            "model": getattr(assistant, "model", None),
            "chunks": len(chunks),
        }

    @app.get("/metrics")
    def metrics(request: Request):
        return request.app.state.metrics.snapshot()

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest, request: Request, response: Response):
        state = request.app.state

        reason = state.limiter.check(client_ip(request, trust))
        if reason:
            log.warning("rate_limited", extra={"fields": {"reason": reason}})
            if reason == "daily":
                raise HTTPException(429, "Daily demo limit reached. Please try again tomorrow.", headers={"Retry-After": "3600"})
            raise HTTPException(429, "Too many requests. Please wait a minute and try again.", headers={"Retry-After": "60"})

        request_id = uuid.uuid4().hex[:12]
        response.headers["X-Request-ID"] = request_id
        base = {
            "request_id": request_id,
            "q_len": len(req.question),
            "q_hash": hashlib.sha1(req.question.lower().encode()).hexdigest()[:10],
            "mode": req.mode or "default",
        }
        if log_q:
            base["question"] = req.question

        try:
            kwargs = {"mode": req.mode} if req.mode else {}
            result = state.assistant.ask(req.question, **kwargs)
        except Exception:  # noqa: BLE001  (never leak provider errors or keys to the client)
            state.metrics.record_error()
            log.exception("ask_failed", extra={"fields": base})
            raise HTTPException(
                503,
                "The assistant is temporarily unavailable. Please try again shortly.",
                headers={"X-Request-ID": request_id},
            )

        state.metrics.record(result["answered"], result["total_ms"], llm_used=bool(result.get("contexts")))
        log.info(
            "ask",
            extra={
                "fields": {
                    **base,
                    "answered": result["answered"],
                    "best_similarity": result["best_similarity"],
                    "n_sources": len(result["sources"]),
                    "retrieve_ms": result["retrieve_ms"],
                    "total_ms": result["total_ms"],
                }
            },
        )
        return AskResponse(
            request_id=request_id,
            answer=result["answer"],
            answered=result["answered"],
            sources=result["sources"],
            best_similarity=result["best_similarity"],
            retrieve_ms=result["retrieve_ms"],
            total_ms=result["total_ms"],
        )

    return app


app = create_app()
