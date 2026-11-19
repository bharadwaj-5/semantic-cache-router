"""FastAPI application — the main entry point."""

from __future__ import annotations

import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient

from app.config import (
    CACHE_SIMILARITY_THRESHOLD,
    CORPUS_VERSION,
    ESCALATION_MIN_RETRIEVAL_SCORE,
    LARGE_MODEL,
    LOG_LEVEL,
    SMALL_MODEL,
)
from app.models import (
    CacheEntry,
    IngestRequest,
    QueryRequest,
    QueryResponse,
    RequestLog,
    StatsResponse,
)
from app.embeddings import embed
from app.cache import SemanticCache
from app.router import classify_query, needs_escalation
from app.rag import ingest_documents, rag_pipeline
from app.logger import init_db, log_request, get_stats, get_all_logs

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Shared Qdrant client (in-memory) ────────────────────────
qdrant = QdrantClient(location=":memory:")
cache = SemanticCache(client=qdrant)


# ── Lifespan ─────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    # Auto-ingest sample docs if present
    sample_dir = Path(__file__).resolve().parent.parent / "data" / "documents"
    if sample_dir.exists() and any(sample_dir.iterdir()):
        try:
            count = ingest_documents(str(sample_dir), qdrant)
            logger.info("Auto-ingested %d chunks from sample docs", count)
        except Exception as e:
            logger.warning("Auto-ingest failed: %s", e)
    yield
    # Shutdown — nothing to clean up for in-memory


app = FastAPI(
    title="Semantic Cache Router",
    description=(
        "Cost-aware LLM routing with semantic caching. "
        "Routes simple queries to a small model and complex ones to a large model, "
        "caches answers by query similarity, and proves the savings on a dashboard."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Core endpoint ────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
def answer_query(req: QueryRequest):
    """The main answer endpoint — cache → route → generate → log."""
    t0 = time.time()
    query = req.query

    # 1️⃣  Semantic cache lookup
    q_vec = embed(query)
    hit = cache.lookup(q_vec, corpus_version=CORPUS_VERSION)

    if hit:
        entry = hit["entry"]
        latency_ms = round((time.time() - t0) * 1000, 2)
        sources = entry.get("sources", [])
        resp = QueryResponse(
            query=query,
            answer=entry["answer"],
            sources=sources,
            model_used=entry.get("model_used", "cache"),
            route="cache_hit",
            cache_hit=True,
            latency_ms=latency_ms,
            cost=0.0,
            tokens_in=0,
            tokens_out=0,
            quality_score=entry.get("quality_score", 0.0),
        )
        # Log
        log_request(RequestLog(
            query=query,
            model_used="cache",
            latency_ms=latency_ms,
            cache_hit=True,
            route="cache_hit",
            quality_score=resp.quality_score,
            corpus_version=CORPUS_VERSION,
            answer=resp.answer,
        ))
        return resp

    # 2️⃣  Route the query
    route = classify_query(query)
    model = SMALL_MODEL if route == "simple" else LARGE_MODEL

    # 3️⃣  RAG pipeline
    result = rag_pipeline(query, model=model, client=qdrant)

    # 4️⃣  Escalation check (only for simple-route answers)
    actual_route = route
    if route == "simple" and needs_escalation(
        result["answer"],
        result["scores"],
        min_score=ESCALATION_MIN_RETRIEVAL_SCORE,
    ):
        logger.info("Escalating query to large model: %s", query[:60])
        result = rag_pipeline(query, model=LARGE_MODEL, client=qdrant)
        actual_route = "escalated"

    latency_ms = round((time.time() - t0) * 1000, 2)

    # 5️⃣  Cache the result
    cache.store(
        q_vec,
        CacheEntry(
            query=query,
            answer=result["answer"],
            sources=result["sources"],
            model_used=model if actual_route != "escalated" else LARGE_MODEL,
            corpus_version=CORPUS_VERSION,
        ),
    )

    # 6️⃣  Build response
    resp = QueryResponse(
        query=query,
        answer=result["answer"],
        sources=result["sources"],
        model_used=model if actual_route != "escalated" else LARGE_MODEL,
        route=actual_route,
        cache_hit=False,
        latency_ms=latency_ms,
        cost=result["cost"],
        tokens_in=result["tokens_in"],
        tokens_out=result["tokens_out"],
        quality_score=result["quality_score"],
    )

    # 7️⃣  Log
    log_request(RequestLog(
        query=query,
        model_used=resp.model_used,
        tokens_in=resp.tokens_in,
        tokens_out=resp.tokens_out,
        cost=resp.cost,
        latency_ms=latency_ms,
        cache_hit=False,
        route=actual_route,
        quality_score=resp.quality_score,
        corpus_version=CORPUS_VERSION,
        answer=resp.answer,
    ))

    return resp


# ── Ingest endpoint ──────────────────────────────────────────
@app.post("/ingest")
def ingest(req: IngestRequest):
    """Re-ingest documents and bump corpus version (invalidates cache)."""
    try:
        count = ingest_documents(req.directory, qdrant)
        cache.invalidate()
        return {
            "status": "ok",
            "chunks_ingested": count,
            "corpus_version": CORPUS_VERSION,
            "cache_cleared": True,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── Stats endpoint ───────────────────────────────────────────
@app.get("/stats", response_model=StatsResponse)
def stats():
    """Return aggregate metrics for the dashboard."""
    return get_stats()


# ── Logs endpoint (for the dashboard) ────────────────────────
@app.get("/logs")
def logs():
    """Return raw request logs."""
    return get_all_logs()


# ── Health ───────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "cache_entries": cache.count,
        "corpus_version": CORPUS_VERSION,
        "small_model": SMALL_MODEL,
        "large_model": LARGE_MODEL,
        "cache_threshold": CACHE_SIMILARITY_THRESHOLD,
    }
