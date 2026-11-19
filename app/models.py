"""Pydantic models shared across the application."""

from __future__ import annotations

from pydantic import BaseModel, Field
from datetime import datetime, timezone


# ── API ──────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    """Incoming user query."""
    query: str = Field(..., min_length=1, description="User question")
    user_id: str = Field(default="default", description="Tenant / user identifier")


class SourceChunk(BaseModel):
    """A retrieved document chunk returned as a source."""
    text: str
    document: str
    score: float


class QueryResponse(BaseModel):
    """Full response returned to the caller."""
    query: str
    answer: str
    sources: list[SourceChunk] = []
    model_used: str
    route: str              # "simple" | "complex" | "escalated"
    cache_hit: bool
    latency_ms: float
    cost: float
    tokens_in: int
    tokens_out: int
    quality_score: float    # 0‑1 heuristic based on retrieval scores


# ── Internal ─────────────────────────────────────────────────
class CacheEntry(BaseModel):
    """Payload stored in the Qdrant cache collection."""
    query: str
    answer: str
    sources: list[SourceChunk] = []
    model_used: str
    corpus_version: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RequestLog(BaseModel):
    """One row in the SQLite request_logs table."""
    id: int | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    query: str
    model_used: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    cache_hit: bool = False
    route: str = ""
    quality_score: float = 0.0
    corpus_version: str = ""
    answer: str = ""


class IngestRequest(BaseModel):
    """Request body for the /ingest endpoint."""
    directory: str = Field(
        default="data/documents",
        description="Path to a folder of .md / .txt files to ingest",
    )


class StatsResponse(BaseModel):
    """Aggregate statistics returned by /stats."""
    total_requests: int
    cache_hits: int
    cache_hit_rate: float
    avg_cost: float
    total_cost: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    route_split: dict[str, int]
